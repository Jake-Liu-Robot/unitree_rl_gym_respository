"""MuJoCo wind model verification script.

Verifies that MuJoCoWindModel (pure numpy, single-env) produces statistics
consistent with the Isaac Gym wind_model.py reference (eval_wind_model_v3.py).

No Isaac Gym, legged_gym, mujoco, or torch imports.
Only: numpy, math, os, sys, argparse.

Usage:
    cd /workspace/unitree_rl_gym
    python legged_gym/g1_wind_test/eval_wind_model_mujoco.py --seed 42
"""

import sys
import os
import math
import argparse
import numpy as np

# Import MuJoCoWindModel and WIND_CFG from sibling script (no legged_gym needed)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from eval_wind_robustness_mujoco import MuJoCoWindModel, WIND_CFG  # noqa: E402

# ============================================================
# CLI
# ============================================================

parser = argparse.ArgumentParser()
parser.add_argument("--seed",    type=int,  default=42)
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()

PASS_FAIL = []   # list of (part_label, name, passed, got, expected, tol_desc)


def record(part, name, got, expected, tol, tol_desc=""):
    passed = abs(got - expected) <= tol
    PASS_FAIL.append((part, name, passed, got, expected, tol_desc))
    return passed


def section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# Part 1: Wind velocity statistics per curriculum level
# ============================================================
section("PART 1: WIND VELOCITY STATISTICS PER CURRICULUM LEVEL")

N_ENVS   = 1024
N_STEPS  = 4000    # 20s at dt=0.005
DT       = 0.005

# Reference midpoints from curriculum_levels (Isaac Gym uses uniform[lo, hi])
ref_mean = {
    0: 0.0,   # [0, 0]
    1: 2.0,   # [1, 3]
    2: 3.5,   # [2, 5]
    3: 6.0,   # [4, 8]
    4: 9.5,   # [7, 12]
    5: 14.0,  # [10, 18]
}

for level in range(6):
    rng = np.random.default_rng(args.seed + level * 17)
    speeds_all = []

    for env_i in range(N_ENVS):
        wm = MuJoCoWindModel(WIND_CFG, level, np.random.default_rng(args.seed + level * 17 + env_i))
        wm.reset_episode()
        ep_speeds = []
        for _ in range(N_STEPS):
            vel = wm.step(DT)
            ep_speeds.append(math.sqrt(vel[0]**2 + vel[1]**2 + vel[2]**2))
        speeds_all.append(np.mean(ep_speeds))

    mean_speed = float(np.mean(speeds_all))
    max_speed  = float(np.max(speeds_all))
    clamp      = WIND_CFG["speed_clamp_per_level"][level]

    print(f"  Level {level}: mean_speed={mean_speed:5.2f} m/s  "
          f"max_episode_mean={max_speed:.2f} m/s  clamp={clamp:.0f} m/s")

    if level == 0:
        p = record("P1", f"L{level}_zero_wind", mean_speed, 0.0, 0.1,
                   "abs <= 0.1 (L0 must be exactly zero wind)")
    else:
        lo, hi = WIND_CFG["curriculum_levels"][level]
        midpoint = (lo + hi) / 2.0
        p = record("P1", f"L{level}_mean_speed", mean_speed, midpoint, midpoint * 0.5,
                   f"within 50% of midpoint {midpoint:.1f} m/s")

    if args.verbose:
        status = "PASS" if p else "FAIL"
        print(f"    -> {status}")

# ============================================================
# Part 2: OU process autocorrelation
# ============================================================
section("PART 2: OU PROCESS — STATIONARITY AND AUTOCORRELATION")

N_ENVS2  = 256
N_STEPS2 = 8000   # 40s at dt=0.005
LAG      = 200    # 1s at dt=0.005

# Theoretical OU autocorr at lag τ: exp(-theta * tau)
# For speed: theta=0.5 (midpoint of [0.2,1.0] but use default 0.5)
# For angle: theta_dir=0.2
# However since theta is randomized, we use population avg.
# Expected at 1s lag (from eval_wind_model_v3.py comment):
#   speed ~ exp(-0.5) = 0.607
#   angle ~ exp(-0.2) = 0.819
EXP_SPEED_CORR = math.exp(-0.5)   # 0.607
EXP_ANGLE_CORR = math.exp(-0.2)   # 0.819

print(f"  Expected: speed_autocorr(1s) ~ exp(-0.5) = {EXP_SPEED_CORR:.3f}")
print(f"  Expected: angle_autocorr(1s) ~ exp(-0.2) = {EXP_ANGLE_CORR:.3f}")

# Collect OU time-series for all envs, then compute per-env autocorrelation
ou_speed_series = np.zeros((N_ENVS2, N_STEPS2))
ou_angle_series = np.zeros((N_ENVS2, N_STEPS2))

for env_i in range(N_ENVS2):
    wm = MuJoCoWindModel(WIND_CFG, 3, np.random.default_rng(args.seed + 1000 + env_i))
    wm.reset_episode()
    for t in range(N_STEPS2):
        wm.step(DT)
        ou_speed_series[env_i, t] = wm.ou_speed
        ou_angle_series[env_i, t] = wm.ou_angle

# Check stationarity: compare stats at t=0.5s, 5s, 20s, 40s
for t_idx, label in [(100, "t=0.5s"), (1000, "t=5s"), (4000, "t=20s"), (7999, "t=40s")]:
    s_mean = ou_speed_series[:, t_idx].mean()
    s_std  = ou_speed_series[:, t_idx].std()
    a_mean = ou_angle_series[:, t_idx].mean()
    a_std  = ou_angle_series[:, t_idx].std()
    print(f"  {label}: speed_OU mean={s_mean:.4f} std={s_std:.4f} | "
          f"angle_OU mean={a_mean:.4f} std={a_std:.4f} "
          f"({a_std * 180 / math.pi:.1f} deg)")

# Compute autocorrelation at 1s lag using samples from t=4000 (stationary regime)
t0 = 4000
t1 = t0 + LAG
speed_corrs = []
angle_corrs = []
for env_i in range(N_ENVS2):
    s0 = ou_speed_series[env_i, t0]
    s1 = ou_speed_series[env_i, t1]
    a0 = ou_angle_series[env_i, t0]
    a1 = ou_angle_series[env_i, t1]
    speed_corrs.append((s0, s1))
    angle_corrs.append((a0, a1))

speed_corrs = np.array(speed_corrs)
angle_corrs = np.array(angle_corrs)

if speed_corrs[:, 0].std() > 1e-8 and speed_corrs[:, 1].std() > 1e-8:
    speed_corr = float(np.corrcoef(speed_corrs[:, 0], speed_corrs[:, 1])[0, 1])
else:
    speed_corr = 0.0

if angle_corrs[:, 0].std() > 1e-8 and angle_corrs[:, 1].std() > 1e-8:
    angle_corr = float(np.corrcoef(angle_corrs[:, 0], angle_corrs[:, 1])[0, 1])
else:
    angle_corr = 0.0

print(f"  Autocorrelation at 1s lag: speed={speed_corr:.3f}, angle={angle_corr:.3f}")
print(f"  Expected: speed~exp(-0.5)={EXP_SPEED_CORR:.3f}, angle~exp(-0.2)={EXP_ANGLE_CORR:.3f}")

record("P2", "ou_speed_autocorr_1s", speed_corr, EXP_SPEED_CORR, 0.15,
       "within 0.15 of exp(-0.5)=0.607")
record("P2", "ou_angle_autocorr_1s", angle_corr, EXP_ANGLE_CORR, 0.15,
       "within 0.15 of exp(-0.2)=0.819")

# ============================================================
# Part 3: Gust event statistics
# ============================================================
section("PART 3: GUST EVENT STATISTICS")

N_ENVS3   = 1024
N_STEPS3  = 20000   # 100s at dt=0.005

for level in [0, 2, 5]:
    gust_starts        = 0
    gust_active_steps  = 0

    for env_i in range(N_ENVS3):
        wm = MuJoCoWindModel(WIND_CFG, level, np.random.default_rng(args.seed + 2000 + level * 100 + env_i))
        wm.reset_episode()
        prev_active = False
        for t in range(N_STEPS3):
            wm.step(DT)
            now_active = wm.gust_active
            if now_active and not prev_active:
                gust_starts += 1
            if now_active:
                gust_active_steps += 1
            prev_active = now_active

    gust_rate  = gust_starts / (N_ENVS3 * N_STEPS3 * DT)
    duty_cycle = gust_active_steps / (N_ENVS3 * N_STEPS3)
    expected_prob = WIND_CFG["gust_prob"] * (1.0 + WIND_CFG["gust_prob_scale_per_level"] * level)
    print(f"  Level {level}: gust_rate={gust_rate:.3f}/s  "
          f"(expected~{expected_prob:.3f}/s)  duty_cycle={duty_cycle * 100:.1f}%")

    record("P3", f"L{level}_gust_rate", gust_rate, expected_prob, expected_prob * 0.4,
           f"within 40% of expected {expected_prob:.3f}/s")

# ============================================================
# Part 4: Analytical checks — P1 (directional area), P2 (height), P3 (CoP torque)
# ============================================================
section("PART 4: ANALYTICAL CHECKS — P1 / P2 / P3")

print()
print("  P1: Direction-dependent projected area (ellipsoidal model)")
A_front = WIND_CFG["frontal_area_front"]
A_side  = WIND_CFG["frontal_area_side"]
A_top   = WIND_CFG["frontal_area_top"]

ref_P1 = {
    0:   A_front,  # cos(0)=1, sin(0)=0
    30:  math.sqrt((A_front * math.cos(math.radians(30)))**2 +
                   (A_side  * math.sin(math.radians(30)))**2),
    45:  math.sqrt((A_front * math.cos(math.radians(45)))**2 +
                   (A_side  * math.sin(math.radians(45)))**2),
    60:  math.sqrt((A_front * math.cos(math.radians(60)))**2 +
                   (A_side  * math.sin(math.radians(60)))**2),
    90:  A_side,   # cos(90)=0, sin(90)=1
}

for angle_deg, expected_A in ref_P1.items():
    theta   = math.radians(angle_deg)
    dx      = abs(math.cos(theta))
    dy      = abs(math.sin(theta))
    dz      = 0.0
    A_calc  = math.sqrt((A_front * dx)**2 + (A_side * dy)**2 + (A_top * dz)**2)
    ratio   = A_calc / A_front * 100
    ok      = abs(A_calc - expected_A) < 1e-6
    flag    = "" if ok else " <-- MISMATCH"
    print(f"    angle={angle_deg:3d}°: A_eff={A_calc:.4f} m² ({ratio:.0f}% of frontal)"
          f"  expected={expected_A:.4f}{flag}")

# Key check: 45°
A_45_calc = math.sqrt((A_front * math.cos(math.radians(45)))**2 +
                      (A_side  * math.sin(math.radians(45)))**2)
A_45_ref  = 0.3863   # pre-computed: sqrt((0.50*cos45)²+(0.22*sin45)²)
record("P4", "P1_area_at_45deg", A_45_calc, A_45_ref, 0.001,
       "abs <= 0.001 m² of 0.3863")

print()
print("  P2: Height-dependent wind profile (power law)")
ref_h = WIND_CFG["reference_height"]
alpha = WIND_CFG["height_exponent"]
for body_name, z in [("pelvis", 0.74), ("thigh", 0.54), ("shin", 0.35), ("ankle", 0.08)]:
    ratio  = max(z / ref_h, WIND_CFG["min_height_ratio"])
    factor = ratio ** alpha
    print(f"    {body_name:8s} (z={z:.2f}m): v_scale={factor:.4f} "
          f"({factor * 100:.1f}%)  F_scale={factor**2:.4f} ({factor**2 * 100:.1f}%)")

# Key check: pelvis at z=0.74m
pelvis_z   = 0.74
pelvis_rat = max(pelvis_z / ref_h, WIND_CFG["min_height_ratio"])
pelvis_fac = pelvis_rat ** alpha
ref_pelvis = 0.9620  # (0.74/0.85)^0.28
record("P4", "P2_pelvis_height_factor", pelvis_fac, ref_pelvis, 0.005,
       "abs <= 0.005 of 0.9620")

print()
print("  P3: Center-of-pressure tipping torque")
cop_offsets   = WIND_CFG["cop_z_offsets"]
fractions     = WIND_CFG["force_body_fractions"]
body_mass_ref = 35.0
BW            = body_mass_ref * 9.81
for F_total in [5, 30, 60, 100, 150]:
    total_torque = sum(F_total * fractions[i] * cop_offsets[i] for i in range(5))
    lean_rad     = math.radians(5)
    g_restoring  = body_mass_ref * 9.81 * 0.85 * math.sin(lean_rad)
    ratio_pct    = total_torque / g_restoring * 100
    print(f"    F_total={F_total:4.0f}N: CoP_torque={total_torque:.3f} Nm  "
          f"gravity_restoring@5deg={g_restoring:.1f} Nm  ratio={ratio_pct:.1f}%")

# Key check: F=60N
# 60*(0.55*0.10 + 0.12*0.005 + 0.12*0.005 + 0.08*0.002 + 0.08*0.002)
# = 60*(0.055 + 0.0006 + 0.0006 + 0.00016 + 0.00016) = 60*0.05652 = 3.3912
F60_torque    = sum(60.0 * fractions[i] * cop_offsets[i] for i in range(5))
ref_60_torque = 3.3912
record("P4", "P3_cop_torque_at_60N", F60_torque, ref_60_torque, 0.01,
       "abs <= 0.01 Nm of 3.3912")

# ============================================================
# Part 5: PASS/FAIL comparison summary
# ============================================================
section("PART 5: PASS/FAIL COMPARISON SUMMARY")

all_parts = {}
for (part, name, passed, got, expected, tol_desc) in PASS_FAIL:
    all_parts.setdefault(part, []).append((name, passed, got, expected, tol_desc))

total_pass = sum(1 for (_, _, p, _, _, _) in PASS_FAIL if p)
total_all  = len(PASS_FAIL)

for part in sorted(all_parts):
    print(f"\n  {part}:")
    for (name, passed, got, expected, tol_desc) in all_parts[part]:
        status  = "PASS" if passed else "FAIL"
        print(f"    [{status}] {name}")
        print(f"           got={got:.4f}  expected={expected:.4f}  tol={tol_desc}")

print()
print(f"  Result: {total_pass}/{total_all} checks passed")
if total_pass == total_all:
    print("  ** ALL CHECKS PASSED **")
else:
    n_fail = total_all - total_pass
    print(f"  ** {n_fail} CHECK(S) FAILED — review output above **")

print()
print("=" * 70)
print("VERIFICATION COMPLETE")
print("=" * 70)
