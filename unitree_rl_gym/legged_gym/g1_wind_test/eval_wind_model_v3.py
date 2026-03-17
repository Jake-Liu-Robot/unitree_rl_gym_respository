"""Comprehensive evaluation of the v3 wind model physics.

Evaluates:
  Part 1: Wind velocity statistics per curriculum level
  Part 2: OU process stationarity and autocorrelation
  Part 3: Gust event statistics
  Part 4: Full simulation aerodynamic force analysis
  Part 5-8: Analytical evaluation of P1-P3 improvements
"""

import isaacgym  # must be imported before torch
import torch
import numpy as np
from legged_gym.envs import *
from legged_gym.utils import task_registry
from legged_gym.envs.g1_wind.wind_model import WindModel
from legged_gym.envs.g1_wind.g1_wind_config import G1WindRoughCfg

torch.set_printoptions(precision=4, sci_mode=False)
cfg = G1WindRoughCfg()

# ============================================================
# Part 1: Wind velocity model statistical evaluation
# ============================================================
print("=" * 70)
print("PART 1: WIND VELOCITY MODEL — STATISTICAL ANALYSIS")
print("=" * 70)

for level in range(6):
    n_envs = 1024
    wm = WindModel(n_envs, "cuda:0", cfg.wind)
    curriculum = torch.full((n_envs,), level, dtype=torch.long, device="cuda:0")
    env_ids = torch.arange(n_envs, device="cuda:0")
    wm.reset_envs(env_ids, curriculum)

    dt = 0.005
    n_steps = 4000  # 20 seconds
    speeds = []
    for t in range(n_steps):
        vel = wm.step(dt)
        speed = torch.norm(vel, dim=1)
        speeds.append(speed.cpu())

    speeds = torch.stack(speeds)
    mean_speed = speeds.mean().item()
    std_speed = speeds.std().item()
    max_speed = speeds.max().item()
    ti = std_speed / max(mean_speed, 0.01)

    A_mean = (cfg.wind.frontal_area_front + cfg.wind.frontal_area_side) / 2
    F_mean = 0.5 * cfg.wind.air_density * cfg.wind.drag_coefficient * A_mean * mean_speed ** 2
    F_max = 0.5 * cfg.wind.air_density * cfg.wind.drag_coefficient * A_mean * max_speed ** 2
    clamp = cfg.wind.force_clamp_per_level[level]

    print(
        f"  Level {level}: speed={mean_speed:5.2f}+/-{std_speed:.2f} m/s, "
        f"max={max_speed:.1f}, TI={ti:.2f}, "
        f"F_mean={F_mean:.1f}N, F_max(unclamped)={F_max:.1f}N, clamp={clamp:.0f}N"
    )

# ============================================================
# Part 2: OU process behavior
# ============================================================
print()
print("=" * 70)
print("PART 2: OU PROCESS — STATIONARITY AND AUTOCORRELATION")
print("=" * 70)

wm = WindModel(4096, "cuda:0", cfg.wind)
curriculum = torch.full((4096,), 3, dtype=torch.long, device="cuda:0")
wm.reset_envs(torch.arange(4096, device="cuda:0"), curriculum)

dt = 0.005
ou_speeds = []
ou_angles = []
for t in range(8000):
    wm.step(dt)
    ou_speeds.append(wm.ou_speed_state.cpu().clone())
    ou_angles.append(wm.ou_angle_state.cpu().clone())

ou_speeds = torch.stack(ou_speeds)
ou_angles = torch.stack(ou_angles)

for t_idx, label in [(100, "t=0.5s"), (1000, "t=5s"), (4000, "t=20s"), (7999, "t=40s")]:
    s = ou_speeds[t_idx]
    a = ou_angles[t_idx]
    print(
        f"  {label}: speed_OU mean={s.mean():.4f} std={s.std():.4f} | "
        f"angle_OU mean={a.mean():.4f} std={a.std():.4f} ({a.std() * 180 / 3.14159:.1f} deg)"
    )

lag = 200  # 1s at dt=0.005
speed_corr = torch.corrcoef(torch.stack([ou_speeds[4000], ou_speeds[4000 + lag]]))[0, 1].item()
angle_corr = torch.corrcoef(torch.stack([ou_angles[4000], ou_angles[4000 + lag]]))[0, 1].item()
print(f"  Autocorrelation at 1s lag: speed={speed_corr:.3f}, angle={angle_corr:.3f}")
print(f"  Expected: speed~exp(-0.5)=0.607, angle~exp(-0.2)=0.819")

# ============================================================
# Part 3: Gust statistics
# ============================================================
print()
print("=" * 70)
print("PART 3: GUST EVENT STATISTICS")
print("=" * 70)

for level in [0, 2, 5]:
    wm = WindModel(4096, "cuda:0", cfg.wind)
    curriculum = torch.full((4096,), level, dtype=torch.long, device="cuda:0")
    wm.reset_envs(torch.arange(4096, device="cuda:0"), curriculum)

    dt = 0.005
    gust_starts = 0
    gust_active_steps = 0
    total_steps = 20000  # 100s
    prev_active = torch.zeros(4096, dtype=torch.bool, device="cuda:0")
    for t in range(total_steps):
        wm.step(dt)
        newly_active = wm.gust_active & ~prev_active
        gust_starts += newly_active.sum().item()
        gust_active_steps += wm.gust_active.sum().item()
        prev_active = wm.gust_active.clone()

    gust_rate = gust_starts / (4096 * total_steps * dt)
    duty_cycle = gust_active_steps / (4096 * total_steps)
    expected_prob = cfg.wind.gust_prob * (1 + cfg.wind.gust_prob_scale_per_level * level)
    print(
        f"  Level {level}: gust_rate={gust_rate:.3f}/s (expected~{expected_prob:.3f}), "
        f"duty_cycle={duty_cycle * 100:.1f}%"
    )

# ============================================================
# Part 4: Full simulation — aerodynamic force
# ============================================================
print()
print("=" * 70)
print("PART 4: PER-BODY AERODYNAMICS — FULL SIMULATION")
print("=" * 70)

env_cfg, _ = task_registry.get_cfgs(name="g1_wind")
env_cfg.env.num_envs = 64
env_cfg.terrain.num_rows = 4
env_cfg.terrain.num_cols = 4
env_cfg.terrain.mesh_type = "plane"

for level in [0, 2, 3, 5]:
    env_cfg.wind.curriculum_start_level = level
    env, _ = task_registry.make_env(name="g1_wind", args=None, env_cfg=env_cfg)
    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)

    for _ in range(50):
        env.step(actions)

    forces = []
    vels = []
    for _ in range(200):
        env.step(actions)
        forces.append(env.wind_model.wind_force.clone())
        vels.append(env.wind_model.get_wind_velocity().clone())

    forces = torch.stack(forces)
    vels = torch.stack(vels)
    force_mags = torch.norm(forces, dim=2)
    vel_mags = torch.norm(vels, dim=2)

    clamp = env_cfg.wind.force_clamp_per_level[level]
    clamped_frac = (force_mags >= clamp * 0.99).float().mean().item()

    force_z_mean = forces[:, :, 2].abs().mean().item()
    force_xy_mean = torch.norm(forces[:, :, :2], dim=2).mean().item()
    z_ratio = force_z_mean / max(force_xy_mean, 0.01) * 100

    print(
        f"  Level {level}: |F|={force_mags.mean():.2f}+/-{force_mags.std():.2f} N, "
        f"max={force_mags.max():.1f}N, clamped={clamped_frac * 100:.1f}%, "
        f"|v|={vel_mags.mean():.2f} m/s, F_z/F_xy={z_ratio:.1f}%"
    )

    env.gym.destroy_sim(env.sim)
    del env

# ============================================================
# Part 5-8: Analytical evaluations
# ============================================================
print()
print("=" * 70)
print("PART 5: P1 — DIRECTIONAL AREA EFFECT")
print("=" * 70)
A_front = cfg.wind.frontal_area_front
A_side = cfg.wind.frontal_area_side
for angle_deg in [0, 30, 45, 60, 90]:
    theta = np.radians(angle_deg)
    # Elliptical projection (used in code): always ≤ A_front
    A_ellip = np.sqrt((A_front * np.cos(theta)) ** 2 + (A_side * np.sin(theta)) ** 2)
    # Old cross-flow (for comparison): can exceed A_front
    A_cross = A_front * abs(np.cos(theta)) + A_side * abs(np.sin(theta))
    ratio = A_ellip / A_front * 100
    print(
        f"  angle={angle_deg:3d} deg: A_ellip={A_ellip:.3f} m2 ({ratio:.0f}% of frontal)"
        f"  [old cross-flow={A_cross:.3f}]"
    )

print()
print("=" * 70)
print("PART 6: P2 — HEIGHT PROFILE SCALING")
print("=" * 70)
ref_h = cfg.wind.reference_height
alpha = cfg.wind.height_exponent
for name, z in [("pelvis", 0.74), ("thigh", 0.54), ("shin", 0.35), ("ankle", 0.08)]:
    ratio = max(z / ref_h, 0.05)
    factor = ratio ** alpha
    print(
        f"  {name:8s} (z={z:.2f}m): v_scale={factor:.4f} ({factor * 100:.1f}%), "
        f"F_scale={factor ** 2:.4f} ({factor ** 2 * 100:.1f}%)"
    )

print()
print("=" * 70)
print("PART 7: FORCE CLAMP vs BODY WEIGHT")
print("=" * 70)
body_mass = 35.0
body_weight = body_mass * 9.81
for level in range(6):
    clamp = cfg.wind.force_clamp_per_level[level]
    frac = clamp / body_weight * 100
    speeds = cfg.wind.curriculum_levels[level]
    print(f"  L{level}: clamp={clamp:4.0f}N ({frac:5.1f}% BW), wind={speeds[0]:.0f}-{speeds[1]:.0f} m/s")

print()
print("=" * 70)
print("PART 8: P3 — CoP TIPPING TORQUE")
print("=" * 70)
cop_offsets = cfg.wind.cop_z_offsets
fractions = cfg.wind.force_body_fractions
for F_total in [5, 30, 60, 100, 150]:
    total_torque = sum(F_total * fractions[i] * cop_offsets[i] for i in range(5))
    lean_rad = np.radians(5)
    g_torque = body_mass * 9.81 * 0.85 * np.sin(lean_rad)
    print(
        f"  F_total={F_total:4.0f}N: CoP_torque={total_torque:.3f} Nm, "
        f"gravity restoring@5deg={g_torque:.1f} Nm, "
        f"ratio={total_torque / g_torque * 100:.1f}%"
    )

print()
print("=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)
