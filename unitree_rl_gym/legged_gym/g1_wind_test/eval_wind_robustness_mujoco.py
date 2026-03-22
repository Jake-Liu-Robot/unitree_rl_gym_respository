"""
MuJoCo wind robustness evaluation for g1_wind policies.

Replicates eval_wind_robustness.py suites A-F without Isaac Gym or legged_gym.
Uses deploy/deploy_mujoco/configs/g1.yaml for sim/control parameters.

Usage:
    python eval_wind_robustness_mujoco.py \\
        --policy logs/g1_wind/exported/policies/policy_lstm_1.pt \\
        --suite all --test_level 3,4,5 --num_episodes 50 --seed 42 \\
        --output exp3_full_method_mujoco.json

Allowed dependencies: mujoco, torch, numpy, math, json, argparse, os, yaml
"""

import os
import math
import json
import argparse
import yaml
import numpy as np
import torch
import mujoco

# ============================================================
# Path setup — all paths derived from __file__, no legged_gym
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
G1_YAML    = os.path.join(REPO_ROOT, "deploy/deploy_mujoco/configs/g1.yaml")
OUTPUT_DIR = os.path.join(REPO_ROOT, "test_results/mujoco")

# ============================================================
# Wind config dict — hardcoded from g1_wind_config.py (ground truth)
# ============================================================

WIND_CFG = {
    # Physical constants (g1_wind_config.py class wind)
    "air_density":           1.225,
    "drag_coefficient":      1.1,

    # P1: direction-dependent projected area
    "frontal_area_front":    0.50,
    "frontal_area_side":     0.22,
    "frontal_area_top":      0.10,

    # P2: height-dependent wind profile (power law boundary layer)
    "height_profile_enabled": True,
    "height_exponent":        0.28,
    "reference_height":       0.85,
    "min_height_ratio":       0.05,

    # P3: center-of-pressure z-offset per body (body-local, m)
    "cop_z_offsets": [0.10, 0.005, 0.005, 0.002, 0.002],

    # Multi-body force distribution
    "force_body_names":     ["pelvis", "left_hip_yaw_link", "right_hip_yaw_link",
                             "left_knee_link", "right_knee_link"],
    "force_body_fractions": [0.55, 0.12, 0.12, 0.08, 0.08],

    # Layer 2: OU default parameters (used when ou_randomize=False)
    "ou_theta":      0.5,
    "ou_sigma":      0.18,
    "ou_theta_dir":  0.2,
    "ou_sigma_dir":  0.15,

    # Layer 2: per-episode OU parameter randomization ranges
    "ou_theta_range":     [0.2,  1.0],
    "ou_sigma_range":     [0.05, 0.25],
    "ou_theta_dir_range": [0.05, 0.5],
    "ou_sigma_dir_range": [0.02, 0.25],

    # Layer 3: gust parameters
    "gust_prob":                  0.1,
    "gust_prob_scale_per_level":  0.1,
    "gust_speed_range":           [2.0, 6.0],
    "gust_speed_scale_per_level": 0.2,
    "gust_angle_range":           [-1.047, 1.047],   # ±60° = ±π/3
    "gust_duration_range":        [1.5, 3.0],
    "gust_ramp_up":               0.3,
    "gust_ramp_down":             0.5,

    # Per-level speed clamp (m/s)
    "speed_clamp_per_level": [0.0, 5.0, 8.0, 13.0, 20.0, 28.0],

    # Curriculum levels: [min_speed, max_speed] m/s
    "curriculum_levels": [
        [0.0,  0.0],
        [1.0,  3.0],
        [2.0,  5.0],
        [4.0,  8.0],
        [7.0, 12.0],
        [10.0, 18.0],
    ],

    # Per-level force clamp (N)
    "force_clamp_per_level": [5.0, 15.0, 30.0, 60.0, 100.0, 150.0],
}

# ============================================================
# Per-experiment policy paths (relative to REPO_ROOT)
# ============================================================

EXP_POLICIES = {
    1: "logs/g1_wind_baseline/exported/policies/policy_lstm_1.pt",
    2: "logs/g1_wind_push_only/exported/policies/policy_lstm_1.pt",
    3: "logs/g1_wind/exported/policies/policy_lstm_1.pt",
    4: "logs/g1_wind_no_curriculum/exported/policies/policy_lstm_1.pt",
    5: "logs/g1_wind_no_reward/exported/policies/policy_lstm_1.pt",
}

# ============================================================
# Scenario definitions — copied verbatim from eval_wind_robustness.py
# ============================================================

WIND_MODES = {
    "B1_steady":     {"ou_speed": False, "ou_dir": False, "gusts": False,
                      "label": "Steady (Layer1 only)"},
    "B2_turbulent":  {"ou_speed": True,  "ou_dir": True,  "gusts": False,
                      "label": "Turbulent (L1+L2)"},
    "B3_gusts_only": {"ou_speed": False, "ou_dir": False, "gusts": True,
                      "label": "Gusts only (L1+L3)"},
    "B4_full":       {"ou_speed": True,  "ou_dir": True,  "gusts": True,
                      "label": "Full model (L1+L2+L3)"},
    "B5_pure_gusts": {"ou_speed": False, "ou_dir": False, "gusts": True,
                      "base_speed_override": 0.0,
                      "label": "Pure gusts (L3 only)"},
}

WIND_DIRECTIONS = {
    "C1_front":    {"angle": 0.0,           "label": "Front (0\u00b0)"},
    "C2_side":     {"angle": math.pi / 2,   "label": "Side (90\u00b0)"},
    "C3_back":     {"angle": math.pi,       "label": "Back (180\u00b0)"},
    "C4_diagonal": {"angle": math.pi / 4,   "label": "Diagonal (45\u00b0)"},
    "C5_random":   {"angle": None,          "label": "Random"},
}

OU_EXTREMES = {
    "D1_calm":        {"theta": 1.0, "sigma": 0.05, "theta_dir": 1.0,  "sigma_dir": 0.02,
                       "label": "Calm (near-constant)"},
    "D2_turbulent":   {"theta": 0.2, "sigma": 0.4,  "theta_dir": 0.2,  "sigma_dir": 0.2,
                       "label": "High turbulence (OOD)"},
    "D3_locked_dir":  {"theta": 0.5, "sigma": 0.25, "theta_dir": 1.0,  "sigma_dir": 0.02,
                       "label": "Locked direction"},
    "D4_erratic_dir": {"theta": 0.5, "sigma": 0.25, "theta_dir": 0.05, "sigma_dir": 0.25,
                       "label": "Erratic direction"},
    "D5_default":     {"theta": 0.5, "sigma": 0.25, "theta_dir": 0.2,  "sigma_dir": 0.15,
                       "label": "Training default"},
}

OOD_PATTERNS = {
    "E1_step":     {"label_fmt": "Step change (0\u2192{peak:.0f} m/s at t=5s)"},
    "E2_periodic": {"label_fmt": "Periodic ({lo:.0f}-{hi:.0f} m/s, T=4s)"},
    "E3_reversal": {"label_fmt": "Dir reversal (180\u00b0 at t=5s, {peak:.0f} m/s)"},
}

COMMAND_SCENARIOS = {
    "F1_standing":   {"vx": 0.0, "vy": 0.0,
                      "label": "Standing (cmd=0)"},
    "F2_slow_fwd":   {"vx": 0.3, "vy": 0.0,
                      "label": "Slow forward (0.3)"},
    "F3_normal_fwd": {"vx": 0.6, "vy": 0.0,
                      "label": "Normal forward (0.6)"},
    "F4_fast_fwd":   {"vx": 1.0, "vy": 0.0,
                      "label": "Fast forward (1.0)"},
    "F5_lateral":    {"vx": 0.0, "vy": 0.5,
                      "label": "Lateral walk (vy=0.5)"},
    "F6_turning":    {"vx": 0.5, "vy": 0.0, "heading_offset": 1.0,
                      "label": "Turning (vx=0.5+yaw)"},
    "F7_headwind":   {"vx": 0.5, "vy": 0.0, "wind_angle": math.pi,
                      "label": "Headwind walk"},
    "F8_tailwind":   {"vx": 0.5, "vy": 0.0, "wind_angle": 0.0,
                      "label": "Tailwind walk"},
}

# ============================================================
# Math helpers — copied verbatim from deploy_mujoco.py
# ============================================================

def get_gravity_orientation(quaternion):
    """Gravity direction in body frame. quaternion = [w, x, y, z]."""
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]
    gravity_orientation    = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands."""
    return (target_q - q) * kp + (target_dq - dq) * kd

# ============================================================
# MuJoCoWindModel — pure numpy, single environment
# ============================================================

class MuJoCoWindModel:
    """3-layer wind model for single-environment MuJoCo evaluation.

    Ported from wind_model.py. Uses numpy RNG instead of isaacgym utils.

    Flags for suite control (set before reset_episode()):
        enable_ou:    bool  — enable Layer 2 OU speed + direction processes
        enable_gusts: bool  — enable Layer 3 gust events
        fixed_angle:  float or None — fix base wind angle (None = random)
    """

    def __init__(self, cfg: dict, level: int, rng: np.random.Generator):
        self.cfg   = cfg
        self.level = level
        self.rng   = rng

        # Suite-control flags
        self.enable_ou    = True
        self.enable_gusts = True
        self.fixed_angle  = None   # float (radians) or None

        # Layer 1 state
        self.base_speed = 0.0
        self.base_angle = 0.0

        # Layer 2 state
        self.ou_speed     = 0.0
        self.ou_angle     = 0.0
        self.ou_theta     = cfg["ou_theta"]
        self.ou_sigma     = cfg["ou_sigma"]
        self.ou_theta_dir = cfg["ou_theta_dir"]
        self.ou_sigma_dir = cfg["ou_sigma_dir"]

        # Layer 3 state
        self.gust_active   = False
        self.gust_elapsed  = 0.0
        self.gust_duration = 0.0
        self.gust_speed    = 0.0
        self.gust_angle    = 0.0

        # Last computed wind velocity (for force reporting)
        self.wind_vel       = np.zeros(3)
        self.wind_force_mag = 0.0   # set by aerodynamics code after each step

    # ----------------------------------------------------------
    def reset_episode(self):
        """Sample new base wind + OU params for a new episode."""
        cfg = self.cfg
        lvl = min(self.level, len(cfg["curriculum_levels"]) - 1)
        lo, hi = cfg["curriculum_levels"][lvl]
        self.base_speed = float(self.rng.uniform(lo, hi))

        if self.fixed_angle is not None:
            self.base_angle = float(self.fixed_angle)
        else:
            self.base_angle = float(self.rng.uniform(0.0, 2.0 * math.pi))

        # Layer 2: reset OU states + randomize params per episode
        self.ou_speed = 0.0
        self.ou_angle = 0.0
        self.ou_theta     = float(self.rng.uniform(*cfg["ou_theta_range"]))
        self.ou_sigma     = float(self.rng.uniform(*cfg["ou_sigma_range"]))
        self.ou_theta_dir = float(self.rng.uniform(*cfg["ou_theta_dir_range"]))
        self.ou_sigma_dir = float(self.rng.uniform(*cfg["ou_sigma_dir_range"]))

        # Layer 3: reset gust
        self.gust_active   = False
        self.gust_elapsed  = 0.0
        self.gust_duration = 0.0
        self.gust_speed    = 0.0
        self.gust_angle    = 0.0

    # ----------------------------------------------------------
    def step(self, dt: float) -> np.ndarray:
        """Advance wind model one physics substep. Returns wind_vel [3,] (z=0)."""
        cfg     = self.cfg
        sqrt_dt = math.sqrt(dt)

        # Layer 2: OU speed and direction processes
        if self.enable_ou:
            self.ou_speed += (
                -self.ou_theta * self.ou_speed * dt
                + self.ou_sigma * sqrt_dt * self.rng.standard_normal()
            )
            self.ou_angle += (
                -self.ou_theta_dir * self.ou_angle * dt
                + self.ou_sigma_dir * sqrt_dt * self.rng.standard_normal()
            )

        ou_speed_contrib = self.ou_speed if self.enable_ou else 0.0
        ou_angle_contrib = self.ou_angle if self.enable_ou else 0.0

        speed_clamp_list = cfg["speed_clamp_per_level"]
        clamp = speed_clamp_list[min(self.level, len(speed_clamp_list) - 1)]
        effective_speed = max(0.0, self.base_speed + ou_speed_contrib)
        effective_angle = self.base_angle + ou_angle_contrib

        base_vel = np.array([
            effective_speed * math.cos(effective_angle),
            effective_speed * math.sin(effective_angle),
            0.0,
        ])

        # Layer 3: gusts
        gust_vel = np.zeros(3)
        if self.enable_gusts:
            gust_vel = self._update_gusts(dt)

        # Clamp total velocity magnitude (matches wind_model.py behavior)
        total_vel = base_vel + gust_vel
        total_speed = math.sqrt(total_vel[0]**2 + total_vel[1]**2 + total_vel[2]**2)
        if total_speed > clamp:
            total_vel = total_vel * (clamp / max(total_speed, 1e-8))
        self.wind_vel = total_vel
        return self.wind_vel.copy()

    # ----------------------------------------------------------
    def _update_gusts(self, dt: float) -> np.ndarray:
        cfg = self.cfg
        if self.gust_active:
            self.gust_elapsed += dt
            if self.gust_elapsed >= self.gust_duration:
                self.gust_active  = False
                self.gust_elapsed = 0.0
        else:
            gust_prob = cfg["gust_prob"] * (
                1.0 + cfg["gust_prob_scale_per_level"] * self.level
            )
            if self.rng.random() < gust_prob * dt:
                self.gust_duration = float(self.rng.uniform(*cfg["gust_duration_range"]))
                self.gust_elapsed  = 0.0
                speed_scale = (
                    1.0 + cfg["gust_speed_scale_per_level"] * self.level
                )
                self.gust_speed = (
                    float(self.rng.uniform(*cfg["gust_speed_range"])) * speed_scale
                )
                angle_offset   = float(self.rng.uniform(*cfg["gust_angle_range"]))
                self.gust_angle = self.base_angle + angle_offset
                self.gust_active = True

        if not self.gust_active:
            return np.zeros(3)

        envelope = self._gust_envelope()
        return np.array([
            self.gust_speed * envelope * math.cos(self.gust_angle),
            self.gust_speed * envelope * math.sin(self.gust_angle),
            0.0,
        ])

    def _gust_envelope(self) -> float:
        """Trapezoidal ramp-up / sustain / ramp-down envelope, clamped [0, 1]."""
        t   = self.gust_elapsed
        dur = self.gust_duration
        ru  = self.cfg["gust_ramp_up"]   + 1e-6
        rd  = self.cfg["gust_ramp_down"] + 1e-6
        return max(0.0, min(1.0, min(t / ru, (dur - t) / rd)))

    def override_ou_params(self, theta, sigma, theta_dir, sigma_dir):
        """Override OU params after reset_episode() for Suite D."""
        self.ou_theta     = theta
        self.ou_sigma     = sigma
        self.ou_theta_dir = theta_dir
        self.ou_sigma_dir = sigma_dir

# ============================================================
# MuJoCo setup helpers
# ============================================================

def load_sim(repo_root: str):
    """Load g1.yaml, resolve paths, return (m, d, sim_cfg)."""
    with open(G1_YAML, "r") as f:
        sim_cfg = yaml.load(f, Loader=yaml.FullLoader)

    xml_path    = sim_cfg["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", repo_root)
    policy_path = sim_cfg["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", repo_root)
    sim_cfg["_xml_path"]    = xml_path
    sim_cfg["_policy_path"] = policy_path

    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = sim_cfg["simulation_dt"]
    return m, d, sim_cfg


def resolve_body_ids(m, body_names: list) -> dict:
    """Resolve body names to MuJoCo body IDs. Exits if any name is missing."""
    ids = {}
    for name in body_names:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid == -1:
            print(f"[ERROR] Body '{name}' not found in MuJoCo model.")
            print("Available bodies:")
            print([m.body(i).name for i in range(m.nbody)])
            raise SystemExit(1)
        ids[name] = bid
    return ids


def reset_episode(m, d, default_angles):
    """Reset MuJoCo data to a standing pose with default joint angles."""
    mujoco.mj_resetData(m, d)
    d.qpos[2]   = 0.78       # nominal pelvis height (G1)
    d.qpos[3]   = 1.0        # quaternion w — upright
    d.qpos[4:7] = 0.0        # quaternion xyz
    d.qpos[7:]  = default_angles
    d.xfrc_applied[:] = 0.0
    mujoco.mj_forward(m, d)

# ============================================================
# Policy loading
# ============================================================

def load_policy(policy_path: str, num_obs: int):
    """Load TorchScript LSTM policy; fall back to pre-train MLP if needed."""
    policy = torch.jit.load(policy_path, map_location="cpu")
    policy.eval()
    print(f"[INFO] Policy loaded: {policy_path}")
    return policy


_reset_policy_warned = False

def reset_policy(policy, num_obs: int):
    """Reset LSTM hidden state at the start of each episode."""
    global _reset_policy_warned
    if hasattr(policy, "reset"):
        policy.reset()
    else:
        # Fallback: warm-up with zero obs to flush any stale hidden state
        dummy = torch.zeros(1, num_obs)
        with torch.no_grad():
            for _ in range(3):
                policy(dummy)
        if not _reset_policy_warned:
            print("[WARN] policy.reset() not found, using zero warm-up fallback (warned once)")
            _reset_policy_warned = True


def policy_infer(policy, obs: np.ndarray) -> np.ndarray:
    """Run one inference step; tries policy(obs) then policy.act_inference(obs)."""
    obs_tensor = torch.from_numpy(obs).unsqueeze(0).float()
    with torch.no_grad():
        try:
            action = policy(obs_tensor).detach().numpy().squeeze()
        except Exception:
            action = policy.act_inference(obs_tensor).detach().numpy().squeeze()
    return action

# ============================================================
# Per-body aerodynamic force (P0/P1/P2/P3)
# ============================================================

def apply_wind_forces(d, body_ids: dict, wind_vel: np.ndarray,
                      level: int) -> float:
    """Compute and apply aerodynamic forces to d.xfrc_applied.

    Returns total force magnitude (N) for metric tracking.

    xfrc_applied layout (per spec): [0:3] = force, [3:6] = torque.
    Zero-fills the array first (MuJoCo does NOT auto-clear it).

    Physics:
      P0: relative velocity  v_rel = scaled_wind - body_lin_vel
      P1: ellipsoidal projected area via d.xmat rotation matrix
      P2: height-dependent wind scaling (power law)
      P3: CoP torque from body-local z offset
    """
    cfg     = WIND_CFG
    rho     = cfg["air_density"]
    Cd      = cfg["drag_coefficient"]
    A_front = cfg["frontal_area_front"]
    A_side  = cfg["frontal_area_side"]
    A_top   = cfg["frontal_area_top"]
    ref_h   = cfg["reference_height"]
    alpha   = cfg["height_exponent"]
    min_r   = cfg["min_height_ratio"]
    fracs   = cfg["force_body_fractions"]
    cop_zs  = cfg["cop_z_offsets"]
    f_clamp = cfg["force_clamp_per_level"]
    lv_clamp = f_clamp[min(level, len(f_clamp) - 1)]

    d.xfrc_applied[:] = 0.0   # MuJoCo does NOT clear this automatically

    total_force = np.zeros(3)

    for i, name in enumerate(cfg["force_body_names"]):
        bid      = body_ids[name]
        fraction = fracs[i]
        cop_z    = cop_zs[i]

        # P2: height scaling
        z = float(d.xpos[bid][2])
        if cfg["height_profile_enabled"]:
            height_factor = max(z / ref_h, min_r) ** alpha
        else:
            height_factor = 1.0
        scaled_wind = wind_vel * height_factor

        # P0: relative velocity (d.cvel layout: [rot(3), lin(3)])
        body_lin_vel = d.cvel[bid][3:6].copy()
        v_rel        = scaled_wind - body_lin_vel
        v_rel_speed  = float(np.linalg.norm(v_rel)) + 1e-8
        v_rel_dir    = v_rel / v_rel_speed

        # P1: ellipsoidal projected area using body rotation matrix
        # d.xmat[bid] is the 3×3 body-to-world rotation matrix (row-major).
        # .T transforms world-frame v_rel_dir into body frame.
        R_world2body = d.xmat[bid].reshape(3, 3).T
        v_rel_body   = R_world2body @ v_rel_dir
        dx = abs(float(v_rel_body[0]))
        dy = abs(float(v_rel_body[1]))
        dz = abs(float(v_rel_body[2]))
        A_eff = math.sqrt(
            (A_front * dx) ** 2 + (A_side * dy) ** 2 + (A_top * dz) ** 2
        )

        # Force magnitude (per-body clamp)
        force_mag = min(
            0.5 * rho * Cd * A_eff * fraction * v_rel_speed ** 2,
            lv_clamp * fraction,
        )
        force_vec = v_rel_dir * force_mag

        # P3: CoP torque — offset [0, 0, cop_z] is along world z (upright approx)
        torque = np.cross(np.array([0.0, 0.0, cop_z]), force_vec)

        # Write into xfrc_applied (per spec: [0:3]=force, [3:6]=torque)
        d.xfrc_applied[bid, 0:3] += force_vec
        d.xfrc_applied[bid, 3:6] += torque

        total_force += force_vec

    return float(np.linalg.norm(total_force))

# ============================================================
# Core evaluation loop
# ============================================================

def evaluate_scenario(
    m, d, policy, sim_cfg, body_ids,
    wind_model: MuJoCoWindModel,
    num_episodes: int = 50,
    max_steps:    int = 1000,
    ood_pattern:  str = None,
    ood_peak_speed: float = 10.0,
    fix_commands: dict = None,
    scenario_seed: int = None,
) -> dict:
    """Run num_episodes sequential episodes under the current wind_model config.

    Args:
        wind_model: already configured (enable_ou/enable_gusts/fixed_angle set by caller)
        ood_pattern: "step" | "periodic" | "reversal" | None
        ood_peak_speed: used by OOD patterns
        fix_commands: dict with keys vx, vy [, heading_offset, wind_angle]
        scenario_seed: if not None, re-seed the wind model's RNG for reproducibility

    Returns dict with survival_rate, mean_episode_length, mean_tracking_error,
    mean_wind_force_N, and stub reward fields (0.0 — no reward shaping in MuJoCo).
    """
    if scenario_seed is not None:
        wind_model.rng = np.random.default_rng(scenario_seed)

    # Unpack sim config
    sim_dt     = sim_cfg["simulation_dt"]
    decimation = sim_cfg["control_decimation"]
    ctrl_dt    = sim_dt * decimation
    kps        = np.array(sim_cfg["kps"],            dtype=np.float32)
    kds        = np.array(sim_cfg["kds"],            dtype=np.float32)
    default_angles = np.array(sim_cfg["default_angles"], dtype=np.float32)
    ang_vel_scale  = sim_cfg["ang_vel_scale"]
    dof_pos_scale  = sim_cfg["dof_pos_scale"]
    dof_vel_scale  = sim_cfg["dof_vel_scale"]
    action_scale   = sim_cfg["action_scale"]
    cmd_scale      = np.array(sim_cfg["cmd_scale"], dtype=np.float32)
    num_actions    = sim_cfg["num_actions"]
    num_obs        = sim_cfg["num_obs"]
    period         = 0.8      # gait phase period (s)
    ood_mid        = ood_peak_speed / 2.0
    ood_amp        = ood_peak_speed / 2.0

    # Command
    default_cmd = np.array(sim_cfg.get("cmd_init", [0.5, 0.0, 0.0]), dtype=np.float32)
    if fix_commands is not None:
        cmd = np.array([
            fix_commands.get("vx",  float(default_cmd[0])),
            fix_commands.get("vy",  float(default_cmd[1])),
            fix_commands.get("heading_offset", 0.0),  # yaw command
        ], dtype=np.float32)
    else:
        cmd = default_cmd.copy()
    if len(cmd) < 3:
        cmd = np.append(cmd, 0.0).astype(np.float32)

    # Fix wind direction for F7/F8 scenarios
    wind_angle_override = None
    if fix_commands is not None and "wind_angle" in fix_commands:
        wind_angle_override = float(fix_commands["wind_angle"])

    episode_lengths  = []
    episode_tracking = []
    episode_forces   = []

    for _ep in range(num_episodes):
        # --- Episode reset ---
        reset_episode(m, d, default_angles)
        wind_model.reset_episode()
        reset_policy(policy, num_obs)

        # Apply wind direction override (fixed angle takes priority over reset)
        if wind_model.fixed_angle is not None:
            wind_model.base_angle = float(wind_model.fixed_angle)
        if wind_angle_override is not None:
            wind_model.base_angle = wind_angle_override

        action      = np.zeros(num_actions, dtype=np.float32)
        target_dof_pos = default_angles.copy()
        obs         = np.zeros(num_obs, dtype=np.float32)

        ep_track       = 0.0
        ep_force       = 0.0
        step_counter   = 0     # total physics steps this episode
        ctrl_step      = 0     # control steps completed
        total_force_mag = 0.0

        for ctrl_step in range(1, max_steps + 1):
            t_ep = (ctrl_step - 1) * ctrl_dt

            # OOD pattern: override wind model state before physics substeps
            if ood_pattern == "step":
                wind_model.base_speed = 0.0 if t_ep < 5.0 else ood_peak_speed
            elif ood_pattern == "periodic":
                wind_model.base_speed = (
                    ood_mid + ood_amp * math.sin(2.0 * math.pi / 4.0 * t_ep)
                )
            elif ood_pattern == "reversal":
                wind_model.base_speed = ood_peak_speed
                wind_model.base_angle = 0.0 if t_ep < 5.0 else math.pi

            # Run control_decimation physics substeps
            for _sub in range(decimation):
                # PD torque
                tau = pd_control(
                    target_dof_pos, d.qpos[7:], kps,
                    np.zeros(num_actions, dtype=np.float32), d.qvel[6:], kds,
                )
                d.ctrl[:] = tau

                # Wind force (advance model one physics step, then apply)
                wind_vel = wind_model.step(sim_dt)
                total_force_mag = apply_wind_forces(d, body_ids, wind_vel, wind_model.level)
                wind_model.wind_force_mag = total_force_mag

                mujoco.mj_step(m, d)
                step_counter += 1

            # --- Policy inference (after all substeps) ---
            qj    = d.qpos[7:].copy()
            dqj   = d.qvel[6:].copy()
            quat  = d.qpos[3:7].copy()    # [w, x, y, z]
            omega = d.qvel[3:6].copy()    # angular velocity (MuJoCo: body frame for free joint)

            grav   = get_gravity_orientation(quat)
            omega_s = omega * ang_vel_scale
            qj_s   = (qj - default_angles) * dof_pos_scale
            dqj_s  = dqj * dof_vel_scale

            # Phase: counter incremented every physics step
            count  = step_counter * sim_dt
            phase  = count % period / period
            sin_ph = math.sin(2.0 * math.pi * phase)
            cos_ph = math.cos(2.0 * math.pi * phase)

            obs[0:3]                              = omega_s
            obs[3:6]                              = grav
            obs[6:9]                              = cmd * cmd_scale
            obs[9:9 + num_actions]                = qj_s
            obs[9 + num_actions:9 + 2*num_actions]    = dqj_s
            obs[9 + 2*num_actions:9 + 3*num_actions]  = action   # action that just ran physics (matches Isaac Gym self.actions)
            obs[9 + 3*num_actions]                = sin_ph
            obs[9 + 3*num_actions + 1]            = cos_ph

            action      = policy_infer(policy, obs)
            target_dof_pos = action * action_scale + default_angles

            # Tracking error: body-frame linear velocity vs command
            # d.qvel[0:2] is world-frame; rotate to body frame via yaw before comparing.
            # cmd is in body frame (matching Isaac Gym env.base_lin_vel convention).
            qw, qx, qy, qz = quat
            yaw = math.atan2(2.0 * (qw * qz + qx * qy),
                             1.0 - 2.0 * (qy * qy + qz * qz))
            cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
            vx_body =  cos_yaw * float(d.qvel[0]) + sin_yaw * float(d.qvel[1])
            vy_body = -sin_yaw * float(d.qvel[0]) + cos_yaw * float(d.qvel[1])
            ep_track += math.sqrt(
                (vx_body - float(cmd[0])) ** 2
                + (vy_body - float(cmd[1])) ** 2
            )
            ep_force += total_force_mag

            # Termination: fallen (base too low or tilt too large)
            fallen = (float(d.qpos[2]) < 0.3) or (abs(float(grav[2])) < 0.5)
            if fallen:
                break   # ctrl_step retains current value

        # Record episode (ctrl_step = steps completed before fall or max)
        if ctrl_step > 0:
            episode_lengths.append(ctrl_step)
            episode_tracking.append(ep_track / ctrl_step)
            episode_forces.append(ep_force   / ctrl_step)

    # --- Aggregate metrics ---
    n = len(episode_lengths)
    if n == 0:
        return {
            "survival_rate": 0.0, "survival_se": 0.0,
            "mean_episode_length": 0.0, "std_episode_length": 0.0,
            "mean_reward_per_step": 0.0, "std_reward_per_step": 0.0,
            "mean_tracking_error": 9.99, "std_tracking_error": 0.0,
            "mean_wind_force_N": 0.0, "reward_components": {},
        }

    lengths  = np.array(episode_lengths)
    tracking = np.array(episode_tracking)
    forces   = np.array(episode_forces)
    surv     = float(np.mean(lengths >= max_steps))
    surv_se  = float(np.sqrt(surv * (1 - surv) / max(n, 1)))

    return {
        "survival_rate":       surv,
        "survival_se":         surv_se,
        "mean_episode_length": float(np.mean(lengths)),
        "std_episode_length":  float(np.std(lengths)),
        "mean_reward_per_step": 0.0,
        "std_reward_per_step":  0.0,
        "mean_tracking_error": float(np.mean(tracking)),
        "std_tracking_error":  float(np.std(tracking)),
        "mean_wind_force_N":   float(np.mean(forces)),
        "reward_components":   {},
    }

# ============================================================
# Test suite runners
# ============================================================

def _make_wind_model(level: int, base_seed: int, scenario_idx: int) -> MuJoCoWindModel:
    seed = base_seed + scenario_idx * 1000 if base_seed is not None else None
    rng  = np.random.default_rng(seed)
    return MuJoCoWindModel(WIND_CFG, level, rng)


def run_suite_levels(m, d, policy, sim_cfg, body_ids, args,
                     scenario_counter: list) -> dict:
    """Suite A: Full wind model level sweep L0–L5."""
    print_header("Suite A: Wind Level Sweep (Full Model)")
    level_names = ["No wind", "Light (1-3)", "Light-Med (2-5)",
                   "Medium (4-8)", "Strong (7-12)", "Extreme (10-18)"]
    results = {}
    for level in range(6):
        wm = _make_wind_model(level, args.seed, scenario_counter[0])
        scenario_counter[0] += 1
        r = evaluate_scenario(m, d, policy, sim_cfg, body_ids, wm,
                              num_episodes=args.num_episodes, max_steps=args.max_steps)
        key = f"A_level{level}"
        results[key] = r
        print_row(key, level_names[level], r)
    return results


def run_suite_modes(m, d, policy, sim_cfg, body_ids, args,
                    test_level: int, scenario_counter: list) -> dict:
    """Suite B: Wind mode decomposition at fixed level."""
    print_header(f"Suite B: Wind Modes (Level {test_level})")
    results = {}
    for key, mode in WIND_MODES.items():
        wm = _make_wind_model(test_level, args.seed, scenario_counter[0])
        scenario_counter[0] += 1
        wm.enable_ou    = mode.get("ou_speed", True) or mode.get("ou_dir", True)
        wm.enable_gusts = mode.get("gusts", True)
        base_speed_override = mode.get("base_speed_override", None)
        r = evaluate_scenario(m, d, policy, sim_cfg, body_ids, wm,
                              num_episodes=args.num_episodes, max_steps=args.max_steps)
        # Apply base_speed_override by patching reset: override after reset_episode
        if base_speed_override is not None:
            # Re-run with base_speed forced to 0 every episode
            wm2 = _make_wind_model(test_level, args.seed, scenario_counter[0] - 1)
            wm2.enable_ou    = wm.enable_ou
            wm2.enable_gusts = wm.enable_gusts
            orig_reset = wm2.reset_episode
            def _patched_reset(bso=base_speed_override, orig=orig_reset):
                orig()
                wm2.base_speed = bso
            wm2.reset_episode = _patched_reset
            r = evaluate_scenario(m, d, policy, sim_cfg, body_ids, wm2,
                                  num_episodes=args.num_episodes, max_steps=args.max_steps)
        results[key] = r
        print_row(key, mode["label"], r)
    return results


def run_suite_directions(m, d, policy, sim_cfg, body_ids, args,
                         test_level: int, scenario_counter: list) -> dict:
    """Suite C: Fixed wind directions, steady (no OU, no gusts)."""
    print_header(f"Suite C: Wind Directions (Level {test_level}, Steady)")
    results = {}
    for key, cfg_d in WIND_DIRECTIONS.items():
        wm = _make_wind_model(test_level, args.seed, scenario_counter[0])
        scenario_counter[0] += 1
        wm.enable_ou    = False
        wm.enable_gusts = False
        wm.fixed_angle  = cfg_d["angle"]   # None for C5_random
        r = evaluate_scenario(m, d, policy, sim_cfg, body_ids, wm,
                              num_episodes=args.num_episodes, max_steps=args.max_steps)
        results[key] = r
        print_row(key, cfg_d["label"], r)
    return results


def run_suite_ou_extremes(m, d, policy, sim_cfg, body_ids, args,
                          test_level: int, scenario_counter: list) -> dict:
    """Suite D: OU parameter extremes."""
    print_header(f"Suite D: OU Parameter Extremes (Level {test_level})")
    results = {}
    for key, params in OU_EXTREMES.items():
        wm = _make_wind_model(test_level, args.seed, scenario_counter[0])
        scenario_counter[0] += 1
        # Override OU params right after reset_episode() in evaluate_scenario
        orig_reset = wm.reset_episode
        def _ou_reset(p=params, orig=orig_reset, wm_ref=wm):
            orig()
            wm_ref.override_ou_params(p["theta"], p["sigma"],
                                      p["theta_dir"], p["sigma_dir"])
        wm.reset_episode = _ou_reset
        r = evaluate_scenario(m, d, policy, sim_cfg, body_ids, wm,
                              num_episodes=args.num_episodes, max_steps=args.max_steps)
        results[key] = r
        print_row(key, params["label"], r)
    return results


def run_suite_ood(m, d, policy, sim_cfg, body_ids, args,
                  test_level: int, scenario_counter: list) -> dict:
    """Suite E: Out-of-distribution wind patterns (step / periodic / reversal)."""
    print_header(f"Suite E: OOD Patterns (Level {test_level})")
    curriculum_levels = WIND_CFG["curriculum_levels"]
    peak_speed = max(
        curriculum_levels[min(test_level, len(curriculum_levels) - 1)][1], 10.0
    )
    results = {}
    for key, pat in OOD_PATTERNS.items():
        pattern_name = key.split("_")[1]   # "step", "periodic", "reversal"
        if pattern_name in ("step", "reversal"):
            label = pat["label_fmt"].format(peak=peak_speed)
        else:
            label = pat["label_fmt"].format(lo=0, hi=peak_speed)

        wm = _make_wind_model(test_level, args.seed, scenario_counter[0])
        scenario_counter[0] += 1
        wm.enable_ou    = False
        wm.enable_gusts = False
        wm.fixed_angle  = 0.0   # frontal, matches eval_wind_robustness.py
        r = evaluate_scenario(m, d, policy, sim_cfg, body_ids, wm,
                              num_episodes=args.num_episodes, max_steps=args.max_steps,
                              ood_pattern=pattern_name, ood_peak_speed=peak_speed)
        results[key] = r
        print_row(key, label, r)
    return results


def run_suite_commands(m, d, policy, sim_cfg, body_ids, args,
                       test_level: int, scenario_counter: list) -> dict:
    """Suite F: Command variations under full wind model."""
    print_header(f"Suite F: Command Variations (Level {test_level})")
    results = {}
    for key, scenario in COMMAND_SCENARIOS.items():
        wm = _make_wind_model(test_level, args.seed, scenario_counter[0])
        scenario_counter[0] += 1
        fix_dir = scenario.get("wind_angle", None)
        if fix_dir is not None:
            wm.fixed_angle  = fix_dir
            wm.enable_ou    = False
            wm.enable_gusts = False
        fix_cmd = {k: v for k, v in scenario.items()
                   if k not in ("label", "wind_angle")}
        r = evaluate_scenario(m, d, policy, sim_cfg, body_ids, wm,
                              num_episodes=args.num_episodes, max_steps=args.max_steps,
                              fix_commands=fix_cmd)
        results[key] = r
        print_row(key, scenario["label"], r)
    return results

# ============================================================
# Output formatting
# ============================================================

def print_header(title: str):
    print(f"\n{'='*100}")
    print(f"  {title}")
    print(f"{'='*100}")
    print(f"{'ID':>18} {'Description':>25} {'Survival%':>12} {'EpLen':>10} "
          f"{'TrackErr':>12} {'Wind(N)':>8}")
    print("-" * 100)


def print_row(test_id: str, label: str, r: dict):
    surv_str  = f"{r['survival_rate']*100:.1f}\u00b1{r['survival_se']*100:.1f}%"
    track_str = f"{r['mean_tracking_error']:.4f}\u00b1{r['std_tracking_error']:.3f}"
    print(f"{test_id:>18} {label:>25} {surv_str:>12} "
          f"{r['mean_episode_length']:>6.0f}\u00b1{r['std_episode_length']:<3.0f} "
          f"{track_str:>12} {r['mean_wind_force_N']:>8.2f}")


def save_results_json(all_results: dict, output_path: str, metadata: dict = None):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    payload = {"metadata": metadata or {}, "results": all_results}
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved to: {output_path}")

# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="MuJoCo wind robustness evaluation (Isaac-Gym-free)"
    )
    parser.add_argument("--exp",          type=int,  default=None, choices=[1, 2, 3, 4, 5],
                        help="Experiment ID 1-5 (selects policy from EXP_POLICIES)")
    parser.add_argument("--policy",       type=str,  default=None,
                        help="Explicit path to policy .pt (overrides --exp)")
    parser.add_argument("--suite",        type=str,  default="all",
                        choices=["all", "levels", "modes", "directions",
                                 "ou", "ood", "commands"])
    parser.add_argument("--test_level",   type=str,  default="3",
                        help="Levels for suites B-F: int, comma-list, or 'all' (=3,4,5)")
    parser.add_argument("--num_episodes", type=int,  default=50)
    parser.add_argument("--max_steps",    type=int,  default=1000,
                        help="Max control steps per episode (1000 = 20 s at 50 Hz)")
    parser.add_argument("--output",       type=str,  default=None,
                        help="JSON filename saved to test_results/mujoco/")
    parser.add_argument("--seed",         type=int,  default=None)
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # --- Load sim ---
    m, d, sim_cfg = load_sim(REPO_ROOT)
    num_obs = sim_cfg["num_obs"]

    # --- Resolve wind body IDs ---
    body_ids = resolve_body_ids(m, WIND_CFG["force_body_names"])
    print(f"Wind bodies resolved: {body_ids}")

    # --- Load policy ---
    if args.policy:
        policy_path = (args.policy if os.path.isabs(args.policy)
                       else os.path.join(REPO_ROOT, args.policy))
    elif args.exp is not None:
        policy_path = os.path.join(REPO_ROOT, EXP_POLICIES[args.exp])
    else:
        lstm_path = os.path.join(REPO_ROOT, EXP_POLICIES[3])
        policy_path = (lstm_path if os.path.exists(lstm_path)
                       else sim_cfg["_policy_path"])
    policy = load_policy(policy_path, num_obs)

    # --- Parse test levels ---
    if args.test_level == "all":
        test_levels = [3, 4, 5]
    else:
        test_levels = [int(x) for x in args.test_level.split(",")]

    suite = args.suite

    # Scenario counter shared across all suites (for per-scenario seeding)
    scenario_counter = [0]
    all_results = {}

    # Suite A: L0-L5 level sweep
    if suite in ("all", "levels"):
        r = run_suite_levels(m, d, policy, sim_cfg, body_ids, args, scenario_counter)
        all_results.update(r)

    # Suites B-F: at each requested level
    for tl in test_levels:
        if suite in ("all", "modes"):
            r = run_suite_modes(m, d, policy, sim_cfg, body_ids, args, tl,
                                scenario_counter)
            all_results.update({f"{k}_L{tl}": v for k, v in r.items()})

        if suite in ("all", "directions"):
            r = run_suite_directions(m, d, policy, sim_cfg, body_ids, args, tl,
                                     scenario_counter)
            all_results.update({f"{k}_L{tl}": v for k, v in r.items()})

        if suite in ("all", "ou"):
            r = run_suite_ou_extremes(m, d, policy, sim_cfg, body_ids, args, tl,
                                      scenario_counter)
            all_results.update({f"{k}_L{tl}": v for k, v in r.items()})

        if suite in ("all", "ood"):
            r = run_suite_ood(m, d, policy, sim_cfg, body_ids, args, tl,
                              scenario_counter)
            all_results.update({f"{k}_L{tl}": v for k, v in r.items()})

        if suite in ("all", "commands"):
            r = run_suite_commands(m, d, policy, sim_cfg, body_ids, args, tl,
                                   scenario_counter)
            all_results.update({f"{k}_L{tl}": v for k, v in r.items()})

    print(f"\n{'='*100}")
    print(f"  Completed {len(all_results)} test scenarios  "
          f"(expected 84 for suite=all, test_level=all)")
    print(f"{'='*100}")

    # --- Save JSON ---
    if args.output:
        out_path = os.path.join(OUTPUT_DIR, args.output)
        metadata = {
            "policy":       policy_path,
            "suite":        suite,
            "test_levels":  test_levels,
            "num_episodes": args.num_episodes,
            "max_steps":    args.max_steps,
            "seed":         args.seed,
            "scenario_count": len(all_results),
        }
        save_results_json(all_results, out_path, metadata=metadata)


if __name__ == "__main__":
    main()
