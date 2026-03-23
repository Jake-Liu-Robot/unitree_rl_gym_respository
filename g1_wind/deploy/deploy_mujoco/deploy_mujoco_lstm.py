"""
MuJoCo deployment script for the G1 wind-robust walking policy (LSTM).

Features:
  - LSTM policy inference with internal hidden state management
  - 3-layer wind model (base + OU turbulence + gusts) applied as body forces
  - Wind direction arrow visualized in the MuJoCo viewer
  - Console status printout every 2 seconds

Usage:
  python deploy/deploy_mujoco/deploy_mujoco_lstm.py g1_wind.yaml
  python deploy/deploy_mujoco/deploy_mujoco_lstm.py g1_wind.yaml --wind_level 5
  python deploy/deploy_mujoco/deploy_mujoco_lstm.py g1_wind.yaml --wind_level 0  # no wind
"""

import time
import argparse

import numpy as np
import torch
import yaml
import mujoco
import mujoco.viewer

from legged_gym import LEGGED_GYM_ROOT_DIR


# ──────────────────────────────────────────────────────────────────────────────
# Wind Model  (NumPy single-env version of wind_model.py)
# ──────────────────────────────────────────────────────────────────────────────

class WindModel:
    SPEED_RANGES = [
        [0.0,  0.0],   # L0  — no wind
        [1.0,  3.0],   # L1  — light
        [2.0,  5.0],   # L2  — light-medium
        [4.0,  8.0],   # L3  — medium
        [7.0, 12.0],   # L4  — strong + gusts
        [10.0, 18.0],  # L5  — extreme (gale, ~43% body weight)
        [15.0, 24.0],  # L6  — violent storm
        [20.0, 30.0],  # L7  — hurricane cat-1
        [25.0, 38.0],  # L8  — hurricane cat-2
        [30.0, 46.0],  # L9  — hurricane cat-3
        [35.0, 55.0],  # L10 — hurricane cat-4
    ]
    SPEED_CLAMPS = [0.0, 5.0, 8.0, 13.0, 20.0, 28.0,
                    36.0, 45.0, 55.0, 65.0, 75.0]

    def __init__(self, level=3, dt=0.02):
        self.dt    = dt
        self.level = max(0, min(10, level))
        self.wind_velocity = np.zeros(3)
        self.reset()

    def set_level(self, level):
        self.level = max(0, min(10, level))
        self.reset()

    def reset(self):
        lo, hi = self.SPEED_RANGES[self.level]
        self.base_speed = np.random.uniform(lo, hi)
        self.base_angle = np.random.uniform(0, 2 * np.pi)

        # OU process parameters (randomized per episode)
        self.ou_speed     = 0.0
        self.ou_angle     = 0.0
        self.ou_theta     = np.random.uniform(0.2, 1.0)
        self.ou_sigma     = np.random.uniform(0.05, 0.25)
        self.ou_theta_dir = np.random.uniform(0.05, 0.5)
        self.ou_sigma_dir = np.random.uniform(0.02, 0.25)

        # Gust state
        self.gust_active   = False
        self.gust_speed    = 0.0
        self.gust_dir      = np.zeros(3)
        self.gust_elapsed  = 0.0
        self.gust_duration = 0.0
        self.wind_velocity = np.zeros(3)

    def step(self):
        dt = self.dt

        # Layer 2: OU speed fluctuation
        self.ou_speed += (- self.ou_theta * self.ou_speed * dt
                          + self.ou_sigma * np.sqrt(dt) * np.random.randn())
        eff_speed = float(np.clip(self.base_speed + self.ou_speed,
                                  0.0, self.SPEED_CLAMPS[self.level]))

        # Layer 2: OU directional drift
        self.ou_angle += (- self.ou_theta_dir * self.ou_angle * dt
                          + self.ou_sigma_dir * np.sqrt(dt) * np.random.randn())
        angle    = self.base_angle + self.ou_angle
        base_vel = eff_speed * np.array([np.cos(angle), np.sin(angle), 0.0])

        # Layer 3: Gust events
        if not self.gust_active and np.random.random() < 0.1 * dt:
            self.gust_active   = True
            self.gust_elapsed  = 0.0
            self.gust_speed    = np.random.uniform(2.0, 6.0)
            g_angle            = angle + np.random.uniform(-np.pi / 3, np.pi / 3)
            self.gust_dir      = np.array([np.cos(g_angle), np.sin(g_angle), 0.0])
            self.gust_duration = np.random.uniform(1.5, 3.0)

        gust_vel = np.zeros(3)
        if self.gust_active:
            self.gust_elapsed += dt
            t, dur = self.gust_elapsed, self.gust_duration
            ru, rd = 0.3, 0.5
            if   t < ru:        envelope = t / ru
            elif t < dur - rd:  envelope = 1.0
            elif t < dur:       envelope = (dur - t) / rd
            else:               envelope = 0.0; self.gust_active = False
            gust_vel = self.gust_speed * envelope * self.gust_dir

        self.wind_velocity = base_vel + gust_vel
        return self.wind_velocity.copy()


# ──────────────────────────────────────────────────────────────────────────────
# Per-body aerodynamic force computation
# ──────────────────────────────────────────────────────────────────────────────

# Bodies that receive wind forces and their parameters (from g1_wind_env.py)
_WIND_BODIES    = ['pelvis',
                   'left_hip_yaw_link', 'right_hip_yaw_link',
                   'left_knee_link',    'right_knee_link']
_FRACTIONS      = [0.55, 0.12, 0.12, 0.08, 0.08]
_COP_OFFSETS    = [0.10, 0.005, 0.005, 0.002, 0.002]   # body-local z offset (m)
_FORCE_CLAMPS   = [5, 15, 30, 60, 100, 150,
                   250, 380, 550, 750, 1000]             # per-level max total force (N)

RHO     = 1.225   # air density  (kg/m³)
CD      = 1.1     # drag coefficient
A_FRONT = 0.50    # projected area — frontal  (m²)
A_SIDE  = 0.22    # projected area — lateral  (m²)
A_TOP   = 0.10    # projected area — vertical (m²)
Z_REF   = 0.85    # reference height — pelvis (m)
ALPHA   = 0.28    # boundary-layer exponent (urban terrain)


def _quat_z_axis(q):
    """Return body-frame z-axis in world frame given MuJoCo quaternion [w,x,y,z]."""
    w, x, y, z = q
    return np.array([2*(x*z + w*y),
                     2*(y*z - w*x),
                     1 - 2*(x*x + y*y)])


def _quat_rot_inv(q, v):
    """Rotate world-frame vector v into body frame using quaternion [w,x,y,z]."""
    w, x, y, z = q
    R = np.array([[1-2*(y*y+z*z), 2*(x*y+w*z),   2*(x*z-w*y)],
                  [2*(x*y-w*z),   1-2*(x*x+z*z), 2*(y*z+w*x)],
                  [2*(x*z+w*y),   2*(y*z-w*x),   1-2*(x*x+y*y)]])
    return R.T @ v   # R.T = world→body


def apply_wind_forces(m, d, wind_vel, level):
    """Compute aerodynamic forces and write them to d.xfrc_applied.

    Forces are applied at the center of pressure (not COM) so that
    the resulting torque tilts the robot realistically.
    Returns the total applied force magnitude (N).
    """
    d.xfrc_applied[:] = 0.0

    raw_forces = []   # (body_id, F_world, cop_pos, body_pos)
    total_raw  = 0.0

    for name, fraction, cop_z in zip(_WIND_BODIES, _FRACTIONS, _COP_OFFSETS):
        body_id  = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        body_pos = d.xpos[body_id].copy()
        body_vel = d.cvel[body_id, 3:6].copy()   # linear vel, world frame

        # Height-dependent wind speed (boundary layer power law)
        z          = max(body_pos[2], 0.1)
        v_wind_z   = wind_vel * (z / Z_REF) ** ALPHA
        v_rel      = v_wind_z - body_vel
        v_rel_mag  = float(np.linalg.norm(v_rel))

        if v_rel_mag < 0.01:
            raw_forces.append((body_id, np.zeros(3), body_pos, body_pos))
            continue

        v_rel_hat = v_rel / v_rel_mag

        # 3D ellipsoidal projected area in body-local frame
        v_local = _quat_rot_inv(d.xquat[body_id], v_rel_hat)
        dx, dy, dz = abs(v_local[0]), abs(v_local[1]), abs(v_local[2])
        A_eff = float(np.sqrt((A_FRONT * dx)**2 + (A_SIDE * dy)**2 + (A_TOP * dz)**2))

        F_mag   = 0.5 * RHO * CD * A_eff * fraction * v_rel_mag**2
        F_world = F_mag * v_rel_hat

        # Center of pressure: offset along body-local z-axis
        cop_pos = body_pos + cop_z * _quat_z_axis(d.xquat[body_id])

        total_raw += F_mag
        raw_forces.append((body_id, F_world, cop_pos, body_pos))

    # Uniform clamp to per-level maximum
    clamp = _FORCE_CLAMPS[level]
    scale = min(1.0, clamp / (total_raw + 1e-6))

    for body_id, F_world, cop_pos, body_pos in raw_forces:
        F = F_world * scale
        torque = np.cross(cop_pos - body_pos, F)
        d.xfrc_applied[body_id, :3] += torque   # torque (world frame)
        d.xfrc_applied[body_id, 3:6] += F        # force  (world frame)

    return total_raw * scale


# ──────────────────────────────────────────────────────────────────────────────
# Wind arrow visualization
# ──────────────────────────────────────────────────────────────────────────────

# Arrow colour per wind level: blue → cyan → green → yellow → orange → red → purple → white
_LEVEL_RGBA = [
    [0.4, 0.4, 1.0, 0.9],   # L0  blue
    [0.0, 0.8, 1.0, 0.9],   # L1  cyan
    [0.0, 1.0, 0.4, 0.9],   # L2  green
    [1.0, 1.0, 0.0, 0.9],   # L3  yellow
    [1.0, 0.5, 0.0, 0.9],   # L4  orange
    [1.0, 0.1, 0.1, 0.9],   # L5  red
    [0.8, 0.0, 0.2, 0.9],   # L6  dark red
    [0.6, 0.0, 0.6, 0.9],   # L7  purple
    [0.8, 0.0, 1.0, 0.9],   # L8  violet
    [1.0, 0.0, 0.8, 0.9],   # L9  magenta
    [1.0, 1.0, 1.0, 0.9],   # L10 white
]

# Arrow drawn at an elevated position behind the robot
_ARROW_ORIGIN = np.array([-1.5, 0.0, 1.0], dtype=np.float64)
_ARROW_MIN_LEN = 0.3    # always at least this long so it's visible at low speeds
_ARROW_MAX_LEN = 1.5    # cap so it doesn't extend too far at high speeds
_ARROW_WIDTH   = 0.06   # shaft thickness


def draw_wind_arrow(viewer, wind_vel, level):
    """Draw a wind-direction arrow in a fixed corner of the scene (thread-safe)."""
    speed = float(np.linalg.norm(wind_vel))

    with viewer.lock():
        viewer.user_scn.ngeom = 0
        if speed < 0.01:
            return

        wind_dir  = wind_vel / speed
        # Scale length with speed but enforce min/max so it's always readable
        arrow_len = float(np.clip(speed / 5.0 * _ARROW_MAX_LEN,
                                  _ARROW_MIN_LEN, _ARROW_MAX_LEN))
        arrow_end = _ARROW_ORIGIN + wind_dir * arrow_len

        geom = viewer.user_scn.geoms[0]
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW,
                             _ARROW_WIDTH, _ARROW_ORIGIN, arrow_end)
        geom.rgba[:] = _LEVEL_RGBA[level]
        viewer.user_scn.ngeom = 1


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_gravity_orientation(quat):
    """Project gravity vector into base frame from quaternion [w, x, y, z]."""
    qw, qx, qy, qz = quat
    return np.array([ 2*(-qz*qx + qw*qy),
                     -2*( qz*qy + qw*qx),
                      1 - 2*(qw*qw + qz*qz)])


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str,
                        help="Config filename inside deploy/deploy_mujoco/configs/")
    parser.add_argument("--wind_level", type=int, default=3,
                        help="Wind curriculum level 0-5 (default: 3)")
    args = parser.parse_args()

    # ── Load config ──────────────────────────────────────────────────────────
    cfg_path = f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{args.config_file}"
    with open(cfg_path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    policy_path      = cfg["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
    xml_path         = cfg["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
    sim_duration     = cfg["simulation_duration"]
    sim_dt           = cfg["simulation_dt"]
    ctrl_decimation  = cfg["control_decimation"]
    kps              = np.array(cfg["kps"],            dtype=np.float32)
    kds              = np.array(cfg["kds"],            dtype=np.float32)
    default_angles   = np.array(cfg["default_angles"], dtype=np.float32)
    ang_vel_scale    = cfg["ang_vel_scale"]
    dof_pos_scale    = cfg["dof_pos_scale"]
    dof_vel_scale    = cfg["dof_vel_scale"]
    action_scale     = cfg["action_scale"]
    cmd_scale        = np.array(cfg["cmd_scale"],      dtype=np.float32)
    num_actions      = cfg["num_actions"]
    num_obs          = cfg["num_obs"]
    cmd              = np.array(cfg["cmd_init"],       dtype=np.float32)

    # ── Init buffers ─────────────────────────────────────────────────────────
    action         = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs            = np.zeros(num_obs,     dtype=np.float32)
    counter        = 0

    # ── Load MuJoCo model ────────────────────────────────────────────────────
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = sim_dt

    # ── Load LSTM policy ─────────────────────────────────────────────────────
    policy = torch.jit.load(policy_path)
    policy.reset_memory()

    # ── Wind model ───────────────────────────────────────────────────────────
    control_dt = sim_dt * ctrl_decimation   # 0.02 s
    wind_model = WindModel(level=args.wind_level, dt=control_dt)
    wind_vel   = np.zeros(3)

    print("\n=== G1 Wind-Robust Walking — MuJoCo Deployment ===")
    print(f"  Policy     : {policy_path}")
    print(f"  Wind level : {args.wind_level}  "
          f"({wind_model.SPEED_RANGES[args.wind_level]} m/s)")
    print(f"  Command    : vx={cmd[0]:.1f}  vy={cmd[1]:.1f}  yaw={cmd[2]:.1f}")
    print("==================================================\n")

    # ── Simulation loop ──────────────────────────────────────────────────────
    with mujoco.viewer.launch_passive(m, d) as viewer:
        viewer.opt.label = mujoco.mjtLabel.mjLABEL_NONE                        # suppress body/geom name labels
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False          # suppress contact force arrows
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False          # suppress contact point dots
        start = time.time()

        while viewer.is_running() and time.time() - start < sim_duration:
            step_start = time.time()

            # ── Control step (every ctrl_decimation physics steps) ──────────
            if counter % ctrl_decimation == 0:

                # Build observation
                qj      = (d.qpos[7:] - default_angles) * dof_pos_scale
                dqj     = d.qvel[6:] * dof_vel_scale
                omega   = d.qvel[3:6] * ang_vel_scale
                gravity = get_gravity_orientation(d.qpos[3:7])

                t_sim     = counter * sim_dt
                phase     = (t_sim % 0.8) / 0.8
                sin_phase = np.sin(2 * np.pi * phase)
                cos_phase = np.cos(2 * np.pi * phase)

                # Yaw correction: steer back toward target heading (0 = facing +X)
                quat = d.qpos[3:7]  # MuJoCo quaternion [w, x, y, z]
                siny = 2.0 * (quat[0]*quat[3] + quat[1]*quat[2])
                cosy = 1.0 - 2.0 * (quat[2]**2 + quat[3]**2)
                yaw  = np.arctan2(siny, cosy)
                yaw_err = np.arctan2(np.sin(-yaw), np.cos(-yaw))  # error to target_yaw=0
                cmd[2] = float(np.clip(yaw_err * 2.0, -1.0, 1.0))

                obs[:3]                              = omega
                obs[3:6]                             = gravity
                obs[6:9]                             = cmd * cmd_scale
                obs[9:9 + num_actions]               = qj
                obs[9 + num_actions:9 + 2*num_actions]   = dqj
                obs[9 + 2*num_actions:9 + 3*num_actions] = action
                obs[9 + 3*num_actions:9 + 3*num_actions + 2] = [sin_phase, cos_phase]

                # LSTM policy inference
                obs_tensor     = torch.from_numpy(obs).unsqueeze(0)
                action         = policy(obs_tensor).detach().numpy().squeeze()
                target_dof_pos = action * action_scale + default_angles

                # Step wind model and apply forces for this control cycle
                wind_vel   = wind_model.step()
                wind_force = apply_wind_forces(m, d, wind_vel, wind_model.level)

                # Update wind arrow in viewer
                draw_wind_arrow(viewer, wind_vel, wind_model.level)

                # Console status every 2 s
                ctrl_step = counter // ctrl_decimation
                if ctrl_step % 100 == 0:
                    speed = float(np.linalg.norm(wind_vel[:2]))
                    print(f"t={t_sim:5.1f}s  |  wind={speed:4.1f} m/s  "
                          f"force={wind_force:5.1f} N  level={wind_model.level}  "
                          f"cmd=[{cmd[0]:.1f}, {cmd[1]:.1f}, {cmd[2]:.1f}]")

            # ── PD control → physics step ───────────────────────────────────
            tau      = pd_control(target_dof_pos, d.qpos[7:], kps,
                                  np.zeros_like(kds),    d.qvel[6:], kds)
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)
            counter  += 1

            viewer.sync()

            # Real-time pacing
            elapsed = time.time() - step_start
            if sim_dt - elapsed > 0:
                time.sleep(sim_dt - elapsed)
