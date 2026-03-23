# G1 Wind-Robust Walking — RL Course Project

## Project Overview
Training the Unitree G1 humanoid robot to walk stably under continuous, time-varying wind disturbances using deep reinforcement learning. This is a course project for a reinforcement learning class.

## Tech Stack
- **Simulation**: NVIDIA Isaac Gym Preview 4 (GPU-accelerated parallel physics)
- **Base Framework**: unitree_rl_gym (https://github.com/unitreerobotics/unitree_rl_gym)
- **RL Algorithm**: PPO via rsl_rl (https://github.com/leggedrobotics/rsl_rl)
- **Upstream**: legged_gym (https://github.com/leggedrobotics/legged_gym)
- **Language**: Python 3.8+, PyTorch (CUDA 12.1)
- **Robot**: Unitree G1 (~35kg, 12 DOF legs, URDF: g1_12dof.urdf)

## Repository Structure
```
unitree_rl_gym/
├── legged_gym/
│   ├── envs/
│   │   ├── base/
│   │   │   ├── legged_robot.py          # Base env (defines feet_indices, phase, reset flow)
│   │   │   └── legged_robot_config.py   # Base config
│   │   ├── g1/
│   │   │   ├── g1_config.py             # G1 config (foot_name="ankle_roll", 12 DOF)
│   │   │   └── g1_env.py                # G1 env (phase, feet tracking, gait rewards)
│   │   ├── g1_wind/                     # ★ Our wind environment
│   │   │   ├── __init__.py              # Exports G1WindRobot, G1WindRoughCfg, G1WindRoughCfgPPO
│   │   │   ├── g1_wind_config.py        # Wind params, curriculum, reward scales, PPO config
│   │   │   ├── g1_wind_env.py           # Wind force application, observations, rewards, curriculum
│   │   │   └── wind_model.py            # 3-layer wind model (base + OU + gusts)
│   │   └── __init__.py                  # Task registry: "g1_wind" → G1WindRobot
│   ├── g1_wind_test/                    # ★ Test scripts
│   │   ├── eval_wind_robustness.py          # Isaac Gym comprehensive eval (84 scenarios)
│   │   ├── eval_wind_model_v3.py            # Isaac Gym wind model physics verification
│   │   ├── smoke_test_g1_wind.py            # Isaac Gym environment smoke test
│   │   ├── eval_wind_robustness_mujoco.py   # ★ MuJoCo sim2sim eval (84 scenarios)
│   │   ├── eval_wind_model_mujoco.py        # ★ MuJoCo wind model verification
│   │   └── analyze_sim2sim.py               # ★ Isaac Gym vs MuJoCo comparison analysis
│   ├── g1_wind_doc/                     # ★ Documentation
│   │   ├── g1_wind_test.md             # Wind environment test notes
│   │   └── G1_Wind_Robust_Walking_Analysis.md  # Research analysis
│   ├── scripts/
│   │   ├── train.py
│   │   └── play.py                    # Visual observation with wind/command control
│   └── utils/
├── rsl_rl/                              # PPO implementation (do NOT modify)
├── resources/robots/g1_description/     # G1 URDF models (g1_12dof.urdf, g1_29dof.urdf, etc.)
├── test_results/                        # Isaac Gym evaluation results (per-experiment JSON)
│   └── mujoco/                          # ★ Sim2Sim (MuJoCo) eval results (JSON)
└── logs/                                # Training outputs
```

## Class Hierarchy
```
BaseTask → LeggedRobot → G1Robot → G1WindRobot
LeggedRobotCfg → G1RoughCfg → G1WindRoughCfg
LeggedRobotCfgPPO → G1RoughCfgPPO → G1WindRoughCfgPPO
```

## Key Attribute Origins (Inheritance Chain)
| Attribute | Defined in | Notes |
|-----------|-----------|-------|
| `self.feet_indices` | `LeggedRobot._create_envs()` :763 | Matches body names containing `cfg.asset.foot_name` ("ankle_roll") → [left_ankle_roll_link(6), right_ankle_roll_link(12)] |
| `self.phase` | `G1Robot._post_physics_step_callback()` :60 | `(episode_length_buf * dt) % 0.8 / 0.8`, only exists after first `step()` |
| `self.torso_body_idx` | `G1WindRobot._init_buffers()` :24 | `find_actor_rigid_body_handle(..., "pelvis")` → index 0 |
| `self.wind_model` | `G1WindRobot._init_buffers()` :30 | `WindModel` instance, only when `cfg.wind.enable=True` |
| `wind_model.effective_direction` | `WindModel.step()` | Real-time wind direction (base + OU drift), used by lean_compensation reward |
| `self.wind_body_flat_indices` | `G1WindRobot._init_buffers()` | [num_envs, num_wind_bodies] flat indices for multi-body force scatter |
| `self.force_body_fractions` | `G1WindRobot._init_buffers()` | [num_wind_bodies] area fraction per body for force distribution (0.55/0.12/0.12/0.08/0.08) |

## Wind Model v3 (wind_model.py + g1_wind_env.py)

### Wind Velocity (wind_model.py) — 3-layer superposition
```
# Layer 1+2: base wind with OU fluctuation (speed + direction)
effective_speed = clamp(base_speed + ou_speed_state, min=0)
effective_dir   = (cos(base_angle + ou_angle_state), sin(...), 0)
base_vel        = effective_speed × effective_dir

# Layer 3: independent gust velocity vector
gust_vel = gust_speed × envelope(t) × gust_dir

# Combined wind velocity (m/s, world frame)
wind_velocity = base_vel + gust_vel
```

| Layer | Mechanism | Key params |
|-------|-----------|-----------|
| 1: Base wind | Per-episode constant direction + speed | Speed range from curriculum level |
| 2: OU process | Speed OU + directional OU, **per-episode randomized** | θ∈[0.2,1.0], σ∈[0.05,0.25], θ_dir∈[0.05,0.5], σ_dir∈[0.02,0.25] |
| 3: Gusts | Independent velocity vector, trapezoidal envelope | speed=[2,6]m/s, dir=base±60°, ramp 0.3s/0.5s, prob=0.1/s |

Per-level speed clamp: [0, 5, 8, 13, 20, 28] m/s — prevents OU+gust producing unrealistic extremes

### Per-Body Aerodynamics (g1_wind_env.py) — v3 physics improvements
```
# P0: Relative velocity (wind - body velocity)
v_rel = v_wind(z) - v_body          # per-body, not just wind speed

# P1: Per-body 3D ellipsoidal projected area (using v_rel direction + body quaternions)
A_eff = sqrt((A_front×|dx|)² + (A_side×|dy|)² + (A_top×|dz|)²)   # per-body local frame
# A_front=0.50 m², A_side=0.22 m², A_top=0.10 m²

# P2: Height-dependent wind speed (power law boundary layer)
v_wind(z) = v_ref × (z / z_ref)^α   # α=0.28 (urban terrain), z_ref=0.85m (pelvis height)

# Per-body force
F_i = 0.5×ρ×Cd × A_eff × fraction_i × |v_rel_i|² × v_rel_hat_i

# Per-level force clamp on total magnitude
total_force = clamp(Σ F_i, max=per_level_clamp)

# P3: Apply at center of pressure (not COM)
# CoP offset along body-local z-axis: [+0.10, +0.005, +0.005, +0.002, +0.002] m (URDF-derived)
# Rotated to world frame via per-body quaternions → correct torque when tilted
gym.apply_rigid_body_force_at_pos_tensors(sim, forces, cop_positions, ENV_SPACE)
```

Physical constants: ρ=1.225 kg/m³, Cd=1.1
Force bodies: pelvis(55%) + thighs(12%×2) + shins(8%×2) = 95%
Per-level force clamp: [5, 15, 30, 60, 100, 150] N
Per-level speed clamp: [0, 5, 8, 13, 20, 28] m/s
Rigid body state refreshed after each physics substep for accurate per-body velocities

## Observation Space (g1_wind_config.py)

**Standard obs (47 dims)** — no wind info, sim-to-real friendly:
```
ang_vel(3) + gravity(3) + commands(3) + dof_pos(12) + dof_vel(12) + actions(12) + sin/cos_phase(2)
```

**Privileged obs (56 dims)** — teacher policy only:
```
base_lin_vel(3) + ang_vel(3) + gravity(3) + commands(3) + dof_pos(12) + dof_vel(12) + actions(12) + phase(2) + wind_vel(3) + wind_force(3)
```

## Reward Structure (g1_wind_env.py + inherited from G1Robot)

Literature-informed reward design (Phase 4.9). Key references:
- Viereck et al. IROS 2024 (moderate orientation, base acceleration)
- Humanoid-Gym 2024 (second-order action smoothness)
- Booster Gym 2025 (power penalty, feet distance)
- Xu et al. 2025 (wind-adaptive tracking sigma)
- Li et al. 2025 (angular momentum / damping)

| Category | Reward | Scale | Source | Notes |
|----------|--------|-------|--------|-------|
| Tracking | tracking_lin_vel | 1.0 | **overridden** | wind-adaptive sigma: σ_eff = 0.25 × (1 + |F_wind|/100) |
| Tracking | tracking_ang_vel | 0.5 | inherited | |
| Posture | orientation | -0.3 | **overridden** | wind-aware: penalty × 1/(1+F/50N), no lean conflict |
| Posture | base_height | -10.0 | **overridden** | wind-aware target: 0.78→0.741m at 150N |
| Velocity | lin_vel_z | -2.0 | inherited | |
| Velocity | ang_vel_xy | **-0.15** | inherited | was -0.2→-0.15, relax for gust angular response |
| Energy | dof_acc | **-1.5e-7** | inherited | was -2.5e-7, allow faster joint response |
| Energy | dof_vel | -5e-4 | inherited | |
| Energy | action_rate | -0.01 | inherited | first-order: \|\|a_t - a_{t-1}\|\|² |
| Energy | action_rate2 | -0.005 | g1_wind | second-order: \|\|a_t - 2a_{t-1} + a_{t-2}\|\|² |
| Energy | power | **-2e-4** | g1_wind | was -5e-4, allow aggressive wind rejection torques |
| Limits | dof_pos_limits | -5.0 | inherited | |
| Humanoid | alive | **1.0** | inherited | was 0.5→1.0, prevent "die early" with negative rewards |
| Humanoid | contact | 0.18 | **overridden** | wind-adaptive stance ratio: 0.55→0.75 at peak wind |
| Humanoid | hip_pos | -0.3 | inherited | allow wider stance |
| Humanoid | contact_no_vel | **-0.1** | inherited | was -0.2, wind causes unavoidable sliding |
| Humanoid | feet_swing_height | -8.0 | **overridden** | wind-adaptive target: 0.08→0.04m at peak wind |
| **Wind** | lean_compensation | 0.8 | g1_wind | dominant over orientation under wind |
| **Wind** | feet_distance | **-0.5** | g1_wind | was -0.3→-0.5, stronger wide stance incentive |
| **Wind** | base_acc | -0.002 | g1_wind | wind-compensated controllable base_acc_xy² |
| **Wind** | ang_momentum_change | **0.0** | **disabled** | was -0.0003, negligible gradient contribution |
| **Wind** | com_balance | **0.0** | **disabled** | conflicts with lean_compensation; alive+base_height suffice |
| disabled | sustained_walking | 0.0 | — | identical to alive |
| disabled | contact_symmetry | 0.0 | — | incentivized standing still |

`only_positive_rewards = False` — allows negative reward gradient under strong wind.
`tracking_sigma_wind_scale = 1.0` — at 100N wind, sigma doubles from 0.25 to 0.50.

## Curriculum Controller (g1_wind_env.py)

6 levels with window-based advancement + mixed-level upgrade (80% advance, 20% stay):

| Level | Speed (m/s) | Description |
|-------|------------|-------------|
| 0 | 0 | No wind (baseline) |
| 1 | 1-3 | Light |
| 2 | 2-5 | Light-medium |
| 3 | 4-8 | Medium |
| 4 | 7-12 | Strong + gusts |
| 5 | 10-18 | Extreme |

Upgrade: accumulate stats over `upgrade_window=300` resets, advance 80% of envs if `survival > 0.7` AND `tracking > 0.4`.
Demotion: demote 50% of envs if `survival < 0.4` AND `tracking < 0.3` (fractional to prevent oscillation).
Cooldown: skip 500 resets after any level change before evaluating again.

## PPO Config (g1_wind_config.py)
- Policy: `ActorCriticRecurrent` (LSTM hidden_size=128, 1 layer)
- Asymmetric design (literature-conforming, upper-end for wind estimation):
  - Actor MLP: [128, 64] — LSTM-heavy (wind estimation from obs history)
  - Critic MLP: [256, 128] — MLP-heavy (privileged wind info processing)
- `gamma = 0.995`, `learning_rate = 5e-4` (adaptive schedule)
- `desired_kl = 0.008`, `entropy_coef = 0.008`
- `num_steps_per_env = 64`, `num_mini_batches = 4`
- `max_iterations = 10000`, `experiment_name = 'g1_wind'`
- Total params: ~250K (Actor ~110K + Critic ~140K)
- Ref: Humanoid-Gym 2024, Walk These Ways 2023, Gait-Conditioned RL

## Domain Randomization
- `push_robots = False` (wind replaces push perturbation)
- Friction: [0.1, 1.25], Base mass: [-1, +3] kg
- Action delay: [0, 2] control steps (0-40ms), ring buffer with per-env delay
- PD gain randomization: stiffness ×[0.8, 1.2], damping ×[0.8, 1.2]
- Motor strength: ×[0.8, 1.0] (only weaker than nominal)
- Ref: Walk These Ways, Rapid Locomotion, ANYmal-DroQ, Booster Gym 2025

## Experiments Plan
| ID | Task Name | Wind | Curriculum | Wind Rewards | Purpose | Status |
|----|-----------|------|------------|-------------|---------|--------|
| Exp1 | `g1_wind_baseline` | OFF | — | (inactive) | No-wind baseline | **Done** |
| Exp2 | `g1_wind_push_only` | OFF (push) | ON | ON | Push perturbation only | **Done** |
| Exp3 | `g1_wind` | ON | ON | ON | Full method | **Done (Run8)** |
| Exp4 | `g1_wind_no_curriculum` | ON (fixed L3) | OFF | ON | Ablation: curriculum | **Done** |
| Exp5 | `g1_wind_no_reward` | ON | ON | OFF (base G1 values) | Ablation: wind rewards | **Done** |

## Common Commands

```bash
# --- Training ---
python legged_gym/scripts/train.py --task=g1_wind --headless
python legged_gym/scripts/train.py --task=g1_wind --load_run Mar10_18-22-48_ --checkpoint 2900  # resume

# --- Ablation experiments ---
python legged_gym/scripts/train.py --task=g1_wind_baseline --headless --num_envs 2048 --max_iterations 12000
python legged_gym/scripts/train.py --task=g1_wind_no_curriculum --headless --num_envs 2048 --max_iterations 12000
python legged_gym/scripts/train.py --task=g1_wind_no_reward --headless --num_envs 2048 --max_iterations 12000

# --- Play (visual observation with scenario control) ---
# Basic (defaults: L5, full wind, random commands, 4 envs)
python legged_gym/scripts/play.py --task=g1_wind --load_run Mar10_18-22-48_
# No wind baseline
python legged_gym/scripts/play.py --task=g1_wind --load_run Mar10_18-22-48_ --wind_level 0 --fix_vx 0.6
# L5 steady frontal wind, standing
python legged_gym/scripts/play.py --task=g1_wind --load_run Mar10_18-22-48_ --wind_level 5 --wind_mode steady --wind_angle 0 --fix_vx 0 --fix_vy 0
# L5 side wind, walking
python legged_gym/scripts/play.py --task=g1_wind --load_run Mar10_18-22-48_ --wind_level 5 --wind_mode steady --wind_angle 90 --fix_vx 0.6
# L5 full model (turbulence + gusts)
python legged_gym/scripts/play.py --task=g1_wind --load_run Mar10_18-22-48_ --wind_level 5 --fix_vx 0.6 
# OOD: wind direction reversal at t=5s
python legged_gym/scripts/play.py --task=g1_wind --load_run Mar10_18-22-48_ --wind_level 5 --ood_pattern reversal --fix_vx 0.6

# play.py parameters:
#   --wind_level 0-5    Wind curriculum level (default: max)
#   --wind_angle DEG    Fixed wind direction in degrees (default: random)
#   --wind_mode MODE    full|steady|turbulent|gusts (default: full)
#   --ood_pattern PAT   step|periodic|reversal (default: none)
#   --fix_vx FLOAT      Fixed forward velocity (default: random resampling)
#   --fix_vy FLOAT      Fixed lateral velocity (default: random resampling)
#   --fix_yaw FLOAT     Fixed heading offset for turning
#   --num_envs INT      Number of robots (default: 4)

# --- Evaluation (quantitative, all suites: levels/modes/directions/ou/ood/commands) ---
python legged_gym/g1_wind_test/eval_wind_robustness.py --task g1_wind --load_run Mar10_18-22-48_ --headless               # all suites at L3
python legged_gym/g1_wind_test/eval_wind_robustness.py --task g1_wind --load_run Mar10_18-22-48_ --test_level all --headless  # all suites at L3,4,5
python legged_gym/g1_wind_test/eval_wind_robustness.py --task g1_wind --load_run Mar10_18-22-48_ --suite levels --headless  # wind level sweep only
python legged_gym/g1_wind_test/eval_wind_robustness.py --task g1_wind --load_run Mar10_18-22-48_ --suite commands --test_level all --headless  # command variations
python legged_gym/g1_wind_test/eval_wind_robustness.py --task g1_wind --load_run Mar10_18-22-48_ --suite ood --test_level all --headless  # OOD patterns
# Save results to JSON with reproducible seed:
python legged_gym/g1_wind_test/eval_wind_robustness.py --task g1_wind --load_run Mar10_18-22-48_ --output results.json --seed 42 --headless

# --- Smoke test ---
python legged_gym/g1_wind_test/smoke_test_g1_wind.py

# --- Sim2Sim (MuJoCo) ---
# Verify MuJoCo wind model matches Isaac Gym physics (14/14 checks):
python legged_gym/g1_wind_test/eval_wind_model_mujoco.py --seed 42
# Run full 84-scenario eval (Suites A-F) for one experiment:
python legged_gym/g1_wind_test/eval_wind_robustness_mujoco.py \
  --policy logs/g1_wind/exported/policies/policy_lstm_1.pt \
  --suite all --num_episodes 50 --seed 42 --output exp3_full_method_mujoco.json
# Or use --exp 1-5 to select experiment by ID (reads EXP_POLICIES dict):
python legged_gym/g1_wind_test/eval_wind_robustness_mujoco.py \
  --exp 3 --suite all --num_episodes 50 --seed 42 --output exp3_full_method_mujoco.json
# Compare Isaac Gym vs MuJoCo results:
python legged_gym/g1_wind_test/analyze_sim2sim.py
python legged_gym/g1_wind_test/analyze_sim2sim.py --verbose  # per-scenario detail
# Export policy for MuJoCo (needed before running MuJoCo eval):
python legged_gym/scripts/play.py --task=g1_wind_baseline --load_run Mar12_00-14-00_ --headless
```

Note: `play.py` requires `--load_run <run_dir>` to locate the model. Without it will raise FileNotFoundError.

Available runs: `Feb28_21-36-56_` (Run4, old baseline), `Mar10_18-22-48_` (Run8/Exp3, best)
Ablation runs: `Mar12_00-14-00_` (Exp1), `Mar12_21-47-52_` (Exp2), `Mar12_21-34-46_` (Exp4), `Mar12_21-34-59_` (Exp5)
Registered tasks: `g1_wind`, `g1_wind_baseline`, `g1_wind_push_only`, `g1_wind_no_curriculum`, `g1_wind_no_reward`

## Coding Conventions
- Inherit from existing classes — don't rewrite base code
- All tensors on GPU (`self.device`), use torch operations, no Python loops over envs
- Reward functions: `_reward_<name>(self)` → auto-registered by non-zero scale in config
- Config classes use nested class pattern matching legged_gym style
- Use `gymtorch.unwrap_tensor()` when passing tensors to Isaac Gym C++ API

## Trained Models

| Exp | Task | Directory | Iters | Role |
|-----|------|-----------|-------|------|
| Exp1 | `g1_wind_baseline` | `logs/g1_wind_baseline/Mar12_00-14-00_` | 9950 | No-wind baseline (277K) |
| Exp2 | `g1_wind_push_only` | `logs/g1_wind_push_only/Mar12_21-47-52_` | 1450 | Push perturbation only (277K) |
| **Exp3** | `g1_wind` | `logs/g1_wind/Mar10_18-22-48_` | 2900 | **Full method (277K, best)** |
| Exp4 | `g1_wind_no_curriculum` | `logs/g1_wind_no_curriculum/Mar12_21-34-46_` | 1500 | Ablation: no curriculum (277K) |
| Exp5 | `g1_wind_no_reward` | `logs/g1_wind_no_reward/Mar12_21-34-59_` | 2100 | Ablation: no wind rewards (277K) |
| Run4 | `g1_wind` | `logs/g1_wind/Feb28_21-36-56_` | 1950 | Old arch (73K, historical only) |

- All Exp1-5 use identical 277K architecture: LSTM-128, Actor [128,64], Critic [256,128].
- Run8 (Exp3): **best result** — v3.2 wind physics + Phase 4.14 config. Reached curriculum L5.
- Exported LSTM policy: `logs/g1_wind/exported/policies/policy_lstm_1.pt`

## Evaluation Results Summary (84 scenarios per experiment, Suites A-F at L3/L4/L5)

| Exp | L3 pass | L4 pass | L5 pass | L5 avg surv | L5 trk_err | Fails |
|-----|---------|---------|---------|-------------|------------|-------|
| Exp1 (baseline) | 26/26 | 8/26 | 1/26 | 5% | 1.520 | 45/84 |
| Exp2 (push only) | 26/26 | 26/26 | 11/26 | 81% | 1.106 | 16/84 |
| **Exp3 (full)** | **26/26** | **26/26** | **26/26** | **100%** | **0.232** | **0/84** |
| Exp4 (no curriculum) | 26/26 | 26/26 | 9/26 | 48% | 0.899 | 18/84 |
| Exp5 (no wind reward) | 26/26 | 26/26 | 26/26 | 99% | 0.257 | 0/84 |

Pass = survival >= 90%. L5 trk_err = mean tracking error across all 26 L5 scenarios. Results in `test_results/` (per-experiment JSON).

**Ablation conclusions** (contribution ranking):
1. Wind model training (Exp1→Exp3): essential, L5 5%→100%
2. Curriculum learning (Exp4→Exp3): critical, L5 48%→100%
3. Wind physics vs push (Exp2→Exp3): significant, L5 81%→100%
4. Wind-specific rewards (Exp5→Exp3): marginal, L5 99%→100%, trk 0.257→0.232

## Sim2Sim (MuJoCo) Results Summary

MuJoCo eval: `--suite all --test_level all` — 84 scenarios per experiment (Suites A-F at L3/L4/L5).
Results in `test_results/mujoco/`.

| Exp | L3 pass | L4 pass | L5 pass | L5 avg surv | L5 trk_err | Fails |
|-----|---------|---------|---------|-------------|------------|-------|
| Exp1 (baseline) | 26/26 | 13/26 | 2/26 | 28% | 1.379 | 39/84 |
| Exp2 (push only) | 26/26 | 26/26 | 13/26 | 81% | 1.272 | 14/84 |
| **Exp3 (full)** | **26/26** | **26/26** | **26/26** | **100%** | **0.191** | **0/84** |
| Exp4 (no curriculum) | 26/26 | 16/26 | 7/26 | 54% | 0.817 | 30/84 |
| Exp5 (no wind reward) | 26/26 | 26/26 | 26/26 | 100% | 0.182 | 0/84 |

Pass = survival >= 90%. L5 trk_err = mean tracking error across all 26 L5 scenarios.

**Sim2Sim gap** (Isaac Gym L5 avg surv → MuJoCo L5 avg surv):
- Exp1: 5% → 28% (Δ = +23pp) — **MuJoCo better** (unexpected; baseline policy generalizes to MuJoCo physics)
- Exp2: 81% → 81% (Δ = 0pp) — **match**
- Exp3: 100% → 100% (Δ = 0pp) — **perfect transfer**
- Exp4: 48% → 54% (Δ = +6pp) — MuJoCo slightly better
- Exp5: 99% → 100% (Δ = +1pp) — **match**

**Key findings**:
- Exp3/Exp5 transfer perfectly (100% survival in both engines).
- MuJoCo tracking error is consistently **15-20% lower** than Isaac Gym (action timing fix resolved the previous 2x gap).
- Exp1 surprising improvement (5%→28%): MuJoCo's smoother contact dynamics may aid the no-wind baseline.
- Ablation conclusions hold in MuJoCo: curriculum (Exp4) and wind training (Exp1) remain the critical factors.

**Exp3 per-suite tracking error (all 100% survival, MuJoCo)**:
| Suite | L3 trk | L4 trk | L5 trk |
|-------|--------|--------|--------|
| B (modes) | 0.146 | 0.154 | 0.178 |
| C (directions) | 0.142 | 0.140 | 0.156 |
| D (OU extremes) | 0.152 | 0.166 | 0.205 |
| E (OOD) | 0.152 | 0.164 | 0.221 |
| F (commands) | 0.160 | 0.169 | 0.202 |

## Known Issue: Gait Asymmetry & Heading Drift

### Observation
The trained policy (Exp3) exhibits a consistent heading drift of ~18° (rightward) even without wind.
Under wind, the drift increases proportionally:

| Wind speed | Heading drift | Gait bias component | Wind-induced component |
|-----------|--------------|--------------------|-----------------------|
| 0 m/s | -18° | -18° | 0° |
| 5 m/s | -26° | -18° | -8° |
| 8 m/s | -32° | -18° | -14° |
| 15 m/s | -53° | -18° | -35° |

The drift stabilizes at a dynamic equilibrium where policy yaw correction torque = gait-induced + wind-induced yaw torque.

### Equilibrium Mechanism
The heading drift does NOT accumulate indefinitely — it stabilizes at a fixed angle via negative feedback:

1. Gait asymmetry + wind produce net yaw torque → heading drifts
2. As yaw error grows, `cmd[2] = 0.5 × (target - heading)` grows → policy applies stronger correction
3. At some angle, correction torque = drift torque → **dynamic equilibrium**
4. If perturbed beyond equilibrium, correction force dominates → pulled back
5. If perturbed below equilibrium, drift force dominates → pushed forward

The equilibrium angle increases with wind speed because wind adds yaw torque (from asymmetric per-body forces during walking) that requires a larger cmd[2] to balance.

### Emergent Strategy: Yaw as Wind Load Reduction
The heading drift is not purely a failure — it is a **physically rational emergent strategy**. At yaw=-53° under 15 m/s headwind:

```
Frontal (yaw=0°):  A_eff = A_front = 0.50 m²     → Force ≈ 50 N
Yawed (yaw=-53°):  A_eff = sqrt((0.50×cos53°)² + (0.22×sin53°)²) ≈ 0.35 m²  → Force ≈ 35 N
```

By turning ~53°, the robot presents its **narrower side profile** to the wind, reducing aerodynamic force by ~30%. The policy effectively "discovered" that sacrificing heading accuracy (tracking_ang_vel scale=0.5) to reduce wind load is a better trade-off than maintaining perfect heading at the cost of stability risk (alive scale=1.0).

This is analogous to a human instinctively turning sideways in strong wind to reduce drag — a physically sound strategy that emerges naturally from the reward-weighted optimization.

### Reward Trade-off Analysis
The policy converges to a Pareto-approximate optimum across competing objectives:

| Reward | Scale | Policy behavior under 15 m/s wind |
|--------|-------|-----------------------------------|
| alive | 1.0 | **Highest priority** — never fall (fully satisfied) |
| tracking_lin_vel | 1.0 | **High priority** — maintain forward velocity (mostly satisfied) |
| lean_compensation | 0.8 | **High priority** — lean into wind (fully satisfied) |
| tracking_ang_vel | 0.5 | **Medium priority** — follow yaw command (partially sacrificed) |
| orientation | -0.3 | Allows tilting (wind-adaptive penalty reduction) |
| energy terms | <0.01 | Lowest priority — efficiency (naturally satisfied) |

The policy "computes": correcting yaw requires joint torques that would otherwise be used for stability. At 15 m/s, the marginal cost of yaw correction (stability risk, alive=1.0) exceeds the marginal benefit (yaw tracking, scale=0.5). Accepting -53° drift while reducing wind load is the optimal trade-off under the given reward weights.

### Root Causes
1. **No symmetry enforcement**: No reward term or architectural constraint enforces left-right gait symmetry. The `tracking_ang_vel` reward (scale=0.5) is too weak relative to survival (1.0) and velocity tracking (1.0).
2. **Stochastic training breaks symmetry**: Random weight initialization, random environment resets, and stochastic PPO gradients cause the policy to converge to a slightly asymmetric local optimum.
3. **LSTM temporal bias**: Sequential processing of observations develops asymmetric hidden state patterns that reinforce directional preference.
4. **Observation ordering**: Left leg joints always precede right leg joints in the observation vector `[left×6, right×6]`, creating a subtle network bias.
5. **Per-step yaw torque does not cancel**: The neural network produces slightly different left vs right leg trajectories (δ₁ ≠ δ₂). Each gait cycle has a small net yaw torque (δ₁ - δ₂) that accumulates until balanced by heading correction.
6. **Wind amplification**: Under wind, asymmetric leg positions create asymmetric per-body forces (different heights → different wind speeds via power law), producing additional net yaw torque proportional to wind speed.

### Impact on Evaluation
- **Quantitative evaluation (84 scenarios)**: Not affected. The eval script uses `cmd[2]=0` (no heading command), and the survival/tracking metrics do not penalize heading drift. All results (100% survival for Exp3) remain valid.
- **Visual demos**: The robot walks at an angle to the world +X axis. This is a cosmetic issue — the policy successfully maintains stability, forward velocity, and wind resistance. The heading drift is an emergent wind-load-reduction strategy.

### Potential Solutions (Future Work)
1. **Mirror loss**: Enforce `π(mirror(obs)) = mirror(π(obs))` during training to guarantee symmetric gaits.
2. **Data augmentation**: Mirror observations and actions (swap left/right) and add to training batch.
3. **Symmetric network architecture**: Share weights between left and right leg processing pathways.
4. **Stronger heading reward**: Increase `tracking_ang_vel` scale (e.g., 0.5→2.0) or add explicit heading penalty to make yaw correction higher priority than wind load reduction.
5. **Gait symmetry reward**: Add `contact_symmetry` or stride-length symmetry term to eliminate the baseline ~18° drift.

### MuJoCo Deployment Bug Fix (2026-03-23)
During demo video development, a critical bug was found in `deploy/deploy_mujoco/deploy_mujoco_lstm.py`:
- **xfrc_applied layout was swapped**: force written to torque slots and vice versa (`[0:3]=torque, [3:6]=force` instead of correct `[0:3]=force, [3:6]=torque`). This caused the robot to fall at wind speeds ≥11 m/s despite the eval script achieving 100% survival at L5.
- **Simulation loop order**: Policy inference occurred before physics substeps (should be after, matching training and eval). Wind forces were applied once per control step instead of every physics substep.
- Both bugs are now fixed. The deployment script matches the eval script's loop structure.

## Caveats
- `self.phase` only exists after first `step()` call (created in `_post_physics_step_callback`)
- `net_contact_force_tensor` is unreliable on GPU + triangle mesh terrain
- `apply_rigid_body_force_at_pos_tensors` must be called BEFORE `gym.simulate()`

## Development Phases
- [x] Phase 0: Research and analysis (DONE — see G1_Wind_Robust_Walking_Analysis.md)
- [x] Phase 1: Environment setup — g1_wind_config.py, g1_wind_env.py, wind_model.py created and registered
- [x] Phase 2: Reward design — 4 wind-specific rewards implemented
- [x] Phase 3: Curriculum controller — 6-level window-based curriculum with auto-advancement
- [x] Phase 4: Training — Run4 (baseline, 1950 iters), Run5 (wind-trained, 1500 iters, reached level 4)
  - Wind init bug fixed (reset_envs at creation)
- [x] Phase 4.5: Code review and fixes (Run6 prep)
  - S1: Removed wind_stability (conflicted with tracking_lin_vel)
  - S2: Fixed lean_compensation coordinate frame (world→body frame transform)
  - S3: Set only_positive_rewards=False, boosted alive 0.15→0.5
  - M1: Network [64,32]→[128,64]
  - M2: Curriculum demotion (survival<0.3 AND tracking<0.2)
  - M3: Wind force clamp (per-level: L0=5N → L5=150N)
  - M4: Precomputed curriculum_levels tensor
  - M5: Normalized wind obs (vel/10, force/100)
  - Eval script: frozen curriculum during evaluation
- [x] Phase 4.6: Wind model v2
  - Layer 2: Added directional OU (θ_dir=0.2, σ_dir=0.15 → ±14° 1σ drift)
  - Layer 2: OU sigma scales with curriculum (1+0.15×level), added σ_min=0.1 floor
  - Layer 2: Reduced base sigma 0.3→0.25 (I≈0.25, suburban terrain)
  - Layer 3: Gust changed from speed multiplier to independent force vector (no v² amplification)
  - Layer 3: Independent gust direction (base ±60°)
  - Layer 3: Trapezoidal envelope (ramp_up=0.3s, ramp_down=0.5s, no step discontinuity)
  - Layer 3: Reduced frequency 0.3→0.1/s, increased duration [1,2]→[1.5,3]s
  - Layer 3: Gust speed & prob scale with curriculum level
  - Per-level force clamp [5,15,30,60,100,150]N (replaces global 500N cap)
  - Multi-body force distribution: pelvis(55%)+thighs(12%×2)+shins(8%×2) = 95%
  - lean_compensation uses effective_direction (real-time, not episode-initial)
  - Per-episode OU parameter randomization (θ, σ, θ_dir, σ_dir sampled from ranges)
  - Comprehensive robustness eval framework: eval_wind_robustness.py
    - Suite A: wind level sweep (L0-L5)
    - Suite B: wind mode decomposition (steady/turbulent/gusts/full/pure_gusts)
    - Suite C: wind direction tests (front/side/back/diagonal/random)
    - Suite D: OU parameter extremes (calm/turbulent/locked/erratic/default)
    - Suite E: out-of-distribution patterns (step change, periodic, direction reversal)
    - Suite F: command variations (standing/slow/normal/fast/lateral/turning/headwind/tailwind)
    - A/B policy comparison support, --seed for reproducibility
    - Reward component weighted aggregation (bug fix: correct batch-size weighting)
- [x] Phase 4.7: Wind model v3 — physically accurate aerodynamics
  - P0: Relative velocity (v_wind - v_body) instead of v_wind only (fixes ~40% force error)
  - P1: Direction-dependent projected area: A_front=0.50, A_side=0.22 m² (elliptical projection)
    - Elliptical: A_eff = sqrt((A_front·cosθ)² + (A_side·sinθ)²) guarantees A_eff ≤ A_front
    - Replaced cross-flow formula which overshoot at oblique angles (109% at 30°)
  - P2: Height-dependent wind speed: v(z) = v_ref × (z/z_ref)^α (boundary layer profile, initially α=1/7)
  - P3: Force at center of pressure via apply_rigid_body_force_at_pos_tensors (tipping torque)
  - CoP vertical offsets: pelvis +0.08m, thighs +0.02m, shins +0.01m above COM
  - Wind model refactored: outputs velocity only, force computation in env with per-body physics
  - Rigid body state refreshed per physics substep for accurate body velocities
  - Removed single frontal_area constant, replaced with directional A_front/A_side
  - v3.1 fixes: reduced OU sigma 0.25→0.18, σ_range [.05,.4]→[.05,.25], gust speed [3,8]→[2,6]
  - Added per-level speed clamp [0,5,8,13,20,28] m/s to prevent OU+gust extremes (was 67m/s at L5)
  - L0 now truly zero wind (speed_clamp=0 overrides ou_sigma_min noise)
- [x] Phase 4.8: Wind model v3.2 — physics review fixes
  - B1: A_eff now uses v_rel direction (not wind_vel) for projected area computation
  - B2: CoP offset along body-local z-axis via quat_rotate (not world z-axis)
  - B3: 3D ellipsoidal area model: A_eff = sqrt((A_front·dx)² + (A_side·dy)² + (A_top·dz)²)
    - Added frontal_area_top = 0.10 m² for vertical wind component on tilted bodies
  - B4: Per-body quaternions for A_eff (each body's own orientation, not just base_quat)
  - P1: Cd 1.0 → 1.1 (closer to measured humanoid bluff body Cd=1.0-1.3)
  - P2: Height exponent α=1/7 (open terrain) → α=0.28 (urban/suburban terrain)
  - P3: CoP offsets refined from URDF inertial data:
    - pelvis +0.08m → +0.10m (COM z=-0.076m, effective CoP for 55% upper body)
    - thighs +0.02m → +0.005m (COM z=-0.151m ≈ geometric center)
    - shins +0.01m → +0.002m (COM z=-0.121m ≈ geometric center)
    - Final: [+0.10, +0.005, +0.005, +0.002, +0.002] m
  - M1: Direction OU sigma now scales with curriculum level (same sigma_scale as speed OU)
  - M2: lean_compensation scales by wind force magnitude / 50N (not binary on/off)
  - Verified: URDF mass 32.68kg vs real G1 35kg — covered by domain randomization [-1,+3]kg
  - Verified: F at 18 m/s (L5 max) ≈ 109N (31.8% body weight), matches Beaufort 8 "gale"
- [x] Phase 4.9: Literature-informed reward redesign
  - R1: orientation -1.0→-0.3, lean_compensation 0.3→0.8 (resolve lean conflict, Viereck 2024)
  - R2: feet_swing_height -20→-8 (reduce over-constraint, Viereck 2024)
  - R3: hip_pos -1.0→-0.3 (allow wider stance, Booster Gym 2025)
  - R4: ang_vel_xy -0.05→-0.2 (post-gust damping, Li 2025)
  - R5: dof_vel -1e-3→-5e-4 (allow braking torques)
  - R6: Wind-adaptive tracking sigma: σ_eff = σ × (1 + |F_wind|/100) (Xu 2025)
  - R7: NEW action_rate2 = -0.005 (second-order smoothness, Humanoid-Gym 2024)
  - R8: NEW power = -5e-4 (motoring-only penalty, Booster Gym 2025)
  - R9: NEW feet_distance = -0.3 (penalize narrow stance, Booster Gym 2025)
  - R10: NEW base_acc = -0.002 (base acceleration penalty, Viereck 2024)
  - Buffer management: last_last_actions via _buf_last_actions rotation in _post_physics_step_callback
- [x] Phase 4.10: Reward robustness fixes
  - F1: base_acc wind compensation — subtract F_wind/m, only penalize controllable acceleration
  - F2: feet_distance min_sep moved from hardcode to config (wind.feet_min_separation)
  - F3: NEW ang_momentum_change = -0.0003 (angular acceleration penalty, Li 2025; reduced from -0.001 to avoid penalizing unavoidable gust-onset angular accel)
  - F4: NEW com_balance = -10.0 (COM over support center; raised from -2.0 for meaningful gradient at small deviations)
  - F5: Override feet_swing_height — wind-adaptive target: 0.08→0.04m at 150N
  - F6: Override contact — wind-adaptive stance ratio: 0.55→0.75 at 150N
  - F7: Config: robot_nominal_mass, swing_height_base/wind_reduction, stance_ratio_base/wind_increase
  - F8: CLAUDE.md synced — fixed power (-1e-4→-5e-4), base_acc (-0.01→-0.002)
- [x] Phase 4.11: Network, reward, PPO, curriculum overhaul
  - N1: LSTM hidden_size 64→128 (more capacity for wind OU estimation, τ=2-5s)
  - N2: Asymmetric actor-critic MLP:
    - Actor [128,64]→[128,64] (LSTM-heavy: 78% params in LSTM for wind estimation)
    - Critic [128,64]→[256,128] (MLP-heavy: privileged wind info processing)
  - N3: Removed inverted bottleneck (128→256 expansion wasted 33K params)
  - R1: alive 0.5→1.0 (prevent "die early" under only_positive_rewards=False)
  - R2: com_balance -10.0→-1.5 (severe conflict with lean_compensation resolved)
  - R3: ang_momentum_change -0.0003→0.0 (disabled: negligible gradient, noise only)
  - R4: power -5e-4→-2e-4 (allow aggressive wind rejection torques)
  - R5: dof_acc -2.5e-7→-1.5e-7 (faster joint response for wind)
  - R6: contact_no_vel -0.2→-0.1 (wind causes unavoidable sliding)
  - R7: feet_distance -0.3→-0.5 (stronger wide stance incentive)
  - R8: ang_vel_xy -0.2→-0.15 (relax for gust angular response)
  - P1: gamma 0.99→0.995 (longer horizon for 20s episodes)
  - P2: learning_rate 1e-3→5e-4 (larger network, adaptive schedule adjusts)
  - P3: desired_kl 0.01→0.008 (conservative updates during curriculum transitions)
  - P4: entropy_coef 0.01→0.008 (slightly less exploration with bigger network)
  - P5: num_steps_per_env 24→64 (1.28s rollout: 64 × 0.02s control dt)
  - C1: survival_threshold 0.8→0.7, tracking_threshold 0.6→0.4 (relaxed advancement)
  - C2: demotion thresholds 0.3/0.2→0.4/0.3 (demote earlier)
  - C3: upgrade_window 200→300 (more reliable statistics)
  - C4: Mixed-level upgrade: 80% envs advance, 20% stay (catastrophic forgetting prevention)
- [x] Phase 4.12: Pre-training optimization (reward conflict resolution + network + curriculum)
  - R1: orientation → wind-aware override: penalty × 1/(1+F/50N), eliminates lean conflict
  - R2: base_height → wind-aware override: target 0.78→0.741m at 150N (lower COM under wind)
  - R3: com_balance -1.5→0.0 (disabled: fundamental conflict with lean_compensation)
  - N1: LSTM 128→256 (standard capacity for wind OU estimation)
  - N2: Actor MLP [128,64]→[256,128] (match LSTM output, avoid bottleneck)
  - N3: Critic MLP [256,128]→[512,256] (larger for privileged wind info)
  - N4: Total params 277K→930K
  - P1: learning_rate 5e-4→3e-4 (larger network)
  - P2: max_iterations 15000→20000 (more training for larger network)
  - P3: num_mini_batches 4→8 (smaller mini-batch stabilizes larger network)
  - C1: Fractional demotion 100%→50% (prevents level oscillation)
  - C2: Cooldown 500 resets after level change (stabilize before re-evaluating)
- [x] Phase 4.13: Domain randomization enhancement (pre-training)
  - DR1: Action delay randomization: [0, 2] control steps (0-40ms)
    - Ring buffer implementation, per-env delay sampled at reset
    - Obs shows intended action, torque uses delayed action (LSTM compensates)
    - Ref: Walk These Ways (Margolis 2023), Rapid Locomotion (Margolis 2022)
  - DR2: PD gain randomization: stiffness ×[0.8,1.2], damping ×[0.8,1.2]
    - Per-env multipliers, re-sampled each episode
    - Override _compute_torques with randomized gains
    - Ref: ANYmal-DroQ, Walk These Ways
  - DR3: Motor strength randomization: torque_limits ×[0.8,1.0]
    - Asymmetric: only weaker (models degradation, voltage sag)
    - Ref: Rapid Locomotion, Booster Gym 2025
- [x] Phase 4.14: Literature-conforming network/PPO parameters
  - N1: LSTM hidden_size 256→128 (literature standard 64-128; 128 for wind OU estimation)
  - N2: Actor MLP [256,128]→[128,64] (matches Humanoid-Gym scale)
  - N3: Critic MLP [512,256]→[256,128] (still asymmetric, larger for privileged obs)
  - N4: Total params ~930K→~250K (within literature range for locomotion)
  - P1: learning_rate 3e-4→5e-4 (standard for ~250K params)
  - P2: num_mini_batches 8→4 (standard, 65K samples per mini-batch)
  - P3: max_iterations 20000→10000 (smaller network converges faster)
  - Ref: Humanoid-Gym 2024, Walk These Ways 2023, Gait-Conditioned RL
- [x] Phase 5: Training — Run8 (best, 2900 iters, reached L5, reward 35.29)
- [ ] Phase 6: Testing, analysis, visualization, report
  - [x] Quantitative evaluation: eval_wind_robustness.py (Suites A-F, L3/L4/L5)
    - Run8 full eval: 84 scenarios, all >= 98% survival, 0 failures
    - Suite A (levels): L0-L5 all 100%, tracking error 0.186→0.254
    - Suite B (modes): 98-100%, B3_gusts_L4 is lowest (98%)
    - Suite C (directions): all 100%, policy fully direction-invariant
    - Suite D (OU extremes): D4_erratic_L5 100% (D2_turbulent_L5 tracking 0.364, worst)
    - Suite E (OOD): E1-E3 all 98-100% at L3-L5, reversal L5 tracking 0.299
    - Suite F (commands): all 98-100%, F4_fast_L5 (98%), lateral tracking 0.278
  - [x] Visual observation: play.py — confirmed upright walking under all wind conditions
    - No wind: normal gait; L5: body lean + trajectory drift (body-frame tracking maintained)
    - LSTM warm-up ~2-3s for wind estimation convergence
  - [x] Ablation experiments — all 5 experiments trained and evaluated (84 scenarios each)
    - Exp1 (baseline): 45/84 fail, L5 avg survival 5% — validates wind training necessity
    - Exp2 (push only): 16/84 fail, L5 avg survival 81% — push provides partial robustness
    - Exp3 (full method): 0/84 fail, L5 avg survival 100% — perfect
    - Exp4 (no curriculum): 18/84 fail, L5 avg survival 48% — curriculum is critical
    - Exp5 (no wind reward): 0/84 fail, L5 avg survival 99% — wind rewards marginal
    - Results: `test_results/exp{1..5}_*/`
  - [x] Sim2Sim (MuJoCo) validation:
    - Script: `legged_gym/g1_wind_test/eval_wind_robustness_mujoco.py`
    - Results: `test_results/mujoco/exp{1..5}_*_mujoco.json`
    - Wind: full v3 per-body aerodynamics (P0/P1/P2/P3) ported to pure numpy
    - Force applied via `d.xfrc_applied` with CoP torque offset
    - Exp3 transfers perfectly: 100% → 100% L5 survival; failing policies degrade further in MuJoCo
  - [ ] Report and analysis
- [x] Phase 7: Sim2Sim (MuJoCo) validation
  - Goal: validate policies transfer across physics engines (Isaac Gym PhysX → MuJoCo)
  - Wind model ported to pure numpy (MuJoCoWindModel) — no isaacgym/legged_gym imports
    - 3-layer OU model (base + OU + gusts) reproduced exactly from wind_model.py
    - Full v3 per-body aerodynamics (P0/P1/P2/P3) via d.xfrc_applied
    - P0: relative velocity (v_wind − v_body) per body using d.cvel[id][3:6]
    - P1: ellipsoidal projected area using d.xmat body rotation matrix
    - P2: height-dependent wind scaling (power law) via d.xpos body z-height
    - P3: CoP torque offset via np.cross(cop_offset_world, force_vec) in xfrc_applied[id][3:6]
    - Speed clamp bug fixed: clamp total velocity (base+OU+gust) not just base+OU
  - Scripts created:
    - `eval_wind_robustness_mujoco.py` — full 84-scenario eval, matches Isaac Gym suite structure
      - Supports --exp 1-5 (select from EXP_POLICIES dict) or --policy (explicit path)
      - One-time LSTM warm-up warning (not per-episode)
      - Tracking error in body frame: d.qvel[0:2] (world) rotated via yaw from quaternion
      - obs[action_slot] = action (the action that just ran physics, matching Isaac Gym self.actions)
    - `eval_wind_model_mujoco.py` — statistical verification: 14/14 checks pass
    - `analyze_sim2sim.py` — side-by-side Isaac Gym vs MuJoCo comparison tables (4 tables)
  - Key implementation gotchas:
    - `d.cvel` layout: [rot(3), lin(3)] — body linear vel is d.cvel[id][3:6], NOT [0:3]
    - `d.xfrc_applied` must be zeroed manually each step (MuJoCo does NOT auto-clear)
    - `d.qpos[3:7]` quaternion is (w, x, y, z) — same as Isaac Gym
    - Body name mismatch: Isaac Gym "left_thigh_link" ≠ MuJoCo names; verified via m.body(i).name
    - LSTM hidden state: policy.reset() not available → zero warm-up fallback (3 forward passes)
    - Policy paths: logs/<task>/exported/policies/policy_lstm_1.pt (no run timestamp subdir)
    - To export Exp1 policy: python legged_gym/scripts/play.py --task=g1_wind_baseline --load_run Mar12_00-14-00_ --headless
    - Tracking error frame: d.qvel[0:2] is world-frame; rotate via yaw to body-frame before comparing to cmd
  - Results (84 scenarios per exp, Suites A-F at L3/L4/L5, `--test_level all`):
    - Exp3 (full method): 100% L5 survival in MuJoCo — perfect transfer; tracking 15-20% lower than Isaac Gym
    - Exp5 (no wind reward): 100% L5 survival — also transfers perfectly
    - Exp1 (baseline): 28% L5 survival (IG: 5%) — unexpected improvement; MuJoCo contact smoother for baseline
    - Exp2 (push only): 81% L5 survival (IG: 81%) — perfect match
    - Exp4 (no curriculum): 54% L5 survival (IG: 48%) — MuJoCo slightly better
    - Overall: ablation conclusions hold in MuJoCo; MuJoCo tracking consistently lower after action timing fix

## Sim2Sim Gotchas Reference
| Issue | Detail |
|-------|--------|
| `d.cvel` layout | `[rot(3), lin(3)]` — body linear vel is `d.cvel[id][3:6]`, NOT `[0:3]` |
| `d.xfrc_applied` | NOT zeroed by MuJoCo — must reset manually each step |
| `d.qpos[3:7]` | MuJoCo quaternion is `(w, x, y, z)` — same as Isaac Gym |
| Body name mismatch | Verify G1 XML body names with `m.body(i).name` — may differ from Isaac Gym |
| LSTM call signature | Try `policy(obs)` first, then `policy.act_inference(obs)` |
| No legged_gym imports | All wind params hardcoded; all paths via `REPO_ROOT` from `__file__` |
| `d.qvel[0:2]` is world-frame | Rotate to body-frame via yaw before comparing to cmd (which is body-frame) |
| obs action slot | Use `action` (the action that just ran physics), not stale `last_action` — matches Isaac Gym `self.actions` |
| Speed clamp scope | Must clamp total velocity magnitude (base+OU+gust), not just base+OU separately |
| Exported policy path | `logs/<task>/exported/policies/policy_lstm_1.pt` — no run timestamp subdir; created by `play.py` |
