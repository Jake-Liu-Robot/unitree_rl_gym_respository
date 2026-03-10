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
│   │   ├── eval_g1_wind.py             # Quantitative evaluation across wind levels
│   │   ├── eval_wind_robustness.py     # Comprehensive robustness eval (modes/dirs/OU/OOD)
│   │   └── smoke_test_g1_wind.py       # Quick env verification
│   ├── g1_wind_doc/                     # ★ Documentation
│   │   ├── g1_wind_test.md             # Wind environment test notes
│   │   └── G1_Wind_Robust_Walking_Analysis.md  # Research analysis
│   ├── scripts/
│   │   ├── train.py
│   │   └── play.py
│   └── utils/
├── rsl_rl/                              # PPO implementation (do NOT modify)
├── resources/robots/g1_description/     # G1 URDF models (g1_12dof.urdf, g1_29dof.urdf, etc.)
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

| Category | Reward | Scale | Source |
|----------|--------|-------|--------|
| Base walking | tracking_lin_vel | 1.0 | inherited |
| Base walking | tracking_ang_vel | 0.5 | inherited |
| Base walking | orientation | -1.0 | inherited |
| Base walking | base_height | -10.0 | inherited |
| Energy | dof_acc / dof_vel / action_rate | -2.5e-7 / -1e-3 / -0.01 | inherited |
| Humanoid | alive / contact / hip_pos / feet_swing_height | 0.5 / 0.18 / -1.0 / -20.0 | inherited (alive boosted) |
| **Wind-specific** | **lean_compensation** | **0.3** | reward leaning against wind (scales by wind_force_mag/50N, uses effective_direction with OU drift) |
| ~~removed~~ | ~~wind_stability~~ | — | removed: conflicted with tracking_lin_vel |
| disabled | sustained_walking | 0.0 | disabled: identical to alive |
| disabled | contact_symmetry | 0.0 | disabled: incentivized standing still |

`only_positive_rewards = False` — allows negative reward gradient under strong wind.

## Curriculum Controller (g1_wind_env.py)

6 levels with window-based advancement (all envs advance together):

| Level | Speed (m/s) | Description |
|-------|------------|-------------|
| 0 | 0 | No wind (baseline) |
| 1 | 1-3 | Light |
| 2 | 2-5 | Light-medium |
| 3 | 4-8 | Medium |
| 4 | 7-12 | Strong + gusts |
| 5 | 10-18 | Extreme |

Upgrade: accumulate stats over `upgrade_window=200` resets, advance if `survival > 0.8` AND `tracking > 0.6`.
Demotion: demote if `survival < 0.3` AND `tracking < 0.2` (prevents getting stuck at too-hard level).

## PPO Config (g1_wind_config.py)
- Policy: `ActorCriticRecurrent` (LSTM, hidden_size=64, 1 layer)
- Actor/Critic: [128, 64] hidden dims, ELU activation
- `entropy_coef = 0.01`, `max_iterations = 15000`
- `experiment_name = 'g1_wind'`

## Domain Randomization
- `push_robots = False` (wind replaces push perturbation)
- Friction: [0.1, 1.25], Base mass: [-1, +3] kg

## Experiments Plan
| ID | Name | Wind | Push | Purpose |
|----|------|------|------|---------|
| Exp1 | Baseline | OFF | OFF | Reference |
| Exp2 | Push-Only | OFF | ON | Traditional robustness |
| Exp3 | Wind-Trained | ON+Curriculum | OFF | Our method |
| Exp4 | No-Curriculum | ON (fixed medium) | OFF | Ablation |
| Exp5 | No-Wind-Reward | ON+Curriculum | OFF | Ablation (base rewards only) |

## Common Commands

```bash
# --- Training ---
python legged_gym/scripts/train.py --task=g1_wind --headless
python legged_gym/scripts/train.py --task=g1_wind --load_run Mar02_21-49-27_ --checkpoint 3750  # resume

# --- Play (visualize) --- must specify --load_run
python legged_gym/scripts/play.py --task=g1_wind --load_run Mar02_21-49-27_
python legged_gym/scripts/play.py --task=g1_wind --load_run Mar02_21-49-27_ --checkpoint 3750

# --- Evaluation (basic level sweep) ---
python legged_gym/g1_wind_test/eval_g1_wind.py --task g1_wind --load_run Mar02_21-49-27_ --headless

# --- Robustness evaluation (comprehensive) ---
python legged_gym/g1_wind_test/eval_wind_robustness.py --task g1_wind --load_run Mar02_21-49-27_ --headless               # all suites
python legged_gym/g1_wind_test/eval_wind_robustness.py --task g1_wind --load_run Mar02_21-49-27_ --suite modes --headless   # wind mode decomposition
python legged_gym/g1_wind_test/eval_wind_robustness.py --task g1_wind --load_run Mar02_21-49-27_ --suite ou --headless      # OU extremes
python legged_gym/g1_wind_test/eval_wind_robustness.py --task g1_wind --load_run Mar02_21-49-27_ --suite ood --headless     # out-of-distribution
# A/B comparison between two policies:
python legged_gym/g1_wind_test/eval_wind_robustness.py --task g1_wind --load_run <run_A> --load_run2 <run_B> --headless

# --- Smoke test ---
python legged_gym/g1_wind_test/smoke_test_g1_wind.py
```

Note: `play.py` requires `--load_run <run_dir>` to locate the model. Without it will raise FileNotFoundError.

Available runs: `Feb28_21-36-56_` (Run4, baseline), `Feb28_22-07-25_` (Run5, wind-trained), `Mar02_21-49-27_` (Run6, latest)

## Coding Conventions
- Inherit from existing classes — don't rewrite base code
- All tensors on GPU (`self.device`), use torch operations, no Python loops over envs
- Reward functions: `_reward_<name>(self)` → auto-registered by non-zero scale in config
- Config classes use nested class pattern matching legged_gym style
- Use `gymtorch.unwrap_tensor()` when passing tensors to Isaac Gym C++ API

## Trained Models (logs/g1_wind/)

| Run | Directory | Iters | Curriculum | mean_reward | mean_ep_len | Role |
|-----|-----------|-------|------------|-------------|-------------|------|
| Run4 | Feb28_21-36-56_ | 1950 | Level 0 (no wind) | 21.25 | 980 | Baseline (Exp1) |
| Run5 | Feb28_22-07-25_ | 1500 | Level 4 (strong) | 1.71 | 290 | Wind-trained (Exp3) |

- Run4: curriculum never advanced, effectively a no-wind baseline. Good tracking (0.84), long episodes.
- Run5: curriculum reached level 4 (7-12 m/s). Shorter episodes under wind, tracking degraded (0.13).
- Exported LSTM policy: `logs/g1_wind/exported/policies/policy_lstm_1.pt`
- Baseline G1 (no wind env): `logs/g1/Feb20_16-37-13_/` (only model_0.pt, 1 iter)
- Early aborted runs: Feb28_21-29-05_, Feb28_21-31-16_, Feb28_21-33-53_ (only model_0.pt each)

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
    - Suite A: wind level sweep
    - Suite B: wind mode decomposition (steady/turbulent/gusts/full)
    - Suite C: wind direction tests (front/side/back/diagonal/random)
    - Suite D: OU parameter extremes (calm/turbulent/locked/erratic)
    - Suite E: out-of-distribution patterns (step change, periodic)
    - A/B policy comparison support
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
- [ ] Phase 5: Retrain (Run6), evaluate, compare with Run4/Run5
- [ ] Phase 6: Testing, analysis, visualization, report
  - Quantitative evaluation (survival rate, tracking accuracy per wind level)
  - Cross-test: Run4 vs Run5 vs Run6 under same wind conditions
  - Robustness eval: modes × directions × OU extremes × OOD patterns
  - Ablation experiments (Exp2/Exp4/Exp5) if needed
  - Visualization and report
