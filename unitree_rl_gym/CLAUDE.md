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
│   │   └── smoke_test_g1_wind.py       # Quick env verification
│   ├── g1_wind_doc/                     # ★ Documentation
│   │   ├── README.md                   # Wind environment documentation (Chinese)
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

## Wind Force Model (Implemented in wind_model.py)

Three-layer superposition, all GPU-vectorized:
```
effective_speed = clamp(base_speed + ou_state, min=0) × gust_factor
force = 0.5 × ρ × Cd × A × effective_speed²
wind_force = direction × force   # [num_envs, 3]
```

| Layer | Mechanism | Key params |
|-------|-----------|-----------|
| 1: Base wind | Per-episode constant direction (xy plane) + speed | Speed range from curriculum level |
| 2: OU process | Ornstein-Uhlenbeck fluctuation | θ=0.5, σ=0.3×base_speed |
| 3: Gusts | Random bursts via Poisson-like trigger | 2-3× multiplier, 1-2s duration, prob=0.3/s |

Physical constants: ρ=1.225 kg/m³, Cd=1.0, A=0.5 m²

### Force Application (g1_wind_env.py)
- `step()` overrides base: applies wind force BEFORE `gym.simulate()` on every physics substep
- Force tensor: sparse [num_envs × num_bodies, 3], only pelvis body gets force
- Precomputed `torso_indices = arange(num_envs) * num_bodies + torso_body_idx`
- Coordinate frame: `gymapi.ENV_SPACE`

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
| Humanoid | alive / contact / hip_pos / feet_swing_height | 0.15 / 0.18 / -1.0 / -20.0 | inherited |
| **Wind-specific** | **wind_stability** | **-0.5** | penalize vel tracking error (squared) |
| **Wind-specific** | **lean_compensation** | **0.2** | reward leaning against wind direction |
| **Wind-specific** | **sustained_walking** | **0.1** | per-step survival bonus |
| **Wind-specific** | **contact_symmetry** | **0.05** | reward symmetric L/R foot contact |

`only_positive_rewards = True` — negative total reward is clipped to 0.

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

## PPO Config (g1_wind_config.py)
- Policy: `ActorCriticRecurrent` (LSTM, hidden_size=64, 1 layer)
- Actor/Critic: [64, 32] hidden dims, ELU activation
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
- `apply_rigid_body_force_tensors` must be called BEFORE `gym.simulate()`

## Development Phases
- [x] Phase 0: Research and analysis (DONE — see G1_Wind_Robust_Walking_Analysis.md)
- [x] Phase 1: Environment setup — g1_wind_config.py, g1_wind_env.py, wind_model.py created and registered
- [x] Phase 2: Reward design — 4 wind-specific rewards implemented
- [x] Phase 3: Curriculum controller — 6-level window-based curriculum with auto-advancement
- [x] Phase 4: Training — Run4 (baseline, 1950 iters), Run5 (wind-trained, 1500 iters, reached level 4)
  - Wind init bug fixed (reset_envs at creation)
- [ ] Phase 5: Testing, analysis, visualization, report
  - Quantitative evaluation (survival rate, tracking accuracy per wind level)
  - Cross-test: Run4 vs Run5 under same wind conditions
  - Ablation experiments (Exp2/Exp4/Exp5) if needed
  - Visualization and report
