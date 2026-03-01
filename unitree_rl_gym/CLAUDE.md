# G1 Wind-Robust Walking — RL Course Project

## Project Overview
Training the Unitree G1 humanoid robot to walk stably under continuous, time-varying wind disturbances using deep reinforcement learning. This is a course project for a reinforcement learning class.

## Tech Stack
- **Simulation**: NVIDIA Isaac Gym Preview 3 (GPU-accelerated parallel physics)
- **Base Framework**: unitree_rl_gym (https://github.com/unitreerobotics/unitree_rl_gym)
- **RL Algorithm**: PPO via rsl_rl (https://github.com/leggedrobotics/rsl_rl)
- **Upstream**: legged_gym (https://github.com/leggedrobotics/legged_gym)
- **Language**: Python 3.8+, PyTorch
- **Robot**: Unitree G1 (~35kg, 23-29 DOF humanoid)

## Repository Structure (Target)
```
unitree_rl_gym/
├── legged_gym/
│   ├── envs/
│   │   ├── base/
│   │   │   ├── legged_robot.py          # Base env class (upstream, minimal changes)
│   │   │   └── legged_robot_config.py   # Base config
│   │   ├── g1/
│   │   │   ├── g1_config.py             # Original G1 config (reference only)
│   │   │   └── g1_env.py                # Original G1 env
│   │   ├── g1_wind/                     # ★ NEW — Our wind environment
│   │   │   ├── __init__.py
│   │   │   ├── g1_wind_config.py        # Wind params, reward scales, curriculum
│   │   │   ├── g1_wind_env.py           # Wind model + force application + rewards
│   │   │   └── wind_model.py            # OU process wind generation (optional separate file)
│   │   └── __init__.py                  # Register g1_wind task here
│   ├── scripts/
│   │   ├── train.py
│   │   └── play.py
│   └── utils/
├── rsl_rl/                              # PPO implementation (do NOT modify)
├── resources/robots/g1/                 # G1 URDF model
└── logs/                                # Training outputs
```

## Key Technical Details

### Wind Force Model (3-layer)
1. **Base wind**: Per-episode sampled direction + speed (constant within episode)
2. **OU process fluctuation**: Ornstein-Uhlenbeck process for temporally-correlated variation
   - dv = θ(μ - v)dt + σ dW, with θ≈0.5, σ≈0.3×base_speed
3. **Gust events**: Random short bursts (2-3x base force, 1-2s duration)

Wind force = 0.5 × ρ × Cd × A × v², applied to torso rigid body via:
```python
gym.apply_rigid_body_force_tensors(sim, forceTensor, None, gymapi.ENV_SPACE)
```
This API applies force for ONE physics timestep only — must call every step.

### Force Tensor Indexing
```python
# forceTensor shape: [num_envs * num_bodies, 3]
# To apply force to torso body index `torso_idx` in each env:
forces = torch.zeros(num_envs * num_bodies, 3, device=device)
indices = torch.arange(num_envs, device=device) * num_bodies + torso_idx
forces[indices] = wind_force_per_env  # [num_envs, 3]
```

### Existing Push Mechanism (Reference)
In `legged_robot.py`, `_push_robots()` applies instantaneous velocity perturbation:
```python
self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, ...)
```
Our wind approach replaces this with continuous force application.

### G1 Config Baseline (from g1_config.py)
```python
class domain_rand:
    randomize_friction = True
    friction_range = [0.1, 1.25]
    randomize_base_mass = True
    added_mass_range = [-1., 3.]
    push_robots = True
    push_interval_s = 5
    max_push_vel_xy = 1.5
```

### Reward Structure (4 layers)
1. **Base walking**: velocity tracking, orientation, base height (keep from original)
2. **Energy/smoothness**: torque penalty, action rate, dof acceleration
3. **Wind-specific**: lean compensation, CoM stability, sustained walking, contact symmetry
4. **Safety**: termination penalty, stumble penalty

### Curriculum (6 levels)
Level 0→5: No wind → Constant light → Variable light → Medium+variable → Strong+gusts → Extreme
Auto-upgrade when survival_rate > threshold AND velocity_tracking > threshold

### Experiments Plan
| ID | Name | Wind | Push | Purpose |
|----|------|------|------|---------|
| Exp1 | Baseline | OFF | OFF | Reference |
| Exp2 | Push-Only | OFF | ON | Traditional robustness |
| Exp3 | Wind-Trained | ON+Curriculum | OFF | Our method |
| Exp4 | No-Curriculum | ON (fixed medium) | OFF | Ablation |
| Exp5 | No-Wind-Reward | ON+Curriculum | OFF | Ablation (base rewards only) |

## Coding Conventions
- Inherit from existing classes (LeggedRobot, LeggedRobotCfg) — don't rewrite
- All tensors on GPU (self.device), use torch operations, no Python loops over envs
- Reward functions follow naming: `_reward_<name>(self)` → auto-registered by non-zero scale
- Config classes use nested class pattern matching legged_gym style
- Use gymtorch.unwrap_tensor() when passing tensors to Isaac Gym C++ API

## Important Caveats
- `net_contact_force_tensor` is unreliable on GPU + triangle mesh terrain
- Use force sensors (`create_asset_force_sensor`) on feet only if contact detection needed
- `apply_rigid_body_force_tensors` must be called BEFORE `gym.simulate()`
- G1 body names need verification from URDF — check with `get_asset_rigid_body_name()`
- rsl_rl supports both MLP and LSTM policies; LSTM may help with wind estimation from history

## Development Phases
- [x] Phase 0: Research and analysis (DONE — see G1_Wind_Robust_Walking_Analysis.md)
- [ ] Phase 1: Environment setup (g1_wind_config.py, g1_wind_env.py, wind_model.py)
- [ ] Phase 2: Reward design and implementation
- [ ] Phase 3: Curriculum controller
- [ ] Phase 4: Training and experiments
- [ ] Phase 5: Analysis, visualization, report
