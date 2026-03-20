# Wind-Robust Walking for the Unitree G1

Training the Unitree G1 humanoid robot to walk stably under continuous, time-varying wind disturbances using deep reinforcement learning.

---

## Overview

Most humanoid locomotion policies are trained in calm simulation environments and fail under real-world aerodynamic disturbances. This project develops a wind-robust walking policy for the **Unitree G1** (~35 kg, 12 DOF legs) by:

- Modeling physically accurate wind forces (boundary layer profile, per-body aerodynamics, turbulence + gusts)
- Training with a 6-level wind curriculum using PPO + LSTM
- Ablating the contribution of each component (wind model, curriculum, wind-specific rewards)

**Key result:** The full method achieves **100% survival across all 84 test scenarios** at wind speeds up to 18 m/s (Beaufort 8 gale), compared to 5% for a no-wind baseline.

---

## Results

| Experiment | L3 Pass | L4 Pass | L5 Pass | L5 Avg Survival | L5 Tracking Err | Failures |
|---|---|---|---|---|---|---|
| Exp1 — No-wind baseline | 26/26 | 8/26 | 1/26 | 5% | 1.617 | 45/84 |
| Exp2 — Push perturbation only | 26/26 | 26/26 | 11/26 | 81% | 1.180 | 16/84 |
| **Exp3 — Full method** | **26/26** | **26/26** | **26/26** | **100%** | **0.254** | **0/84** |
| Exp4 — No curriculum | 26/26 | 26/26 | 9/26 | 48% | 0.910 | 18/84 |
| Exp5 — No wind rewards | 26/26 | 26/26 | 26/26 | 99% | 0.272 | 0/84 |

Pass = survival ≥ 90% over a 20-second episode. Evaluated across 6 suites (wind levels, modes, directions, OU extremes, OOD patterns, command variations) at levels L3/L4/L5.

**Ablation conclusions:**
1. **Wind training** (Exp1→Exp3): essential — L5 survival 5% → 100%
2. **Curriculum learning** (Exp4→Exp3): critical — L5 survival 48% → 100%
3. **Wind physics vs push** (Exp2→Exp3): significant — L5 survival 81% → 100%
4. **Wind-specific rewards** (Exp5→Exp3): marginal — L5 survival 99% → 100%

---

<p align="center">
  <img src="doc/isaac_gym/ablation_contribution.png" alt="Ablation Component Contributions" width="80%">
</p>
 
### L5 Survival Heatmap — All Suites (26 scenarios)
 
<p align="center">
  <img src="doc/isaac_gym/l5_heatmap.png" alt="L5 Survival Heatmap across all test suites" width="70%">
</p>
 
### Reward Component Analysis
 
<p align="center">
  <img src="doc/isaac_gym/l5_rewards.png" alt="Key Reward Components at L5" width="90%">
</p>

----

## Tech Stack

| Component | Details |
|---|---|
| **Simulation** | NVIDIA Isaac Gym Preview 4 (GPU-accelerated parallel physics) |
| **Base framework** | [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym) |
| **RL algorithm** | PPO via [rsl_rl](https://github.com/leggedrobotics/rsl_rl) |
| **Policy** | ActorCriticRecurrent — LSTM-128, Actor MLP [128,64], Critic MLP [256,128] (~250K params) |
| **Robot** | Unitree G1, 12 DOF legs, URDF: `g1_12dof.urdf` |
| **Language** | Python 3.8+, PyTorch (CUDA 12.1) |

---

## Repository Structure

```
Wind_Robust_Walking_for_the_Unitree_G1/
├── unitree_rl_gym/                      # Main project
│   ├── legged_gym/
│   │   ├── envs/
│   │   │   ├── g1_wind/                 # Wind environment
│   │   │   │   ├── g1_wind_config.py    # Wind params, curriculum, PPO config
│   │   │   │   ├── g1_wind_env.py       # Wind force application, rewards, curriculum
│   │   │   │   └── wind_model.py        # 3-layer wind model (base + OU + gusts)
│   │   │   └── base/                    # Base LeggedRobot env
│   │   ├── g1_wind_test/
│   │   │   ├── eval_wind_robustness.py  # Comprehensive evaluation (84 scenarios)
│   │   │   └── smoke_test_g1_wind.py    # Quick env verification
│   │   ├── g1_wind_doc/                 # Research documentation
│   │   └── scripts/
│   │       ├── train.py
│   │       └── play.py                  # Visualization with wind/command control
│   ├── resources/robots/g1_description/ # G1 URDF models
│   └── logs/                            # Training outputs & exported policies
├── IsaacGym_Preview_4_Package/          # NVIDIA Isaac Gym (dependency)
├── rsl_rl/                              # PPO implementation (dependency, do not modify)
├── unitree_sdk2_python/                 # Unitree SDK2 (robot communication)
└── docker/                              # Docker environment setup
```

---

## Wind Model

Three-layer superposition producing physically realistic disturbances:

| Layer | Mechanism | Parameters |
|---|---|---|
| **Base wind** | Per-episode constant direction + speed from curriculum level | Speed range per level |
| **OU turbulence** | Speed OU + directional OU, per-episode randomized | θ∈[0.2,1.0], σ∈[0.05,0.25] |
| **Gusts** | Independent velocity vector, trapezoidal envelope | speed=[2,6] m/s, prob=0.1/s |

Per-level speed clamp: `[0, 5, 8, 13, 20, 28]` m/s

**Per-body aerodynamics (v3.2):**
- Relative velocity: `v_rel = v_wind(z) - v_body` per body
- 3D ellipsoidal projected area: `A_eff = √((A_front·dx)² + (A_side·dy)² + (A_top·dz)²)`
- Height-dependent wind profile: `v(z) = v_ref × (z/z_ref)^0.28` (urban terrain)
- Force applied at center of pressure (not COM) for correct tipping torque
- Force bodies: pelvis (55%) + thighs (12%×2) + shins (8%×2)
- Per-level force clamp: `[5, 15, 30, 60, 100, 150]` N

At L5 max speed (18 m/s), peak force ≈ 109 N ≈ 31.8% of robot body weight — equivalent to a Beaufort 8 gale.

---

## Curriculum

6-level wind curriculum with window-based advancement:

| Level | Speed (m/s) | Description |
|---|---|---|
| 0 | 0 | No wind (baseline) |
| 1 | 1–3 | Light breeze |
| 2 | 2–5 | Light–medium |
| 3 | 4–8 | Medium |
| 4 | 7–12 | Strong + gusts |
| 5 | 10–18 | Extreme (gale) |

- **Advance**: 80% of envs advance if `survival > 0.7` AND `tracking > 0.4` over 300-reset window
- **Demote**: 50% of envs demote if `survival < 0.4` AND `tracking < 0.3`
- **Cooldown**: 500 resets after any level change before re-evaluating

---

## Setup

### Option 1: Docker (recommended)

```bash
cd docker
./docker_build.sh
./docker_run.sh
```

### Option 2: Conda

```bash
conda create -n unitree-rl python=3.8
conda activate unitree-rl

# Install Isaac Gym
cd IsaacGym_Preview_4_Package/isaacgym/python
pip install -e .

# Install rsl_rl
cd rsl_rl
pip install -e .

# Install main project
cd unitree_rl_gym
pip install -e .
```

All commands below should be run from the `unitree_rl_gym/` directory.

---

## Usage

### Training

```bash
# Full method (Exp3)
python legged_gym/scripts/train.py --task=g1_wind --headless

# Resume from checkpoint
python legged_gym/scripts/train.py --task=g1_wind --load_run Mar10_18-22-48_ --checkpoint 2900

# Ablation experiments
python legged_gym/scripts/train.py --task=g1_wind_baseline --headless --num_envs 2048 --max_iterations 12000
python legged_gym/scripts/train.py --task=g1_wind_no_curriculum --headless --num_envs 2048 --max_iterations 12000
python legged_gym/scripts/train.py --task=g1_wind_no_reward --headless --num_envs 2048 --max_iterations 12000
```

### Visualization (Play)

```bash
# Default: L5, full wind model, 4 environments
python legged_gym/scripts/play.py --task=g1_wind --load_run Mar10_18-22-48_

# No wind
python legged_gym/scripts/play.py --task=g1_wind --load_run Mar10_18-22-48_ --wind_level 0 --fix_vx 0.6

# L5 steady frontal wind, standing
python legged_gym/scripts/play.py --task=g1_wind --load_run Mar10_18-22-48_ --wind_level 5 --wind_mode steady --wind_angle 0 --fix_vx 0 --fix_vy 0

# L5 side wind, walking
python legged_gym/scripts/play.py --task=g1_wind --load_run Mar10_18-22-48_ --wind_level 5 --wind_mode steady --wind_angle 90 --fix_vx 0.6

# OOD: wind direction reversal at t=5s
python legged_gym/scripts/play.py --task=g1_wind --load_run Mar10_18-22-48_ --wind_level 5 --ood_pattern reversal --fix_vx 0.6
```

| Flag | Description |
|---|---|
| `--wind_level 0-5` | Wind curriculum level |
| `--wind_angle DEG` | Fixed wind direction in degrees |
| `--wind_mode MODE` | `full` \| `steady` \| `turbulent` \| `gusts` |
| `--ood_pattern PAT` | `step` \| `periodic` \| `reversal` |
| `--fix_vx FLOAT` | Fixed forward velocity command |
| `--fix_vy FLOAT` | Fixed lateral velocity command |
| `--num_envs INT` | Number of parallel robots |

### Evaluation

```bash
# All suites at L3/L4/L5 (84 scenarios)
python legged_gym/g1_wind_test/eval_wind_robustness.py --task g1_wind --load_run Mar10_18-22-48_ --test_level all --headless

# Single suite
python legged_gym/g1_wind_test/eval_wind_robustness.py --task g1_wind --load_run Mar10_18-22-48_ --suite levels --headless

# Save to JSON with reproducible seed
python legged_gym/g1_wind_test/eval_wind_robustness.py --task g1_wind --load_run Mar10_18-22-48_ --output results.json --seed 42 --headless
```

---

## Trained Models

| Exp | Task | Run Directory | Iters | Notes |
|---|---|---|---|---|
| Exp1 | `g1_wind_baseline` | `logs/g1_wind_baseline/Mar12_00-14-00_` | 9950 | No-wind baseline |
| Exp2 | `g1_wind_push_only` | `logs/g1_wind_push_only/Mar12_21-47-52_` | 1450 | Push perturbation only |
| **Exp3** | `g1_wind` | `logs/g1_wind/Mar10_18-22-48_` | 2900 | **Full method (best)** |
| Exp4 | `g1_wind_no_curriculum` | `logs/g1_wind_no_curriculum/Mar12_21-34-46_` | 1500 | Ablation: no curriculum |
| Exp5 | `g1_wind_no_reward` | `logs/g1_wind_no_reward/Mar12_21-34-59_` | 2100 | Ablation: no wind rewards |

All experiments use identical 277K architecture: LSTM-128, Actor MLP [128,64], Critic MLP [256,128].

Exported LSTM policy: `logs/g1_wind/exported/policies/policy_lstm_1.pt`

---

## Acknowledgements

- [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym) — base training framework
- [legged_gym](https://github.com/leggedrobotics/legged_gym) — Isaac Gym locomotion framework
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl) — PPO implementation
- Viereck et al. IROS 2024, Humanoid-Gym 2024, Booster Gym 2025, Xu et al. 2025, Li et al. 2025 — reward design references
