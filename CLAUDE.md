# Unitree RL Gym — Workspace Root

## Workspace Structure
```
Unitree_rl_gym/                          # Workspace root (git repo)
├── unitree_rl_gym/                      # ★ Main project — see unitree_rl_gym/CLAUDE.md for details
│   ├── legged_gym/                      # Env definitions, training scripts, wind environment
│   ├── resources/                       # Robot URDF models
│   └── logs/                            # Training outputs
├── IsaacGym_Preview_4_Package/          # NVIDIA Isaac Gym Preview 4 (physics simulator)
│   └── isaacgym/                        # Python package (pip install -e .)
├── rsl_rl/                              # RSL RL library — PPO implementation (do NOT modify)
│   └── rsl_rl/                          # Python package (pip install -e .)
└── unitree_sdk2_python/                 # Unitree SDK2 (robot communication, not used in training)
```

## Key Notes
- **Main project code** is in `unitree_rl_gym/` — refer to `unitree_rl_gym/CLAUDE.md` for detailed architecture, class hierarchy, reward design, and training instructions
- `IsaacGym_Preview_4_Package/`, `rsl_rl/`, `unitree_sdk2_python/` are **gitignored** dependencies installed via `pip install -e .`
- **Do NOT modify** `rsl_rl/` or `IsaacGym_Preview_4_Package/` — they are upstream libraries

## Environment
- **Conda env**: `unitree-rl` (`conda activate unitree-rl`)
- **Python**: 3.8+, **PyTorch**: CUDA 12.1
- All commands should be run from `unitree_rl_gym/` directory

## Quick Reference
```bash
cd unitree_rl_gym
python legged_gym/scripts/train.py --task=g1_wind --headless
python legged_gym/scripts/play.py --task=g1_wind --load_run Mar02_21-49-27_
```
