# Wind Force Arrow Visualization — Isaac Gym & MuJoCo

Per-body wind force arrow visualization for G1 wind-robust walking policies under L5 headwind.

## Visualization

| Arrow | Position | Meaning |
|-------|----------|---------|
| Per-body force (x5) | Center of pressure (CoP) | Actual clamped force applied to pelvis, thighs, shins |
| Wind field (x5) | 2m upwind, horizontal row | Wind velocity direction indicator |

Force bodies: pelvis (55%), left/right thigh (12% each), left/right shin (8% each).

## Test Configuration

| Parameter | Value |
|-----------|-------|
| Wind level | L5 (10-18 m/s) |
| Wind mode | steady (180deg headwind) |
| Command | vx=0.6 m/s, heading=0 (+X) |
| Heading controller | cmd[2] = clip(0.5 * heading_error, -1, 1) |
| Duration | 15 seconds |
| Resolution | 1280x720 @ 30fps |

## Videos

| File | Engine | Experiment |
|------|--------|-----------|
| `isaac_gym/exp3_headwind_L5.mp4` | Isaac Gym | Exp3 full method |
| `isaac_gym/exp2_headwind_L5.mp4` | Isaac Gym | Exp2 push only |
| `mujoco/exp3_headwind_L5.mp4` | MuJoCo | Exp3 full method |
| `mujoco/exp2_headwind_L5.mp4` | MuJoCo | Exp2 push only |

## Headwind (180deg) Comparison at t=14s

| | Isaac Gym Exp3 | MuJoCo Exp3 | Isaac Gym Exp2 | MuJoCo Exp2 |
|---|---|---|---|---|
| Yaw | **-59.3°** | **-59.3°** | +2.6° | -16.9° |
| cmd2 | +0.52 | +0.52 | -0.02 | +0.15 |
| X progress | +2.89m | +3.83m | -6.46m | -2.49m |
| Force | ~28N | ~27N | ~39N | ~39N |
| Strategy | Sidebody | Sidebody | Head-on | Head-on |

**Findings**:
- Exp3 yaw and cmd2 match perfectly between Isaac Gym and MuJoCo (-59.3°, +0.52)
- Exp3 sidebody strategy reduces drag ~30% (28N vs 39N) and advances against headwind
- Exp2 is pushed backwards in both engines (-6.5m / -2.5m)

## Reproduction

```bash
cd unitree_rl_gym

# Isaac Gym — Exp3 headwind
python legged_gym/scripts/play.py --task=g1_wind --load_run Mar10_18-22-48_ \
  --wind_level 5 --wind_mode steady --wind_angle 180 --fix_vx 0.6 --num_envs 1 \
  --record demo_videos/wind_arrow_viz/isaac_gym/exp3_headwind_L5.mp4 --record_duration 15

# Isaac Gym — Exp2 headwind
python legged_gym/scripts/play.py --task=g1_wind --load_run Mar12_21-47-52_ \
  --model_experiment g1_wind_push_only \
  --wind_level 5 --wind_mode steady --wind_angle 180 --fix_vx 0.6 --num_envs 1 \
  --record demo_videos/wind_arrow_viz/isaac_gym/exp2_headwind_L5.mp4 --record_duration 15

# MuJoCo — Exp3 headwind
python deploy/deploy_mujoco/deploy_mujoco_lstm.py g1_wind.yaml \
  --wind_level 5 --wind_mode steady --wind_angle 180 --cmd_vx 0.6 --seed 42 \
  --record demo_videos/wind_arrow_viz/mujoco/exp3_headwind_L5.mp4 --record_duration 15

# MuJoCo — Exp2 headwind
python deploy/deploy_mujoco/deploy_mujoco_lstm.py g1_wind_push_only.yaml \
  --wind_level 5 --wind_mode steady --wind_angle 180 --cmd_vx 0.6 --seed 42 \
  --record demo_videos/wind_arrow_viz/mujoco/exp2_headwind_L5.mp4 --record_duration 15
```
