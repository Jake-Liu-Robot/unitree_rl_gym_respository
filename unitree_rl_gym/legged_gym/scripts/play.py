import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import math
import argparse

import isaacgym
from isaacgym import gymapi
from isaacgym.torch_utils import quat_apply
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch


def play(args, play_args):
    # When model_experiment is set, we do cross-experiment evaluation:
    # - Environment created from args.task (e.g., g1_wind with wind enabled)
    # - Model loaded from a different experiment directory (e.g., g1_wind_push_only)
    model_experiment = play_args.model_experiment

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, play_args.num_envs)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_action_delay = False
    env_cfg.domain_rand.randomize_pd_gains = False
    env_cfg.domain_rand.randomize_motor_strength = False

    env_cfg.env.test = True

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    # --- Auto-detect network dims from checkpoint to handle old runs ---
    from legged_gym.utils import get_load_path
    if model_experiment:
        log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', model_experiment)
    else:
        log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
    _checkpoint = args.checkpoint if args.checkpoint is not None else -1
    resume_path = get_load_path(log_root, load_run=args.load_run, checkpoint=_checkpoint)
    print("Loading model from:", resume_path)
    loaded = torch.load(resume_path, map_location='cpu')
    if 'model_state_dict' in loaded:
        sd = loaded['model_state_dict']
        # Infer actor hidden dims from checkpoint weights
        actor_layers = sorted([k for k in sd if k.startswith('actor.') and 'weight' in k])
        if actor_layers:
            actor_dims = [sd[k].shape[0] for k in actor_layers[:-1]]  # exclude output layer
            critic_layers = sorted([k for k in sd if k.startswith('critic.') and 'weight' in k])
            critic_dims = [sd[k].shape[0] for k in critic_layers[:-1]]
            if actor_dims != train_cfg.policy.actor_hidden_dims:
                print(f"[compat] Overriding actor_hidden_dims: {train_cfg.policy.actor_hidden_dims} -> {actor_dims}")
                train_cfg.policy.actor_hidden_dims = actor_dims
            if critic_dims != train_cfg.policy.critic_hidden_dims:
                print(f"[compat] Overriding critic_hidden_dims: {train_cfg.policy.critic_hidden_dims} -> {critic_dims}")
                train_cfg.policy.critic_hidden_dims = critic_dims

    # For cross-experiment, disable auto-resume (we load manually with strict=False)
    if model_experiment:
        train_cfg.runner.resume = False
    else:
        train_cfg.runner.resume = True

    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)

    # Cross-experiment: manually load compatible weights (actor OK, critic may differ)
    if model_experiment:
        saved_state = loaded['model_state_dict']
        model_state = ppo_runner.alg.actor_critic.state_dict()
        compatible_state = {}
        skipped = []
        for k, v in saved_state.items():
            if k in model_state and v.shape == model_state[k].shape:
                compatible_state[k] = v
            else:
                skipped.append(k)
        ppo_runner.alg.actor_critic.load_state_dict(compatible_state, strict=False)
        print(f"  Cross-experiment load: {len(compatible_state)} params loaded, "
              f"{len(skipped)} skipped (size mismatch)")
        if skipped:
            print(f"    Skipped: {skipped}")
    del loaded

    policy = ppo_runner.get_inference_policy(device=env.device)
    actor_critic = ppo_runner.alg.actor_critic

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY and not model_experiment:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    # ================================================================
    # Wind scenario setup
    # ================================================================
    has_wind = hasattr(env, 'wind_model') and env.cfg.wind.enable
    wind_model = env.wind_model if has_wind else None

    if has_wind:
        # Freeze curriculum
        env.cfg.wind.upgrade_window = 999999

        # Set wind level
        if play_args.wind_level is not None:
            env.wind_curriculum_level[:] = play_args.wind_level
        # else: default (max level in test mode)

        # Reset all envs with new level
        env_ids = torch.arange(env.num_envs, device=env.device)
        env.reset_idx(env_ids)

        # Wind mode: control which layers are active
        if play_args.wind_mode == "steady":
            wind_model.disable_ou = True
            wind_model.disable_gusts = True
        elif play_args.wind_mode == "turbulent":
            wind_model.disable_ou = False
            wind_model.disable_gusts = True
        elif play_args.wind_mode == "gusts":
            wind_model.disable_ou = True
            wind_model.disable_gusts = False
        # "full" (default): both enabled, no changes needed

        # Fixed wind direction
        if play_args.wind_angle is not None:
            angle_rad = math.radians(play_args.wind_angle)
            wind_model.base_angle[:] = angle_rad
            wind_model.base_direction[:, 0] = math.cos(angle_rad)
            wind_model.base_direction[:, 1] = math.sin(angle_rad)
            wind_model.base_direction[:, 2] = 0.0
            wind_model.effective_direction[:] = wind_model.base_direction[:]

        # Fixed wind speed (overrides random sampling)
        if play_args.wind_speed is not None:
            wind_model.base_speed[:] = play_args.wind_speed

        obs = env.get_observations()

    # ================================================================
    # Command override setup
    # ================================================================
    fix_commands = None
    if play_args.fix_vx is not None or play_args.fix_vy is not None:
        fix_commands = {
            "vx": play_args.fix_vx if play_args.fix_vx is not None else 0.0,
            "vy": play_args.fix_vy if play_args.fix_vy is not None else 0.0,
        }
        if play_args.fix_yaw is not None:
            fix_commands["heading_offset"] = play_args.fix_yaw

        # Disable command resampling
        env._orig_resample_commands = env._resample_commands
        env._resample_commands = lambda env_ids: None

        # Set target heading to 0 (+X) if not explicitly specified
        if "heading_offset" not in fix_commands:
            env.commands[:, 3] = 0.0  # target heading = face +X

        # Apply initial commands
        _apply_fixed_commands(env, fix_commands)
        env.compute_observations()
        clip_obs = env.cfg.normalization.clip_observations
        obs = torch.clip(env.obs_buf, -clip_obs, clip_obs)

    # ================================================================
    # OOD pattern setup
    # ================================================================
    ood_pattern = play_args.ood_pattern
    if ood_pattern and has_wind:
        # Compute peak speed from curriculum level
        wl = play_args.wind_level if play_args.wind_level is not None else 5
        curriculum_levels = env.cfg.wind.curriculum_levels
        if wl < len(curriculum_levels):
            ood_peak_speed = curriculum_levels[wl][1]
        else:
            ood_peak_speed = 18.0
        ood_peak_speed = max(ood_peak_speed, 10.0)
        # Disable OU and gusts for clean OOD patterns
        wind_model.disable_ou = True
        wind_model.disable_gusts = True

    # ================================================================
    # Print scenario info
    # ================================================================
    level_str = str(play_args.wind_level) if play_args.wind_level is not None else "max"
    print(f"\n--- Play Scenario ---")
    print(f"  Wind level : {level_str}")
    print(f"  Wind mode  : {play_args.wind_mode}")
    print(f"  Wind angle : {play_args.wind_angle if play_args.wind_angle is not None else 'random'}deg")
    if ood_pattern and has_wind:
        print(f"  OOD pattern: {ood_pattern} (peak={ood_peak_speed:.0f} m/s)")
    elif ood_pattern:
        print(f"  OOD pattern: {ood_pattern} (wind disabled, ignored)")
    if fix_commands:
        print(f"  Commands   : vx={fix_commands['vx']}, vy={fix_commands['vy']}"
              + (f", yaw={fix_commands.get('heading_offset', 'auto')}" if 'heading_offset' in fix_commands else ""))
    else:
        print(f"  Commands   : random (resampled every ~10s)")
    print(f"  Num envs   : {env.num_envs}")
    print(f"---------------------\n")

    # ================================================================
    # Video recording setup
    # ================================================================
    video_writer = None
    record_frames = play_args.record is not None
    record_cam = None
    if record_frames:
        import cv2
        from isaacgym import gymapi
        rec_fps = play_args.record_fps
        rec_steps = int(play_args.record_duration / (env.cfg.sim.dt * env.cfg.control.decimation))
        render_every = max(1, int(1.0 / rec_fps / (env.cfg.sim.dt * env.cfg.control.decimation)))
        # Create offscreen camera sensor in env 0
        cam_props = gymapi.CameraProperties()
        cam_props.width = 1280
        cam_props.height = 720
        cam_props.enable_tensors = False
        record_cam = env.gym.create_camera_sensor(env.envs[0], cam_props)
        print(f"Created offscreen camera ({cam_props.width}x{cam_props.height})")

    # ================================================================
    # Main loop
    # ================================================================
    dt_control = env.cfg.sim.dt * env.cfg.control.decimation
    ep_step = torch.zeros(env.num_envs, device=env.device)

    total_iters = rec_steps if record_frames else 10 * int(env.max_episode_length)
    for i in range(total_iters):
        # --- OOD pattern: override wind before step ---
        if ood_pattern and has_wind:
            t = ep_step * dt_control
            if ood_pattern == "step":
                wind_model.base_speed = torch.where(
                    t < 5.0,
                    torch.zeros_like(t),
                    torch.full_like(t, ood_peak_speed)
                )
            elif ood_pattern == "periodic":
                mid = ood_peak_speed / 2.0
                amp = ood_peak_speed / 2.0
                wind_model.base_speed = mid + amp * torch.sin(2 * math.pi / 4.0 * t)
            elif ood_pattern == "reversal":
                angle = torch.where(t < 5.0, torch.zeros_like(t),
                                    torch.full_like(t, math.pi))
                wind_model.base_speed[:] = ood_peak_speed
                wind_model.base_angle[:] = angle
                wind_model.base_direction[:, 0] = torch.cos(angle)
                wind_model.base_direction[:, 1] = torch.sin(angle)
                wind_model.base_direction[:, 2] = 0.0

        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())
        ep_step += 1

        # Diagnostic: print yaw and position every 2s
        if i % 100 == 0:
            pos = env.root_states[0, :3].cpu().numpy()
            forward = env.root_states[0, 7:10].cpu().numpy()  # lin vel
            fwd_vec = torch.tensor([[1, 0, 0]], dtype=torch.float, device=env.device)
            heading_vec = quat_apply(env.base_quat[0:1], fwd_vec)
            heading_angle = math.degrees(math.atan2(heading_vec[0, 1].item(), heading_vec[0, 0].item()))
            cmd2 = env.commands[0, 2].item()
            target_h = math.degrees(env.commands[0, 3].item())
            print(f"  t={i*dt_control:.1f}s  yaw={heading_angle:+6.1f}°  target={target_h:+6.1f}°  cmd2={cmd2:+.2f}  pos=[{pos[0]:+.2f},{pos[1]:+.2f}]")

        # Camera tracking: rear-side view (see both lean and forward progress)
        robot_pos = env.root_states[0, :3].cpu().numpy()
        cam_offset = np.array([-2.0, -2.0, 1.0])  # behind-right, above
        cam_pos = robot_pos + cam_offset
        cam_target = robot_pos + np.array([0, 0, 0.5])  # look at pelvis height
        if not args.headless:
            env.set_camera(cam_pos.tolist(), cam_target.tolist())
        if record_cam is not None:
            env.gym.set_camera_location(record_cam, env.envs[0],
                                        gymapi.Vec3(*cam_pos),
                                        gymapi.Vec3(*cam_target))

        # Override commands after step
        if fix_commands is not None:
            _apply_fixed_commands(env, fix_commands)
            env.compute_observations()
            clip_obs = env.cfg.normalization.clip_observations
            obs = torch.clip(env.obs_buf, -clip_obs, clip_obs)

        # Reset LSTM hidden states and episode step counter for terminated envs
        if dones.any():
            actor_critic.memory_a.reset(dones)
            ep_step[dones] = 0

            # Re-apply wind overrides for reset envs
            if has_wind and play_args.wind_angle is not None:
                done_ids = dones.nonzero(as_tuple=False).flatten()
                angle_rad = math.radians(play_args.wind_angle)
                for idx in done_ids:
                    i_env = idx.item()
                    wind_model.base_angle[i_env] = angle_rad
                    wind_model.base_direction[i_env, 0] = math.cos(angle_rad)
                    wind_model.base_direction[i_env, 1] = math.sin(angle_rad)
                    wind_model.base_direction[i_env, 2] = 0.0

        # Record frame
        if record_frames and i % render_every == 0:
            env.gym.fetch_results(env.sim, True)
            env.gym.step_graphics(env.sim)
            env.gym.render_all_camera_sensors(env.sim)
            img = env.gym.get_camera_image(env.sim, env.envs[0], record_cam, gymapi.IMAGE_COLOR)
            if img is not None:
                frame = np.frombuffer(img, dtype=np.uint8).reshape(720, 1280, 4)  # RGBA
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                if video_writer is None:
                    h, w = frame_bgr.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(play_args.record, fourcc, rec_fps, (w, h))
                    print(f"Recording {play_args.record_duration}s → {play_args.record} ({w}x{h} @ {rec_fps}fps)")
                video_writer.write(frame_bgr)

    if video_writer is not None:
        video_writer.release()
        print(f"Done! Video saved to: {play_args.record}")


def _apply_fixed_commands(env, fix_commands):
    """Override velocity commands."""
    env.commands[:, 0] = fix_commands["vx"]
    env.commands[:, 1] = fix_commands["vy"]
    if "heading_offset" in fix_commands:
        forward = quat_apply(env.base_quat, env.forward_vec)
        cur_heading = torch.atan2(forward[:, 1], forward[:, 0])
        env.commands[:, 3] = cur_heading + fix_commands["heading_offset"]


if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False

    # Pre-parse play-specific args before get_args()
    play_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    play_parser.add_argument("--wind_level", type=int, default=None,
                             help="Wind curriculum level 0-5 (default: max)")
    play_parser.add_argument("--wind_angle", type=float, default=None,
                             help="Fixed wind direction in degrees (default: random)")
    play_parser.add_argument("--wind_speed", type=float, default=None,
                             help="Fixed wind speed in m/s (overrides random sampling)")
    play_parser.add_argument("--wind_mode", type=str, default="full",
                             choices=["full", "steady", "turbulent", "gusts"],
                             help="Wind model mode")
    play_parser.add_argument("--ood_pattern", type=str, default=None,
                             choices=["step", "periodic", "reversal"],
                             help="OOD wind pattern")
    play_parser.add_argument("--fix_vx", type=float, default=None,
                             help="Fixed forward velocity command")
    play_parser.add_argument("--fix_vy", type=float, default=None,
                             help="Fixed lateral velocity command")
    play_parser.add_argument("--fix_yaw", type=float, default=None,
                             help="Fixed heading offset for turning")
    play_parser.add_argument("--num_envs", type=int, default=4,
                             help="Number of environments (default: 4)")
    play_parser.add_argument("--model_experiment", type=str, default=None,
                             help="Load model from a different experiment directory (cross-experiment eval)")
    play_parser.add_argument("--record", type=str, default=None,
                             help="Record to MP4 file (e.g. --record video.mp4)")
    play_parser.add_argument("--record_duration", type=float, default=10.0,
                             help="Recording duration in seconds (default: 10)")
    play_parser.add_argument("--record_fps", type=int, default=30,
                             help="Recording FPS (default: 30)")
    play_args, remaining_argv = play_parser.parse_known_args()

    # Recording requires graphics — strip --headless so camera sensor works
    if play_args.record and '--headless' in remaining_argv:
        remaining_argv.remove('--headless')
        print("[record] Ignoring --headless (camera sensor needs graphics device)")

    # Patch sys.argv so get_args() only sees its own arguments
    import sys
    sys.argv = [sys.argv[0]] + remaining_argv

    args = get_args()
    play(args, play_args)
