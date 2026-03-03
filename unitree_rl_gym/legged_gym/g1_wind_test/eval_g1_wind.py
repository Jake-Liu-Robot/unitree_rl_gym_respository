"""
Quantitative evaluation of g1_wind models across different wind levels.

Uses Isaac Gym's native --load_run / --checkpoint / --headless flags.

Usage:
    # Evaluate the wind-trained model (Run5) across all wind levels
    python eval_g1_wind.py --task g1_wind --load_run Feb28_22-07-25_ --headless

    # Evaluate the baseline model (Run4)
    python eval_g1_wind.py --task g1_wind --load_run Feb28_21-36-56_ --headless

    # Evaluate specific checkpoint and specific wind levels
    python eval_g1_wind.py --task g1_wind --load_run Feb28_22-07-25_ --checkpoint 1000 --levels 0 3 5 --headless
"""

import isaacgym
import torch
import numpy as np
import os
import argparse

from legged_gym.envs import *
from legged_gym.utils import task_registry, get_args, get_load_path, class_to_dict
from rsl_rl.runners import OnPolicyRunner


def load_policy(env, train_cfg, load_run, checkpoint, device):
    """Load a trained policy from a specific run."""
    train_cfg.runner.resume = True
    train_cfg_dict = class_to_dict(train_cfg)
    # log_dir=None avoids creating a new tensorboard log
    runner = OnPolicyRunner(env, train_cfg_dict, log_dir=None, device=device)

    # logs/ is under unitree_rl_gym/ (two levels up from g1_wind_test/)
    log_root = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', '..', 'logs', train_cfg.runner.experiment_name
    )
    resume_path = get_load_path(log_root, load_run=load_run, checkpoint=checkpoint)
    print(f"Loading model from: {resume_path}")
    runner.load(resume_path, load_optimizer=False)
    return runner.get_inference_policy(device=device)


def evaluate_wind_level(env, policy, wind_level, num_episodes=50, max_steps=1000):
    """Run evaluation at a fixed wind level, collect statistics.

    Returns dict with:
        - survival_rate: fraction of episodes that survived max_steps
        - mean_episode_length: average episode length in steps
        - mean_tracking_error: average xy velocity tracking error
        - mean_reward: average per-step reward
        - mean_wind_force_mag: average wind force magnitude (N)
    """
    device = env.device
    num_envs = env.num_envs

    # Force all envs to the target wind level
    env.wind_curriculum_level[:] = wind_level

    # Freeze curriculum: save and override the upgrade window to prevent advancement
    saved_upgrade_window = env.cfg.wind.upgrade_window
    env.cfg.wind.upgrade_window = 999999  # effectively disable curriculum changes

    # Reset all envs by forcing termination
    env.reset_buf[:] = 1
    env_ids = torch.arange(num_envs, device=device)
    env.reset_idx(env_ids)
    obs = env.get_observations()

    # Accumulators
    episode_lengths = []
    episode_rewards = []
    episode_tracking_errors = []
    episode_wind_forces = []
    episodes_collected = 0

    # Per-env running stats
    ep_len = torch.zeros(num_envs, device=device)
    ep_rew = torch.zeros(num_envs, device=device)
    ep_track_err = torch.zeros(num_envs, device=device)
    ep_wind_force = torch.zeros(num_envs, device=device)

    step = 0
    while episodes_collected < num_episodes:
        with torch.no_grad():
            actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())

        # Per-step metrics
        track_err = torch.sum(
            torch.square(env.commands[:, :2] - env.base_lin_vel[:, :2]), dim=1
        ).sqrt()
        wind_mag = torch.norm(env.wind_model.wind_force, dim=1)

        ep_len += 1
        ep_rew += rews
        ep_track_err += track_err
        ep_wind_force += wind_mag

        # Collect finished episodes
        done_ids = dones.nonzero(as_tuple=False).flatten()
        if len(done_ids) > 0:
            for idx in done_ids:
                i = idx.item()
                length = ep_len[i].item()
                if length > 0:
                    episode_lengths.append(length)
                    episode_rewards.append(ep_rew[i].item() / length)
                    episode_tracking_errors.append(ep_track_err[i].item() / length)
                    episode_wind_forces.append(ep_wind_force[i].item() / length)
                    episodes_collected += 1

                # Reset per-env accumulators
                ep_len[i] = 0
                ep_rew[i] = 0
                ep_track_err[i] = 0
                ep_wind_force[i] = 0

                if episodes_collected >= num_episodes:
                    break

        step += 1
        # Safety: avoid infinite loop if robot never falls
        if step > max_steps * 3:
            # Collect remaining in-progress episodes
            for i in range(num_envs):
                if ep_len[i] > 0 and episodes_collected < num_episodes:
                    length = ep_len[i].item()
                    episode_lengths.append(length)
                    episode_rewards.append(ep_rew[i].item() / length)
                    episode_tracking_errors.append(ep_track_err[i].item() / length)
                    episode_wind_forces.append(ep_wind_force[i].item() / length)
                    episodes_collected += 1
            break

    # Restore curriculum window
    env.cfg.wind.upgrade_window = saved_upgrade_window
    # Re-freeze level for consistency (in case it drifted before freeze took effect)
    env.wind_curriculum_level[:] = wind_level

    episode_lengths = np.array(episode_lengths[:num_episodes])
    episode_rewards = np.array(episode_rewards[:num_episodes])
    episode_tracking_errors = np.array(episode_tracking_errors[:num_episodes])
    episode_wind_forces = np.array(episode_wind_forces[:num_episodes])

    return {
        "survival_rate": float(np.mean(episode_lengths >= max_steps)),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "std_episode_length": float(np.std(episode_lengths)),
        "mean_reward_per_step": float(np.mean(episode_rewards)),
        "mean_tracking_error": float(np.mean(episode_tracking_errors)),
        "mean_wind_force_N": float(np.mean(episode_wind_forces)),
    }


def main():
    # Use Isaac Gym's native arg parser (same as play.py) to avoid CUDA conflicts
    args = get_args()

    # Our custom args — read from command line manually since get_args() doesn't know them
    eval_parser = argparse.ArgumentParser(add_help=False)
    eval_parser.add_argument("--levels", type=int, nargs="+", default=None)
    eval_parser.add_argument("--num_episodes", type=int, default=50)
    eval_parser.add_argument("--max_steps", type=int, default=1000)
    eval_args, _ = eval_parser.parse_known_args()

    wind_levels = eval_args.levels if eval_args.levels else list(range(6))

    # --- Create environment (same overrides as play.py) ---
    env_cfg, train_cfg = task_registry.get_cfgs(name="g1_wind")
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 64)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.env.test = True
    env_cfg.env.episode_length_s = eval_args.max_steps * env_cfg.sim.dt * env_cfg.control.decimation

    env, _ = task_registry.make_env(name="g1_wind", args=args, env_cfg=env_cfg)
    print(f"Environment created: {env.num_envs} envs, device={env.device}")

    # --- Load policy (use args.load_run from --load_run flag) ---
    load_run = args.load_run if args.load_run else "Feb28_22-07-25_"
    checkpoint = args.checkpoint if args.checkpoint is not None else -1
    policy = load_policy(env, train_cfg, load_run, checkpoint, args.rl_device)

    # --- Evaluate across wind levels ---
    level_names = ["No wind", "Light (1-3)", "Light-Med (2-5)",
                   "Medium (4-8)", "Strong (7-12)", "Extreme (10-18)"]

    print(f"\n{'='*85}")
    print(f"  Evaluating: {load_run}  |  {eval_args.num_episodes} episodes/level  |  max {eval_args.max_steps} steps")
    print(f"{'='*85}")
    print(f"{'Level':>6} {'Description':>15} {'Survival%':>10} {'EpLen':>10} {'Reward/step':>12} "
          f"{'TrackErr':>10} {'WindForce(N)':>13}")
    print(f"{'-'*85}")

    all_results = {}
    for level in wind_levels:
        results = evaluate_wind_level(
            env, policy, level,
            num_episodes=eval_args.num_episodes,
            max_steps=eval_args.max_steps
        )
        all_results[level] = results

        name = level_names[level] if level < len(level_names) else f"Level {level}"
        print(f"{level:>6} {name:>15} {results['survival_rate']*100:>9.1f}% "
              f"{results['mean_episode_length']:>10.1f} {results['mean_reward_per_step']:>12.4f} "
              f"{results['mean_tracking_error']:>10.4f} {results['mean_wind_force_N']:>13.2f}")

    print(f"{'='*85}")

    # --- Summary ---
    print(f"\nSummary for {load_run}:")
    no_wind = all_results.get(0, {})
    strong = all_results.get(4, {})
    if no_wind and strong:
        print(f"  No-wind survival: {no_wind.get('survival_rate', 0)*100:.0f}%  |  "
              f"Strong-wind survival: {strong.get('survival_rate', 0)*100:.0f}%")
        if no_wind.get('mean_tracking_error', 0) > 0:
            degradation = strong.get('mean_tracking_error', 0) / no_wind['mean_tracking_error']
            print(f"  Tracking error degradation (level4/level0): {degradation:.2f}x")


if __name__ == "__main__":
    main()
