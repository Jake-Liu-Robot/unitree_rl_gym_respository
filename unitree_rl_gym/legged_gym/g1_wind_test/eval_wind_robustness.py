"""
Comprehensive wind robustness evaluation for g1_wind policies.

Tests policy across multiple dimensions:
  A: Wind speed levels (0-5)
  B: Wind modes (steady, turbulent, gusts-only, full model)
  C: Wind directions (front, side, back, diagonal, random)
  D: OU parameter extremes (calm, turbulent, locked heading, erratic heading)
  E: Out-of-distribution wind patterns (step change, periodic)

Usage:
    # Run all in-distribution tests at Level 3
    python eval_wind_robustness.py --task g1_wind --load_run Mar02_21-49-27_ --headless

    # Run specific test suite
    python eval_wind_robustness.py --task g1_wind --load_run Mar02_21-49-27_ --suite modes --headless

    # Compare two policies
    python eval_wind_robustness.py --task g1_wind --load_run Mar02_21-49-27_ --load_run2 Feb28_21-36-56_ --headless
"""

import isaacgym
import torch
import numpy as np
import os
import argparse
import math

from legged_gym.envs import *
from legged_gym.utils import task_registry, get_args, get_load_path, class_to_dict
from rsl_rl.runners import OnPolicyRunner


# ============================================================
# Test Scenario Definitions
# ============================================================

# B: Wind modes — control which layers are active
WIND_MODES = {
    "B1_steady":      {"ou_speed": False, "ou_dir": False, "gusts": False, "label": "Steady (Layer1 only)"},
    "B2_turbulent":   {"ou_speed": True,  "ou_dir": True,  "gusts": False, "label": "Turbulent (L1+L2)"},
    "B3_gusts_only":  {"ou_speed": False, "ou_dir": False, "gusts": True,  "label": "Gusts only (L1+L3)"},
    "B4_full":        {"ou_speed": True,  "ou_dir": True,  "gusts": True,  "label": "Full model (L1+L2+L3)"},
    "B5_pure_gusts":  {"ou_speed": False, "ou_dir": False, "gusts": True,  "label": "Pure gusts (L3 only)", "base_speed_override": 0.0},
}

# C: Wind directions (angle in radians, None = random)
WIND_DIRECTIONS = {
    "C1_front":    {"angle": 0.0,       "label": "Front (0°)"},
    "C2_side":     {"angle": math.pi/2, "label": "Side (90°)"},
    "C3_back":     {"angle": math.pi,   "label": "Back (180°)"},
    "C4_diagonal": {"angle": math.pi/4, "label": "Diagonal (45°)"},
    "C5_random":   {"angle": None,      "label": "Random"},
}

# D: OU parameter extremes (override per-env params)
OU_EXTREMES = {
    "D1_calm":       {"theta": 1.0, "sigma": 0.05, "theta_dir": 1.0, "sigma_dir": 0.02,
                      "label": "Calm (near-constant)"},
    "D2_turbulent":  {"theta": 0.2, "sigma": 0.4,  "theta_dir": 0.2, "sigma_dir": 0.2,
                      "label": "High turbulence"},
    "D3_locked_dir": {"theta": 0.5, "sigma": 0.25, "theta_dir": 1.0, "sigma_dir": 0.02,
                      "label": "Locked direction"},
    "D4_erratic_dir":{"theta": 0.5, "sigma": 0.25, "theta_dir": 0.05,"sigma_dir": 0.25,
                      "label": "Erratic direction"},
    "D5_default":    {"theta": 0.5, "sigma": 0.25, "theta_dir": 0.2, "sigma_dir": 0.15,
                      "label": "Training default"},
}

# E: Out-of-distribution patterns
OOD_PATTERNS = {
    "E1_step":     {"label": "Step change (0→10 m/s at t=5s)"},
    "E2_periodic": {"label": "Periodic (A·sin(ωt), T=4s)"},
}


def load_policy(env, train_cfg, load_run, checkpoint, device):
    """Load a trained policy from a specific run."""
    train_cfg.runner.resume = True
    train_cfg_dict = class_to_dict(train_cfg)
    runner = OnPolicyRunner(env, train_cfg_dict, log_dir=None, device=device)

    log_root = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', '..', 'logs', train_cfg.runner.experiment_name
    )
    resume_path = get_load_path(log_root, load_run=load_run, checkpoint=checkpoint)
    print(f"Loading model from: {resume_path}")
    runner.load(resume_path, load_optimizer=False)
    return runner.get_inference_policy(device=device)


# ============================================================
# Core Evaluation Function
# ============================================================

def evaluate_scenario(env, policy, wind_level, num_episodes=50, max_steps=1000,
                      fix_direction=None, mode_overrides=None, ou_overrides=None,
                      ood_pattern=None):
    """Run evaluation under a specific wind scenario.

    Args:
        env: G1WindRobot environment
        policy: trained policy (inference mode)
        wind_level: curriculum level (0-5)
        num_episodes: episodes to collect
        max_steps: max steps per episode
        fix_direction: fixed wind angle (rad) or None for random
        mode_overrides: dict with keys ou_speed/ou_dir/gusts (bool) to enable/disable layers
        ou_overrides: dict with keys theta/sigma/theta_dir/sigma_dir to override OU params
        ood_pattern: "step" or "periodic" for out-of-distribution wind

    Returns:
        dict with evaluation metrics
    """
    device = env.device
    num_envs = env.num_envs
    wind_model = env.wind_model

    # Force wind level and freeze curriculum
    env.wind_curriculum_level[:] = wind_level
    saved_upgrade_window = env.cfg.wind.upgrade_window
    env.cfg.wind.upgrade_window = 999999

    # Save original config for restoration
    saved_ou_randomize = env.cfg.wind.ou_randomize

    # --- Apply mode overrides (disable layers) ---
    if mode_overrides:
        # We'll control layers by zeroing their contributions in a custom step wrapper
        pass  # handled in the step loop below

    # --- Apply OU overrides (disable randomization, set fixed params) ---
    if ou_overrides:
        env.cfg.wind.ou_randomize = False

    # Reset all envs
    env.reset_buf[:] = 1
    env_ids = torch.arange(num_envs, device=device)
    env.reset_idx(env_ids)

    # Apply fixed direction if specified
    if fix_direction is not None:
        wind_model.base_angle[:] = fix_direction
        wind_model.base_direction[:, 0] = math.cos(fix_direction)
        wind_model.base_direction[:, 1] = math.sin(fix_direction)
        wind_model.base_direction[:, 2] = 0.0
        wind_model.effective_direction[:] = wind_model.base_direction[:]

    # Apply fixed base_speed override (for B5_pure_gusts)
    if mode_overrides and "base_speed_override" in mode_overrides:
        wind_model.base_speed[:] = mode_overrides["base_speed_override"]

    # Apply OU param overrides
    if ou_overrides:
        wind_model.env_ou_theta[:] = ou_overrides["theta"]
        wind_model.env_ou_sigma[:] = ou_overrides["sigma"]
        wind_model.env_ou_theta_dir[:] = ou_overrides["theta_dir"]
        wind_model.env_ou_sigma_dir[:] = ou_overrides["sigma_dir"]

    obs = env.get_observations()

    # Accumulators
    episode_lengths = []
    episode_rewards = []
    episode_tracking_errors = []
    episode_wind_forces = []
    episodes_collected = 0

    ep_len = torch.zeros(num_envs, device=device)
    ep_rew = torch.zeros(num_envs, device=device)
    ep_track_err = torch.zeros(num_envs, device=device)
    ep_wind_force = torch.zeros(num_envs, device=device)

    # For OOD: track global step count
    global_step = 0
    dt_control = env.cfg.sim.dt * env.cfg.control.decimation  # control dt

    step = 0
    while episodes_collected < num_episodes:
        # --- OOD pattern: override wind model state before step ---
        if ood_pattern == "step":
            # Step change: no wind for first 5s, then 10 m/s
            t = global_step * dt_control
            if t < 5.0:
                wind_model.base_speed[:] = 0.0
            else:
                wind_model.base_speed[:] = 10.0

        elif ood_pattern == "periodic":
            # Periodic wind: speed = 5 + 5*sin(2π/4 * t)
            t = global_step * dt_control
            wind_model.base_speed[:] = 5.0 + 5.0 * math.sin(2 * math.pi / 4.0 * t)

        # --- Mode overrides: zero out disabled layers before step ---
        if mode_overrides:
            if not mode_overrides.get("ou_speed", True):
                wind_model.ou_speed_state[:] = 0.0
            if not mode_overrides.get("ou_dir", True):
                wind_model.ou_angle_state[:] = 0.0
            if not mode_overrides.get("gusts", True):
                wind_model.gust_active[:] = False
                wind_model.gust_speed[:] = 0.0

        with torch.no_grad():
            actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())

        global_step += 1

        # Per-step metrics
        track_err = torch.sum(
            torch.square(env.commands[:, :2] - env.base_lin_vel[:, :2]), dim=1
        ).sqrt()
        wind_mag = torch.norm(wind_model.wind_force, dim=1)

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

                ep_len[i] = 0
                ep_rew[i] = 0
                ep_track_err[i] = 0
                ep_wind_force[i] = 0

                if episodes_collected >= num_episodes:
                    break

            # Re-apply overrides after env resets
            if fix_direction is not None:
                for idx in done_ids:
                    i = idx.item()
                    wind_model.base_angle[i] = fix_direction
                    wind_model.base_direction[i, 0] = math.cos(fix_direction)
                    wind_model.base_direction[i, 1] = math.sin(fix_direction)
                    wind_model.base_direction[i, 2] = 0.0

            if ou_overrides:
                for idx in done_ids:
                    i = idx.item()
                    wind_model.env_ou_theta[i] = ou_overrides["theta"]
                    wind_model.env_ou_sigma[i] = ou_overrides["sigma"]
                    wind_model.env_ou_theta_dir[i] = ou_overrides["theta_dir"]
                    wind_model.env_ou_sigma_dir[i] = ou_overrides["sigma_dir"]

            if mode_overrides and "base_speed_override" in mode_overrides:
                for idx in done_ids:
                    wind_model.base_speed[idx.item()] = mode_overrides["base_speed_override"]

        step += 1
        if step > max_steps * 3:
            for i in range(num_envs):
                if ep_len[i] > 0 and episodes_collected < num_episodes:
                    length = ep_len[i].item()
                    episode_lengths.append(length)
                    episode_rewards.append(ep_rew[i].item() / length)
                    episode_tracking_errors.append(ep_track_err[i].item() / length)
                    episode_wind_forces.append(ep_wind_force[i].item() / length)
                    episodes_collected += 1
            break

    # Restore
    env.cfg.wind.upgrade_window = saved_upgrade_window
    env.cfg.wind.ou_randomize = saved_ou_randomize

    episode_lengths = np.array(episode_lengths[:num_episodes])
    episode_rewards = np.array(episode_rewards[:num_episodes])
    episode_tracking_errors = np.array(episode_tracking_errors[:num_episodes])
    episode_wind_forces = np.array(episode_wind_forces[:num_episodes])

    return {
        "survival_rate": float(np.mean(episode_lengths >= max_steps)),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "mean_reward_per_step": float(np.mean(episode_rewards)),
        "mean_tracking_error": float(np.mean(episode_tracking_errors)),
        "mean_wind_force_N": float(np.mean(episode_wind_forces)),
    }


# ============================================================
# Test Suites
# ============================================================

def run_suite_levels(env, policy, args):
    """Suite A: Standard level sweep (same as eval_g1_wind.py but with full model)."""
    print_header("Suite A: Wind Level Sweep (Full Model)")
    level_names = ["No wind", "Light (1-3)", "Light-Med (2-5)",
                   "Medium (4-8)", "Strong (7-12)", "Extreme (10-18)"]
    results = {}
    for level in range(6):
        r = evaluate_scenario(env, policy, level,
                              num_episodes=args.num_episodes, max_steps=args.max_steps)
        results[f"A_level{level}"] = r
        print_row(f"L{level}", level_names[level], r)
    return results


def run_suite_modes(env, policy, args, test_level=3):
    """Suite B: Wind mode decomposition at a fixed level."""
    print_header(f"Suite B: Wind Modes (Level {test_level})")
    results = {}
    for key, mode in WIND_MODES.items():
        overrides = {k: mode[k] for k in ["ou_speed", "ou_dir", "gusts"]}
        if "base_speed_override" in mode:
            overrides["base_speed_override"] = mode["base_speed_override"]
        r = evaluate_scenario(env, policy, test_level,
                              num_episodes=args.num_episodes, max_steps=args.max_steps,
                              mode_overrides=overrides)
        results[key] = r
        print_row(key, mode["label"], r)
    return results


def run_suite_directions(env, policy, args, test_level=3):
    """Suite C: Fixed wind directions at a fixed level."""
    print_header(f"Suite C: Wind Directions (Level {test_level}, Steady)")
    results = {}
    for key, d in WIND_DIRECTIONS.items():
        r = evaluate_scenario(env, policy, test_level,
                              num_episodes=args.num_episodes, max_steps=args.max_steps,
                              fix_direction=d["angle"],
                              mode_overrides={"ou_speed": False, "ou_dir": False, "gusts": False})
        results[key] = r
        print_row(key, d["label"], r)
    return results


def run_suite_ou_extremes(env, policy, args, test_level=3):
    """Suite D: OU parameter extremes at a fixed level."""
    print_header(f"Suite D: OU Parameter Extremes (Level {test_level})")
    results = {}
    for key, params in OU_EXTREMES.items():
        ou_overrides = {k: params[k] for k in ["theta", "sigma", "theta_dir", "sigma_dir"]}
        r = evaluate_scenario(env, policy, test_level,
                              num_episodes=args.num_episodes, max_steps=args.max_steps,
                              ou_overrides=ou_overrides)
        results[key] = r
        print_row(key, params["label"], r)
    return results


def run_suite_ood(env, policy, args, test_level=3):
    """Suite E: Out-of-distribution wind patterns."""
    print_header(f"Suite E: Out-of-Distribution Patterns (Level {test_level})")
    results = {}
    for key, p in OOD_PATTERNS.items():
        pattern_name = key.split("_")[1]  # "step" or "periodic"
        r = evaluate_scenario(env, policy, test_level,
                              num_episodes=args.num_episodes, max_steps=args.max_steps,
                              fix_direction=0.0,
                              mode_overrides={"ou_speed": False, "ou_dir": False, "gusts": False},
                              ood_pattern=pattern_name)
        results[key] = r
        print_row(key, p["label"], r)
    return results


# ============================================================
# Output Formatting
# ============================================================

def print_header(title):
    print(f"\n{'='*95}")
    print(f"  {title}")
    print(f"{'='*95}")
    print(f"{'ID':>15} {'Description':>25} {'Survival%':>10} {'EpLen':>8} "
          f"{'Reward/s':>9} {'TrackErr':>9} {'Wind(N)':>8}")
    print(f"{'-'*95}")


def print_row(test_id, label, r):
    print(f"{test_id:>15} {label:>25} {r['survival_rate']*100:>9.1f}% "
          f"{r['mean_episode_length']:>8.1f} {r['mean_reward_per_step']:>9.4f} "
          f"{r['mean_tracking_error']:>9.4f} {r['mean_wind_force_N']:>8.2f}")


# ============================================================
# Main
# ============================================================

def main():
    args = get_args()

    eval_parser = argparse.ArgumentParser(add_help=False)
    eval_parser.add_argument("--suite", type=str, default="all",
                             choices=["all", "levels", "modes", "directions", "ou", "ood"],
                             help="Which test suite to run")
    eval_parser.add_argument("--test_level", type=int, default=3,
                             help="Wind level for single-level suites (B/C/D/E)")
    eval_parser.add_argument("--num_episodes", type=int, default=50)
    eval_parser.add_argument("--max_steps", type=int, default=1000)
    eval_parser.add_argument("--load_run2", type=str, default=None,
                             help="Second policy for A/B comparison")
    eval_args, _ = eval_parser.parse_known_args()

    # --- Create environment ---
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

    # --- Load policy ---
    load_run = args.load_run if args.load_run else "Mar02_21-49-27_"
    checkpoint = args.checkpoint if args.checkpoint is not None else -1
    policy = load_policy(env, train_cfg, load_run, checkpoint, args.rl_device)
    print(f"\nEvaluating policy: {load_run}")

    # --- Run selected suites ---
    suite = eval_args.suite
    tl = eval_args.test_level
    all_results = {}

    if suite in ("all", "levels"):
        all_results.update(run_suite_levels(env, policy, eval_args))
    if suite in ("all", "modes"):
        all_results.update(run_suite_modes(env, policy, eval_args, tl))
    if suite in ("all", "directions"):
        all_results.update(run_suite_directions(env, policy, eval_args, tl))
    if suite in ("all", "ou"):
        all_results.update(run_suite_ou_extremes(env, policy, eval_args, tl))
    if suite in ("all", "ood"):
        all_results.update(run_suite_ood(env, policy, eval_args, tl))

    print(f"\n{'='*95}")
    print(f"  Completed {len(all_results)} test scenarios for {load_run}")
    print(f"{'='*95}")

    # --- Optional: Compare with second policy ---
    if eval_args.load_run2:
        print(f"\n\n{'#'*95}")
        print(f"  Comparing with second policy: {eval_args.load_run2}")
        print(f"{'#'*95}")
        policy2 = load_policy(env, train_cfg, eval_args.load_run2, checkpoint, args.rl_device)

        all_results2 = {}
        if suite in ("all", "levels"):
            all_results2.update(run_suite_levels(env, policy2, eval_args))
        if suite in ("all", "modes"):
            all_results2.update(run_suite_modes(env, policy2, eval_args, tl))
        if suite in ("all", "directions"):
            all_results2.update(run_suite_directions(env, policy2, eval_args, tl))
        if suite in ("all", "ou"):
            all_results2.update(run_suite_ou_extremes(env, policy2, eval_args, tl))
        if suite in ("all", "ood"):
            all_results2.update(run_suite_ood(env, policy2, eval_args, tl))

        # Print comparison
        print(f"\n{'='*95}")
        print(f"  Survival Rate Comparison: {load_run} vs {eval_args.load_run2}")
        print(f"{'='*95}")
        print(f"{'Scenario':>20} {'Policy A':>10} {'Policy B':>10} {'Delta':>10}")
        print(f"{'-'*55}")
        for key in all_results:
            if key in all_results2:
                sa = all_results[key]["survival_rate"] * 100
                sb = all_results2[key]["survival_rate"] * 100
                delta = sa - sb
                sign = "+" if delta >= 0 else ""
                print(f"{key:>20} {sa:>9.1f}% {sb:>9.1f}% {sign}{delta:>9.1f}%")


if __name__ == "__main__":
    main()
