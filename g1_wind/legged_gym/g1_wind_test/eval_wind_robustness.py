"""
Comprehensive wind robustness evaluation for g1_wind policies.

Tests policy across multiple dimensions:
  A: Wind speed levels (0-5)
  B: Wind modes (steady, turbulent, gusts-only, full model)
  C: Wind directions (front, side, back, diagonal, random)
  D: OU parameter extremes (calm, turbulent, locked heading, erratic heading)
  E: Out-of-distribution wind patterns (step change, periodic, direction reversal)
  F: Command variations (standing, slow/fast walk, lateral, turning, head/tailwind)

Usage:
    # Run all suites at L3,L4,L5 (comprehensive test)
    python eval_wind_robustness.py --task g1_wind --load_run Mar10_18-22-48_ --test_level all --headless

    # Run all suites at default level 3 only
    python eval_wind_robustness.py --task g1_wind --load_run Mar10_18-22-48_ --headless

    # Run specific suite at specific levels
    python eval_wind_robustness.py --task g1_wind --load_run Mar10_18-22-48_ --suite modes --test_level 4,5 --headless

    # Compare two policies
    python eval_wind_robustness.py --task g1_wind --load_run Mar10_18-22-48_ --load_run2 Feb28_21-36-56_ --headless

    # Save results to JSON
    python eval_wind_robustness.py --task g1_wind --load_run Mar10_18-22-48_ --output results.json --headless
"""

import isaacgym
import torch
import numpy as np
import os
import json
import argparse
import math
from isaacgym.torch_utils import quat_apply
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
                      "label": "High turbulence (OOD)"},
    "D3_locked_dir": {"theta": 0.5, "sigma": 0.25, "theta_dir": 1.0, "sigma_dir": 0.02,
                      "label": "Locked direction"},
    "D4_erratic_dir":{"theta": 0.5, "sigma": 0.25, "theta_dir": 0.05,"sigma_dir": 0.25,
                      "label": "Erratic direction"},
    "D5_default":    {"theta": 0.5, "sigma": 0.25, "theta_dir": 0.2, "sigma_dir": 0.15,
                      "label": "Training default"},
}

# E: Out-of-distribution patterns
# Wind speeds are scaled to the test level's curriculum range at runtime.
OOD_PATTERNS = {
    "E1_step":     {"label_fmt": "Step change (0\u2192{peak:.0f} m/s at t=5s)"},
    "E2_periodic": {"label_fmt": "Periodic ({lo:.0f}-{hi:.0f} m/s, T=4s)"},
    "E3_reversal": {"label_fmt": "Dir reversal (180\u00b0 at t=5s, {peak:.0f} m/s)"},
}

# F: Command variations — fixed velocity commands under wind
# All within training range: vx,vy ~ U[-1,1], yaw from heading, resampled every 10s
COMMAND_SCENARIOS = {
    "F1_standing":   {"vx": 0.0, "vy": 0.0,
                      "label": "Standing (cmd=0)"},
    "F2_slow_fwd":   {"vx": 0.3, "vy": 0.0,
                      "label": "Slow forward (0.3)"},
    "F3_normal_fwd": {"vx": 0.6, "vy": 0.0,
                      "label": "Normal forward (0.6)"},
    "F4_fast_fwd":   {"vx": 1.0, "vy": 0.0,
                      "label": "Fast forward (1.0)"},
    "F5_lateral":    {"vx": 0.0, "vy": 0.5,
                      "label": "Lateral walk (vy=0.5)"},
    "F6_turning":    {"vx": 0.5, "vy": 0.0, "heading_offset": 1.0,
                      "label": "Turning (vx=0.5+yaw)"},
    "F7_headwind":   {"vx": 0.5, "vy": 0.0, "wind_angle": math.pi,
                      "label": "Headwind walk"},
    "F8_tailwind":   {"vx": 0.5, "vy": 0.0, "wind_angle": 0.0,
                      "label": "Tailwind walk"},
}


def load_policy(env, train_cfg, load_run, checkpoint, device, model_experiment=None):
    """Load a trained policy from a specific run.

    Args:
        model_experiment: override experiment directory name for model loading.
            When evaluating a baseline model in the g1_wind env, the critic
            dimensions won't match; strict=False is used automatically so that
            only the actor weights (which share the same architecture) are loaded.

    Returns:
        (policy_fn, actor_critic): inference policy callable and the
        ActorCriticRecurrent module (needed for LSTM hidden state reset).
    """
    train_cfg.runner.resume = True
    train_cfg_dict = class_to_dict(train_cfg)
    runner = OnPolicyRunner(env, train_cfg_dict, log_dir=None, device=device)

    experiment_name = model_experiment or train_cfg.runner.experiment_name
    log_root = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', '..', 'logs', experiment_name
    )
    resume_path = get_load_path(log_root, load_run=load_run, checkpoint=checkpoint)
    print(f"Loading model from: {resume_path}")

    # When loading cross-experiment (e.g. baseline model into g1_wind env),
    # critic dimensions may differ. Filter out size-mismatched keys and load
    # only compatible weights (actor weights share the same architecture).
    loaded_dict = torch.load(resume_path)
    saved_state = loaded_dict['model_state_dict']
    if model_experiment and model_experiment != "g1_wind":
        model_state = runner.alg.actor_critic.state_dict()
        compatible_state = {}
        skipped = []
        for k, v in saved_state.items():
            if k in model_state and v.shape == model_state[k].shape:
                compatible_state[k] = v
            else:
                skipped.append(k)
        missing, unexpected = runner.alg.actor_critic.load_state_dict(
            compatible_state, strict=False
        )
        print(f"  Cross-experiment load: {len(compatible_state)} params loaded, "
              f"{len(skipped)} skipped (size mismatch)")
        if skipped:
            print(f"    Skipped: {skipped}")
    else:
        runner.alg.actor_critic.load_state_dict(saved_state)
    runner.current_learning_iteration = loaded_dict['iter']

    policy = runner.get_inference_policy(device=device)
    actor_critic = runner.alg.actor_critic
    return policy, actor_critic


def _apply_fixed_commands(env, fix_commands):
    """Override velocity commands for fixed-command evaluation.

    Sets vx, vy directly. For turning (heading_offset), continuously
    sets target heading ahead of current heading to produce constant yaw.
    For non-turning, heading_command mode auto-maintains initial heading.
    """
    env.commands[:, 0] = fix_commands["vx"]
    env.commands[:, 1] = fix_commands["vy"]
    if "heading_offset" in fix_commands:
        forward = quat_apply(env.base_quat, env.forward_vec)
        cur_heading = torch.atan2(forward[:, 1], forward[:, 0])
        env.commands[:, 3] = cur_heading + fix_commands["heading_offset"]


# ============================================================
# Core Evaluation Function
# ============================================================

def evaluate_scenario(env, policy, wind_level, num_episodes=50, max_steps=1000,
                      fix_direction=None, mode_overrides=None, ou_overrides=None,
                      ood_pattern=None, ood_peak_speed=10.0, actor_critic=None,
                      fix_commands=None):
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
        ood_pattern: "step", "periodic", or "reversal" for out-of-distribution wind
        ood_peak_speed: peak wind speed for OOD patterns (m/s), scales with test level
        actor_critic: ActorCriticRecurrent module for LSTM hidden state reset
        fix_commands: dict with vx, vy [, heading_offset] to override velocity commands

    Returns:
        dict with evaluation metrics including per-component reward breakdown
    """
    # Derived OOD speeds from peak
    ood_mid_speed = ood_peak_speed / 2.0
    ood_amp_speed = ood_peak_speed / 2.0
    device = env.device
    num_envs = env.num_envs
    has_wind = hasattr(env, 'wind_model')
    wind_model = env.wind_model if has_wind else None

    # Force wind level and freeze curriculum
    if has_wind:
        env.wind_curriculum_level[:] = wind_level
        saved_upgrade_window = env.cfg.wind.upgrade_window
        env.cfg.wind.upgrade_window = 999999

        # Save original config for restoration
        saved_ou_randomize = env.cfg.wind.ou_randomize
        saved_disable_ou = wind_model.disable_ou
        saved_disable_gusts = wind_model.disable_gusts

        # --- Apply mode overrides via wind model flags (clean substep-level disable) ---
        if mode_overrides:
            wind_model.disable_ou = (
                not mode_overrides.get("ou_speed", True)
                and not mode_overrides.get("ou_dir", True)
            )
            wind_model.disable_gusts = not mode_overrides.get("gusts", True)

        # --- Apply OU overrides (disable randomization, set fixed params) ---
        if ou_overrides:
            env.cfg.wind.ou_randomize = False

    # Reset all envs
    env.reset_buf[:] = 1
    env_ids = torch.arange(num_envs, device=device)
    env.reset_idx(env_ids)

    # Apply fixed direction if specified
    if has_wind and fix_direction is not None:
        wind_model.base_angle[:] = fix_direction
        wind_model.base_direction[:, 0] = math.cos(fix_direction)
        wind_model.base_direction[:, 1] = math.sin(fix_direction)
        wind_model.base_direction[:, 2] = 0.0
        wind_model.effective_direction[:] = wind_model.base_direction[:]

    # Apply fixed base_speed override (for B5_pure_gusts)
    if has_wind and mode_overrides and "base_speed_override" in mode_overrides:
        wind_model.base_speed[:] = mode_overrides["base_speed_override"]

    # Apply OU param overrides
    if has_wind and ou_overrides:
        wind_model.env_ou_theta[:] = ou_overrides["theta"]
        wind_model.env_ou_sigma[:] = ou_overrides["sigma"]
        wind_model.env_ou_theta_dir[:] = ou_overrides["theta_dir"]
        wind_model.env_ou_sigma_dir[:] = ou_overrides["sigma_dir"]

    obs = env.get_observations()

    # --- Apply fixed commands: disable resampling + override ---
    saved_resample = None
    if fix_commands is not None:
        saved_resample = env._resample_commands
        env._resample_commands = lambda env_ids: None  # no-op
        _apply_fixed_commands(env, fix_commands)
        env.compute_observations()
        clip_obs = env.cfg.normalization.clip_observations
        obs = torch.clip(env.obs_buf, -clip_obs, clip_obs)

    # Initialize LSTM hidden states (first forward pass) then reset
    if actor_critic is not None:
        with torch.no_grad():
            policy(obs.detach())
        actor_critic.memory_a.reset(torch.ones(num_envs, dtype=torch.bool, device=device))

    # Accumulators
    episode_lengths = []
    episode_rewards = []
    episode_tracking_errors = []
    episode_wind_forces = []
    episodes_collected = 0

    # Per-component reward tracking (from env.extras["episode"])
    # Each entry is (snapshot_dict, weight) where weight = number of done envs
    # in that step. extras["episode"]["rew_*"] is already batch-averaged, so we
    # need weighted aggregation to avoid over-counting small batches.
    rew_component_samples = []

    ep_len = torch.zeros(num_envs, device=device)
    ep_rew = torch.zeros(num_envs, device=device)
    ep_track_err = torch.zeros(num_envs, device=device)
    ep_wind_force = torch.zeros(num_envs, device=device)

    # Per-env episode step counter for OOD patterns (reset with each episode)
    ep_step = torch.zeros(num_envs, device=device)
    dt_control = env.cfg.sim.dt * env.cfg.control.decimation  # control dt

    step = 0
    while episodes_collected < num_episodes:
        # --- OOD pattern: override wind model state using per-env episode time ---
        # Wind speeds scale to the test level's curriculum max speed
        if has_wind and ood_pattern == "step":
            t = ep_step * dt_control
            wind_model.base_speed = torch.where(
                t < 5.0,
                torch.zeros_like(t),
                torch.full_like(t, ood_peak_speed)
            )
        elif has_wind and ood_pattern == "periodic":
            t = ep_step * dt_control
            wind_model.base_speed = ood_mid_speed + ood_amp_speed * torch.sin(2 * math.pi / 4.0 * t)
        elif has_wind and ood_pattern == "reversal":
            # Direction flips 180 deg at t=5s, speed stays constant
            t = ep_step * dt_control
            angle = torch.where(t < 5.0, torch.zeros_like(t),
                                torch.full_like(t, math.pi))
            wind_model.base_speed[:] = ood_peak_speed
            wind_model.base_angle[:] = angle
            wind_model.base_direction[:, 0] = torch.cos(angle)
            wind_model.base_direction[:, 1] = torch.sin(angle)
            wind_model.base_direction[:, 2] = 0.0

        with torch.no_grad():
            actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())

        ep_step += 1

        # --- Override commands after step (resampling is no-op, but
        #     heading_command mode recomputed yaw inside step) ---
        if fix_commands is not None:
            _apply_fixed_commands(env, fix_commands)
            env.compute_observations()
            clip_obs = env.cfg.normalization.clip_observations
            obs = torch.clip(env.obs_buf, -clip_obs, clip_obs)

        # Reset LSTM hidden states for terminated envs
        if actor_critic is not None and dones.any():
            actor_critic.memory_a.reset(dones)

        # Per-step metrics
        track_err = torch.sum(
            torch.square(env.commands[:, :2] - env.base_lin_vel[:, :2]), dim=1
        ).sqrt()
        wind_mag = torch.norm(wind_model.wind_force, dim=1) if has_wind else torch.zeros(num_envs, device=device)

        ep_len += 1
        ep_rew += rews
        ep_track_err += track_err
        ep_wind_force += wind_mag

        # Collect finished episodes
        done_ids = dones.nonzero(as_tuple=False).flatten()
        if len(done_ids) > 0:
            # Capture per-component rewards from extras (set in env.reset_idx)
            # extras["episode"]["rew_*"] is batch-averaged over done envs,
            # so store once with weight = batch size for correct aggregation.
            if "episode" in env.extras:
                snapshot = {}
                for key, val in env.extras["episode"].items():
                    if key.startswith("rew_"):
                        snapshot[key] = val.item() if isinstance(val, torch.Tensor) else val
                if snapshot:
                    rew_component_samples.append((snapshot, len(done_ids)))

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
                ep_step[i] = 0

                if episodes_collected >= num_episodes:
                    break

            # Re-apply overrides after env resets
            if has_wind and fix_direction is not None:
                for idx in done_ids:
                    i = idx.item()
                    wind_model.base_angle[i] = fix_direction
                    wind_model.base_direction[i, 0] = math.cos(fix_direction)
                    wind_model.base_direction[i, 1] = math.sin(fix_direction)
                    wind_model.base_direction[i, 2] = 0.0

            if has_wind and ou_overrides:
                for idx in done_ids:
                    i = idx.item()
                    wind_model.env_ou_theta[i] = ou_overrides["theta"]
                    wind_model.env_ou_sigma[i] = ou_overrides["sigma"]
                    wind_model.env_ou_theta_dir[i] = ou_overrides["theta_dir"]
                    wind_model.env_ou_sigma_dir[i] = ou_overrides["sigma_dir"]

            if has_wind and mode_overrides and "base_speed_override" in mode_overrides:
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
    if has_wind:
        env.cfg.wind.upgrade_window = saved_upgrade_window
        env.cfg.wind.ou_randomize = saved_ou_randomize
        wind_model.disable_ou = saved_disable_ou
        wind_model.disable_gusts = saved_disable_gusts
    if saved_resample is not None:
        env._resample_commands = saved_resample

    episode_lengths = np.array(episode_lengths[:num_episodes])
    episode_rewards = np.array(episode_rewards[:num_episodes])
    episode_tracking_errors = np.array(episode_tracking_errors[:num_episodes])
    episode_wind_forces = np.array(episode_wind_forces[:num_episodes])

    # Survival rate with binomial standard error
    n = len(episode_lengths)
    survival_rate = float(np.mean(episode_lengths >= max_steps))
    survival_se = float(np.sqrt(survival_rate * (1 - survival_rate) / max(n, 1)))

    # Aggregate reward components (weighted by batch size)
    reward_components = {}
    if rew_component_samples:
        all_keys = set()
        for s, w in rew_component_samples:
            all_keys.update(s.keys())
        total_weight = sum(w for _, w in rew_component_samples)
        for key in sorted(all_keys):
            weighted_sum = sum(s.get(key, 0.0) * w for s, w in rew_component_samples)
            reward_components[key] = float(weighted_sum / total_weight)

    return {
        "survival_rate": survival_rate,
        "survival_se": survival_se,
        "mean_episode_length": float(np.mean(episode_lengths)),
        "std_episode_length": float(np.std(episode_lengths)),
        "mean_reward_per_step": float(np.mean(episode_rewards)),
        "std_reward_per_step": float(np.std(episode_rewards)),
        "mean_tracking_error": float(np.mean(episode_tracking_errors)),
        "std_tracking_error": float(np.std(episode_tracking_errors)),
        "mean_wind_force_N": float(np.mean(episode_wind_forces)),
        "reward_components": reward_components,
    }


# ============================================================
# Test Suites
# ============================================================

def run_suite_levels(env, policy, args, actor_critic=None):
    """Suite A: Standard level sweep (full model)."""
    print_header("Suite A: Wind Level Sweep (Full Model)")
    level_names = ["No wind", "Light (1-3)", "Light-Med (2-5)",
                   "Medium (4-8)", "Strong (7-12)", "Extreme (10-18)"]
    results = {}
    for level in range(6):
        r = evaluate_scenario(env, policy, level,
                              num_episodes=args.num_episodes, max_steps=args.max_steps,
                              actor_critic=actor_critic)
        results[f"A_level{level}"] = r
        print_row(f"L{level}", level_names[level], r)
    return results


def run_suite_modes(env, policy, args, test_level=3, actor_critic=None):
    """Suite B: Wind mode decomposition at a fixed level."""
    print_header(f"Suite B: Wind Modes (Level {test_level})")
    results = {}
    for key, mode in WIND_MODES.items():
        overrides = {k: mode[k] for k in ["ou_speed", "ou_dir", "gusts"]}
        if "base_speed_override" in mode:
            overrides["base_speed_override"] = mode["base_speed_override"]
        r = evaluate_scenario(env, policy, test_level,
                              num_episodes=args.num_episodes, max_steps=args.max_steps,
                              mode_overrides=overrides, actor_critic=actor_critic)
        results[key] = r
        print_row(key, mode["label"], r)
    return results


def run_suite_directions(env, policy, args, test_level=3, actor_critic=None):
    """Suite C: Fixed wind directions at a fixed level (steady wind only)."""
    print_header(f"Suite C: Wind Directions (Level {test_level}, Steady)")
    results = {}
    for key, d in WIND_DIRECTIONS.items():
        r = evaluate_scenario(env, policy, test_level,
                              num_episodes=args.num_episodes, max_steps=args.max_steps,
                              fix_direction=d["angle"],
                              mode_overrides={"ou_speed": False, "ou_dir": False, "gusts": False},
                              actor_critic=actor_critic)
        results[key] = r
        print_row(key, d["label"], r)
    return results


def run_suite_ou_extremes(env, policy, args, test_level=3, actor_critic=None):
    """Suite D: OU parameter extremes at a fixed level."""
    print_header(f"Suite D: OU Parameter Extremes (Level {test_level})")
    results = {}
    for key, params in OU_EXTREMES.items():
        ou_overrides = {k: params[k] for k in ["theta", "sigma", "theta_dir", "sigma_dir"]}
        r = evaluate_scenario(env, policy, test_level,
                              num_episodes=args.num_episodes, max_steps=args.max_steps,
                              ou_overrides=ou_overrides, actor_critic=actor_critic)
        results[key] = r
        print_row(key, params["label"], r)
    return results


def run_suite_ood(env, policy, args, test_level=3, actor_critic=None):
    """Suite E: Out-of-distribution wind patterns scaled to test level."""
    print_header(f"Suite E: Out-of-Distribution Patterns (Level {test_level})")

    # Scale OOD peak speed to the test level's curriculum max
    curriculum_levels = env.cfg.wind.curriculum_levels
    if test_level < len(curriculum_levels):
        peak_speed = curriculum_levels[test_level][1]  # max speed for this level
    else:
        peak_speed = 18.0  # fallback to L5 max
    # Ensure minimum of 10 m/s so low levels still test something meaningful
    peak_speed = max(peak_speed, 10.0)

    results = {}
    for key, p in OOD_PATTERNS.items():
        pattern_name = key.split("_")[1]  # "step" or "periodic"
        # Generate dynamic label
        if pattern_name == "step" or pattern_name == "reversal":
            label = p["label_fmt"].format(peak=peak_speed)
        else:
            label = p["label_fmt"].format(lo=0, hi=peak_speed)
        r = evaluate_scenario(env, policy, test_level,
                              num_episodes=args.num_episodes, max_steps=args.max_steps,
                              fix_direction=0.0,
                              mode_overrides={"ou_speed": False, "ou_dir": False, "gusts": False},
                              ood_pattern=pattern_name, ood_peak_speed=peak_speed,
                              actor_critic=actor_critic)
        results[key] = r
        print_row(key, label, r)
    return results



def run_suite_commands(env, policy, args, test_level=3, actor_critic=None):
    """Suite F: Command variations at a fixed wind level (full wind model)."""
    print_header(f"Suite F: Command Variations (Level {test_level})")
    results = {}
    for key, scenario in COMMAND_SCENARIOS.items():
        fix_cmds = {k: scenario[k] for k in scenario
                    if k not in ("label", "wind_angle")}
        fix_dir = scenario.get("wind_angle", None)
        r = evaluate_scenario(env, policy, test_level,
                              num_episodes=args.num_episodes, max_steps=args.max_steps,
                              fix_commands=fix_cmds, fix_direction=fix_dir,
                              actor_critic=actor_critic)
        results[key] = r
        print_row(key, scenario["label"], r)
    return results


# ============================================================
# Output Formatting
# ============================================================

def print_header(title):
    print(f"\n{'='*105}")
    print(f"  {title}")
    print(f"{'='*105}")
    print(f"{'ID':>15} {'Description':>25} {'Survival%':>12} {'EpLen':>10} "
          f"{'Reward/s':>9} {'TrackErr':>12} {'Wind(N)':>8}")
    print(f"{'-'*105}")


def print_row(test_id, label, r):
    surv_str = f"{r['survival_rate']*100:.1f}±{r['survival_se']*100:.1f}%"
    track_str = f"{r['mean_tracking_error']:.4f}±{r['std_tracking_error']:.3f}"
    print(f"{test_id:>15} {label:>25} {surv_str:>12} "
          f"{r['mean_episode_length']:>6.0f}±{r['std_episode_length']:<3.0f} "
          f"{r['mean_reward_per_step']:>9.4f} {track_str:>12} {r['mean_wind_force_N']:>8.2f}")


def print_reward_breakdown(results, top_n=8):
    """Print top reward components across all scenarios in a suite."""
    # Aggregate all reward components across scenarios
    all_components = {}
    for scenario_key, r in results.items():
        for comp_key, val in r.get("reward_components", {}).items():
            if comp_key not in all_components:
                all_components[comp_key] = {}
            all_components[comp_key][scenario_key] = val

    if not all_components:
        return

    # Sort by absolute magnitude of the mean across scenarios
    comp_means = {}
    for comp_key, scenario_vals in all_components.items():
        comp_means[comp_key] = np.mean(list(scenario_vals.values()))

    sorted_comps = sorted(comp_means.items(), key=lambda x: abs(x[1]), reverse=True)

    print(f"\n  Reward Component Breakdown (top {top_n}):")
    print(f"  {'Component':>25}", end="")
    scenario_keys = list(results.keys())
    for sk in scenario_keys:
        short = sk[-8:] if len(sk) > 8 else sk
        print(f" {short:>10}", end="")
    print()
    print(f"  {'-'*25}", end="")
    for _ in scenario_keys:
        print(f" {'-'*10}", end="")
    print()

    for comp_key, mean_val in sorted_comps[:top_n]:
        name = comp_key.replace("rew_", "")
        print(f"  {name:>25}", end="")
        for sk in scenario_keys:
            val = all_components[comp_key].get(sk, 0.0)
            print(f" {val:>10.4f}", end="")
        print()


def save_results_json(all_results, output_path, metadata=None):
    """Save all results to a JSON file."""
    output = {
        "metadata": metadata or {},
        "results": all_results,
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")


# ============================================================
# Main
# ============================================================

def main():
    # Pre-extract eval-specific args before get_args() (which uses strict parsing)
    eval_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    eval_parser.add_argument("--suite", type=str, default="all",
                             choices=["all", "levels", "modes", "directions", "ou", "ood", "commands"],
                             help="Which test suite to run")
    eval_parser.add_argument("--test_level", type=str, default="3",
                             help="Wind level(s) for B/C/D/E: single int, comma-separated, or 'all' for 3,4,5")
    eval_parser.add_argument("--num_episodes", type=int, default=50)
    eval_parser.add_argument("--max_steps", type=int, default=1000)
    eval_parser.add_argument("--load_run2", type=str, default=None,
                             help="Second policy for A/B comparison")
    eval_parser.add_argument("--output", type=str, default=None,
                             help="Output JSON file path for results")
    eval_parser.add_argument("--seed", type=int, default=None,
                             help="Random seed for reproducibility")
    eval_parser.add_argument("--model_experiment", type=str, default=None,
                             help="Override experiment directory for model loading "
                                  "(e.g. 'g1_wind_baseline' to load baseline model into g1_wind env)")
    eval_args, remaining_argv = eval_parser.parse_known_args()

    # Patch sys.argv so get_args() only sees arguments it recognizes
    import sys
    sys.argv = [sys.argv[0]] + remaining_argv

    args = get_args()

    # --- Set random seed for reproducibility ---
    if eval_args.seed is not None:
        torch.manual_seed(eval_args.seed)
        np.random.seed(eval_args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(eval_args.seed)

    # --- Create environment ---
    # Always use g1_wind env config (with wind) for evaluation so all policies
    # are tested under identical wind conditions. The --task arg is only used
    # to determine where to load the model from (experiment_name in train_cfg).
    task_name = args.task if args.task else "g1_wind"
    env_cfg, _ = task_registry.get_cfgs(name="g1_wind")
    _, train_cfg = task_registry.get_cfgs(name=task_name)
    # Auto-set model_experiment when task differs from g1_wind
    if not eval_args.model_experiment and task_name != "g1_wind":
        eval_args.model_experiment = train_cfg.runner.experiment_name
        print(f"  [NOTE] Loading model from experiment: {eval_args.model_experiment}")
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 64)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_action_delay = False
    env_cfg.domain_rand.randomize_pd_gains = False
    env_cfg.domain_rand.randomize_motor_strength = False
    env_cfg.env.test = True
    env_cfg.env.episode_length_s = eval_args.max_steps * env_cfg.sim.dt * env_cfg.control.decimation

    env, _ = task_registry.make_env(name="g1_wind", args=args, env_cfg=env_cfg)
    has_wind = True  # g1_wind env always has wind
    print(f"Environment created: g1_wind, {env.num_envs} envs, device={env.device}")

    # --- Load policy ---
    load_run = args.load_run if args.load_run else "Mar10_18-22-48_"
    checkpoint = args.checkpoint if args.checkpoint is not None else -1
    model_experiment = eval_args.model_experiment
    policy, actor_critic = load_policy(env, train_cfg, load_run, checkpoint, args.rl_device,
                                       model_experiment=model_experiment)
    print(f"\nEvaluating policy: {load_run}" +
          (f" (from {model_experiment})" if model_experiment else ""))

    # --- Parse test levels ---
    suite = eval_args.suite
    tl_str = eval_args.test_level
    if tl_str == "all":
        test_levels = [3, 4, 5]
    else:
        test_levels = [int(x) for x in tl_str.split(",")]

    all_results = {}

    # Suite A: always a full level sweep (ignores test_level)
    if suite in ("all", "levels"):
        results = run_suite_levels(env, policy, eval_args, actor_critic=actor_critic)
        all_results.update(results)
        print_reward_breakdown(results)

    # Suites B-E: run at each requested test level
    for tl in test_levels:
        if suite in ("all", "modes"):
            results = run_suite_modes(env, policy, eval_args, tl, actor_critic=actor_critic)
            # Prefix keys with level for uniqueness
            results = {f"{k}_L{tl}": v for k, v in results.items()}
            all_results.update(results)
            print_reward_breakdown(results)

        if suite in ("all", "directions"):
            results = run_suite_directions(env, policy, eval_args, tl, actor_critic=actor_critic)
            results = {f"{k}_L{tl}": v for k, v in results.items()}
            all_results.update(results)
            print_reward_breakdown(results)

        if suite in ("all", "ou"):
            results = run_suite_ou_extremes(env, policy, eval_args, tl, actor_critic=actor_critic)
            results = {f"{k}_L{tl}": v for k, v in results.items()}
            all_results.update(results)
            print_reward_breakdown(results)

        if suite in ("all", "ood"):
            results = run_suite_ood(env, policy, eval_args, tl, actor_critic=actor_critic)
            results = {f"{k}_L{tl}": v for k, v in results.items()}
            all_results.update(results)
            print_reward_breakdown(results)

        if suite in ("all", "commands"):
            results = run_suite_commands(env, policy, eval_args, tl, actor_critic=actor_critic)
            results = {f"{k}_L{tl}": v for k, v in results.items()}
            all_results.update(results)
            print_reward_breakdown(results)

    print(f"\n{'='*105}")
    print(f"  Completed {len(all_results)} test scenarios for {load_run}")
    print(f"{'='*105}")

    # --- Optional: Compare with second policy ---
    all_results2 = {}
    if eval_args.load_run2:
        print(f"\n\n{'#'*105}")
        print(f"  Comparing with second policy: {eval_args.load_run2}")
        print(f"{'#'*105}")
        policy2, actor_critic2 = load_policy(env, train_cfg, eval_args.load_run2, checkpoint, args.rl_device)

        if suite in ("all", "levels"):
            all_results2.update(run_suite_levels(env, policy2, eval_args, actor_critic=actor_critic2))
        for tl in test_levels:
            if suite in ("all", "modes"):
                r = run_suite_modes(env, policy2, eval_args, tl, actor_critic=actor_critic2)
                all_results2.update({f"{k}_L{tl}": v for k, v in r.items()})
            if suite in ("all", "directions"):
                r = run_suite_directions(env, policy2, eval_args, tl, actor_critic=actor_critic2)
                all_results2.update({f"{k}_L{tl}": v for k, v in r.items()})
            if suite in ("all", "ou"):
                r = run_suite_ou_extremes(env, policy2, eval_args, tl, actor_critic=actor_critic2)
                all_results2.update({f"{k}_L{tl}": v for k, v in r.items()})
            if suite in ("all", "ood"):
                r = run_suite_ood(env, policy2, eval_args, tl, actor_critic=actor_critic2)
                all_results2.update({f"{k}_L{tl}": v for k, v in r.items()})
            if suite in ("all", "commands"):
                r = run_suite_commands(env, policy2, eval_args, tl, actor_critic=actor_critic2)
                all_results2.update({f"{k}_L{tl}": v for k, v in r.items()})

        # Print comparison
        print(f"\n{'='*105}")
        print(f"  Comparison: {load_run} (A) vs {eval_args.load_run2} (B)")
        print(f"{'='*105}")
        print(f"{'Scenario':>25} {'Surv A':>10} {'Surv B':>10} {'Delta':>10} "
              f"{'Track A':>10} {'Track B':>10}")
        print(f"{'-'*85}")
        for key in sorted(all_results.keys()):
            if key in all_results2:
                sa = all_results[key]["survival_rate"] * 100
                sb = all_results2[key]["survival_rate"] * 100
                delta = sa - sb
                ta = all_results[key]["mean_tracking_error"]
                tb = all_results2[key]["mean_tracking_error"]
                sign = "+" if delta >= 0 else ""
                print(f"{key:>25} {sa:>9.1f}% {sb:>9.1f}% {sign}{delta:>8.1f}% "
                      f"{ta:>10.4f} {tb:>10.4f}")

    # --- Save results to JSON ---
    if eval_args.output:
        metadata = {
            "policy_a": load_run,
            "policy_b": eval_args.load_run2,
            "suite": suite,
            "test_levels": test_levels,
            "num_episodes": eval_args.num_episodes,
            "max_steps": eval_args.max_steps,
            "num_envs": env.num_envs,
        }
        json_results = {"policy_a": all_results}
        if all_results2:
            json_results["policy_b"] = all_results2
        save_results_json(json_results, eval_args.output, metadata=metadata)


if __name__ == "__main__":
    main()
