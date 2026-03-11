"""
Comprehensive wind robustness evaluation for g1_wind policies.

Tests policy across multiple dimensions:
  A: Wind speed levels (0-5)
  B: Wind modes (steady, turbulent, gusts-only, full model)
  C: Wind directions (front, side, back, diagonal, random)
  D: OU parameter extremes (calm, turbulent, locked heading, erratic heading)
  E: Out-of-distribution wind patterns (step change, periodic)

Usage:
    # Run all suites (default level 3 for B-E)
    python eval_wind_robustness.py --task g1_wind --load_run Mar10_18-22-48_ --headless

    # Run specific suite at level 5
    python eval_wind_robustness.py --task g1_wind --load_run Mar10_18-22-48_ --suite levels --headless

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
OOD_PATTERNS = {
    "E1_step":     {"label": "Step change (0→10 m/s at t=5s)"},
    "E2_periodic": {"label": "Periodic (A·sin(ωt), T=4s)"},
}


def load_policy(env, train_cfg, load_run, checkpoint, device):
    """Load a trained policy from a specific run.

    Returns:
        (policy_fn, actor_critic): inference policy callable and the
        ActorCriticRecurrent module (needed for LSTM hidden state reset).
    """
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
    policy = runner.get_inference_policy(device=device)
    actor_critic = runner.alg.actor_critic
    return policy, actor_critic


# ============================================================
# Core Evaluation Function
# ============================================================

def evaluate_scenario(env, policy, wind_level, num_episodes=50, max_steps=1000,
                      fix_direction=None, mode_overrides=None, ou_overrides=None,
                      ood_pattern=None, actor_critic=None):
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
        actor_critic: ActorCriticRecurrent module for LSTM hidden state reset

    Returns:
        dict with evaluation metrics including per-component reward breakdown
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
    saved_disable_ou = wind_model.disable_ou
    saved_disable_gusts = wind_model.disable_gusts

    # --- Apply mode overrides via wind model flags (clean substep-level disable) ---
    # Note: disable_ou controls both speed and direction OU together.
    # All current test scenarios use them as a pair (both on or both off).
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
        if ood_pattern == "step":
            t = ep_step * dt_control
            wind_model.base_speed = torch.where(
                t < 5.0,
                torch.zeros_like(t),
                torch.full_like(t, 10.0)
            )
        elif ood_pattern == "periodic":
            t = ep_step * dt_control
            wind_model.base_speed = 5.0 + 5.0 * torch.sin(2 * math.pi / 4.0 * t)

        with torch.no_grad():
            actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())

        ep_step += 1

        # Reset LSTM hidden states for terminated envs
        if actor_critic is not None and dones.any():
            actor_critic.memory_a.reset(dones)

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
            # Capture per-component rewards from extras (set in env.reset_idx)
            if "episode" in env.extras:
                snapshot = {}
                for key, val in env.extras["episode"].items():
                    if key.startswith("rew_"):
                        snapshot[key] = val.item() if isinstance(val, torch.Tensor) else val
                if snapshot:
                    n_done = len(done_ids)
                    for _ in range(n_done):
                        rew_component_samples.append(snapshot)

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
    wind_model.disable_ou = saved_disable_ou
    wind_model.disable_gusts = saved_disable_gusts

    episode_lengths = np.array(episode_lengths[:num_episodes])
    episode_rewards = np.array(episode_rewards[:num_episodes])
    episode_tracking_errors = np.array(episode_tracking_errors[:num_episodes])
    episode_wind_forces = np.array(episode_wind_forces[:num_episodes])

    # Survival rate with binomial standard error
    n = len(episode_lengths)
    survival_rate = float(np.mean(episode_lengths >= max_steps))
    survival_se = float(np.sqrt(survival_rate * (1 - survival_rate) / max(n, 1)))

    # Aggregate reward components
    reward_components = {}
    if rew_component_samples:
        all_keys = set()
        for s in rew_component_samples:
            all_keys.update(s.keys())
        for key in sorted(all_keys):
            vals = [s.get(key, 0.0) for s in rew_component_samples]
            reward_components[key] = float(np.mean(vals))

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
    """Suite E: Out-of-distribution wind patterns."""
    print_header(f"Suite E: Out-of-Distribution Patterns (Level {test_level})")
    results = {}
    for key, p in OOD_PATTERNS.items():
        pattern_name = key.split("_")[1]  # "step" or "periodic"
        r = evaluate_scenario(env, policy, test_level,
                              num_episodes=args.num_episodes, max_steps=args.max_steps,
                              fix_direction=0.0,
                              mode_overrides={"ou_speed": False, "ou_dir": False, "gusts": False},
                              ood_pattern=pattern_name, actor_critic=actor_critic)
        results[key] = r
        print_row(key, p["label"], r)
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
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")


# ============================================================
# Main
# ============================================================

def main():
    args = get_args()

    eval_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    eval_parser.add_argument("--suite", type=str, default="all",
                             choices=["all", "levels", "modes", "directions", "ou", "ood"],
                             help="Which test suite to run")
    eval_parser.add_argument("--test_level", type=int, default=3,
                             help="Wind level for single-level suites (B/C/D/E)")
    eval_parser.add_argument("--num_episodes", type=int, default=50)
    eval_parser.add_argument("--max_steps", type=int, default=1000)
    eval_parser.add_argument("--load_run2", type=str, default=None,
                             help="Second policy for A/B comparison")
    eval_parser.add_argument("--output", type=str, default=None,
                             help="Output JSON file path for results")
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
    env_cfg.domain_rand.randomize_action_delay = False
    env_cfg.domain_rand.randomize_pd_gains = False
    env_cfg.domain_rand.randomize_motor_strength = False
    env_cfg.env.test = True
    env_cfg.env.episode_length_s = eval_args.max_steps * env_cfg.sim.dt * env_cfg.control.decimation

    env, _ = task_registry.make_env(name="g1_wind", args=args, env_cfg=env_cfg)
    print(f"Environment created: {env.num_envs} envs, device={env.device}")

    # --- Load policy ---
    load_run = args.load_run if args.load_run else "Mar10_18-22-48_"
    checkpoint = args.checkpoint if args.checkpoint is not None else -1
    policy, actor_critic = load_policy(env, train_cfg, load_run, checkpoint, args.rl_device)
    print(f"\nEvaluating policy: {load_run}")

    # --- Run selected suites ---
    suite = eval_args.suite
    tl = eval_args.test_level
    all_results = {}

    if suite in ("all", "levels"):
        results = run_suite_levels(env, policy, eval_args, actor_critic=actor_critic)
        all_results.update(results)
        print_reward_breakdown(results)

    if suite in ("all", "modes"):
        results = run_suite_modes(env, policy, eval_args, tl, actor_critic=actor_critic)
        all_results.update(results)
        print_reward_breakdown(results)

    if suite in ("all", "directions"):
        results = run_suite_directions(env, policy, eval_args, tl, actor_critic=actor_critic)
        all_results.update(results)
        print_reward_breakdown(results)

    if suite in ("all", "ou"):
        results = run_suite_ou_extremes(env, policy, eval_args, tl, actor_critic=actor_critic)
        all_results.update(results)
        print_reward_breakdown(results)

    if suite in ("all", "ood"):
        results = run_suite_ood(env, policy, eval_args, tl, actor_critic=actor_critic)
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
        if suite in ("all", "modes"):
            all_results2.update(run_suite_modes(env, policy2, eval_args, tl, actor_critic=actor_critic2))
        if suite in ("all", "directions"):
            all_results2.update(run_suite_directions(env, policy2, eval_args, tl, actor_critic=actor_critic2))
        if suite in ("all", "ou"):
            all_results2.update(run_suite_ou_extremes(env, policy2, eval_args, tl, actor_critic=actor_critic2))
        if suite in ("all", "ood"):
            all_results2.update(run_suite_ood(env, policy2, eval_args, tl, actor_critic=actor_critic2))

        # Print comparison
        print(f"\n{'='*105}")
        print(f"  Comparison: {load_run} (A) vs {eval_args.load_run2} (B)")
        print(f"{'='*105}")
        print(f"{'Scenario':>20} {'Surv A':>10} {'Surv B':>10} {'Delta':>10} "
              f"{'Track A':>10} {'Track B':>10}")
        print(f"{'-'*75}")
        for key in all_results:
            if key in all_results2:
                sa = all_results[key]["survival_rate"] * 100
                sb = all_results2[key]["survival_rate"] * 100
                delta = sa - sb
                ta = all_results[key]["mean_tracking_error"]
                tb = all_results2[key]["mean_tracking_error"]
                sign = "+" if delta >= 0 else ""
                print(f"{key:>20} {sa:>9.1f}% {sb:>9.1f}% {sign}{delta:>8.1f}% "
                      f"{ta:>10.4f} {tb:>10.4f}")

    # --- Save results to JSON ---
    if eval_args.output:
        metadata = {
            "policy_a": load_run,
            "policy_b": eval_args.load_run2,
            "suite": suite,
            "test_level": tl,
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
