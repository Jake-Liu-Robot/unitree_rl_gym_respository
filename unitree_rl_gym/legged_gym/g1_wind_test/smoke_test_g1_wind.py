"""Smoke test: create g1_wind environment, run 10 steps, print wind force."""

import isaacgym  # must be imported before torch
import torch
from legged_gym.envs import *
from legged_gym.utils import task_registry

def main():
    # --- 1. Create environment ---
    env_cfg, train_cfg = task_registry.get_cfgs(name="g1_wind")

    # Use minimal envs for fast testing
    env_cfg.env.num_envs = 4
    env_cfg.terrain.num_rows = 2
    env_cfg.terrain.num_cols = 2
    env_cfg.terrain.mesh_type = "plane"  # flat terrain, faster creation

    # Force curriculum level 3 (medium wind) so we can see non-zero forces
    env_cfg.wind.curriculum_start_level = 3

    env, _ = task_registry.make_env(name="g1_wind", args=None, env_cfg=env_cfg)
    print(f"[OK] Environment created: {env.num_envs} envs, device={env.device}")
    print(f"[OK] Obs shape: {env.obs_buf.shape}, Privileged obs shape: {env.privileged_obs_buf.shape}")

    # --- 2. Verify key attributes (available before first step) ---
    print(f"[OK] torso_body_idx (pelvis): {env.torso_body_idx}")
    print(f"[OK] feet_indices: {env.feet_indices}")
    print(f"[OK] wind_model enabled: {env.cfg.wind.enable}")
    print(f"[OK] wind_curriculum_level: {env.wind_curriculum_level}")

    # --- 3. Run 10 steps, print wind force each step ---
    # Note: self.phase is created in _post_physics_step_callback on first step()
    print("\n--- Running 10 steps ---")
    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)

    for step in range(10):
        obs, priv_obs, rew, done, info = env.step(actions)

        # Verify phase exists after first step
        if step == 0:
            print(f"[OK] phase shape (after 1st step): {env.phase.shape}")

        # Get current wind force from model
        wind_force = env.wind_model.wind_force  # [num_envs, 3]
        wind_vel = env.wind_model.get_wind_velocity()

        print(f"Step {step:2d} | "
              f"wind_force[0]=[{wind_force[0, 0]:+7.2f}, {wind_force[0, 1]:+7.2f}, {wind_force[0, 2]:+7.2f}] N | "
              f"wind_vel[0]=[{wind_vel[0, 0]:+6.2f}, {wind_vel[0, 1]:+6.2f}, {wind_vel[0, 2]:+6.2f}] m/s | "
              f"reward={rew[0]:.4f}")

    print("\n[OK] Smoke test PASSED!")


if __name__ == "__main__":
    main()
