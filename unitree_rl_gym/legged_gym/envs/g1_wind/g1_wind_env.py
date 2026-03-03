import numpy as np
import torch
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi

from legged_gym.envs.g1.g1_env import G1Robot
from legged_gym.envs.g1_wind.wind_model import WindModel


class G1WindRobot(G1Robot):
    """G1 humanoid robot with continuous wind disturbance.

    Extends G1Robot with:
    - 3-layer wind force model (base + OU + gusts) applied to torso
    - Wind-specific reward functions
    - Wind curriculum controller
    - Wind velocity in privileged observations
    """

    def _init_buffers(self):
        super()._init_buffers()

        # --- Find torso body index ---
        self.torso_body_idx = self.gym.find_actor_rigid_body_handle(
            self.envs[0], self.actor_handles[0], "pelvis"
        )

        # --- Wind model ---
        if self.cfg.wind.enable:
            self.wind_model = WindModel(self.num_envs, self.device, self.cfg.wind)

            # Per-env curriculum level
            # In test/play mode, use play_level to see wind at a specific difficulty
            start_level = self.cfg.wind.curriculum_start_level
            if hasattr(self.cfg.env, 'test') and self.cfg.env.test:
                if self.cfg.wind.play_level is not None:
                    start_level = self.cfg.wind.play_level
                else:
                    # Default to max level in play mode so we can see full wind
                    start_level = len(self.cfg.wind.curriculum_levels) - 1
            self.wind_curriculum_level = torch.full(
                (self.num_envs,), start_level,
                dtype=torch.long, device=self.device
            )

            # Force tensor for apply_rigid_body_force_tensors
            # Shape: [num_envs * num_bodies, 3]
            self.wind_force_tensor = torch.zeros(
                self.num_envs * self.num_bodies, 3,
                dtype=torch.float, device=self.device
            )

            # Precompute torso indices for all envs: env_i * num_bodies + torso_idx
            self.torso_indices = (
                torch.arange(self.num_envs, device=self.device) * self.num_bodies
                + self.torso_body_idx
            )

            # Curriculum evaluation: accumulate stats over a window of resets
            self.wind_reset_counter = 0
            self.wind_survival_acc = 0.0  # accumulated survival fraction
            self.wind_tracking_acc = 0.0  # accumulated tracking performance

            # Initialize wind for all envs so first episode has wind
            all_env_ids = torch.arange(self.num_envs, device=self.device)
            self.wind_model.reset_envs(all_env_ids, self.wind_curriculum_level)

    def step(self, actions):
        """Override step to apply wind forces before each physics substep."""
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        self.render()

        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(
                self.sim, gymtorch.unwrap_tensor(self.torques)
            )

            # --- Apply wind force BEFORE simulate ---
            if self.cfg.wind.enable:
                self._apply_wind_force()

            self.gym.simulate(self.sim)

            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)

        self.post_physics_step()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def _apply_wind_force(self):
        """Compute wind force and apply to torso via Isaac Gym API."""
        # Advance wind model by one physics substep
        physics_dt = self.sim_params.dt
        wind_force = self.wind_model.step(physics_dt)  # [num_envs, 3]

        # Fill force tensor (sparse: only torso body gets force)
        self.wind_force_tensor[:] = 0.0
        self.wind_force_tensor[self.torso_indices] = wind_force

        # Apply — must be before gym.simulate()
        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self.wind_force_tensor),
            None,  # no torque
            gymapi.ENV_SPACE
        )

    def reset_idx(self, env_ids):
        """Reset environments and wind state."""
        if len(env_ids) == 0:
            return

        # Update curriculum before reset
        if self.cfg.wind.enable:
            self._update_wind_curriculum(env_ids)

            # Reset wind model for these envs
            self.wind_model.reset_envs(env_ids, self.wind_curriculum_level)

        super().reset_idx(env_ids)

        # Always log curriculum info so all ep_infos have consistent keys
        if self.cfg.wind.enable:
            self.extras["episode"]["wind_curriculum_mean"] = self.wind_curriculum_level.float().mean().item()
            self.extras["episode"]["wind_curriculum_max"] = self.wind_curriculum_level.max().item()

    def _update_wind_curriculum(self, env_ids):
        """Accumulate stats over a window of resets, then decide level-up.

        Instead of upgrading immediately per-env, we accumulate the average
        survival and tracking performance across `upgrade_window` resets,
        then upgrade ALL envs by one level if the window average passes thresholds.
        This prevents premature advancement from lucky individual episodes.
        """
        n = len(env_ids)

        # Compute per-env survival fraction
        survival_frac = (
            self.episode_length_buf[env_ids].float() / self.max_episode_length
        )

        # Compute per-env tracking performance (normalized to [0, 1])
        tracking_perf = torch.zeros(n, device=self.device)
        if "tracking_lin_vel" in self.episode_sums:
            scale_with_dt = self.reward_scales.get("tracking_lin_vel", self.dt)
            tracking_perf = (
                self.episode_sums["tracking_lin_vel"][env_ids]
                / (self.episode_length_buf[env_ids].float().clamp(min=1) * scale_with_dt)
            )

        # Accumulate into window
        self.wind_survival_acc += survival_frac.mean().item() * n
        self.wind_tracking_acc += tracking_perf.mean().item() * n
        self.wind_reset_counter += n

        # Evaluate when window is full
        if self.wind_reset_counter >= self.cfg.wind.upgrade_window:
            avg_survival = self.wind_survival_acc / self.wind_reset_counter
            avg_tracking = self.wind_tracking_acc / self.wind_reset_counter

            if (avg_survival > self.cfg.wind.survival_threshold
                    and avg_tracking > self.cfg.wind.tracking_threshold):
                max_level = len(self.cfg.wind.curriculum_levels) - 1
                self.wind_curriculum_level[:] = torch.clamp(
                    self.wind_curriculum_level + 1, max=max_level
                )
            elif (avg_survival < self.cfg.wind.demotion_survival_threshold
                    and avg_tracking < self.cfg.wind.demotion_tracking_threshold):
                # Demote if performing very poorly at current level
                self.wind_curriculum_level[:] = torch.clamp(
                    self.wind_curriculum_level - 1, min=0
                )

            # Reset window
            self.wind_reset_counter = 0
            self.wind_survival_acc = 0.0
            self.wind_tracking_acc = 0.0

    def compute_observations(self):
        """Compute observations, adding wind velocity to privileged obs."""
        sin_phase = torch.sin(2 * np.pi * self.phase).unsqueeze(1)
        cos_phase = torch.cos(2 * np.pi * self.phase).unsqueeze(1)

        # Standard obs (same as G1 — 47 dims, no wind info for sim-to-real)
        self.obs_buf = torch.cat((
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            self.commands[:, :3] * self.commands_scale,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
            sin_phase,
            cos_phase
        ), dim=-1)

        # Privileged obs: includes base_lin_vel + wind velocity + wind force
        if self.cfg.wind.enable:
            wind_vel = self.wind_model.get_wind_velocity()   # [num_envs, 3]
            wind_force = self.wind_model.wind_force           # [num_envs, 3]
            # Normalize: wind_vel ~[0,18] m/s → /10, wind_force ~[0,500] N → /100
            self.privileged_obs_buf = torch.cat((
                self.base_lin_vel * self.obs_scales.lin_vel,
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                self.commands[:, :3] * self.commands_scale,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
                sin_phase,
                cos_phase,
                wind_vel * 0.1,        # 3 dims, normalized
                wind_force * 0.01,     # 3 dims, normalized
            ), dim=-1)
        else:
            self.privileged_obs_buf = torch.cat((
                self.base_lin_vel * self.obs_scales.lin_vel,
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                self.commands[:, :3] * self.commands_scale,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
                sin_phase,
                cos_phase,
            ), dim=-1)

        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    # ========== Wind-specific reward functions ==========

    def _reward_lean_compensation(self):
        """Reward leaning against wind force to maintain balance.

        Both vectors must be in the same frame. We transform wind direction
        from world frame to body frame so it matches projected_gravity.

        In body frame: projected_gravity points opposite to the body's "up".
        When the robot leans into the wind, the gravity vector's horizontal
        component aligns with the wind direction (in body frame).
        We reward negative dot product = leaning against wind.
        """
        if not self.cfg.wind.enable:
            return torch.zeros(self.num_envs, device=self.device)

        gravity_xy = self.projected_gravity[:, :2]  # body frame [num_envs, 2]

        # Transform wind direction from world frame to body frame
        wind_dir_body = quat_rotate_inverse(
            self.base_quat, self.wind_model.base_direction
        )  # [num_envs, 3]
        wind_dir_body_xy = wind_dir_body[:, :2]  # [num_envs, 2]

        # Negative dot product = leaning against the wind force direction
        lean_against_wind = -torch.sum(gravity_xy * wind_dir_body_xy, dim=1)

        # Only reward when wind is actually blowing
        wind_active = self.wind_model.base_speed > 0.5
        return lean_against_wind * wind_active.float()

    def _reward_sustained_walking(self):
        """Reward for remaining alive and walking (not falling).

        Simple per-step survival bonus, scaled by whether the robot
        is actually making progress (non-zero command tracking).
        """
        return torch.ones(self.num_envs, device=self.device)

    def _reward_contact_symmetry(self):
        """Reward symmetric left/right foot contact patterns.

        Under wind, the robot may develop asymmetric gait. This reward
        encourages balanced foot contact timing between left and right legs.
        """
        # Contact detection for left and right feet
        # feet_indices[0] = left_ankle_roll, feet_indices[1] = right_ankle_roll
        left_contact = self.contact_forces[:, self.feet_indices[0], 2] > 1.0
        right_contact = self.contact_forces[:, self.feet_indices[1], 2] > 1.0

        # Reward when both feet have similar contact state (both on or both off)
        symmetry = (left_contact == right_contact).float()
        return symmetry
