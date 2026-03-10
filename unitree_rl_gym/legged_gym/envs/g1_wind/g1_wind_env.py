import numpy as np
import torch
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi

from legged_gym.envs.g1.g1_env import G1Robot
from legged_gym.envs.g1_wind.wind_model import WindModel


class G1WindRobot(G1Robot):
    """G1 humanoid robot with physically accurate wind disturbance (v3).

    Extends G1Robot with:
    - 3-layer wind velocity model (base + OU + gusts)
    - Per-body aerodynamic force computation:
        P0: Relative velocity (v_wind - v_body) for accurate drag
        P1: Direction-dependent projected area (cross-flow principle)
        P2: Height-dependent wind speed (power law boundary layer)
        P3: Force at center of pressure via apply_rigid_body_force_at_pos_tensors
    - Per-curriculum-level force clamping
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

            # --- Aerodynamic force coefficient: 0.5 * ρ * Cd (without area) ---
            self.aero_force_coeff = (
                0.5 * self.cfg.wind.air_density * self.cfg.wind.drag_coefficient
            )

            # --- Force tensor for apply_rigid_body_force_at_pos_tensors ---
            self.wind_force_tensor = torch.zeros(
                self.num_envs * self.num_bodies, 3,
                dtype=torch.float, device=self.device
            )
            # Position tensor (P3): application point for each body
            self.wind_pos_tensor = torch.zeros(
                self.num_envs * self.num_bodies, 3,
                dtype=torch.float, device=self.device
            )

            # --- Multi-body force distribution ---
            self.force_body_fractions = torch.tensor(
                self.cfg.wind.force_body_fractions,
                dtype=torch.float, device=self.device
            )  # [num_wind_bodies]

            # Find and store body indices for wind-receiving bodies
            wind_body_local_indices = []
            for name in self.cfg.wind.force_body_names:
                idx = self.gym.find_actor_rigid_body_handle(
                    self.envs[0], self.actor_handles[0], name
                )
                wind_body_local_indices.append(idx)
            self.wind_body_local_indices = wind_body_local_indices  # Python list for tensor indexing

            # Precompute flat indices: [num_envs, num_wind_bodies]
            env_offsets = torch.arange(self.num_envs, device=self.device) * self.num_bodies
            body_offsets = torch.tensor(wind_body_local_indices, device=self.device)
            self.wind_body_flat_indices = env_offsets.unsqueeze(1) + body_offsets.unsqueeze(0)

            # P3: Center of pressure vertical offsets per wind body
            self.cop_z_offsets = torch.tensor(
                self.cfg.wind.cop_z_offsets,
                dtype=torch.float, device=self.device
            )  # [num_wind_bodies]

            # P3: CoP local offset vectors (along each body's local z-axis)
            self.cop_local_offsets = torch.zeros(
                len(self.cfg.wind.force_body_names), 3,
                dtype=torch.float, device=self.device
            )
            self.cop_local_offsets[:, 2] = self.cop_z_offsets  # [num_wind_bodies, 3]

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

            # Refresh rigid body state after each simulate for accurate
            # body velocities/positions in the next substep's wind computation
            if self.cfg.wind.enable:
                self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.post_physics_step()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def _apply_wind_force(self):
        """Compute and apply aerodynamic forces with full per-body physics.

        P0: Relative velocity — F ∝ |v_wind - v_body|² (not just v_wind²)
        P1: Projected area — per-body 3D ellipsoidal using v_rel direction + body quaternions
        P2: Height profile — v(z) = v_ref × (z/z_ref)^α (boundary layer)
        P3: Center of pressure — CoP offset in body-local frame, rotated to world
        """
        physics_dt = self.sim_params.dt
        wind_vel = self.wind_model.step(physics_dt)  # [num_envs, 3] m/s

        num_wind_bodies = len(self.cfg.wind.force_body_names)

        # --- Get per-body states for wind-receiving bodies ---
        # rigid_body_states_view: [num_envs, num_bodies, 13] (from G1Robot._init_foot)
        # 13 = pos(3) + quat(4) + lin_vel(3) + ang_vel(3)
        wind_body_states = self.rigid_body_states_view[:, self.wind_body_local_indices, :]
        body_pos = wind_body_states[:, :, 0:3]    # [num_envs, num_wind_bodies, 3]
        body_vel = wind_body_states[:, :, 7:10]   # [num_envs, num_wind_bodies, 3]
        body_quat = wind_body_states[:, :, 3:7]   # [num_envs, num_wind_bodies, 4]
        body_heights = body_pos[:, :, 2]           # [num_envs, num_wind_bodies]

        # --- P2: Height-dependent wind speed scaling ---
        if self.cfg.wind.height_profile_enabled:
            ref_h = self.cfg.wind.reference_height
            alpha = self.cfg.wind.height_exponent
            min_ratio = self.cfg.wind.min_height_ratio
            height_factor = (body_heights / ref_h).clamp(min=min_ratio) ** alpha
            # [num_envs, num_wind_bodies]
        else:
            height_factor = torch.ones_like(body_heights)

        # Expand wind_vel for per-body computation: [num_envs, 1, 3] → broadcast
        wind_vel_expanded = wind_vel.unsqueeze(1)  # [num_envs, 1, 3]
        scaled_wind_vel = wind_vel_expanded * height_factor.unsqueeze(2)
        # [num_envs, num_wind_bodies, 3]

        # --- P0: Relative velocity ---
        v_rel = scaled_wind_vel - body_vel  # [num_envs, num_wind_bodies, 3]
        v_rel_speed = torch.norm(v_rel, dim=2)  # [num_envs, num_wind_bodies]
        v_rel_dir = v_rel / v_rel_speed.unsqueeze(2).clamp(min=1e-8)

        # --- P1: Per-body 3D ellipsoidal projected area ---
        # B1 fix: uses v_rel direction (not wind_vel) for area computation
        # B3 fix: includes z-component via A_top for tilted bodies
        # B4 fix: uses per-body quaternions (not just base_quat)
        flat_quat = body_quat.reshape(-1, 4)  # [num_envs * num_wind_bodies, 4]
        flat_v_rel_dir = v_rel_dir.reshape(-1, 3)
        v_rel_body = quat_rotate_inverse(flat_quat, flat_v_rel_dir)
        v_rel_body = v_rel_body.reshape(self.num_envs, num_wind_bodies, 3)

        # A_eff = sqrt((A_front·|dx|)² + (A_side·|dy|)² + (A_top·|dz|)²)
        dx = v_rel_body[:, :, 0].abs()  # [num_envs, num_wind_bodies]
        dy = v_rel_body[:, :, 1].abs()
        dz = v_rel_body[:, :, 2].abs()
        effective_area = torch.sqrt(
            (self.cfg.wind.frontal_area_front * dx) ** 2
            + (self.cfg.wind.frontal_area_side * dy) ** 2
            + (self.cfg.wind.frontal_area_top * dz) ** 2
        )  # [num_envs, num_wind_bodies]

        # --- Compute per-body aerodynamic force ---
        # F_i = 0.5 * ρ * Cd * A_eff_i * fraction_i * |v_rel_i|² × v_rel_hat_i
        fractions = self.force_body_fractions  # [num_wind_bodies]

        force_mag = (
            self.aero_force_coeff
            * effective_area                    # [num_envs, num_wind_bodies]
            * fractions.unsqueeze(0)            # [1, num_wind_bodies]
            * v_rel_speed ** 2                  # [num_envs, num_wind_bodies]
        )  # [num_envs, num_wind_bodies]

        per_body_force = v_rel_dir * force_mag.unsqueeze(2)  # [num_envs, num_wind_bodies, 3]

        # --- Per-level force clamp (on total force magnitude) ---
        total_force = per_body_force.sum(dim=1)  # [num_envs, 3]
        total_mag = torch.norm(total_force, dim=1)  # [num_envs]
        max_level = self.wind_model.force_clamp_tensor.shape[0] - 1
        per_env_clamp = self.wind_model.force_clamp_tensor[
            self.wind_curriculum_level.clamp(max=max_level)
        ]  # [num_envs]
        clamp_ratio = (per_env_clamp / total_mag.clamp(min=1e-8)).clamp(max=1.0)
        per_body_force = per_body_force * clamp_ratio.unsqueeze(1).unsqueeze(2)

        # Store total wind force for observations
        self.wind_model.wind_force = per_body_force.sum(dim=1)  # [num_envs, 3]

        # --- P3: Center of pressure positions ---
        # B2 fix: CoP offset along body-local z-axis (not world z-axis)
        # When robot tilts, CoP follows the tilted body axis correctly
        flat_local_offsets = self.cop_local_offsets.repeat(self.num_envs, 1)
        world_offsets = quat_rotate(flat_quat, flat_local_offsets)
        cop_positions = body_pos + world_offsets.reshape(
            self.num_envs, num_wind_bodies, 3
        )

        # --- Scatter into tensors and apply ---
        self.wind_force_tensor[:] = 0.0
        self.wind_pos_tensor[:] = 0.0
        flat_idx = self.wind_body_flat_indices.reshape(-1)  # [num_envs * num_wind_bodies]
        self.wind_force_tensor[flat_idx] = per_body_force.reshape(-1, 3)
        self.wind_pos_tensor[flat_idx] = cop_positions.reshape(-1, 3)

        # Apply forces at center of pressure positions (automatic torque computation)
        self.gym.apply_rigid_body_force_at_pos_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self.wind_force_tensor),
            gymtorch.unwrap_tensor(self.wind_pos_tensor),
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

        Uses effective_direction (base + OU drift) so the reward tracks
        the real-time wind heading, not just the episode-initial direction.
        """
        if not self.cfg.wind.enable:
            return torch.zeros(self.num_envs, device=self.device)

        gravity_xy = self.projected_gravity[:, :2]  # body frame [num_envs, 2]

        # Transform current wind direction (with OU drift) to body frame
        wind_dir_body = quat_rotate_inverse(
            self.base_quat, self.wind_model.effective_direction
        )  # [num_envs, 3]
        wind_dir_body_xy = wind_dir_body[:, :2]  # [num_envs, 2]

        # Negative dot product = leaning against the wind force direction
        lean_against_wind = -torch.sum(gravity_xy * wind_dir_body_xy, dim=1)

        # M2 fix: scale by wind force magnitude — stronger wind needs more lean
        wind_force_mag = torch.norm(self.wind_model.wind_force, dim=1)
        wind_scale = (wind_force_mag / 50.0).clamp(max=1.0)  # normalize by ~L3 force
        return lean_against_wind * wind_scale

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
