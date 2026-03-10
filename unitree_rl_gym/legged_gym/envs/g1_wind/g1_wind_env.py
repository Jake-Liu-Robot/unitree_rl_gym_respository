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

        # --- Validate wind config consistency ---
        if self.cfg.wind.enable:
            wcfg = self.cfg.wind
            n_bodies = len(wcfg.force_body_names)
            assert len(wcfg.force_body_fractions) == n_bodies, \
                f"force_body_fractions length ({len(wcfg.force_body_fractions)}) != force_body_names ({n_bodies})"
            assert len(wcfg.cop_z_offsets) == n_bodies, \
                f"cop_z_offsets length ({len(wcfg.cop_z_offsets)}) != force_body_names ({n_bodies})"
            n_levels = len(wcfg.curriculum_levels)
            assert len(wcfg.force_clamp_per_level) == n_levels, \
                f"force_clamp_per_level length ({len(wcfg.force_clamp_per_level)}) != curriculum_levels ({n_levels})"
            assert len(wcfg.speed_clamp_per_level) == n_levels, \
                f"speed_clamp_per_level length ({len(wcfg.speed_clamp_per_level)}) != curriculum_levels ({n_levels})"
            assert abs(sum(wcfg.force_body_fractions) - 0.95) < 0.05, \
                f"force_body_fractions sum ({sum(wcfg.force_body_fractions)}) should be ~0.95"

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

            # Precompute flat CoP offsets for _apply_wind_force (avoids per-substep allocation)
            self._flat_cop_offsets = self.cop_local_offsets.repeat(self.num_envs, 1)
            # [num_envs * num_wind_bodies, 3]

            # Curriculum evaluation: accumulate stats over a window of resets
            self.wind_reset_counter = 0
            self.wind_survival_acc = 0.0  # accumulated survival fraction
            self.wind_tracking_acc = 0.0  # accumulated tracking performance
            # Cooldown: skip evaluation for N resets after level change
            self._curriculum_cooldown = 0

            # Initialize wind for all envs so first episode has wind
            all_env_ids = torch.arange(self.num_envs, device=self.device)
            self.wind_model.reset_envs(all_env_ids, self.wind_curriculum_level)

        # --- Domain Randomization buffers ---
        # Action delay: ring buffer stores recent actions, reads delayed version
        # for torque computation. self.actions (in obs) still shows current intended
        # action; only execution is delayed. This trains the LSTM to compensate.
        if getattr(self.cfg.domain_rand, 'randomize_action_delay', False):
            self._max_action_delay = self.cfg.domain_rand.action_delay_range[1]
            self._action_delay_buf = torch.zeros(
                self.num_envs, self._max_action_delay + 1, self.num_actions,
                dtype=torch.float, device=self.device
            )
            self._action_delay = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device
            )
            self._delay_write_idx = 0

        # PD gain randomization: per-env multipliers (re-sampled each episode)
        if getattr(self.cfg.domain_rand, 'randomize_pd_gains', False):
            self._stiffness_mult = torch.ones(
                self.num_envs, 1, dtype=torch.float, device=self.device
            )
            self._damping_mult = torch.ones(
                self.num_envs, 1, dtype=torch.float, device=self.device
            )

        # Motor strength: per-env torque limit multiplier (re-sampled each episode)
        if getattr(self.cfg.domain_rand, 'randomize_motor_strength', False):
            self._motor_strength = torch.ones(
                self.num_envs, 1, dtype=torch.float, device=self.device
            )

        # --- Buffers for new reward functions ---
        # Second-order action smoothness (action_rate2):
        # need last_last_actions = actions(t-2), managed via _buf_last_actions
        self.last_last_actions = torch.zeros(
            self.num_envs, self.num_actions,
            dtype=torch.float, device=self.device
        )
        self._buf_last_actions = torch.zeros(
            self.num_envs, self.num_actions,
            dtype=torch.float, device=self.device
        )
        # Base acceleration reward: track previous body-frame xy velocity
        self._prev_base_lin_vel_xy = torch.zeros(
            self.num_envs, 2,
            dtype=torch.float, device=self.device
        )
        # Angular momentum change reward: track previous angular velocity (xy)
        self._prev_base_ang_vel_xy = torch.zeros(
            self.num_envs, 2,
            dtype=torch.float, device=self.device
        )

    def _post_physics_step_callback(self):
        """Rotate action history buffer before reward computation.

        Called by LeggedRobot.post_physics_step BEFORE compute_reward.
        At this point: self.last_actions = actions(t-1), set by base at end of previous step.
        _buf_last_actions holds actions(t-2) from the previous callback.
        We shift: last_last_actions ← _buf(t-2), then _buf ← last_actions(t-1).
        """
        self.last_last_actions[:] = self._buf_last_actions
        self._buf_last_actions[:] = self.last_actions
        super()._post_physics_step_callback()

    def step(self, actions):
        """Override step to apply action delay, wind forces, and DR torques."""
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

        # --- Action delay: obs shows intended action, execution uses delayed ---
        if getattr(self.cfg.domain_rand, 'randomize_action_delay', False):
            self._action_delay_buf[:, self._delay_write_idx] = self.actions
            read_idx = (self._delay_write_idx - self._action_delay) % (self._max_action_delay + 1)
            exec_actions = self._action_delay_buf[
                torch.arange(self.num_envs, device=self.device), read_idx
            ]
            self._delay_write_idx = (self._delay_write_idx + 1) % (self._max_action_delay + 1)
        else:
            exec_actions = self.actions

        self.render()

        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(exec_actions).view(self.torques.shape)
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
        world_offsets = quat_rotate(flat_quat, self._flat_cop_offsets)
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

    def _compute_torques(self, actions):
        """Compute torques with per-env PD gain and motor strength randomization.

        PD gains are multiplied by per-env stiffness/damping factors (sampled
        at episode reset). Torque limits are scaled by per-env motor strength.
        This accounts for actuator model mismatch in sim-to-real transfer.
        """
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type

        p_gains = self.p_gains  # [num_actions]
        d_gains = self.d_gains  # [num_actions]

        if getattr(self.cfg.domain_rand, 'randomize_pd_gains', False):
            # [num_actions] * [num_envs, 1] → [num_envs, num_actions]
            p_gains = p_gains * self._stiffness_mult
            d_gains = d_gains * self._damping_mult

        if control_type == "P":
            torques = p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos) - d_gains * self.dof_vel
        elif control_type == "V":
            torques = p_gains * (actions_scaled - self.dof_vel) - d_gains * (self.dof_vel - self.last_dof_vel) / self.sim_params.dt
        elif control_type == "T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")

        limits = self.torque_limits  # [num_dof]
        if getattr(self.cfg.domain_rand, 'randomize_motor_strength', False):
            limits = limits * self._motor_strength  # [num_envs, num_dof]

        return torch.clip(torques, -limits, limits)

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

        # Reset new reward buffers
        self.last_last_actions[env_ids] = 0.
        self._buf_last_actions[env_ids] = 0.
        self._prev_base_lin_vel_xy[env_ids] = 0.
        self._prev_base_ang_vel_xy[env_ids] = 0.

        # --- Re-sample domain randomization parameters ---
        n = len(env_ids)

        if getattr(self.cfg.domain_rand, 'randomize_action_delay', False):
            self._action_delay_buf[env_ids] = 0.0
            self._action_delay[env_ids] = torch.randint(
                self.cfg.domain_rand.action_delay_range[0],
                self.cfg.domain_rand.action_delay_range[1] + 1,
                (n,), device=self.device
            )

        if getattr(self.cfg.domain_rand, 'randomize_pd_gains', False):
            self._stiffness_mult[env_ids] = torch_rand_float(
                self.cfg.domain_rand.stiffness_multiplier_range[0],
                self.cfg.domain_rand.stiffness_multiplier_range[1],
                (n, 1), device=self.device
            )
            self._damping_mult[env_ids] = torch_rand_float(
                self.cfg.domain_rand.damping_multiplier_range[0],
                self.cfg.domain_rand.damping_multiplier_range[1],
                (n, 1), device=self.device
            )

        if getattr(self.cfg.domain_rand, 'randomize_motor_strength', False):
            self._motor_strength[env_ids] = torch_rand_float(
                self.cfg.domain_rand.motor_strength_range[0],
                self.cfg.domain_rand.motor_strength_range[1],
                (n, 1), device=self.device
            )

        # Always log curriculum info so all ep_infos have consistent keys
        if self.cfg.wind.enable:
            self.extras["episode"]["wind_curriculum_mean"] = self.wind_curriculum_level.float().mean().item()
            self.extras["episode"]["wind_curriculum_max"] = self.wind_curriculum_level.max().item()

    def _update_wind_curriculum(self, env_ids):
        """Accumulate stats over a window of resets, then decide level change.

        Uses window-based evaluation for reliable statistics.
        - Upgrade: only upgrade_fraction (80%) of envs advance (forgetting prevention)
        - Demotion: only demotion_fraction (50%) of envs demote (prevents oscillation)
        - Cooldown: after any level change, skip evaluation for N resets
        """
        n = len(env_ids)

        # Cooldown: skip accumulation and evaluation after level change
        if self._curriculum_cooldown > 0:
            self._curriculum_cooldown -= n
            return

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

        # Accumulate into window (sum, then divide by counter when evaluating)
        self.wind_survival_acc += survival_frac.sum().item()
        self.wind_tracking_acc += tracking_perf.sum().item()
        self.wind_reset_counter += n

        # Evaluate when window is full
        if self.wind_reset_counter >= self.cfg.wind.upgrade_window:
            avg_survival = self.wind_survival_acc / self.wind_reset_counter
            avg_tracking = self.wind_tracking_acc / self.wind_reset_counter
            level_changed = False

            if (avg_survival > self.cfg.wind.survival_threshold
                    and avg_tracking > self.cfg.wind.tracking_threshold):
                max_level = len(self.cfg.wind.curriculum_levels) - 1
                # Mixed-level upgrade: only upgrade_fraction of envs advance
                upgrade_frac = getattr(self.cfg.wind, 'upgrade_fraction', 1.0)
                if upgrade_frac < 1.0:
                    upgrade_mask = torch.rand(self.num_envs, device=self.device) < upgrade_frac
                    self.wind_curriculum_level[upgrade_mask] = torch.clamp(
                        self.wind_curriculum_level[upgrade_mask] + 1, max=max_level
                    )
                else:
                    self.wind_curriculum_level[:] = torch.clamp(
                        self.wind_curriculum_level + 1, max=max_level
                    )
                level_changed = True
            elif (avg_survival < self.cfg.wind.demotion_survival_threshold
                    and avg_tracking < self.cfg.wind.demotion_tracking_threshold):
                # Fractional demotion: only demote a fraction of envs
                demotion_frac = getattr(self.cfg.wind, 'demotion_fraction', 1.0)
                if demotion_frac < 1.0:
                    demote_mask = torch.rand(self.num_envs, device=self.device) < demotion_frac
                    self.wind_curriculum_level[demote_mask] = torch.clamp(
                        self.wind_curriculum_level[demote_mask] - 1, min=0
                    )
                else:
                    self.wind_curriculum_level[:] = torch.clamp(
                        self.wind_curriculum_level - 1, min=0
                    )
                level_changed = True

            # Reset window + start cooldown if level changed
            self.wind_reset_counter = 0
            self.wind_survival_acc = 0.0
            self.wind_tracking_acc = 0.0
            if level_changed:
                cooldown = getattr(self.cfg.wind, 'upgrade_cooldown', 0)
                self._curriculum_cooldown = cooldown

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
                wind_vel * self.cfg.env.priv_obs_wind_vel_scale,
                wind_force * self.cfg.env.priv_obs_wind_force_scale,
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

    # ========== Overridden base reward functions ==========

    def _reward_orientation(self):
        """Wind-aware orientation penalty.

        At zero wind: full penalty for body tilt (standard upright incentive).
        Under wind: smoothly reduces penalty via 1/(1 + F/F_ref), allowing
        lean_compensation to dominate without conflicting gradients.

        At F_ref=50N: penalty halves. At 100N: penalty is 1/3.
        The smooth 1/(1+x) curve ensures:
          - No hard cutoff (always some upright preference)
          - Continuous & differentiable (clean gradients)
          - Monotonically decreasing (more wind → less penalty)
        """
        gravity_xy = self.projected_gravity[:, :2]
        raw_penalty = torch.sum(torch.square(gravity_xy), dim=1)
        if self.cfg.wind.enable:
            wind_force_mag = torch.norm(self.wind_model.wind_force, dim=1)
            reduction = 1.0 / (1.0 + wind_force_mag / self.cfg.wind.orientation_wind_scale)
            return raw_penalty * reduction
        return raw_penalty

    def _reward_base_height(self):
        """Wind-aware base height penalty.

        Under strong wind, allows the robot to crouch slightly to lower COM,
        reducing the wind's tipping moment (torque = F × h_com).
        Target lowers from 0.78m to ~0.741m at 150N peak force.
        """
        base_target = self.cfg.rewards.base_height_target
        if self.cfg.wind.enable:
            wind_force_mag = torch.norm(self.wind_model.wind_force, dim=1)
            wind_scale = (wind_force_mag / 150.0).clamp(max=1.0)
            target = base_target * (
                1.0 - self.cfg.wind.base_height_wind_reduction * wind_scale
            )
        else:
            target = base_target
        base_height = self.root_states[:, 2]
        return torch.square(base_height - target)

    def _reward_tracking_lin_vel(self):
        """Track linear velocity commands with wind-adaptive sigma.

        Under strong wind, tracking error naturally increases. A fixed tight
        sigma (0.25) gives near-zero reward with no gradient at large errors.
        We widen sigma proportionally to wind force magnitude to maintain
        learning signal throughout curriculum progression.

        Ref: Xu et al. 2025 — multi-objective RL for command tracking vs force compliance.
        """
        lin_vel_error = torch.sum(torch.square(
            self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        if self.cfg.wind.enable:
            wind_force_mag = torch.norm(self.wind_model.wind_force, dim=1)
            # sigma_eff = sigma_base * (1 + scale * |F_wind| / 100N)
            # At 100N: sigma 0.25 → 0.50, 1m/s error reward: 0.018 → 0.135
            sigma = self.cfg.rewards.tracking_sigma * (
                1.0 + self.cfg.rewards.tracking_sigma_wind_scale
                * wind_force_mag / 100.0
            )
        else:
            sigma = self.cfg.rewards.tracking_sigma
        return torch.exp(-lin_vel_error / sigma)

    # ========== Wind-specific reward functions ==========

    def _reward_lean_compensation(self):
        """Reward leaning against the actual wind force direction.

        Uses wind_force direction (includes base + OU + gusts) instead of
        effective_direction (base + OU only). This ensures the robot leans
        against the real instantaneous force, not just the base wind heading.
        Critical for gust response: gusts can arrive from ±60° off base.
        """
        if not self.cfg.wind.enable:
            return torch.zeros(self.num_envs, device=self.device)

        gravity_xy = self.projected_gravity[:, :2]  # body frame [num_envs, 2]

        # Use actual force direction (includes gusts), not just base+OU
        wind_force = self.wind_model.wind_force  # [num_envs, 3]
        wind_force_mag = torch.norm(wind_force, dim=1)
        force_dir = wind_force / wind_force_mag.unsqueeze(1).clamp(min=1e-8)

        # Transform force direction to body frame
        force_dir_body = quat_rotate_inverse(self.base_quat, force_dir)
        force_dir_body_xy = force_dir_body[:, :2]  # [num_envs, 2]

        # Negative dot product = leaning against the wind force direction
        lean_against_wind = -torch.sum(gravity_xy * force_dir_body_xy, dim=1)

        # Scale by wind force magnitude — stronger wind needs more lean
        wind_scale = (wind_force_mag / 50.0).clamp(max=1.0)
        return lean_against_wind * wind_scale

    def _reward_feet_distance(self):
        """Penalize feet being too close together.

        Under wind, a wider stance increases the support polygon and improves
        lateral stability. Returns positive value when feet separation is
        below the minimum threshold (used with negative scale as penalty).

        Ref: Booster Gym 2025 — r_feet_dist = max(d_ref - d_feet, 0), weight=-1.0.
        """
        feet_xy = self.feet_pos[:, :, :2]  # [num_envs, 2, 2]
        separation = torch.norm(feet_xy[:, 0] - feet_xy[:, 1], dim=1)
        min_sep = self.cfg.wind.feet_min_separation
        return (min_sep - separation).clamp(min=0)

    def _reward_base_acc(self):
        """Penalize controllable base acceleration (wind-compensated).

        Subtracts expected wind-induced acceleration (F_wind / m) so only
        the policy's own jerkiness is penalized. Without this, the penalty
        has a 4.3 m/s² floor at L5 (150N/35kg) that the robot cannot avoid.

        Note: base_lin_vel is in body frame (quat_rotate_inverse in base class),
        so wind force must also be transformed to body frame before compensation.

        Ref: Viereck et al. IROS 2024 — base acceleration penalty.
        """
        acc_xy = (self.base_lin_vel[:, :2] - self._prev_base_lin_vel_xy) / self.dt
        self._prev_base_lin_vel_xy = self.base_lin_vel[:, :2].clone()
        if self.cfg.wind.enable:
            # Transform wind force from world frame to body frame (matching base_lin_vel)
            wind_force_body = quat_rotate_inverse(
                self.base_quat, self.wind_model.wind_force
            )
            wind_acc_xy = wind_force_body[:, :2] / self.cfg.wind.robot_nominal_mass
            acc_xy = acc_xy - wind_acc_xy
        return torch.sum(torch.square(acc_xy), dim=1)

    def _reward_action_rate2(self):
        """Penalize second-order action difference for smoother control.

        ||a_t - 2*a_{t-1} + a_{t-2}||² penalizes acceleration of actions,
        producing smoother trajectories than first-order action_rate alone.
        Particularly important for sim-to-real transfer under wind.

        Ref: Humanoid-Gym 2024 (RobotEra) — second-order action smoothness.
        """
        return torch.sum(torch.square(
            self.actions - 2 * self.last_actions + self.last_last_actions
        ), dim=1)

    def _reward_power(self):
        """Penalize positive (motoring) mechanical power, allow braking.

        sum(max(tau * dq, 0)) only penalizes joints doing positive work
        (driving motion), not negative work (braking/resisting). This allows
        the robot to use high torques for disturbance rejection without penalty.

        Ref: Booster Gym 2025 — power = max(tau * dq, 0), weight=-2e-4.
        """
        return torch.sum(torch.clamp(self.torques * self.dof_vel, min=0), dim=1)

    def _reward_sustained_walking(self):
        """Reward for remaining alive (disabled, identical to alive)."""
        return torch.ones(self.num_envs, device=self.device)

    def _reward_contact_symmetry(self):
        """Reward symmetric foot contact patterns (disabled)."""
        left_contact = self.contact_forces[:, self.feet_indices[0], 2] > 1.0
        right_contact = self.contact_forces[:, self.feet_indices[1], 2] > 1.0
        symmetry = (left_contact == right_contact).float()
        return symmetry

    def _reward_ang_momentum_change(self):
        """Penalize rate of change of angular momentum (xy axes).

        dL/dt ≈ I × dω/dt. Penalizing angular acceleration reduces
        rotational jerk from wind gusts, complementing ang_vel_xy which
        penalizes absolute angular velocity.

        Ref: Li et al. 2025 — angular momentum regularization.
        """
        ang_acc_xy = (
            (self.base_ang_vel[:, :2] - self._prev_base_ang_vel_xy) / self.dt
        )
        self._prev_base_ang_vel_xy = self.base_ang_vel[:, :2].clone()
        return torch.sum(torch.square(ang_acc_xy), dim=1)

    def _reward_com_balance(self):
        """Penalize COM projection deviating from support center.

        Support center = contact-weighted mean of feet positions.
        During single support: support center ≈ stance foot.
        During double support: support center ≈ midpoint of feet.

        Ensures the robot repositions its feet to keep COM over the
        support polygon when leaning into wind (lean_compensation).
        Only active when at least one foot is in contact.
        """
        com_xy = self.root_states[:, :2]  # [num_envs, 2]
        feet_xy = self.feet_pos[:, :, :2]  # [num_envs, 2, 2]

        contact = (
            self.contact_forces[:, self.feet_indices, 2] > 1.0
        ).float()  # [num_envs, 2]
        # Contact-weighted center; fallback to mean if both feet in air
        weights = contact / contact.sum(dim=1, keepdim=True).clamp(min=1.0)
        support_center = (feet_xy * weights.unsqueeze(2)).sum(dim=1)

        has_support = (contact.sum(dim=1) > 0).float()
        return (
            torch.sum(torch.square(com_xy - support_center), dim=1)
            * has_support
        )

    # ========== Overridden gait rewards for wind adaptation ==========

    def _reward_feet_swing_height(self):
        """Wind-adaptive swing height target.

        Base target 0.08m. Under strong wind, the target decreases
        (up to 50% at peak force) so the robot can take lower, more
        stable swings instead of being penalized for cautious gaits.
        """
        contact = (
            torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2)
            > 1.0
        )
        if self.cfg.wind.enable:
            wind_force_mag = torch.norm(self.wind_model.wind_force, dim=1)
            wind_scale = (wind_force_mag / 150.0).clamp(max=1.0)
            target = self.cfg.wind.swing_height_base * (
                1.0 - self.cfg.wind.swing_height_wind_reduction * wind_scale
            )  # [num_envs]
            pos_error = (
                torch.square(self.feet_pos[:, :, 2] - target.unsqueeze(1))
                * ~contact
            )
        else:
            pos_error = (
                torch.square(
                    self.feet_pos[:, :, 2] - self.cfg.wind.swing_height_base
                )
                * ~contact
            )
        return torch.sum(pos_error, dim=1)

    def _reward_contact(self):
        """Wind-adaptive gait phase contact reward.

        Under strong wind, the stance phase ratio widens (0.55 → 0.75)
        to allow longer double-support periods, improving stability.
        """
        if self.cfg.wind.enable:
            wind_force_mag = torch.norm(self.wind_model.wind_force, dim=1)
            stance_ratio = self.cfg.wind.stance_ratio_base + (
                self.cfg.wind.stance_ratio_wind_increase
                * (wind_force_mag / 150.0).clamp(max=1.0)
            )  # [num_envs]
        else:
            stance_ratio = self.cfg.wind.stance_ratio_base

        res = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        for i in range(self.feet_num):
            is_stance = self.leg_phase[:, i] < stance_ratio
            contact = self.contact_forces[:, self.feet_indices[i], 2] > 1
            res += ~(contact ^ is_stance)
        return res
