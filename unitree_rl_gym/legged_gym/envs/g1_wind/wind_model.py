"""
3-layer wind model for G1 wind-robust walking (v3).

Layer 1: Base wind — per-episode constant direction + speed (from curriculum)
Layer 2: OU process — speed fluctuation + directional drift (per-episode randomized params)
Layer 3: Gust events — independent velocity vector, trapezoidal envelope, own direction

Wind velocity composition:
    base_vel = (base_speed + ou_speed) × dir(base_angle + ou_angle)   # Layer 1+2
    gust_vel = gust_speed × envelope(t) × gust_direction              # Layer 3
    wind_velocity = base_vel + gust_vel

v3 changes vs v2:
  - Wind model now outputs VELOCITY only (m/s, world frame)
  - Force computation moved to g1_wind_env.py with full per-body aerodynamics:
    P0: Relative velocity (v_wind - v_body) for accurate drag
    P1: Direction-dependent projected area (cross-flow principle)
    P2: Height-dependent wind speed (power law boundary layer)
    P3: Force at center of pressure (not COM) via apply_rigid_body_force_at_pos_tensors

All operations are vectorized GPU tensor ops (no Python loops over envs).
"""

import math
import torch
from isaacgym.torch_utils import torch_rand_float


class WindModel:
    """Vectorized 3-layer wind model for parallel environments (v3).

    Produces wind velocity field. Force computation is handled by the
    environment using per-body aerodynamics.
    """

    def __init__(self, num_envs, device, cfg):
        """
        Args:
            num_envs: Number of parallel environments
            device: Torch device ('cuda:0', 'cpu', etc.)
            cfg: Wind config object (G1WindRoughCfg.wind)
        """
        self.num_envs = num_envs
        self.device = device
        self.cfg = cfg

        # --- Layer 1 state: base wind (constant per episode) ---
        self.base_angle = torch.zeros(num_envs, device=device)
        self.base_direction = torch.zeros(num_envs, 3, device=device)
        self.base_speed = torch.zeros(num_envs, device=device)

        # --- Layer 2 state: OU processes ---
        self.ou_speed_state = torch.zeros(num_envs, device=device)
        self.ou_angle_state = torch.zeros(num_envs, device=device)

        # Per-env randomized OU parameters (sampled each episode in reset_envs)
        self.env_ou_theta = torch.full((num_envs,), cfg.ou_theta, device=device)
        self.env_ou_sigma = torch.full((num_envs,), cfg.ou_sigma, device=device)
        self.env_ou_theta_dir = torch.full((num_envs,), cfg.ou_theta_dir, device=device)
        self.env_ou_sigma_dir = torch.full((num_envs,), cfg.ou_sigma_dir, device=device)

        # --- Layer 3 state: gust events ---
        self.gust_active = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.gust_elapsed = torch.zeros(num_envs, device=device)
        self.gust_duration = torch.zeros(num_envs, device=device)
        self.gust_speed = torch.zeros(num_envs, device=device)
        self.gust_direction = torch.zeros(num_envs, 3, device=device)

        # --- Output buffers ---
        self.wind_velocity = torch.zeros(num_envs, 3, device=device)
        self.wind_force = torch.zeros(num_envs, 3, device=device)  # set by env after force computation
        self.effective_direction = torch.zeros(num_envs, 3, device=device)

        # --- Per-env curriculum level (for parameter scaling) ---
        self.env_level = torch.zeros(num_envs, dtype=torch.long, device=device)

        # --- Precomputed curriculum levels tensor ---
        self.curriculum_levels_tensor = torch.tensor(
            cfg.curriculum_levels, dtype=torch.float, device=device
        )  # [num_levels, 2]

        # --- Per-level force clamp ---
        self.force_clamp_tensor = torch.tensor(
            cfg.force_clamp_per_level, dtype=torch.float, device=device
        )  # [num_levels]

        # --- Per-level wind speed clamp (prevents unrealistic OU+gust extremes) ---
        self.speed_clamp_tensor = torch.tensor(
            cfg.speed_clamp_per_level, dtype=torch.float, device=device
        )  # [num_levels]

    # ================================================================
    # Lifecycle
    # ================================================================

    def reset_envs(self, env_ids, curriculum_level):
        """Reset wind state for specified environments (called on episode reset).

        Args:
            env_ids: Tensor of environment indices to reset
            curriculum_level: Tensor [num_envs], per-env curriculum level
        """
        if len(env_ids) == 0:
            return

        n = len(env_ids)
        self.env_level[env_ids] = curriculum_level[env_ids].long()

        # --- Layer 1: sample base wind ---
        angles = torch_rand_float(0, 2 * math.pi, (n, 1), device=self.device).squeeze(1)
        self.base_angle[env_ids] = angles
        self.base_direction[env_ids, 0] = torch.cos(angles)
        self.base_direction[env_ids, 1] = torch.sin(angles)
        self.base_direction[env_ids, 2] = 0.0
        self.base_speed[env_ids] = self._sample_speed_per_env(curriculum_level, env_ids)

        # --- Layer 2: reset OU states + randomize params ---
        self.ou_speed_state[env_ids] = 0.0
        self.ou_angle_state[env_ids] = 0.0

        if self.cfg.ou_randomize:
            self.env_ou_theta[env_ids] = torch_rand_float(
                self.cfg.ou_theta_range[0], self.cfg.ou_theta_range[1],
                (n, 1), device=self.device
            ).squeeze(1)
            self.env_ou_sigma[env_ids] = torch_rand_float(
                self.cfg.ou_sigma_range[0], self.cfg.ou_sigma_range[1],
                (n, 1), device=self.device
            ).squeeze(1)
            self.env_ou_theta_dir[env_ids] = torch_rand_float(
                self.cfg.ou_theta_dir_range[0], self.cfg.ou_theta_dir_range[1],
                (n, 1), device=self.device
            ).squeeze(1)
            self.env_ou_sigma_dir[env_ids] = torch_rand_float(
                self.cfg.ou_sigma_dir_range[0], self.cfg.ou_sigma_dir_range[1],
                (n, 1), device=self.device
            ).squeeze(1)

        # --- Layer 3: reset gust ---
        self.gust_active[env_ids] = False
        self.gust_elapsed[env_ids] = 0.0
        self.gust_duration[env_ids] = 0.0
        self.gust_speed[env_ids] = 0.0
        self.gust_direction[env_ids] = 0.0

        # Initialize effective direction
        self.effective_direction[env_ids] = self.base_direction[env_ids]

    # ================================================================
    # Step
    # ================================================================

    def step(self, dt):
        """Advance wind model by one physics substep.

        Computes the wind velocity field. Force computation is handled by
        the environment using per-body aerodynamics (relative velocity,
        height profile, directional projected area, center of pressure).

        Args:
            dt: Simulation timestep (seconds)

        Returns:
            wind_velocity: Tensor [num_envs, 3] — velocity in world frame (m/s)
        """
        # --- Layer 2: update OU processes ---
        self._update_ou(dt)

        # --- Layer 3: update gust lifecycle ---
        self._update_gusts(dt)

        # --- Wind velocity (Layer 1 + Layer 2) ---
        effective_speed = torch.clamp(self.base_speed + self.ou_speed_state, min=0.0)

        # Direction with angular OU offset
        current_angle = self.base_angle + self.ou_angle_state
        self.effective_direction[:, 0] = torch.cos(current_angle)
        self.effective_direction[:, 1] = torch.sin(current_angle)
        self.effective_direction[:, 2] = 0.0

        base_vel = self.effective_direction * effective_speed.unsqueeze(1)

        # --- Gust velocity (Layer 3) ---
        gust_envelope = self._compute_gust_envelope()
        gust_vel = self.gust_direction * (self.gust_speed * gust_envelope).unsqueeze(1)

        # --- Combined wind velocity ---
        total_vel = base_vel + gust_vel

        # --- Per-level speed clamp (prevents unrealistic OU+gust extremes) ---
        # L0 clamp=0 ensures truly zero wind at baseline level
        speed = torch.norm(total_vel, dim=1, keepdim=True).clamp(min=1e-8)
        max_level = self.speed_clamp_tensor.shape[0] - 1
        per_env_speed_clamp = self.speed_clamp_tensor[
            self.env_level.clamp(max=max_level)
        ].unsqueeze(1)  # [num_envs, 1]
        clamp_ratio = (per_env_speed_clamp / speed).clamp(max=1.0)
        self.wind_velocity = total_vel * clamp_ratio

        return self.wind_velocity

    # ================================================================
    # Layer 2: OU processes
    # ================================================================

    def _update_ou(self, dt):
        """Update speed and direction Ornstein-Uhlenbeck processes.

        Uses per-env randomized params (env_ou_theta, env_ou_sigma, etc.)
        when ou_randomize=True, otherwise falls back to config defaults.
        """
        cfg = self.cfg
        sqrt_dt = dt ** 0.5

        # Per-env turbulence scaling: higher curriculum = more turbulent
        sigma_scale = 1.0 + cfg.ou_sigma_scale_per_level * self.env_level.float()

        # Speed OU: dv = θ(0-v)dt + σ_eff·√dt·dW
        # σ_eff = env_ou_sigma × base_speed × scale + sigma_min
        effective_sigma = self.env_ou_sigma * self.base_speed * sigma_scale + cfg.ou_sigma_min
        dW_speed = torch.randn(self.num_envs, device=self.device)
        self.ou_speed_state += (
            self.env_ou_theta * (0.0 - self.ou_speed_state) * dt
            + effective_sigma * sqrt_dt * dW_speed
        )

        # Direction OU: dα = θ_dir(0-α)dt + σ_dir_eff·√dt·dW
        # Direction variability also scales with curriculum level (M1)
        dW_dir = torch.randn(self.num_envs, device=self.device)
        effective_sigma_dir = self.env_ou_sigma_dir * sigma_scale
        self.ou_angle_state += (
            self.env_ou_theta_dir * (0.0 - self.ou_angle_state) * dt
            + effective_sigma_dir * sqrt_dt * dW_dir
        )

    # ================================================================
    # Layer 3: gust events
    # ================================================================

    def _update_gusts(self, dt):
        """Advance gust timers, expire finished gusts, trigger new ones."""
        cfg = self.cfg

        # Advance elapsed time for active gusts
        self.gust_elapsed[self.gust_active] += dt

        # Expire finished gusts
        expired = self.gust_active & (self.gust_elapsed >= self.gust_duration)
        self.gust_active[expired] = False
        self.gust_elapsed[expired] = 0.0

        # Trigger new gusts (only where no active gust)
        can_gust = ~self.gust_active
        gust_prob = cfg.gust_prob * (1.0 + cfg.gust_prob_scale_per_level * self.env_level.float())
        roll = torch.rand(self.num_envs, device=self.device)
        new_gust = can_gust & (roll < gust_prob * dt)

        if new_gust.any():
            n_new = new_gust.sum().item()

            # Duration
            self.gust_duration[new_gust] = torch_rand_float(
                cfg.gust_duration_range[0], cfg.gust_duration_range[1],
                (n_new, 1), device=self.device
            ).squeeze(1)
            self.gust_elapsed[new_gust] = 0.0

            # Independent gust speed, scaled by curriculum level
            speed_scale = 1.0 + cfg.gust_speed_scale_per_level * self.env_level[new_gust].float()
            base_gust_speed = torch_rand_float(
                cfg.gust_speed_range[0], cfg.gust_speed_range[1],
                (n_new, 1), device=self.device
            ).squeeze(1)
            self.gust_speed[new_gust] = base_gust_speed * speed_scale

            # Independent direction: base heading + random offset ±60°
            angle_offset = torch_rand_float(
                cfg.gust_angle_range[0], cfg.gust_angle_range[1],
                (n_new, 1), device=self.device
            ).squeeze(1)
            gust_angle = self.base_angle[new_gust] + angle_offset
            self.gust_direction[new_gust, 0] = torch.cos(gust_angle)
            self.gust_direction[new_gust, 1] = torch.sin(gust_angle)
            self.gust_direction[new_gust, 2] = 0.0

            self.gust_active[new_gust] = True

    def _compute_gust_envelope(self):
        """Trapezoidal envelope: ramp-up -> sustain -> ramp-down.

        envelope(t) = min(t / ramp_up, (dur - t) / ramp_down)  clamped [0, 1]
        Gracefully handles short durations where ramp_up + ramp_down > duration.
        """
        envelope = torch.zeros(self.num_envs, device=self.device)
        if not self.gust_active.any():
            return envelope

        active = self.gust_active
        t = self.gust_elapsed[active]
        dur = self.gust_duration[active]
        ru = self.cfg.gust_ramp_up + 1e-6   # avoid /0
        rd = self.cfg.gust_ramp_down + 1e-6

        envelope[active] = torch.min(t / ru, (dur - t) / rd).clamp(0.0, 1.0)
        return envelope

    # ================================================================
    # Helpers
    # ================================================================

    def _sample_speed_per_env(self, curriculum_level, env_ids):
        """Sample wind speed from curriculum level range for each env."""
        levels = self.curriculum_levels_tensor
        max_level = levels.shape[0] - 1
        lvl = curriculum_level[env_ids].long().clamp(max=max_level)
        speed_min = levels[lvl, 0]
        speed_max = levels[lvl, 1]
        rand = torch.rand(len(env_ids), device=self.device)
        return speed_min + rand * (speed_max - speed_min)

    def get_wind_velocity(self):
        """Return current wind velocity vector [num_envs, 3] (m/s).

        Returns the velocity computed during the last step() call.
        Used for privileged observations.
        """
        return self.wind_velocity
