"""
3-layer wind model for G1 wind-robust walking.

Layer 1: Base wind — per-episode constant direction + speed
Layer 2: OU process — temporally-correlated fluctuation (Ornstein-Uhlenbeck)
Layer 3: Gust events — random short bursts (2-3x base force, 1-2s duration)

Wind force = 0.5 * rho * Cd * A * v^2, applied to torso rigid body.
All operations are vectorized GPU tensor ops (no Python loops over envs).
"""

import torch
from isaacgym.torch_utils import torch_rand_float


class WindModel:
    """Vectorized 3-layer wind model for parallel environments."""

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

        # --- Physical constants ---
        self.rho = cfg.air_density         # kg/m^3
        self.Cd = cfg.drag_coefficient     # dimensionless
        self.A = cfg.frontal_area          # m^2

        # --- OU process parameters ---
        self.ou_theta = cfg.ou_theta       # mean reversion rate
        self.ou_sigma = cfg.ou_sigma       # noise intensity (fraction of base speed)

        # --- Gust parameters ---
        self.gust_prob = cfg.gust_prob     # probability of gust per timestep
        self.gust_force_multiplier = cfg.gust_force_multiplier  # 2-3x
        self.gust_duration_range = cfg.gust_duration_range      # [min_s, max_s]

        # --- State buffers ---
        # Base wind: constant per episode
        self.base_direction = torch.zeros(num_envs, 3, device=device)  # unit vector (xy plane)
        self.base_speed = torch.zeros(num_envs, device=device)         # m/s

        # OU fluctuation state
        self.ou_state = torch.zeros(num_envs, device=device)  # speed fluctuation

        # Gust state
        self.gust_active = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.gust_remaining_time = torch.zeros(num_envs, device=device)
        self.gust_multiplier = torch.zeros(num_envs, device=device)

        # Output: final wind force vector per env [num_envs, 3]
        self.wind_force = torch.zeros(num_envs, 3, device=device)

        # Precompute curriculum levels tensor (avoid re-creating each call)
        self.curriculum_levels_tensor = torch.tensor(
            cfg.curriculum_levels, dtype=torch.float, device=device
        )  # [num_levels, 2]

    def reset_envs(self, env_ids, curriculum_level):
        """Reset wind state for specified environments (on episode reset).

        Args:
            env_ids: Tensor of environment indices to reset
            curriculum_level: Tensor [num_envs] or scalar, current curriculum level per env
        """
        if len(env_ids) == 0:
            return

        n = len(env_ids)

        # Sample random horizontal direction (unit vector in xy plane)
        angles = torch_rand_float(0, 2 * 3.14159265, (n, 1), device=self.device).squeeze(1)
        self.base_direction[env_ids, 0] = torch.cos(angles)
        self.base_direction[env_ids, 1] = torch.sin(angles)
        self.base_direction[env_ids, 2] = 0.0

        # Sample base wind speed per-env from its own curriculum level
        self.base_speed[env_ids] = self._sample_speed_per_env(curriculum_level, env_ids)

        # Reset OU state
        self.ou_state[env_ids] = 0.0

        # Reset gust state
        self.gust_active[env_ids] = False
        self.gust_remaining_time[env_ids] = 0.0
        self.gust_multiplier[env_ids] = 0.0

    def step(self, dt):
        """Advance wind model by one timestep, compute wind force.

        Args:
            dt: Simulation timestep (seconds)

        Returns:
            wind_force: Tensor [num_envs, 3] — force in world frame (N)
        """
        # --- Layer 2: OU process update ---
        # dv = theta * (0 - v) * dt + sigma * base_speed * sqrt(dt) * dW
        dW = torch.randn(self.num_envs, device=self.device)
        self.ou_state += (
            self.ou_theta * (0.0 - self.ou_state) * dt
            + self.ou_sigma * self.base_speed * (dt ** 0.5) * dW
        )

        # --- Layer 3: Gust events ---
        self._update_gusts(dt)

        # --- Compute effective wind speed ---
        # effective_speed = base_speed + ou_fluctuation, clamped >= 0
        effective_speed = torch.clamp(self.base_speed + self.ou_state, min=0.0)

        # Apply gust multiplier where active
        gust_factor = torch.where(
            self.gust_active,
            self.gust_multiplier,
            torch.ones_like(self.gust_multiplier)
        )
        effective_speed = effective_speed * gust_factor

        # --- Compute aerodynamic force ---
        # F = 0.5 * rho * Cd * A * v^2
        force_magnitude = 0.5 * self.rho * self.Cd * self.A * effective_speed ** 2
        # Clamp to physical limit (~body weight of 35kg robot = ~350N)
        force_magnitude = torch.clamp(force_magnitude, max=500.0)

        # Apply direction
        self.wind_force = self.base_direction * force_magnitude.unsqueeze(1)

        return self.wind_force

    def _update_gusts(self, dt):
        """Update gust state: decrement timers, randomly trigger new gusts."""
        # Decrement active gust timers
        self.gust_remaining_time -= dt
        expired = self.gust_remaining_time <= 0.0
        self.gust_active[expired] = False
        self.gust_remaining_time[expired] = 0.0

        # Trigger new gusts (only for envs without active gust)
        can_gust = ~self.gust_active
        roll = torch.rand(self.num_envs, device=self.device)
        new_gust = can_gust & (roll < self.gust_prob * dt)

        if new_gust.any():
            n_new = new_gust.sum().item()
            # Random duration
            self.gust_remaining_time[new_gust] = torch_rand_float(
                self.gust_duration_range[0],
                self.gust_duration_range[1],
                (n_new, 1), device=self.device
            ).squeeze(1)
            # Random multiplier
            self.gust_multiplier[new_gust] = torch_rand_float(
                self.gust_force_multiplier[0],
                self.gust_force_multiplier[1],
                (n_new, 1), device=self.device
            ).squeeze(1)
            self.gust_active[new_gust] = True

    def _sample_speed_per_env(self, curriculum_level, env_ids):
        """Sample wind speed per-env based on each env's own curriculum level.

        Args:
            curriculum_level: Tensor [num_envs] of per-env levels
            env_ids: Which envs to reset

        Returns:
            Tensor [len(env_ids)] of sampled wind speeds
        """
        levels = self.curriculum_levels_tensor  # precomputed [num_levels, 2]
        max_level = levels.shape[0] - 1

        lvl = curriculum_level[env_ids].long().clamp(max=max_level)
        speed_min = levels[lvl, 0]  # [n]
        speed_max = levels[lvl, 1]  # [n]

        # Uniform sample in [speed_min, speed_max] per env
        rand = torch.rand(len(env_ids), device=self.device)
        return speed_min + rand * (speed_max - speed_min)

    def get_wind_velocity(self):
        """Return current effective wind velocity vector per env [num_envs, 3].
        Useful for observations / privileged info.
        """
        effective_speed = torch.clamp(self.base_speed + self.ou_state, min=0.0)
        gust_factor = torch.where(
            self.gust_active,
            self.gust_multiplier,
            torch.ones_like(self.gust_multiplier)
        )
        effective_speed = effective_speed * gust_factor
        return self.base_direction * effective_speed.unsqueeze(1)
