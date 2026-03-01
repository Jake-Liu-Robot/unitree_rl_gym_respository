from legged_gym.envs.g1.g1_config import G1RoughCfg, G1RoughCfgPPO


class G1WindRoughCfg(G1RoughCfg):
    """G1 config extended with wind disturbance, curriculum, and wind-specific rewards."""

    class env(G1RoughCfg.env):
        num_observations = 47       # same as G1: ang_vel(3)+gravity(3)+cmd(3)+dof_pos(12)+dof_vel(12)+actions(12)+phase(2)
        num_privileged_obs = 56     # G1 privileged(50) + wind_velocity(3) + wind_force(3)
        num_actions = 12
        episode_length_s = 20

    class wind:
        """3-layer wind model parameters."""
        enable = True

        # Physical constants
        air_density = 1.225         # kg/m^3 (sea level)
        drag_coefficient = 1.0     # dimensionless (approximation for humanoid)
        frontal_area = 0.5          # m^2 (approximate for G1)

        # OU process parameters (Layer 2)
        ou_theta = 0.5              # mean reversion rate (1/s)
        ou_sigma = 0.3              # noise intensity (fraction of base speed)

        # Gust parameters (Layer 3)
        gust_prob = 0.3             # probability per second of gust onset
        gust_force_multiplier = [2.0, 3.0]  # multiplier range
        gust_duration_range = [1.0, 2.0]     # seconds

        # Curriculum levels: (min_speed_m/s, max_speed_m/s)
        # Level 0: no wind
        # Level 1: light constant
        # Level 2: light variable
        # Level 3: medium + variable
        # Level 4: strong + gusts
        # Level 5: extreme
        curriculum_levels = [
            [0.0, 0.0],      # Level 0: no wind
            [1.0, 3.0],      # Level 1: light
            [2.0, 5.0],      # Level 2: light-medium
            [4.0, 8.0],      # Level 3: medium
            [7.0, 12.0],     # Level 4: strong
            [10.0, 18.0],    # Level 5: extreme
        ]

        # Curriculum advancement thresholds
        curriculum_start_level = 0
        survival_threshold = 0.8       # fraction of max episode length
        tracking_threshold = 0.6       # fraction of max tracking reward
        upgrade_window = 200           # num resets to evaluate before level-up

        # For play/test: override starting level (None = use curriculum_start_level)
        play_level = 2

    class domain_rand(G1RoughCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
        # Disable push_robots — wind replaces it as continuous perturbation
        push_robots = False

    class rewards(G1RoughCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.78
        only_positive_rewards = True
        tracking_sigma = 0.25

        class scales(G1RoughCfg.rewards.scales):
            # --- Base walking (inherited, adjusted) ---
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -1.0
            base_height = -10.0

            # --- Energy / smoothness ---
            dof_acc = -2.5e-7
            dof_vel = -1e-3
            action_rate = -0.01
            dof_pos_limits = -5.0

            # --- G1 humanoid-specific (inherited) ---
            alive = 0.15
            hip_pos = -1.0
            contact_no_vel = -0.2
            feet_swing_height = -20.0
            contact = 0.18

            # --- Wind-specific rewards ---
            wind_stability = -0.5       # penalize CoM velocity error under wind
            lean_compensation = 0.2     # reward leaning into wind
            sustained_walking = 0.1     # reward for not falling
            contact_symmetry = 0.05     # reward symmetric L/R foot contact


class G1WindRoughCfgPPO(G1RoughCfgPPO):
    """PPO config for wind-robust G1 training."""

    class policy(G1RoughCfgPPO.policy):
        init_noise_std = 0.8
        actor_hidden_dims = [64, 32]    # slightly larger for wind complexity
        critic_hidden_dims = [64, 32]
        activation = 'elu'
        # LSTM helps with wind estimation from observation history
        rnn_type = 'lstm'
        rnn_hidden_size = 64
        rnn_num_layers = 1

    class algorithm(G1RoughCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(G1RoughCfgPPO.runner):
        policy_class_name = "ActorCriticRecurrent"
        max_iterations = 15000      # more iterations for curriculum training
        run_name = ''
        experiment_name = 'g1_wind'
