from legged_gym.envs.g1.g1_config import G1RoughCfg, G1RoughCfgPPO


class G1WindRoughCfg(G1RoughCfg):
    """G1 config extended with wind disturbance, curriculum, and wind-specific rewards."""

    class env(G1RoughCfg.env):
        num_observations = 47       # same as G1: ang_vel(3)+gravity(3)+cmd(3)+dof_pos(12)+dof_vel(12)+actions(12)+phase(2)
        num_privileged_obs = 56     # G1 privileged(50) + wind_velocity(3) + wind_force(3)
        num_actions = 12
        episode_length_s = 20

    class wind:
        """3-layer wind model parameters (v3)."""
        enable = True

        # Physical constants
        air_density = 1.225         # kg/m^3 (sea level)
        drag_coefficient = 1.1     # dimensionless (humanoid bluff body, literature 1.0-1.3)

        # P1: Direction-dependent projected area (elliptical projection)
        # A_eff = sqrt((A_front·cosθ)² + (A_side·sinθ)²)
        # Guarantees A_side ≤ A_eff ≤ A_front for all angles
        frontal_area_front = 0.50   # m² — front/back facing (wide side)
        frontal_area_side = 0.22    # m² — side facing (narrow side)
        frontal_area_top = 0.10     # m² — top/bottom facing (width × depth ≈ 0.45×0.20)

        # P2: Height-dependent wind profile (atmospheric boundary layer)
        # v(z) = v_ref × (z / z_ref)^α  (power law)
        height_profile_enabled = True
        height_exponent = 0.28      # α for urban/suburban terrain (0.14=open, 0.28=urban, 0.40=dense city)
        reference_height = 0.85     # m (approx pelvis height, wind speed reference point)
        min_height_ratio = 0.05     # clamp min (z/z_ref) to avoid zero

        # P3: Center of pressure offset from COM along body-local z-axis (m)
        # Positive = CoP above COM (creates tipping torque for horizontal wind)
        # pelvis: represents 55% upper body; wind centroid extends above pelvis COM
        #   URDF COM at z=-0.076m, but effective CoP for upper body is ~+0.10m above COM
        # thighs: URDF COM at z=-0.151m ≈ geometric center → CoP-COM ≈ 0
        # shins:  URDF COM at z=-0.121m ≈ geometric center → CoP-COM ≈ 0
        cop_z_offsets = [0.10, 0.005, 0.005, 0.002, 0.002]  # from URDF inertial data

        # Layer 2: OU speed process (defaults, used when ou_randomize=False)
        ou_theta = 0.5              # mean reversion rate (1/s), τ=2s
        ou_sigma = 0.18             # noise intensity (fraction of base speed)
        ou_sigma_min = 0.1          # minimum noise floor (m/s), active even at level 0
        ou_sigma_scale_per_level = 0.15  # sigma multiplier per curriculum level

        # Layer 2: OU direction process (defaults, used when ou_randomize=False)
        ou_theta_dir = 0.2          # mean reversion rate (1/s), τ=5s (slower than speed)
        ou_sigma_dir = 0.15         # noise intensity (rad/s scale), steady-state std ≈ ±14°

        # Layer 2: Per-episode OU parameter randomization
        ou_randomize = True         # enable per-episode sampling of OU params
        ou_theta_range = [0.2, 1.0]         # [slow drift, fast revert]
        ou_sigma_range = [0.05, 0.25]       # [calm, turbulent] (capped to keep TI ≈ 0.3-0.4)
        ou_theta_dir_range = [0.05, 0.5]    # [drifty direction, locked direction]
        ou_sigma_dir_range = [0.02, 0.25]   # [steady heading, erratic heading]

        # Layer 3: Gust events (independent force vector)
        gust_prob = 0.1             # probability per second (avg interval ~10s)
        gust_speed_range = [2.0, 6.0]       # independent gust wind speed (m/s)
        gust_angle_range = [-1.047, 1.047]  # direction offset from base (±60° = ±π/3)
        gust_duration_range = [1.5, 3.0]    # seconds (longer to accommodate envelope)
        gust_ramp_up = 0.3          # envelope rise time (s)
        gust_ramp_down = 0.5        # envelope decay time (s), asymmetric: slower decay
        gust_speed_scale_per_level = 0.2   # gust speed multiplier per curriculum level
        gust_prob_scale_per_level = 0.1    # gust probability multiplier per curriculum level

        # Per-level wind SPEED clamp (m/s) — prevents OU+gust producing unrealistic speeds
        # Set to ~1.6× base range upper bound (realistic 3-second gust factor)
        # L0 = 0: ensures truly zero wind at baseline level
        speed_clamp_per_level = [0.0, 5.0, 8.0, 13.0, 20.0, 28.0]

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
        # Demotion thresholds — drop a level if performance falls too low
        demotion_survival_threshold = 0.3   # demote if survival below this
        demotion_tracking_threshold = 0.2   # AND tracking below this

        # Per-level force clamp (N) — prevents unlearnable episodes from extreme OU
        # Computed from: F = 0.5*ρ*Cd*A*v², then rounded up with headroom
        # L0=5N (micro-perturbation), L5=150N (~44% body weight for 35kg robot)
        force_clamp_per_level = [5.0, 15.0, 30.0, 60.0, 100.0, 150.0]

        # Multi-body force distribution: body names and area fractions
        # Total force is distributed proportionally (fractions sum to ~0.95,
        # remaining 5% is unmodeled: arms, head)
        force_body_names = [
            "pelvis",              # torso — largest exposed area
            "left_hip_yaw_link",   # left thigh
            "right_hip_yaw_link",  # right thigh
            "left_knee_link",      # left shin
            "right_knee_link",     # right shin
        ]
        force_body_fractions = [0.55, 0.12, 0.12, 0.08, 0.08]

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
        only_positive_rewards = False  # allow negative reward for gradient under strong wind
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
            alive = 0.5   # boosted: stronger survival signal with only_positive_rewards=False
            hip_pos = -1.0
            contact_no_vel = -0.2
            feet_swing_height = -20.0
            contact = 0.18

            # --- Wind-specific rewards ---
            # wind_stability removed: duplicated tracking_lin_vel with unbounded penalty
            lean_compensation = 0.3     # reward leaning into wind
            sustained_walking = 0.0     # disabled: identical to alive reward
            contact_symmetry = 0.0      # disabled: incentivized standing still


class G1WindRoughCfgPPO(G1RoughCfgPPO):
    """PPO config for wind-robust G1 training."""

    class policy(G1RoughCfgPPO.policy):
        init_noise_std = 0.8
        actor_hidden_dims = [128, 64]   # larger capacity for wind estimation
        critic_hidden_dims = [128, 64]
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
