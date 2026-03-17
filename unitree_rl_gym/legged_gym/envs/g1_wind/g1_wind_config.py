from legged_gym.envs.g1.g1_config import G1RoughCfg, G1RoughCfgPPO


class G1WindRoughCfg(G1RoughCfg):
    """G1 config extended with wind disturbance, curriculum, and wind-specific rewards."""

    class env(G1RoughCfg.env):
        num_observations = 47       # same as G1: ang_vel(3)+gravity(3)+cmd(3)+dof_pos(12)+dof_vel(12)+actions(12)+phase(2)
        num_privileged_obs = 56     # G1 privileged(50) + wind_velocity(3) + wind_force(3)
        num_actions = 12
        episode_length_s = 20
        # Privileged obs normalization scales for wind channels
        # wind_vel ~[0,28] m/s at L5 → /10 keeps in [-3, 3]
        # wind_force ~[0,150] N at L5 → /100 keeps in [-1.5, 1.5]
        priv_obs_wind_vel_scale = 0.1
        priv_obs_wind_force_scale = 0.01

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

        # Feet separation threshold for feet_distance reward (meters)
        feet_min_separation = 0.20  # matches G1 hip width

        # Nominal robot mass for base_acc wind compensation (kg)
        robot_nominal_mass = 35.0   # URDF 32.68 + domain rand center

        # Wind-adaptive gait parameters
        swing_height_base = 0.08           # base swing height target (m)
        swing_height_wind_reduction = 0.5  # max fraction reduction at peak wind (0.08→0.04m)
        stance_ratio_base = 0.55           # base stance phase ratio
        stance_ratio_wind_increase = 0.20  # max increase at peak wind (0.55→0.75)

        # Enable wind-adaptive reward modifiers (orientation, base_height, contact, swing_height)
        # Set False in ablation experiments to isolate wind-specific reward contributions
        wind_adaptive_rewards = True

        # Wind-adaptive orientation: scale at which orientation penalty halves (N)
        # 1/(1 + F/F_ref): at 50N → 50%, at 100N → 33%, smooth & never fully zero
        orientation_wind_scale = 50.0

        # Wind-adaptive base_height: allow crouching under wind to lower COM
        base_height_wind_reduction = 0.05  # max fraction reduction (0.78→0.741m at 150N)

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
        survival_threshold = 0.7       # fraction of max episode length (relaxed from 0.8)
        tracking_threshold = 0.4       # fraction of max tracking reward (relaxed from 0.6)
        upgrade_window = 300           # num resets to evaluate before level-up (more reliable stats)
        # Demotion thresholds — drop a level if performance falls too low
        demotion_survival_threshold = 0.4   # demote if survival below this (was 0.3, demote earlier)
        demotion_tracking_threshold = 0.3   # AND tracking below this (was 0.2)
        # Mixed-level upgrade: fraction of envs that advance (rest stay for forgetting prevention)
        upgrade_fraction = 0.8
        # Fractional demotion: only demote this fraction of envs (prevents oscillation)
        demotion_fraction = 0.5
        # Cooldown: skip this many resets after level change before evaluating again
        upgrade_cooldown = 500

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

        # Action delay randomization (sim-to-real: computation + communication latency)
        # Delay in control steps; each control step = sim_dt × decimation = 0.02s (50 Hz)
        # 0-2 steps = 0-40ms, matching typical real-world latency
        # Ref: Walk These Ways (Margolis 2023), Rapid Locomotion (Margolis 2022)
        randomize_action_delay = True
        action_delay_range = [0, 2]     # inclusive [min, max] in control steps

        # PD gain randomization (actuator model mismatch)
        # Real motors have ±10-30% variation from nominal stiffness/damping
        # Ref: ANYmal-DroQ, Walk These Ways
        randomize_pd_gains = True
        stiffness_multiplier_range = [0.8, 1.2]    # ±20%
        damping_multiplier_range = [0.8, 1.2]      # ±20%

        # Motor strength randomization (torque limit variation)
        # Models motor degradation, voltage sag under load, thermal derating
        # Asymmetric: can only be weaker than nominal (conservative)
        # Ref: Rapid Locomotion, Booster Gym 2025
        randomize_motor_strength = True
        motor_strength_range = [0.8, 1.0]           # 80-100% of nominal

    class rewards(G1RoughCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.78
        only_positive_rewards = False  # allow negative reward for gradient under strong wind
        tracking_sigma = 0.25
        # Wind-adaptive tracking: sigma_eff = sigma * (1 + scale * |F_wind| / 100N)
        # At 100N wind force, sigma doubles (0.25 → 0.50), maintaining gradient
        # Ref: Xu et al. 2025 (multi-objective RL for force compliance)
        tracking_sigma_wind_scale = 1.0

        class scales(G1RoughCfg.rewards.scales):
            # --- Base walking (inherited, adjusted) ---
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.15          # was -0.2→-0.15: slightly relax for gust angular response
            orientation = -0.3          # wind-aware override in env: smooth decay 1/(1+F/50N)
                                        # no-wind: full penalty; 100N: 33% penalty
            base_height = -10.0         # wind-aware override in env: target lowers under wind

            # --- Energy / smoothness ---
            dof_acc = -1.5e-7           # was -2.5e-7→-1.5e-7: allow faster joint response for wind rejection
            dof_vel = -5e-4
            action_rate = -0.01
            action_rate2 = -0.005       # second-order action smoothness ||a_t - 2*a_{t-1} + a_{t-2}||²
                                        # Ref: Humanoid-Gym 2024
            power = -2e-4               # was -5e-4→-2e-4: allow more aggressive wind rejection torques
                                        # Ref: Booster Gym 2025 (sum(max(tau * dq, 0)))
            dof_pos_limits = -5.0

            # --- G1 humanoid-specific (inherited) ---
            alive = 1.0                 # was 0.5→1.0: stronger survival incentive
                                        # Critical with only_positive_rewards=False to prevent
                                        # "die quickly to avoid negative penalty accumulation"
            hip_pos = -0.3
            contact_no_vel = -0.1       # was -0.2→-0.1: wind causes unavoidable foot sliding
            feet_swing_height = -8.0
            contact = 0.18

            # --- Wind-specific rewards ---
            lean_compensation = 0.8
            feet_distance = -0.5        # was -0.3→-0.5: stronger wide stance incentive for stability
                                        # Ref: Booster Gym 2025 (max(d_ref - d_feet, 0))
            base_acc = -0.002           # wind-compensated base acceleration penalty
                                        # Ref: Viereck et al. 2024
            ang_momentum_change = 0.0   # DISABLED: scale too small (-0.0003) for meaningful gradient,
                                        # adds noise; ang_vel_xy already penalizes angular velocity
            com_balance = 0.0           # DISABLED: fundamentally conflicts with lean_compensation
                                        # alive + base_height implicitly enforce balance
                                        # Ref: Booster Gym 2025, Walk These Ways — no explicit COM reward

            sustained_walking = 0.0     # disabled: identical to alive reward
            contact_symmetry = 0.0      # disabled: incentivized standing still


class G1WindRoughCfgPPO(G1RoughCfgPPO):
    """PPO config for wind-robust G1 training."""

    class policy(G1RoughCfgPPO.policy):
        init_noise_std = 0.8
        # Asymmetric actor-critic design (literature-conforming):
        #   Actor:  LSTM(128) → MLP[128,64] → 12   (LSTM-heavy: wind estimation from history)
        #   Critic: LSTM(128) → MLP[256,128] → 1    (MLP-heavy: privileged wind info processing)
        # Total: ~250K params (Actor ~110K + Critic ~140K)
        # Ref: Humanoid-Gym 2024 (LSTM 64, MLP [32]), Walk These Ways (MLP [128,64,32])
        #      Upper-end of literature range due to wind OU estimation requirement
        actor_hidden_dims = [128, 64]       # match LSTM output dim, standard for locomotion
        critic_hidden_dims = [256, 128]     # larger MLP for privileged wind_vel/wind_force
        activation = 'elu'
        rnn_type = 'lstm'
        rnn_hidden_size = 128       # literature standard (64-128); 128 for wind OU estimation
        rnn_num_layers = 1          # (wind OU τ=2-20s via BPTT over 64 steps)

    class algorithm(G1RoughCfgPPO.algorithm):
        entropy_coef = 0.008
        gamma = 0.995               # long horizon for 20s episodes under curriculum
        learning_rate = 5e-4        # standard for ~250K params (Humanoid-Gym, Walk These Ways)
        desired_kl = 0.008          # conservative updates during curriculum transitions
        num_mini_batches = 4        # standard mini-batch count (4096 envs × 64 steps / 4 = 65K per mini-batch)

    class runner(G1RoughCfgPPO.runner):
        policy_class_name = "ActorCriticRecurrent"
        max_iterations = 10000      # ~250K params converges faster; sufficient for 6-level curriculum
        num_steps_per_env = 64      # longer rollouts for LSTM temporal context
                                    # 64 steps × 0.02s/step = 1.28s (sim.dt=0.005 × decimation=4)
        run_name = ''
        experiment_name = 'g1_wind'


# ============================================================
# Experiment variants (ablation studies)
# ============================================================

class G1WindBaselineCfg(G1WindRoughCfg):
    """Exp1: No-wind baseline. Same architecture/rewards/DR as Run8, only wind disabled."""

    class env(G1WindRoughCfg.env):
        num_observations = 47
        num_privileged_obs = 50     # no wind channels (was 56)
        num_actions = 12
        episode_length_s = 20

    class wind(G1WindRoughCfg.wind):
        enable = False


class G1WindBaselineCfgPPO(G1WindRoughCfgPPO):
    """PPO config for baseline (identical to wind-trained)."""

    class runner(G1WindRoughCfgPPO.runner):
        experiment_name = 'g1_wind_baseline'
        max_iterations = 3000


class G1WindPushOnlyCfg(G1WindRoughCfg):
    """Exp2: Push-only perturbation (traditional baseline). No wind, standard velocity
    impulse push instead. Tests whether generic impulse robustness transfers to wind.
    Push params from G1 default config (g1_config.py), consistent with sim-to-real
    locomotion literature (Walk These Ways, ANYmal, legged_gym)."""

    class env(G1WindRoughCfg.env):
        num_observations = 47
        num_privileged_obs = 50     # no wind channels (was 56)
        num_actions = 12
        episode_length_s = 20

    class wind(G1WindRoughCfg.wind):
        enable = False

    class domain_rand(G1WindRoughCfg.domain_rand):
        # Enable push perturbation (G1 default parameters)
        push_robots = True
        push_interval_s = 5        # push every 5 seconds
        max_push_vel_xy = 1.5      # m/s (G1 default, ~52.5 N·s impulse on 35kg robot)
        # Keep all other domain randomization identical to Exp3
        randomize_friction = True
        friction_range = [0.1, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
        randomize_action_delay = True
        action_delay_range = [0, 2]
        randomize_pd_gains = True
        stiffness_multiplier_range = [0.8, 1.2]
        damping_multiplier_range = [0.8, 1.2]
        randomize_motor_strength = True
        motor_strength_range = [0.8, 1.0]


class G1WindPushOnlyCfgPPO(G1WindRoughCfgPPO):
    """PPO config for push-only baseline."""

    class runner(G1WindRoughCfgPPO.runner):
        experiment_name = 'g1_wind_push_only'
        max_iterations = 3000


class G1WindNoCurriculumCfg(G1WindRoughCfg):
    """Exp4: Wind ON but fixed at L3 (medium, 4-8 m/s). No curriculum advancement."""

    class wind(G1WindRoughCfg.wind):
        curriculum_start_level = 3
        upgrade_window = 999999     # disable auto-advancement
        demotion_fraction = 0.0     # disable demotion


class G1WindNoCurriculumCfgPPO(G1WindRoughCfgPPO):
    """PPO config for no-curriculum ablation."""

    class runner(G1WindRoughCfgPPO.runner):
        experiment_name = 'g1_wind_no_curriculum'
        max_iterations = 3000


class G1WindNoRewardCfg(G1WindRoughCfg):
    """Exp5: Wind ON + curriculum ON, but wind-specific rewards disabled.
    Tests whether base rewards alone can learn wind robustness."""

    class wind(G1WindRoughCfg.wind):
        # Disable wind-adaptive modifiers in overridden reward functions
        # (orientation, base_height, contact, feet_swing_height)
        wind_adaptive_rewards = False

    class rewards(G1WindRoughCfg.rewards):
        # Disable wind-adaptive tracking sigma
        tracking_sigma_wind_scale = 0.0

        class scales(G1WindRoughCfg.rewards.scales):
            # Zero out all wind-specific rewards
            lean_compensation = 0.0
            feet_distance = 0.0
            base_acc = 0.0
            action_rate2 = 0.0

            # Revert wind-tuned scales to G1 base values
            orientation = -1.0          # was -0.3 (wind-reduced)
            ang_vel_xy = -0.05          # was -0.15 (wind-relaxed)
            dof_acc = -2.5e-7           # was -1.5e-7 (wind-relaxed)
            contact_no_vel = -0.2       # was -0.1 (wind-relaxed)
            power = -5e-4              # was -2e-4 (wind-relaxed)
            alive = 0.5                 # was 1.0 (wind-boosted)
            feet_swing_height = -20.0   # was -8.0 (wind-reduced)
            hip_pos = -1.0              # was -0.3 (wind-relaxed)


class G1WindNoRewardCfgPPO(G1WindRoughCfgPPO):
    """PPO config for no-wind-reward ablation."""

    class runner(G1WindRoughCfgPPO.runner):
        experiment_name = 'g1_wind_no_reward'
        max_iterations = 3000
