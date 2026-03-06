from legged_gym.envs.g1.g1_config import G1RoughCfg, G1RoughCfgPPO


class G1StairCfg(G1RoughCfg):
    """G1 stair-walking configuration.

    Inherits all robot, control, and asset settings from G1RoughCfg and
    overrides terrain, rewards, commands, and domain randomisation for
    stair-climbing training.

    OPTION A — BLIND CLIMBING (ACTIVE)
    -----------------------------------
    The robot uses only 47-dim proprioceptive observations (no height map).
    It learns to climb stairs purely via proprioceptive feedback: joint
    positions, velocities, and IMU signals fed into the LSTM.  The curriculum
    shapes a high-clearance conservative gait over 7000–10000 iterations.

    To enable height-map perception (Option B) later:
      * Create a G1StairRobot subclass with _measure_heights(),
        compute_observations() (234-dim), and _get_noise_scale_vec().
      * Set num_observations = 234, num_privileged_obs = 237.
      * Register G1StairRobot in __init__.py for g1_stair.

    IMPORTANT — STEP WIDTH
    ----------------------
    The step_width used in terrain_utils.pyramid_stairs_terrain() is
    hardcoded to 0.31 m inside terrain.py:make_terrain().  There is no
    config knob for it — edit terrain.py directly to use 0.25–0.35 m.
    Step height is controlled by curriculum difficulty:
        step_height = 0.05 + 0.18 * difficulty  ->  0.05 m ... 0.23 m
    The requested maximum of 0.18 m corresponds to difficulty ~0.72,
    which is reached around terrain row 7 of 10.

    IMPORTANT — TERRAIN PROPORTIONS MAPPING
    ----------------------------------------
    make_terrain() interprets the 5-element terrain_proportions list as
    cumulative thresholds for:
        [smooth_slope, rough_slope, stairs_down, stairs_up, discrete]
    There is no explicit "flat" terrain type; the 10 % rough_slope slot
    approximates the requested 10 % flat allocation (rough slope at low
    curriculum difficulty is nearly flat).
    """

    # ------------------------------------------------------------------ #
    #  Terrain                                                             #
    # ------------------------------------------------------------------ #
    class terrain(G1RoughCfg.terrain):
        mesh_type        = 'trimesh'   # converts height-field -> triangle mesh
        curriculum       = True        # difficulty increases with terrain row
        measure_heights  = True        # infrastructure flag; env override needed

        # 17 x 11 = 187 scan points — forward-biased 1.6 m x 1.1 m rectangle
        # centred on the robot.  Only active once env-side code is added.
        measured_points_x = [
            -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,
             0.0,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,
        ]  # 17 points
        measured_points_y = [
            -0.5, -0.4, -0.3, -0.2, -0.1,
             0.0,  0.1,  0.2,  0.3,  0.4,  0.5,
        ]  # 11 points

        num_rows = 10    # difficulty levels  (row 0 = easiest, row 9 = hardest)
        num_cols = 20    # terrain-type columns per difficulty level

        terrain_length = 8.0   # [m] per tile
        terrain_width  = 8.0   # [m] per tile

        static_friction  = 1.0
        dynamic_friction = 1.0
        restitution      = 0.0

        horizontal_scale = 0.1    # [m / pixel] height-field resolution
        vertical_scale   = 0.005  # [m / pixel] height quantisation step
        border_size      = 25     # [m] flat border around the full terrain grid
        slope_treshold   = 0.75   # faces steeper than this become vertical walls

        # Curriculum: robots start in the easier half of the grid.
        max_init_terrain_level = 5

        # Proportion mapping (must sum to 1.0):
        #   index  terrain type       cumulative   columns (of 20)
        #   -----  -----------------  ----------   ---------------
        #     0    smooth_slope          0.10          0-1
        #     1    rough_slope           0.20          2-3
        #     2    stairs_down           0.60          4-11  (40 %)
        #     3    stairs_up             1.00          12-19 (40 %)
        #     4    discrete_obstacles    1.00          -- (none)
        #
        # discrete = 0.00 ensures self.proportions[4] = 1.0, so the
        # "stepping stones / gap / pit" branches in make_terrain() are
        # unreachable and their self.proportions[5/6] index accesses
        # are never executed.
        terrain_proportions = [0.10, 0.10, 0.40, 0.40, 0.00]

    # ------------------------------------------------------------------ #
    #  Observation / action space                                          #
    # ------------------------------------------------------------------ #
    class env(G1RoughCfg.env):
        # 47-dim proprioceptive only (blind climbing — Option A)
        num_observations   = 47
        # critic: 47 + 3 base_lin_vel = 50
        num_privileged_obs = 50
        num_actions        = 12

    # ------------------------------------------------------------------ #
    #  Control (PD gains)                                                  #
    # ------------------------------------------------------------------ #
    class control(G1RoughCfg.control):
        stiffness = {'hip_yaw': 120,
                     'hip_roll': 120,
                     'hip_pitch': 150,
                     'knee': 200,
                     'ankle': 60,
                     }
        damping = {'hip_yaw': 5,
                   'hip_roll': 5,
                   'hip_pitch': 5,
                   'knee': 8,
                   'ankle': 4,
                   }

    # ------------------------------------------------------------------ #
    #  Velocity commands                                                   #
    # ------------------------------------------------------------------ #
    class commands(G1RoughCfg.commands):
        class ranges(G1RoughCfg.commands.ranges):
            lin_vel_x   = [-0.5, 1.0]    # forward-biased; limited reverse on stairs
            lin_vel_y   = [-0.3, 0.3]    # restricted lateral movement
            ang_vel_yaw = [-0.5, 0.5]    # limited turning on stairs
            heading     = [-3.14, 3.14]

    # ------------------------------------------------------------------ #
    #  Domain randomisation                                                #
    # ------------------------------------------------------------------ #
    class domain_rand(G1RoughCfg.domain_rand):
        randomize_friction  = True
        friction_range      = [0.5, 1.25]
        randomize_base_mass = True
        added_mass_range    = [-1., 3.]
        push_robots         = True
        push_interval_s     = 15
        max_push_vel_xy     = 1.0

    # ------------------------------------------------------------------ #
    #  Rewards                                                             #
    # ------------------------------------------------------------------ #
    class rewards(G1RoughCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.78

        # Allow negative total reward during early stair training.
        # With only_positive_rewards=True (base default), the large stair
        # penalties would all be clipped to zero, removing the learning signal.
        only_positive_rewards = False

        class scales(G1RoughCfg.rewards.scales):
            # -- velocity tracking ------------------------------------------------
            tracking_lin_vel = 2.0
            tracking_ang_vel = 1.0

            # -- stability --------------------------------------------------------
            # lin_vel_z reduced vs. flat config (-2.0): some vertical CoM
            # movement is expected and unavoidable when climbing steps.
            lin_vel_z   = -1.0
            ang_vel_xy  = -0.05
            orientation = -0.5

            # -- stepping behaviour -----------------------------------------------
            # feet_air_time: encourages deliberate foot lift on each step.
            # Reward = (air_time - 0.5 s) on first contact; only non-zero
            # when a velocity command is active (norm > 0.1 m/s).
            feet_air_time = 0.5

            # contact: gait-clock phase-matching reward.  Set to 0 because
            # the fixed 0.8 s symmetric gait period is less applicable to
            # variable-cadence stair climbing.
            contact = 0.0

            # -- energy / smoothness ----------------------------------------------
            torques     = -0.00005
            dof_acc     = -2.5e-7
            action_rate = -0.01
            dof_vel     = -1e-3

            # -- safety -----------------------------------------------------------
            # collision: penalise hip/knee contacts (body defined in asset cfg).
            collision   = -1.0

            # stumble: penalise large horizontal foot forces relative to
            # vertical -- exactly what happens when a foot catches a stair edge.
            stumble     = -0.2

            # stand_still: penalise joint deviation from default pose when the
            # commanded velocity is near zero (prevents policy from freezing).
            stand_still = -0.1

            # -- G1-specific (inherited; listed for readability) ------------------
            alive             =  0.5
            hip_pos           = -0.3
            contact_no_vel    = -0.2
            feet_swing_height = -2.0
            dof_pos_limits    = -2.0

            # -- disabled ---------------------------------------------------------
            # base_height: G1 pelvis height changes continuously as the robot
            # ascends/descends stairs, so penalising deviation from a fixed
            # 0.78 m target would conflict with the terrain curriculum.
            base_height = 0.0


class G1StairCfgPPO(G1RoughCfgPPO):
    """PPO training config for g1_stair.

    Keeps the LSTM + small-MLP architecture from G1RoughCfgPPO.
    Only overrides the training schedule and experiment name.
    """

    class policy(G1RoughCfgPPO.policy):
        init_noise_std     = 0.8
        actor_hidden_dims  = [256, 128]
        critic_hidden_dims = [256, 128]
        activation         = 'elu'
        # LSTM frontend (matches G1RoughCfgPPO / ActorCriticRecurrent)
        rnn_type           = 'lstm'
        rnn_hidden_size    = 128
        rnn_num_layers     = 1

    class algorithm(G1RoughCfgPPO.algorithm):
        entropy_coef        = 0.01
        num_learning_epochs = 5
        num_mini_batches    = 4

    class runner(G1RoughCfgPPO.runner):
        policy_class_name = 'ActorCriticRecurrent'
        max_iterations    = 5000
        run_name          = ''
        experiment_name   = 'g1_stair'
