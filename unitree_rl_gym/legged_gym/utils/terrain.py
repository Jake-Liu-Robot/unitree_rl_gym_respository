import numpy as np
from numpy.random import choice
from scipy import interpolate

from isaacgym import terrain_utils
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg


class Terrain:
    # 该函数在legged_robot.py被调用，主要功能是对设置地形的一些参数进行初始化
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:
        self.cfg = cfg
        ## 智能体的个数，获取来源是LeggedRobotCfg类下边的env结构体下边的num_envs
        self.num_robots = num_robots
        ## 赋值地形表示形式
        self.type = cfg.mesh_type
        ## 如果地形无指定或者指定为平面则无需进行之后的赋值
        if self.type in ["none", 'plane']:
            return
        ## 整个环境由一个又一个局部地形构成，局部地形的长宽进行赋值
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        ## 地形的累加比例，对于默认参数terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        ## self.proportions = [0.1, 0.2, 0.55, 0.8, 1.0]
        self.proportions = [np.sum(cfg.terrain_proportions[:i + 1]) for i in range(len(cfg.terrain_proportions))]

        ## 生成的地形块个数
        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        ## 创建一个描述环境的三维数组，数组的大小与地形块的行、列数相关
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        ## 根据水平方向的分辨率得到地形长宽方向上有多少个像素（相当于是栅格化了）
        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        ## 地形边界的像素个数
        self.border = int(cfg.border_size / self.cfg.horizontal_scale)
        ## 最终的综合大环境列像素个数，两侧都有边界
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        ## 最终的综合大环境行像素个数，两侧都有边界
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        ## 初始化高度阈二维数组，数组的维度与行列大小有关
        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)
        ## 如果课程变量是true则执行curiculum函数
        if cfg.curriculum:
            self.curiculum()
        elif cfg.selected:
            ## 该函数基于terrain_kwargs地形类型字典值生成单一地形
            self.selected_terrain()
        else:
            ## 难度和选择完全随机生成地形
            self.randomized_terrain()

        self.heightsamples = self.height_field_raw
        if self.type == "trimesh":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(self.height_field_raw,
                                                                                         self.cfg.horizontal_scale,
                                                                                            self.cfg.vertical_scale,
                                                                                            self.cfg.slope_treshold)
    
    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([0.5, 0.75, 0.9])
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)

    # 该函数在utils / terrain.py中__init__函数被调用，主要功能是按照初始化参数要求生成不同需求的地形。
    def curiculum(self):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                ## 行序号越大difficulty难度越高，难度顾名思义就是地形通行度变差
                difficulty = i / self.cfg.num_rows
                ## 列序号越大choice选择越大，choice结合累加比例选择地形类型。
                choice = j / self.cfg.num_cols + 0.001

                terrain = self.make_terrain(choice, difficulty)
                ## 将生成的地形加入至大环境中，最终储存在self.env_origins数组中，该函数不展开解释
                self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        terrain_type = self.cfg.terrain_kwargs.pop('type')
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain("terrain",
                              width=self.width_per_env_pixels,
                              length=self.width_per_env_pixels,
                              vertical_scale=self.vertical_scale,
                              horizontal_scale=self.horizontal_scale)

            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
            self.add_terrain_to_map(terrain, i, j)

    # 该函数在utils / terrain.py中curiculum函数被调用，主要功能是在issacgym生成地形，其中的terrain_utils为issacgym提供的api。
    def make_terrain(self, choice, difficulty):
        ## 创建一个局部地形，参数包括地形的名字，地形的像素长宽以及分辨率
        terrain = terrain_utils.SubTerrain("terrain",
                                           width=self.width_per_env_pixels,
                                           length=self.width_per_env_pixels,
                                           vertical_scale=self.cfg.vertical_scale,
                                           horizontal_scale=self.cfg.horizontal_scale)
        ## slope坡度与难度正相关
        slope = difficulty * 0.4
        ## 台阶高度也与难度正相关
        step_height = 0.05 + 0.18 * difficulty
        ## 离散障碍物的高度也也与难度正相关
        discrete_obstacles_height = 0.05 + difficulty * 0.2
        ## 石头的大小和难度正相关
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        ## 设定石头之间的距离
        stone_distance = 0.05 if difficulty == 0 else 0.1
        ## 沟的尺寸和难度正相关
        gap_size = 1. * difficulty
        ## 坑的尺寸和难度正相关
        pit_depth = 1. * difficulty
        ## 如果落在第一区间则生成平坦坡
        if choice < self.proportions[0]:
            ## 在第一区间的前一半生成下坡
            if choice < self.proportions[0] / 2:
                slope *= -1
            ## issacgym提供的api就不展开了，其中第一个参数是地形的名称，第二个参数是坡度，第三个参数是在地形的中间生成一个平台，
            ## 让机器人初始化的时候落在上边的，这里设置为3m
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
        ## 如果落在第二区间则生成崎岖坡，相较于第一区间是在坡道基础上叠加了随机地形，后续均为生成各类典型地形
        elif choice < self.proportions[1]:
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005,
                                                 downsampled_scale=0.2)
        elif choice < self.proportions[3]:
            if choice < self.proportions[2]:
                step_height *= -1
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
        elif choice < self.proportions[4]:
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size,
                                                     rectangle_max_size, num_rectangles, platform_size=3.)
        elif choice < self.proportions[5]:
            terrain_utils.stepping_stones_terrain(terrain, stone_size=stepping_stones_size,
                                                  stone_distance=stone_distance,
                                                  max_height=0., platform_size=4.)
        elif choice < self.proportions[6]:
            gap_terrain(terrain, gap_size=gap_size, platform_size=3.)
        else:
            pit_terrain(terrain, depth=pit_depth, platform_size=4.)

        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x = (i + 0.5) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((self.env_length/2. - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length/2. + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width/2. - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width/2. + 1) / terrain.horizontal_scale)
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

def gap_terrain(terrain, gap_size, platform_size=1.):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size
   
    terrain.height_field_raw[center_x-x2 : center_x + x2, center_y-y2 : center_y + y2] = -1000
    terrain.height_field_raw[center_x-x1 : center_x + x1, center_y-y1 : center_y + y1] = 0

def pit_terrain(terrain, depth, platform_size=1.):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth
