import os
from datetime import datetime
from typing import Tuple
import torch
import numpy as np
import sys

from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .helpers import get_args, update_cfg_from_args, class_to_dict, get_load_path, set_seed, parse_sim_params
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class TaskRegistry():
    def __init__(self):
        self.task_classes = {}
        self.env_cfgs = {}
        self.train_cfgs = {}

    ## 根据指定的任务类型，将训练的地形环境、机器人类型、强化学习算法参数进行读取
    def register(self, name: str, task_class: VecEnv, env_cfg: LeggedRobotCfg, train_cfg: LeggedRobotCfgPPO):
        self.task_classes[name] = task_class
        self.env_cfgs[name] = env_cfg
        self.train_cfgs[name] = train_cfg

    def get_task_class(self, name: str) -> VecEnv:
        return self.task_classes[name]
    
    def get_cfgs(self, name) -> Tuple[LeggedRobotCfg, LeggedRobotCfgPPO]:
        train_cfg = self.train_cfgs[name]
        env_cfg = self.env_cfgs[name]
        # copy seed
        env_cfg.seed = train_cfg.seed
        return env_cfg, train_cfg

    ## 该函数中名称后的->代表该函数返回的是VecEnv, LeggedRobotCfg类型的两个参数，也就是对于函数返回值类型的一个声明，类似于C++中的指定。
    def make_env(self, name, args=None, env_cfg=None) -> Tuple[VecEnv, LeggedRobotCfg]:
        """ Creates an environment either from a registered name or from the provided config file.

        Args:
            name (string): Name of a registered env.
            args (Args, optional): Isaac Gym comand line arguments. If None get_args() will be called. Defaults to None.
            env_cfg (Dict, optional): Environment config file used to override the registered config. Defaults to None.

        Raises:
            ValueError: Error if no registered env corresponds to 'name' 

        Returns:
            isaacgym.VecTaskPython: The created environment
            Dict: the corresponding config file
        """
        # if no args passed get command line arguments
        ## 如果用户没有指定参数，那么调用helper.py中的get_args()函数
        if args is None:
            args = get_args()
        # check if there is a registered env with that name
        ## 检查指定训练的机器人模型，环境是否注册存在，这里的注册，我认为就是是否声明以及初始化
        if name in self.task_classes:
            task_class = self.get_task_class(name)
        else:
            raise ValueError(f"Task with name: {name} was not registered")
        if env_cfg is None:
            # load config files(env parameters)
            env_cfg, _ = self.get_cfgs(name)
        # override cfg from args (if specified)
        ## 用终端读入的参数覆盖默认参数
        env_cfg, _ = update_cfg_from_args(env_cfg, None, args)
        ## 设置种子
        ## seed 是指随机数生成器的种子（seed value）。它用于控制随机过程的初始化，确保实验的可重复性。
        set_seed(env_cfg.seed)
        # parse sim params (convert to dict first)
        sim_params = {"sim": class_to_dict(env_cfg.sim)}
        sim_params = parse_sim_params(args, sim_params)
        env = task_class(   cfg=env_cfg,
                            sim_params=sim_params,
                            physics_engine=args.physics_engine,
                            sim_device=args.sim_device,
                            headless=args.headless)
        return env, env_cfg

    def make_alg_runner(self, env, name=None, args=None, train_cfg=None, log_root="default") -> Tuple[OnPolicyRunner, LeggedRobotCfgPPO]:
        """ Creates the training algorithm  either from a registered name or from the provided config file.

        Args:
            env (isaacgym.VecTaskPython): The environment to train (TODO: remove from within the algorithm)
            name (string, optional): Name of a registered env. If None, the config file will be used instead. Defaults to None.
            args (Args, optional): Isaac Gym command line arguments. If None get_args() will be called. Defaults to None.
            train_cfg (Dict, optional): Training config file. If None 'name' will be used to get the config file. Defaults to None.
            log_root (str, optional): Logging directory for Tensorboard. Set to 'None' to avoid logging (at test time for example). 
                                      Logs will be saved in <log_root>/<date_time>_<run_name>. Defaults to "default"=<path_to_LEGGED_GYM>/logs/<experiment_name>.

        Raises:
            ValueError: Error if neither 'name' or 'train_cfg' are provided
            Warning: If both 'name' or 'train_cfg' are provided 'name' is ignored

        Returns:
            PPO: The created algorithm
            Dict: the corresponding config file
        """
        # if no args passed get command line arguments
        ## 如果用户没有指定参数，那么调用helper.py中的get_args()函数
        if args is None:
            args = get_args()
        # if config files are passed use them, otherwise load from the name
        ## 如果训练参数从终端读取后有值的话就使用，否则调用helper.py中的get_args()函数
        if train_cfg is None:
            if name is None:
                raise ValueError("Either 'name' or 'train_cfg' must be not None")
            # load config files
            _, train_cfg = self.get_cfgs(name)
        else:
            if name is not None:
                print(f"'train_cfg' provided -> Ignoring 'name={name}'")
        # override cfg from args (if specified)
        ## 将指定的参数覆盖掉默认的参数
        _, train_cfg = update_cfg_from_args(None, train_cfg, args)

        ## 设置log文件储存的名称以及位置
        if log_root=="default":
            ## LEGGED_GYM_ROOT_DIR在legged_gym里的初始化脚本里定义，就是legged_gym的根目录
            ## log_root最终为logs文件夹下用户指定的试验名称的文件名字，试验名称可在终端启动时指定，默认值在legged_robot_config.py下的
            ## runner结构体中给定
            ## log_dir在此基础上设定系统当前时间，以月日_时_分_秒_运行名称创建目录，如：Jan20_17_02_51_test，代表1月20日17时02分51秒
            log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
            log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)
        elif log_root is None:
            log_dir = None
        else:
            log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)

        ## 创建训练参数字典变量
        train_cfg_dict = class_to_dict(train_cfg)
        ## 调用rsl_rl下的函数，设置PPO强化学习算法相关参数
        runner = OnPolicyRunner(env, train_cfg_dict, log_dir, device=args.rl_device)
        #save resume path before creating a new log_dir
        ## 在创建新的log_dir时保存之前模型储存路径
        resume = train_cfg.runner.resume
        if resume:
            # load previously trained model
            ## 加载之前训练的模型
            resume_path = get_load_path(log_root, load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint)
            print(f"Loading model from: {resume_path}")
            runner.load(resume_path)
        return runner, train_cfg

# make global task registry
task_registry = TaskRegistry()