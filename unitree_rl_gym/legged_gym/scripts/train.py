import os
import numpy as np
from datetime import datetime
import sys

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch

def train(args):
    ## 接收终端启动程序时给定的参数args，返回环境env以及环境初始化参数env_cfg，见2.1节
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ## 进行算法相关设置
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

## 该框架的入口
if __name__ == '__main__':
    ## 首先执行获取参数的函数，该函数的定义在utils/helper.py里
    args = get_args()
    train(args)
