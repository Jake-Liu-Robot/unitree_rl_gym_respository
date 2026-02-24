import torch
from torch import Tensor
import numpy as np
from isaacgym.torch_utils import quat_apply, normalize
from typing import Tuple

# 该脚本主要做数据类型转换，向量投影计算。

# @ torch.jit.script
# 该函数将vec向量沿quat中的z轴旋转量旋转
def quat_apply_yaw(quat, vec):
	## 首先将quat进行副本制作并转化为一个二维向量，其中每个元素有四个分量（四元数）
    quat_yaw = quat.clone().view(-1, 4)
    ## 将该四元数的前两个元素都置为0，四元数为[x,y,z,w]其中x,y元素代表了绕x轴和y轴的旋转分量
    ## 所以将这两个量置0目的就是只保留z轴方向上的旋转分量
    quat_yaw[:, :2] = 0.
    ## 将新的四元数进行归一化
    quat_yaw = normalize(quat_yaw)
    ## 旋转，quat_applys是在torch_utils.py中实现的
    return quat_apply(quat_yaw, vec)

# @ torch.jit.script
def wrap_to_pi(angles):
    angles %= 2*np.pi
    angles -= 2*np.pi * (angles > np.pi)
    return angles

# @ torch.jit.script
def torch_rand_sqrt_float(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], str) -> Tensor
    r = 2*torch.rand(*shape, device=device) - 1
    r = torch.where(r<0., -torch.sqrt(-r), torch.sqrt(r))
    r =  (r + 1.) / 2.
    return (upper - lower) * r + lower