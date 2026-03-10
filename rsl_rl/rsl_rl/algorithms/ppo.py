# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import torch
import torch.nn as nn  # 导入 PyTorch 神经网络模块
import torch.optim as optim  # 导入 PyTorch 优化模块   使用 Adam 优化器来更新 `actor_critic` 网络的参数

from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage

class PPO:
    actor_critic: ActorCritic   # 定义 actor_critic 变量类型为 ActorCritic
    def __init__(self,
                 actor_critic,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",  # 学习率调度策略（`schedule`）
                 desired_kl=0.01,  # 目标 KL 散度（`desired_kl`），用于自适应学习率调整
                 device='cpu',
                 ):

        self.device = device  # 初始化 self.device 为传入的 device 参数

        self.desired_kl = desired_kl  # 初始化 self.desired_kl 为传入的 desired_kl 参数
        self.schedule = schedule  # 初始化 self.schedule 为传入的 schedule 参数
        self.learning_rate = learning_rate  # 初始化 self.learning_rate 为传入的 learning_rate 参数

        # PPO components
        # 最上面PPO类里一系列参数的self初始化
        self.actor_critic = actor_critic   # 传入的神经网络模型，并将其移动到指定的设备（如 CPU 或 GPU）
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()    #一个临时的 `RolloutStorage.Transition` 对象，用于在每个环境步骤中收集数据

        # PPO parameters
        # 最上面PPO类里一系列参数的self初始化
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    # 这个方法根据环境的数量、每个环境收集的转换（transitions）数量、观察空间形状和动作空间形状来创建`RolloutStorage`实例
    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        # `RolloutStorage`负责存储智能体与环境交互产生的数据序列
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device)

    # 这些方法用于切换`actor_critic`网络到评估（测试）模式或训练模式。这对于包含Dropout或Batch Normalization层的网络很重要
    def test_mode(self):  # 定义测试模式函数
        self.actor_critic.test()  # 设置 actor_critic 为测试模式

    def train_mode(self):  # 定义训练模式函数
        self.actor_critic.train()  # 设置 actor_critic 为训练模式

    # 动作选择act: 这是智能体与环境交互的核心。给定当前观察`obs`（用于策略网络）和`critic_obs`（用于价值网络，可能与`obs`相同或包含额外信息），它执行以下操作
    def act(self, obs, critic_obs):  # 定义 act 函数
        if self.actor_critic.is_recurrent:  # 如果 actor_critic 是循环的（RNN/LSTM）
            self.transition.hidden_states = self.actor_critic.get_hidden_states()  # 获取隐藏状态

        # Compute the actions and values
        # 使用 Actor-Critic 网络计算动作，并分离计算图
        self.transition.actions = self.actor_critic.act(obs).detach()
        # 使用 Actor-Critic 网络评估状态价值，并分离计算图
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        # 获取动作的对数概率，并分离计算图
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()

        # 存储动作分布的均值(`action_mean`)和标准差(`action_sigma`)
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()

        # need to record obs and critic_obs before env.step()
        # 在环境步之前需要记录 obs 和 critic_obs
        self.transition.observations = obs  # 记录观察
        self.transition.critic_observations = critic_obs  # 记录评论员观察
        return self.transition.actions # 返回计算出的动作，供环境执行

    # 在环境执行动作后调用此方法
    # 接收奖励(`rewards`)、完成标志(`dones`)和额外信息(`infos`)
    def process_env_step(self, rewards, dones, infos):  # 定义处理环境步函数
        self.transition.rewards = rewards.clone()  # 克隆奖励
        self.transition.dones = dones  # 记录完成状态

        # Bootstrapping on timeouts
        # 对超时进行引导
        # 重要: 它实现了"超时引导 (Bootstrapping on time outs)"。如果一个episode因为达到时间限制而不是因为失败状态而结束（通过`infos["time_outs"]`判断），
        # 它会将最后一步的估计价值（乘以`gamma`）加到奖励中这可以防止智能体因为时间限制而受到不公平的惩罚，并提供更准确的回报估计
        if 'time_outs' in infos:               # 如果信息中有 'time_outs'
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)  # 更新奖励

        # Record the transition
        # 记录过渡
        self.storage.add_transitions(self.transition)  # 添加过渡到存储中
        self.transition.clear()  # 清空 `transition` 对象，为下一步做准备
        self.actor_critic.reset(dones)  # 如果网络是循环的，根据 `dones` 信号重置其隐藏状态

    # 1. 在收集了足够多的transitions后调用
    # 2. 首先，使用`actor_critic`网络评估最后一个状态的价值(`last_values`)
    # 计算回报和优势的方法
    def compute_returns(self, last_critic_obs):
        # 评估最后一个状态的价值
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        # 调用存储器的 compute_returns 方法计算回报和 GAE 优势
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    # 更新策略和价值网络参数的方法
    def update(self):
        # 初始化平均价值损失
        mean_value_loss = 0
        # 初始化平均代理损失
        mean_surrogate_loss = 0

        # 检查 Actor-Critic 网络是否是循环网络
        if self.actor_critic.is_recurrent:
            # 如果是循环网络，使用循环小批量生成器
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
            # 如果不是循环网络
        else:
            # 使用标准小批量生成器
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # 遍历小批量生成器产生的数据
        for (
                obs_batch,  # 观察值小批量
                critic_obs_batch,  # 评论家观察值小批量
                actions_batch,  # 动作小批量
                target_values_batch,  # 目标价值小批量 (用于价值损失计算，通常是回报)
                advantages_batch,  # 优势小批量
                returns_batch,  # 回报小批量 (用于价值损失计算)
                old_actions_log_prob_batch,  # 旧策略下的动作对数概率小批量
                old_mu_batch,  # 旧策略下的动作均值小批量
                old_sigma_batch,  # 旧策略下的动作标准差小批量
                hid_states_batch,  # 隐藏状态小批量 (仅用于循环网络)
                masks_batch,  # 掩码小批量 (仅用于循环网络)
        ) in generator:

                # 使用当前策略重新评估动作 (主要为了获取内部状态如均值、标准差)
                self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
                # 获取当前策略下动作的对数概率
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                # 使用当前策略评估状态价值
                value_batch = self.actor_critic.evaluate(
                    # 传入评论家观察、掩码和隐藏状态
                    critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
                )
                # 获取当前策略的动作均值
                mu_batch = self.actor_critic.action_mean
                # 获取当前策略的动作标准差
                sigma_batch = self.actor_critic.action_std
                # 获取当前策略的熵
                entropy_batch = self.actor_critic.entropy

                # KL (对应数学推导？)
                # KL散度与自适应学习率：如果启用了自适应学习率(`schedule == "adaptive"`）首先，计算当前策略和旧策略（生成数据时的策略）之间的KL散度
                # KL 散度计算 (用于自适应学习率)
                # 如果设置了期望 KL 且调度策略是自适应的
                if self.desired_kl is not None and self.schedule == "adaptive":
                    # 在无梯度计算模式下进行
                    with torch.inference_mode():
                        # 计算当前策略和旧策略之间的 KL 散度
                        kl = torch.sum(
                            # 对数标准差比项 (+1e-5 防止除零)
                            torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                            # 旧标准差平方 + 均值差平方项
                            + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                            # 除以 2 倍当前标准差平方
                            / (2.0 * torch.square(sigma_batch))
                            # 减去 0.5
                            - 0.5,
                            # 沿着最后一个维度求和
                            axis=-1,
                        )
                        # 计算 KL 散度的平均值
                        kl_mean = torch.mean(kl)

                        # 如果 KL 散度远大于期望值
                        if kl_mean > self.desired_kl * 2.0:
                            # 降低学习率 (最小为 1e-5)
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)

                            # 如果 KL 散度远小于期望值 (且大于 0)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            # 提高学习率 (最大为 1e-2)
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)


                # Surrogate loss
                # 代理损失计算 (PPO 核心目标)
                # 计算重要性采样比率 (当前概率 / 旧概率)
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                # 计算未裁剪的代理损失项
                surrogate = -torch.squeeze(advantages_batch) * ratio
                # 计算裁剪后的代理损失项
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                    # 且将比率裁剪到 [1-clip, 1+clip] 范围内
                    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                )
                # 取未裁剪和裁剪后损失中的较大者，并计算平均值
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

                # 价值函数损失计算
                # 如果使用裁剪的价值损失
                if self.use_clipped_value_loss:
                    # 计算裁剪后的价值预测
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                        # 将价值预测与目标价值的差值裁剪到 [-clip, clip]
                        -self.clip_param, self.clip_param
                    )
                    # 计算未裁剪的价值损失 (均方误差)
                    value_losses = (value_batch - returns_batch).pow(2)
                    # 计算裁剪后的价值损失 (均方误差)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    # 取未裁剪和裁剪后损失中的较大者，并计算平均值
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()

                # 如果不使用裁剪的价值损失
                else:
                    # 直接计算价值损失 (均方误差)
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                # 计算总损失 = 代理损失 + 价值损失 - 熵奖励
                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()


                # Gradient step
                # 梯度更新步骤
                # 清空优化器的梯度
                self.optimizer.zero_grad()
                # 反向传播计算梯度
                loss.backward()
                # 对梯度进行裁剪，防止梯度爆炸
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                # 执行一步优化器更新
                self.optimizer.step()
                # 累加当前小批量的价值损失 (转换为 Python float)
                mean_value_loss += value_loss.item()
                # 累加当前小批量的代理损失 (转换为 Python float)
                mean_surrogate_loss += surrogate_loss.item()

        # 计算总的更新次数
        num_updates = self.num_learning_epochs * self.num_mini_batches
        # 计算整个 update 过程中的平均价值损失
        mean_value_loss /= num_updates
        # 计算整个 update 过程中的平均代理损失
        mean_surrogate_loss /= num_updates
        # 清空经验存储器，为下一轮数据收集做准备
        self.storage.clear()

        # 返回平均价值损失和平均代理损失
        return mean_value_loss, mean_surrogate_loss