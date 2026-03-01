# G1 Wind-Robust Walking Environment

训练 Unitree G1 人形机器人在持续时变风扰动下稳定行走的强化学习环境。

## 文件结构

| 文件 | 说明 |
|------|------|
| `wind_model.py` | 3层风力模型（基础风 + OU过程 + 阵风） |
| `g1_wind_config.py` | 环境配置（风参数、课程、奖励权重） |
| `g1_wind_env.py` | 环境主类（风力施加、奖励、课程控制） |
| `__init__.py` | 模块导出 |

## 风力模型 (WindModel)

三层叠加结构，所有运算均为 GPU tensor 向量化操作：

```
最终风速 = (基础风速 + OU波动) × 阵风倍率
风力 F = 0.5 × ρ × Cd × A × v²
```

| 层 | 机制 | 参数 |
|----|------|------|
| Layer 1: 基础风 | 每 episode 采样固定方向+速度 | 水平面随机方向，速度由课程决定 |
| Layer 2: OU过程 | Ornstein-Uhlenbeck 时变波动 | θ=0.5, σ=0.3×base_speed |
| Layer 3: 阵风 | 随机短时爆发 | 2-3x倍率，持续1-2秒 |

物理参数默认值：空气密度 ρ=1.225 kg/m³，阻力系数 Cd=1.0，迎风面积 A=0.5 m²。

## 课程学习 (6级)

| 等级 | 风速范围 (m/s) | 描述 |
|------|---------------|------|
| 0 | 0 | 无风（基线） |
| 1 | 1-3 | 轻风 |
| 2 | 2-5 | 轻-中风 |
| 3 | 4-8 | 中风 |
| 4 | 7-12 | 强风+阵风 |
| 5 | 10-18 | 极端 |

升级条件：存活率 > 80% 且速度跟踪 > 60%，每200次 reset 评估一次。

## 奖励结构

### 继承自 G1 的奖励
- `tracking_lin_vel` (1.0) — 线速度跟踪
- `tracking_ang_vel` (0.5) — 角速度跟踪
- `base_height` (-10.0) — 保持目标高度
- `orientation` (-1.0) — 保持直立
- `contact` (0.18) — 步态相位匹配
- `alive` (0.15) — 存活奖励
- 能量/平滑性惩罚：`dof_acc`, `dof_vel`, `action_rate`, `dof_pos_limits`
- 人形专用：`hip_pos`, `contact_no_vel`, `feet_swing_height`

### 风相关新奖励
| 奖励 | 权重 | 说明 |
|------|------|------|
| `wind_stability` | -0.5 | 惩罚风中速度跟踪偏差 |
| `lean_compensation` | 0.2 | 奖励向风方向倾斜以保持平衡 |
| `sustained_walking` | 0.1 | 每步存活奖励 |
| `contact_symmetry` | 0.05 | 奖励左右脚对称接触 |

## 观测空间

**标准观测 (47维)** — 不含风信息，便于 sim-to-real：
```
ang_vel(3) + gravity(3) + commands(3) + dof_pos(12) + dof_vel(12) + actions(12) + phase(2)
```

**特权观测 (56维)** — 仅训练时使用：
```
标准观测内容 + base_lin_vel(3) + wind_velocity(3) + wind_force(3)
```

## 关键技术细节

### 风力施加
- 在每个物理子步的 `gym.simulate()` **之前**调用 `apply_rigid_body_force_tensors()`
- 风力施加到 `pelvis` 刚体上
- 力张量索引：`env_i * num_bodies + torso_body_idx`
- 坐标系：`gymapi.ENV_SPACE`

### 域随机化
- 关闭 `push_robots`（风力取代脉冲式推力扰动）
- 保留摩擦力随机化 [0.1, 1.25] 和基座质量随机化 [-1, 3] kg

### 策略网络
- LSTM 循环网络 (hidden_size=64)，帮助从历史观测推断风况
- Actor/Critic: [64, 32] 隐藏层，ELU 激活

## 使用方法

```bash
# 训练
python legged_gym/scripts/train.py --task g1_wind --headless

# 推理/可视化
python legged_gym/scripts/play.py --task g1_wind

# 指定环境数和设备
python legged_gym/scripts/train.py --task g1_wind --num-envs 4096 --sim-device cuda:0
```

## 实验计划

| ID | 名称 | 风 | 推力 | 目的 |
|----|------|-----|------|------|
| Exp1 | Baseline | OFF | OFF | 基线参考 |
| Exp2 | Push-Only | OFF | ON | 传统鲁棒性 |
| Exp3 | Wind-Trained | ON+课程 | OFF | 我们的方法 |
| Exp4 | No-Curriculum | ON(固定中等) | OFF | 消融：无课程 |
| Exp5 | No-Wind-Reward | ON+课程 | OFF | 消融：仅基础奖励 |

## 继承关系

```
BaseTask → LeggedRobot → G1Robot → G1WindRobot
BaseConfig → LeggedRobotCfg → G1RoughCfg → G1WindRoughCfg
```
