# 持续风力干扰下的自适应鲁棒行走：G1 人形机器人 RL 项目全面分析

---

## 一、参考资源调研

### 1.1 直接相关研究（持续外力干扰 + 人形机器人）

**核心发现：专门研究持续风力干扰下人形机器人行走的工作极少，这恰好是你的创新空间。**

现有文献中，外力干扰的研究主要集中在以下范式：

**范式 A：瞬时推力（Push Recovery）—— 最主流**

| 工作 | 平台 | 方法 | 要点 |
|------|------|------|------|
| Radosavovic et al., Science Robotics 2024 | Digit（全尺寸人形） | Causal Transformer + 大规模 domain randomization | 用瑜伽球、木棍推机器人，展示了鲁棒恢复能力。但扰动都是瞬时的 |
| Booster Gym, 2025 | Booster T1 | 端到端 PPO + domain randomization | 用 10kg 球撞击，机器人几步内恢复稳定步态。也是瞬时扰动 |
| SoFTA (Hold My Beer), 2025 | Unitree G1 | 解耦频率调度 + RL | 测试了 0.5 m/s 随机速度扰动，每秒施加一次。接近持续扰动但仍是脉冲式 |
| SE-Policy, Nie et al., 2025 | Unitree G1 | 对称等变 RL 策略 | 验证了域随机化下 G1 的鲁棒性，速度跟踪误差改善 40% |

**范式 B：持续外力干扰（四足机器人）—— 存在，但未应用于人形**

| 工作 | 平台 | 方法 | 要点 |
|------|------|------|------|
| Adaptive Quadruped Balance (PMC, 2021) | 定制四足 | SAC + 扰动 curriculum | **最接近你的方向**：设计了持续变化的扰动因素（多频率振动、持续外力），使用最大熵方法训练。但平台是四足，不是人形 |
| Autonomous Evolutionary Mechanism, 2025 | Unitree Go2 | 两阶段框架 + 自适应扰动 | 自动调整训练过程中的扰动强度。使用了 Isaac Gym，但仅用于四足 |

**范式 C：气动力学干扰建模 —— 几乎空白**

经过广泛检索，**没有找到任何专门研究持续风力（aerodynamic disturbance）对人形机器人行走影响的 RL 工作**。这意味着：
- 你的选题具有明确的新颖性（novelty）
- 没有可以直接复用的代码，需要从零设计风力模块
- 最近的参考是 OU 过程在风力建模中的应用（Arenas-López & Badaoui, 2020），以及四足平衡控制中的持续扰动训练

### 1.2 可直接利用的开源项目

**一级资源（直接基础，你的代码将在这些项目上开发）：**

| 项目 | 地址 | 与本项目的关系 |
|------|------|----------------|
| **unitree_rl_gym** | github.com/unitreerobotics/unitree_rl_gym | 主要开发基础。包含 G1 的 URDF、config、训练脚本。已有 `push_robots=True`、`push_interval_s=5`、`max_push_vel_xy=1.5` 的配置 |
| **legged_gym** | github.com/leggedrobotics/legged_gym | unitree_rl_gym 的上游。`legged_robot.py` 中的 `_push_robots()` 方法是你的改写起点 |
| **rsl_rl** | github.com/leggedrobotics/rsl_rl | PPO 算法实现。无需修改，直接使用 |
| **Isaac Gym Preview 3** | NVIDIA 官方 | 物理仿真器。提供 `apply_rigid_body_force_tensors` API，这是施加风力的核心接口 |

**二级资源（参考架构和技巧）：**

| 项目 | 要点 |
|------|------|
| **Humanoid-Gym** (roboterax) | 另一个 G1 训练框架，提供了 sim-to-real 的参考实现 |
| **Booster Gym** | 展示了完整的 push recovery 评估方法，可参考其鲁棒性测试 protocol |
| **Walk These Ways** (Improbable-AI) | Gait-conditioned 策略训练，其 domain randomization 配置可参考 |

### 1.3 关键 API 文档

**Isaac Gym Tensor API — 风力施加的核心：**

```python
# 核心 API：对所有刚体施加力（每个 timestep 调用一次）
gym.apply_rigid_body_force_tensors(
    sim,
    forceTensor,    # shape: [num_envs * num_bodies, 3]，单位：牛顿
    torqueTensor,   # shape: [num_envs * num_bodies, 3]，可为 None
    space            # gymapi.ENV_SPACE 或 gymapi.WORLD_SPACE
)

# 在特定位置施加力（会产生额外力矩）
gym.apply_rigid_body_force_at_pos_tensors(
    sim, forceTensor, posTensor, space
)
```

**重要特性：** 该 API 在每个 physics timestep 施加力，只持续一个 timestep。要实现持续风力，必须在 `post_physics_step()` 中**每个 step 都调用**。

### 1.4 必读论文清单

| # | 论文 | 为什么必读 |
|---|------|-----------|
| 1 | Radosavovic et al., "Real-World Humanoid Locomotion with RL", Science Robotics 2024 | 人形机器人 RL 行走的标杆，push recovery 的评估方法论 |
| 2 | "Robust Humanoid Walking on Compliant and Uneven Terrain", Singh et al., Humanoids 2024 | 最接近的鲁棒行走研究，aperiodic gait 对抗扰动的思路 |
| 3 | "Adaptive Quadruped Balance Control", PMC 2021 | 持续扰动 curriculum 设计的最佳参考 |
| 4 | Arenas-López & Badaoui, "OU Process for Wind Power", 2020 | OU 过程建模风力波动的数学基础 |
| 5 | "Booster Gym: End-to-End RL for Humanoid", 2025 | 人形机器人训练框架参考 + push recovery 评估 |
| 6 | "Coordinated Humanoid Robot Locomotion with SE-Policy", Nie et al., 2025 | G1 平台上的最新 RL 行走工作 |

---

## 二、需要考虑的问题

### 2.1 风力物理建模的真实性

**核心问题：你的风力模型要多精确？**

风力作用于人形机器人有几个维度的复杂性：

- **风力不是单一力**：真实风作用在整个机器人表面，不同部位（头部、躯干、腿部）受力不同。但过于精确的建模会增加仿真开销且对策略训练意义不大。
- **已实现方案（v3.2）**：多刚体气动力模型。风力分布到 5 个刚体：pelvis(55%) + 两大腿(12%×2) + 两小腿(8%×2)。每个刚体独立计算：相对速度 v_rel=v_wind-v_body (P0)、3D 椭球投影面积 A_eff=√((A_front·|dx|)²+(A_side·|dy|)²+(A_top·|dz|)²) 使用逐刚体四元数 (P1)、高度风速剖面 v(z)=v_ref×(z/z_ref)^0.28 (P2)、力施加在压力中心（body-local z 轴偏移，经四元数旋转到世界坐标）(P3)。Cd=1.1。
- **未建模的效果**：翼展效应、上游遮蔽、紊流涡流、空间风场相干性等。这些对 <1m 高的机器人影响在 5-10% 以内，不影响策略训练。

### 2.2 风速参数范围的合理设定

**G1 体重约 35kg，必须确保风力参数物理合理：**

| 蒲福风级 | 风速 (m/s) | 简化风力 F ≈ 0.5ρCdAv² (N) | 占体重比 |
|---------|-----------|---------------------------|---------|
| 4 级（和风） | 5.5-8.0 | 5-10 N | 1.5-3% |
| 6 级（强风） | 10.8-13.8 | 18-30 N | 5-9% |
| 8 级（大风） | 17.2-20.7 | 46-67 N | 13-19% |
| 10 级（狂风） | 24.5-28.4 | 94-126 N | 27-36% |
| 12 级（飓风） | 32.7+ | 167+ N | 48%+ |

注：按 ρ=1.225 kg/m³，Cd=1.2，A=0.35 m² 计算（初始估算值）。
实际实现（v3.2）使用 Cd=1.1，A_front=0.50 m²，A_side=0.22 m²，A_top=0.10 m²（3D 椭球投影）。

**建议训练范围**：从 4 级风（~10N）开始 curriculum，目标达到 8-10 级风（~50-120N）。超过体重 30% 的风力在物理上已经很难保持直立行走（人类在 10 级风中也很难站稳）。

### 2.3 风力与 Push 的本质区别

这是项目需要明确阐述的关键点，也是创新性的核心支撑：

| 维度 | 瞬时 Push | 持续风力 |
|------|----------|---------|
| 力的时间特性 | 脉冲式（1-3 个 timestep） | 持续且时变（贯穿整个 episode） |
| 策略要求 | **反应式**恢复 | **持续自适应**补偿 |
| 力的可预测性 | 完全不可预测 | 有时间相关性（OU 过程） |
| 稳态影响 | 无（恢复后回到正常） | 有（策略需要维持倾斜补偿） |
| Observation 需求 | 可以仅靠 proprioception | 可能需要让策略隐式估计风力方向和大小 |
| 训练信号 | 恢复/未恢复（binary） | 持续的稳定性质量（continuous） |

### 2.4 Observation Space 设计

**策略能"感知"风力吗？**

在真实世界中，机器人没有风速传感器。但策略可以从 proprioception history 中隐式推断外力。需要考虑：

- **基础方案**：仅使用标准 proprioception（关节角度、角速度、IMU 角速度、重力方向）。策略通过历史观测隐式推断风力。这最贴近真实部署。
- **增强方案**：在 observation 中加入最近 N 步的 base 线速度/角速度的历史窗口，给策略更多信息来估计外力。可以用 RNN/LSTM 或 Transformer 来处理。
- **Privileged 方案（训练时）**：在 Asymmetric Actor-Critic 中，Critic 可以直接访问真实风力向量作为 privileged information，Actor 只用 proprioception。这能加速训练。

### 2.5 Reward 设计的平衡

**关键矛盾：** 如果 reward 过度惩罚不稳定，策略可能学会"站着不动等风停"；如果过度奖励前进速度，策略可能在大风中强行前进导致摔倒。

需要平衡的 reward 项包括：速度跟踪（前进）、姿态稳定（不倒）、能量效率（不浪费）、动作平滑（不抖动）。

### 2.6 训练效率与稳定性

- 持续外力比瞬时 push 更难训练——策略在整个 episode 中都面临扰动，早期训练阶段可能完全无法站稳
- 需要精心设计 curriculum，确保初始阶段风力足够小让策略先学会基本行走
- 4096 个并行环境（Isaac Gym 标准配置）每个可以有不同的风力参数，增加训练多样性

### 2.7 评估指标的设计

需要设计一套系统的评估 benchmark，这本身就是一个贡献：

- 最大可行走风速（保持前进且不倒的最大风力）
- 速度跟踪误差 vs 风力强度曲线
- 不同风向下的存活率热力图
- 与无风策略和瞬时 push 策略的对比

---

## 三、技术实现流程

### 3.1 总体架构

```
┌─────────────────────────────────────────────────────────┐
│                    Training Pipeline                      │
│                                                           │
│  ┌──────────────┐    ┌──────────────┐   ┌──────────────┐ │
│  │  Wind Model   │    │  G1 Env      │   │  PPO (rsl_rl)│ │
│  │  (OU Process) │───▶│  (Isaac Gym)  │──▶│  Training    │ │
│  │  + Gust Model │    │  + Rewards    │   │  Algorithm   │ │
│  └──────┬───────┘    └──────────────┘   └──────────────┘ │
│         │                                                  │
│  ┌──────▼───────┐                                         │
│  │  Curriculum   │                                         │
│  │  Controller   │                                         │
│  └──────────────┘                                         │
└─────────────────────────────────────────────────────────┘
```

### 3.2 Phase 1：环境搭建（第 1-2 周）

**Step 1：在 unitree_rl_gym 中创建新的任务环境**

```
legged_gym/envs/g1_wind/
├── g1_wind_config.py    # 风力配置 + reward 配置
├── g1_wind_env.py       # 环境类（继承 LeggedRobot）
└── __init__.py          # 注册任务
```

**Step 2：实现风力模型**

在 `wind_model.py` 中实现三层风速模型（v3 — 输出速度，力计算在 env 中）：

```
三层叠加，矢量加法合成：

Layer 1 (基风): per-episode 恒定 base_angle + base_speed（从课程等级范围中采样）
Layer 2 (OU湍流):
  速度 OU: dv = θ(0-v)dt + σ_eff·√dt·dW,  σ_eff = 0.18×base_speed×(1+0.15×level) + 0.1
  方向 OU: dα = θ_dir(0-α)dt + σ_dir·√dt·dW,  θ_dir=0.2, σ_dir=0.15 → ±14°(1σ)
  Per-episode 参数随机化：θ∈[0.2,1.0], σ∈[0.05,0.25], θ_dir∈[0.05,0.5], σ_dir∈[0.02,0.25]
Layer 3 (阵风): 独立速度向量
  speed = Uniform(2,6) × (1+0.2×level),  dir = base ±60°
  梯形包络: envelope = min(t/0.3, (dur-t)/0.5) clamp [0,1]
  概率: 0.1/s × (1+0.1×level)

wind_velocity = (base_speed + ou_speed) × dir(base_angle + ou_angle)  +  gust_vel
wind_velocity = clamp(‖wind_velocity‖, max=speed_clamp_per_level)
speed_clamp_per_level = [0, 5, 8, 13, 20, 28] m/s
```

关键设计：
- 阵风是独立速度向量叠加（非速度乘数），避免 v² 放大效应
- L0 速度钳制=0，确保无风等级真正零风
- 逐级速度钳制防止 OU 尾部+阵风叠加产生非物理极端值（修复前 L5 曾达 67 m/s）

**Step 3：在环境中施加风力（v3 per-body 气动力）**

`G1WindRobot.step()` 重写了基类的 `step()`，在每个 physics substep（decimation 循环内）的 `gym.simulate()` 之前调用 `_apply_wind_force()`。v3 改进：

```
P0: v_rel = v_wind(z) - v_body                     # 相对速度（逐刚体）
P1: A_eff = sqrt((A_front·|dx|)² + (A_side·|dy|)² + (A_top·|dz|)²)
                                                     # 3D 椭球投影面积（逐刚体四元数）
P2: v_wind(z) = v_ref × (z/z_ref)^0.28             # 高度风速剖面（城市地形 α=0.28）
P3: CoP = COM + quat_rotate(body_quat, [0,0,offset]) # body-local z 轴偏移，产生倾倒力矩

F_i = 0.5×ρ×Cd × A_eff × fraction_i × |v_rel_i|² × v_rel_hat_i
total_force = clamp(Σ F_i, max=force_clamp_per_level)
force_clamp_per_level = [5, 15, 30, 60, 100, 150] N
```

力分布到 5 个刚体：pelvis(55%) + 两大腿(12%×2) + 两小腿(8%×2)。
CoP 偏移（URDF 导出）：[+0.10, +0.005, +0.005, +0.002, +0.002] m（body-local z 轴）。
通过 `apply_rigid_body_force_at_pos_tensors` 在 CoP 位置施力（ENV_SPACE），
Isaac Gym 自动从力臂计算力矩。每个 substep 后刷新 rigid_body_state 以获取准确的刚体速度和四元数。
物理常数：ρ=1.225 kg/m³, Cd=1.1。逐级力钳制：[5, 15, 30, 60, 100, 150] N。

### 3.3 Phase 2：Reward 设计（第 2-3 周）

**Reward 结构分为四层：**

```python
# === 第一层：基础行走 reward（保留 unitree_rl_gym 原有的）===
reward_scales = {
    "tracking_lin_vel": 1.0,     # 速度跟踪
    "tracking_ang_vel": 0.5,     # 角速度跟踪
    "lin_vel_z": -2.0,           # 惩罚垂直振动
    "ang_vel_xy": -0.05,         # 惩罚横滚/俯仰角速度
    "orientation": -1.0,         # 惩罚躯干倾斜
    "base_height": -1.0,         # 维持目标高度
    
    # === 第二层：能量与平滑 ===
    "torques": -0.0001,          # 能量效率
    "dof_acc": -2.5e-7,          # 动作平滑
    "action_rate": -0.01,        # 动作变化率
    
    # === 第三层：风力鲁棒性专用 reward ===
    "wind_lean_compensation": 0.3,   # 奖励向风倾斜
    "com_stability": 0.5,            # 质心在支撑多边形内的程度
    "sustained_walking": 0.2,        # 在风中持续行走的奖励
    "feet_contact_symmetry": 0.1,    # 左右脚接触对称性
    
    # === 第四层：安全终止 ===
    "termination": -10.0,            # 摔倒大惩罚
    "feet_stumble": -1.0,            # 绊倒惩罚
}
```

**风力专用 reward 函数（已实现）：**

`_reward_lean_compensation`：奖励机器人向风源方向倾斜。使用 `effective_direction`（含 OU 方向漂移）转换到 body frame 后与重力投影做点积。reward 按风力大小连续缩放（wind_force_mag / 50N，clamp max=1.0），低风时信号弱，强风时信号满。scale=0.3。

其他风力 reward（sustained_walking, contact_symmetry）已禁用（scale=0.0），因分析发现与已有 reward 冲突或鼓励站立不动。

### 3.4 Phase 3：Curriculum 设计（第 3-4 周）

```
Level 0: 无风（0 m/s）→ 学会基本行走
Level 1: 轻风（1-3 m/s）→ 学会稳态补偿
Level 2: 轻中风（2-5 m/s）→ 学会方向适应（OU 方向漂移 ±14°）
Level 3: 中风（4-8 m/s）→ 适应中等风力 + 阵风
Level 4: 强风（7-12 m/s）→ 阵风更频繁更强（prob×1.4, speed×1.8）
Level 5: 极端（10-18 m/s）→ 强湍流（I≈0.44）+ 强阵风（speed×2.0）
```

**Curriculum 自动升级逻辑（已实现）**：窗口累积式（upgrade_window=200 次 reset），所有 env 一起升/降级。升级条件：survival>0.8 AND tracking>0.6。降级条件：survival<0.3 AND tracking<0.2。

### 3.5 Phase 4：训练与实验（第 4-6 周）

**实验计划：**

| 实验 | 目的 | 配置 |
|------|------|------|
| Exp 1: Baseline | 无风训练的基准策略 | push_robots=False, wind=False |
| Exp 2: Push Only | 传统瞬时推力鲁棒策略 | 原始 push_robots=True 配置 |
| Exp 3: Wind-Trained | 风力训练的策略（你的方法） | wind=True, curriculum=True |
| Exp 4: Ablation - No Curriculum | 无 curriculum 直接强风训练 | wind=True, curriculum=False, 中等风力 |
| Exp 5: Ablation - No Wind Reward | 去掉风力专用 reward | wind=True, 只保留基础 reward |

**每个策略的统一评估 protocol（已实现 → eval_wind_robustness.py）：**

```
Suite A: 风速级别扫描（Level 0-5，完整模型）
Suite B: 风力模式分解
  ├── B1: 纯恒定风（Layer 1 only）— 稳态补偿能力
  ├── B2: 恒定风+湍流（L1+L2）— 持续波动适应
  ├── B3: 恒定风+阵风（L1+L3）— 突发冲击反应
  ├── B4: 完整模型（L1+L2+L3）— 综合表现
  └── B5: 纯阵风（L3 only）— 无基风时的阵风恢复
Suite C: 风向测试
  ├── C1-C4: 固定方向（正面/侧面/背面/斜向 45°）
  └── C5: 随机方向
Suite D: OU 参数极端值
  ├── D1: 极稳定（θ=1.0, σ=0.05）— 近恒定风
  ├── D2: 极湍流（θ=0.2, σ=0.4）— 剧烈波动
  ├── D3: 风向锁定（θ_dir=1.0, σ_dir=0.02）
  ├── D4: 风向飘忽（θ_dir=0.05, σ_dir=0.25）
  └── D5: 训练默认值参考
Suite E: 分布外模式
  ├── E1: 阶跃风（0→10m/s at t=5s）
  └── E2: 周期风（5+5sin(2πt/4)）
```

支持 A/B policy 对比：`--load_run <A> --load_run2 <B>` 自动输出存活率差异表。

### 3.6 Phase 5：分析与报告（第 7-8 周）

**可视化分析目标：**

- 存活率 vs 风力强度曲线（三种策略对比）
- 风力方向 × 风力强度的存活率热力图
- 策略行为可视化：在不同风向下机器人的倾斜姿态
- OU 过程参数敏感性分析
- Curriculum 进度曲线

---

## 四、技术难点

### 4.1 难点一：`apply_rigid_body_force_tensors` 的调用时序

**问题**：Isaac Gym 的 `apply_rigid_body_force_tensors` 只在调用的那个 physics timestep 生效。如果在 `post_physics_step` 中调用，力在下一个 step 才生效；如果在 `pre_physics_step` 中调用，需要确保在 `simulate()` 之前完成。

**解决方案（已实现）**：`G1WindRobot.step()` 重写基类 `step()`，在 decimation 循环内每个 substep 的 `gym.simulate()` 之前调用 `_apply_wind_force()`。每个 substep 都推进 WindModel（OU + 阵风状态更新），确保力在物理仿真前施加。v3 中每个 substep 后额外调用 `refresh_rigid_body_state_tensor()` 以获取准确的刚体位置和速度，用于下一个 substep 的相对速度计算 (P0) 和高度因子计算 (P2)。

### 4.2 难点二：Force Tensor 的形状和索引

**问题**：`apply_rigid_body_force_tensors` 接受的 tensor 形状是 `[num_envs × num_bodies, 3]`，是一个展平的 tensor。你需要正确找到 torso body 在这个展平 tensor 中的索引。

**解决方案（已实现）**：在 `_init_buffers()` 中通过 `find_actor_rigid_body_handle()` 获取所有受风刚体的索引，预计算 `wind_body_flat_indices = [num_envs, num_wind_bodies]` 展平索引。力/位置张量形状 `[num_envs * num_bodies, 3]`，通过 scatter 操作将 5 个风力刚体的力和 CoP 位置写入正确位置。v3 使用 `apply_rigid_body_force_at_pos_tensors` 在 CoP 处施力（自动计算力矩），替代了 v2 的 `apply_rigid_body_force_tensors`（仅在质心施力）。

### 4.3 难点三：OU 过程的参数调优

**问题**：OU 过程的参数直接决定了风力的"感觉"。参数不当会导致变化太快（像白噪声）或太慢（像恒定风）。

**v2 解决方案（已实现）**：

风模型使用两个独立 OU 过程——速度 OU 和方向 OU：

**速度 OU**：dv = θ(0-v)dt + σ_eff·√dt·dW
- **θ_speed=0.5**：τ=2s，对应 1m 高度小尺度湍流的积分时间
- **σ=0.18×base_speed**：稳态湍流强度 TI ≈ 0.34-0.44（v3.1 从 0.25 降至 0.18 以匹配实测数据）
- **σ_min=0.1 m/s**：确保最低噪声（但 L0 通过 speed_clamp=0 强制零风）
- **level_scale=1+0.15×level**：高级别湍流更强（L5 时 TI≈0.44，对应极端粗糙地形）

**方向 OU**（v2 新增）：dα = θ_dir(0-α)dt + σ_dir·√dt·dW
- **θ_dir=0.2**：τ=5s，方向变化慢于速度变化（物理上风向更"惰性"）
- **σ_dir=0.15**：稳态 std = 0.15/√0.4 ≈ 0.24 rad ≈ ±14°(1σ)，符合横向湍流 σ_v≈0.75×σ_u

**离散化**：Euler-Maruyama 方法，dt = physics substep（~0.005s）。

**Per-episode 参数随机化**（v2 新增，ou_randomize=True）：
每次 episode reset 时，从以下范围中独立采样每个 env 的 OU 参数：
- θ_speed ∈ [0.2, 1.0]：有的 episode 波动快回归，有的慢漂移
- σ_speed ∈ [0.05, 0.25]：有的 episode 极稳定，有的极湍流（v3.1 从上限 0.4 降至 0.25 以控制 TI）
- θ_dir ∈ [0.05, 0.5]：有的 episode 风向锁定，有的飘忽
- σ_dir ∈ [0.02, 0.25]：方向波动幅度也随机

效果：每个 episode 的风"性格"不同，策略无法过拟合到单一 OU 节奏，
被迫学会通用的反应式控制策略。本质是 Domain Randomization 应用在风力动力学参数上。

### 4.4 难点四：Reward 权重平衡

**问题**：风力鲁棒性 reward 与基础行走 reward 之间的权重平衡极其敏感。常见失败模式包括：

- **"原地踏步"模式**：如果稳定性 reward 太高，策略学会站着不动（完美稳定但零速度）
- **"蛮力前冲"模式**：如果速度跟踪 reward 太高，策略在大风中强行维持速度导致摔倒
- **"抖动补偿"模式**：如果 action rate 惩罚太低，策略学会高频抖动来对抗风力（能量浪费且不自然）

**解决方案**：

- 初始使用原始 unitree_rl_gym 的 reward 权重，然后逐步引入风力专用 reward
- 使用 tensorboard 监控各 reward 项的 magnitude，确保没有单一 reward 主导
- 关键技巧：在大风条件下适当降低速度跟踪 reward 的权重，允许策略在极端风力下减速但保持平衡

### 4.5 难点五：训练初期的 Bootstrap 问题

**问题**：如果一开始就施加风力，策略在初期几乎立即摔倒（reward 极低），PPO 难以从中学习有效行为。这导致训练长时间停滞。

**解决方案**：

- **必须使用 Curriculum**：Level 0 完全无风，让策略先学会走路
- **保留原始 push_robots 机制**：在 Level 0-1 使用原始的瞬时 push 作为基础鲁棒性训练
- **渐进式引入**：风力从接近零开始（1-2N），缓慢增大
- **大量并行环境**：4096 个环境中，初期大部分处于 Level 0，少量在 Level 1，确保梯度信号足够

### 4.6 难点六：G1 URDF 中 Torso Link 的确认

**问题**：需要确认 G1 URDF 模型中 torso 对应的 rigid body 名称和索引。不同版本的 G1 模型可能有不同的 body 命名。

**解决方案**：在 `_create_envs()` 后打印所有 body 名称和索引，找到 torso（通常名为 `pelvis`、`torso_link` 或 `base_link`）。可以通过以下代码确认：

```python
for i in range(self.gym.get_asset_rigid_body_count(robot_asset)):
    name = self.gym.get_asset_rigid_body_name(robot_asset, i)
    print(f"Body {i}: {name}")
```

---

## 五、基于 unitree_rl_gym + rsl_rl + Isaac Gym 的二次开发可行性分析

### 5.1 结论：完全可行，且是最佳选择

这三个开源项目的组合是实现本项目的理想基础，原因如下：

**unitree_rl_gym 已经提供了 90% 的基础设施：**

- G1 的完整 URDF 模型和关节配置
- 已验证的训练参数（PD gains、default joint positions）
- 完整的 Train → Play → Sim2Sim → Sim2Real 管线
- 已有的 domain randomization 框架（friction、mass randomization）
- 已有的 `push_robots` 机制（你的起点）

**已完成的新增/修改文件：**

```
已实现的文件（4 个文件）：

1. legged_gym/envs/g1_wind/g1_wind_config.py  ✓
   - 继承 G1RoughCfg，wind 配置块（OU 速度+方向参数、阵风参数、curriculum）
   - wind-specific reward scales

2. legged_gym/envs/g1_wind/g1_wind_env.py  ✓
   - 继承 G1Robot，重写 step()/reset_idx()/compute_observations()
   - 风力施加、课程控制器、lean_compensation reward

3. legged_gym/envs/g1_wind/wind_model.py  ✓ (v3.2)
   - 三层风速模型：基风 + OU 速度&方向湍流 + 独立阵风速度向量
   - 输出风速（m/s），力计算移至 g1_wind_env.py（per-body 气动力 P0-P3）
   - Per-episode OU 参数随机化（θ/σ/θ_dir/σ_dir 从范围中采样）
   - 方向 OU sigma 随课程等级缩放（与速度 OU 相同的 sigma_scale）
   - 逐级速度钳制防止极端值

4. legged_gym/envs/__init__.py  ✓
   - 注册 "g1_wind" → G1WindRobot

5. legged_gym/g1_wind_test/eval_wind_robustness.py  ✓
   - 5 维测试矩阵：level × mode × direction × OU extreme × OOD pattern
   - A/B policy 对比功能
```

**rsl_rl 无需修改：**

- PPO 实现已经成熟，直接使用
- 支持 MLP 和 RNN policy，可以尝试 LSTM 来利用历史信息
- Asymmetric Actor-Critic 已有支持，可以直接使用 privileged observation

**Isaac Gym 提供了所需的全部底层 API：**

- `apply_rigid_body_force_tensors`：风力施加
- 4096+ 并行环境：高效训练
- GPU tensor 操作：OU 过程在 GPU 上高效更新

### 5.2 开发路线图

```
Week 1:
├── Fork unitree_rl_gym
├── 确认 G1 URDF body names 和 torso index
├── 实现 WindModel 类（OU 过程 + 基础风 + 阵风）
├── 实现 g1_wind_env.py 的风力施加逻辑
└── 验证：可视化风力方向（viewer 中画箭头）

Week 2:
├── 设计并实现 reward 函数
├── 实现 curriculum controller
├── 调试：在弱风下验证基本训练是否收敛
└── 调试：确认 force tensor 索引正确

Week 3-4:
├── 系统性 reward 权重调优
├── Curriculum 参数调优
├── 运行 Exp 1-3（baseline, push-only, wind-trained）
└── 初步分析训练曲线

Week 5-6:
├── 运行 Exp 4-5（ablation studies）
├── 运行统一评估 protocol
├── 数据收集和可视化
└── Sim2Sim 验证（可选：Isaac Gym → MuJoCo）

Week 7-8:
├── 结果分析和图表制作
├── 撰写报告
└── 准备演示 demo
```

### 5.3 风险与备选方案

| 风险 | 可能性 | 备选方案 |
|------|--------|---------|
| 风力训练完全不收敛 | 低（有 curriculum） | 降低风力上限，或改为更简单的恒定风 |
| OU 过程参数难以调优 | 中 | 简化为阶梯式风力变化（每 N 秒重新采样风速） |
| Reward 权重无法平衡 | 中 | 使用多目标 RL 或 constrained RL 方法 |
| 训练时间过长 | 低 | 减少并行环境数或简化 curriculum 层级 |
| Isaac Gym 力施加 API 行为异常 | 低 | 回退到原始的 velocity perturbation 方式 |

---

## 附录：项目创新性总结

本项目的核心创新点可以从三个层面阐述：

**层面一 — 问题定义的创新**：首次系统性地研究持续时变风力干扰下人形机器人的自适应行走问题。这与传统的瞬时 push recovery 有本质区别——从"受扰后恢复"转变为"在持续扰动中维持稳定"。

**层面二 — 方法论的创新**：提出基于 OU 过程的风力建模方法，设计针对持续外力的 reward 结构和 curriculum 策略。特别是 wind-lean compensation reward 和多层次 curriculum（无风 → 恒定风 → 变向风 → 阵风）的组合。

**层面三 — 评估标准的创新**：建立一套系统的风力鲁棒性 benchmark，包括风向-风力存活率热力图、不同风力模式下的速度跟踪性能、与传统 push recovery 的交叉泛化测试。
