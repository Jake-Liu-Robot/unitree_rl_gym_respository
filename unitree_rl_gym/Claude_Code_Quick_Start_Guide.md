# Claude Code 快速启动指南 — G1 Wind-Robust Walking 项目

---

## 第一步：安装 Claude Code

```bash
# 推荐方式：原生安装器（无需 Node.js）
curl -fsSL https://claude.ai/install.sh | bash

# 验证安装
claude --version
```

如果你更习惯 npm 方式（需要 Node.js 18+）：
```bash
npm install -g @anthropic-ai/claude-code
```

## 第二步：认证

首次运行时会自动弹出浏览器进行认证。你需要以下之一：
- Claude Pro / Max 订阅（固定月费，推荐）
- Anthropic Console API 账户（按用量计费）

```bash
# 或者手动设置 API Key
export ANTHROPIC_API_KEY="sk-ant-..."
```

## 第三步：克隆项目仓库

```bash
# 克隆 unitree_rl_gym
git clone https://github.com/unitreerobotics/unitree_rl_gym.git
cd unitree_rl_gym

# 创建开发分支
git checkout -b feature/wind-robust-walking
```

## 第四步：放入 CLAUDE.md

将我为你生成的 `CLAUDE.md` 文件复制到项目根目录：

```bash
# 假设你从 claude.ai 下载了文件到 ~/Downloads/
cp ~/Downloads/CLAUDE.md ./CLAUDE.md
```

Claude Code 会自动读取项目根目录的 `CLAUDE.md` 作为项目上下文。

## 第五步：启动 Claude Code

```bash
cd unitree_rl_gym
claude
```

## 第六步：开始开发

进入 Claude Code 后，你可以直接用自然语言指令开发。以下是推荐的启动对话：

### 初始化项目结构
```
请帮我在 legged_gym/envs/ 下创建 g1_wind 目录，包含 __init__.py、
g1_wind_config.py、g1_wind_env.py 和 wind_model.py。
先查看现有的 g1_config.py 和 legged_robot.py 了解代码结构，
然后按照 CLAUDE.md 中的设计创建文件骨架。
```

### 确认 G1 URDF Body Names
```
请查看 resources/robots/g1/ 目录下的 URDF 文件，列出所有 rigid body 的名称，
帮我找到 torso/pelvis 对应的 body name。
```

### 实现风力模型
```
请实现 wind_model.py 中的 WindModel 类。
使用 Ornstein-Uhlenbeck 过程建模时变风力，
支持 3 层风力（基础风 + OU波动 + 阵风），
所有操作必须是 GPU tensor 操作。
参考 CLAUDE.md 中的设计。
```

### 实现环境类
```
请实现 g1_wind_env.py，继承 LeggedRobot。
核心功能：
1. 在 step() 中 simulate() 之前施加风力
2. 实现 wind-specific reward 函数
3. 重写 reset_idx() 重置风力参数
4. 实现 curriculum 自动升级逻辑
```

### 注册并测试
```
请在 legged_gym/envs/__init__.py 中注册 g1_wind 任务，
然后帮我写一个简单的测试脚本验证：
1. 环境能正常创建
2. 风力 tensor 形状正确
3. 风力确实被施加到 torso 上
```

---

## 有用的 Claude Code 命令

| 命令 | 说明 |
|------|------|
| `/help` | 查看帮助 |
| `/clear` | 清除对话历史 |
| `/bug` | 报告 bug |
| `Ctrl+C` | 中断当前操作 |
| `Esc` | 取消当前输入 |

## 提示

1. **CLAUDE.md 是你的"记忆"**：Claude Code 每次启动都会读取它。随着项目进展，
   持续更新 CLAUDE.md 中的进度和决策。

2. **让 Claude Code 先读再写**：开发新模块前，先让它读取相关的现有代码
   （如 `请先阅读 legged_robot.py 中的 _push_robots 方法`），再开始实现。

3. **小步迭代**：不要一次让它写所有代码。先写骨架 → 验证编译 → 加功能 → 测试。

4. **利用 git**：Claude Code 可以帮你做 git 操作。每完成一个功能就 commit。
