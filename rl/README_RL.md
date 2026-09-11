# 六绳机械臂强化学习版本

## 文件

- `rope_arm_reach_env.py`：Gymnasium 环境，负责 MuJoCo 仿真、六根绳 PID、奖励和目标采样。
- `train_sac_rope_arm.py`：SAC 训练脚本，带课程学习。
- `eval_sac_rope_arm.py`：加载训练好的 SAC 模型，在 viewer 里按 waypoint 测试。

把三个 `.py` 文件放到你的项目根目录，也就是和 `scene.xml` 同一层。

## 安装依赖

```bash
pip install gymnasium stable-baselines3 mujoco pandas numpy tensorboard
```

## 训练

```bash
python train_sac_rope_arm.py --total-timesteps 300000
```

输出模型：

```text
rl_models/sac_rope_arm.zip
```

## 可视化测试

```bash
python eval_sac_rope_arm.py --model rl_models/sac_rope_arm.zip --deterministic
```

红点是目标点，蓝点是末端点。

## 核心设计

策略输出不是 force，而是六根绳子的目标长度增量：

```text
action = [Δl1, Δl2, Δl3, Δl4, Δl5, Δl6]
```

底层仍然使用 PID：

```text
SAC -> target tendon length increment -> PID -> tendon force -> MuJoCo
```

这样比直接让 RL 输出 force 稳定。

## 课程学习

训练脚本里默认从近目标开始，然后逐步扩大目标范围：

```text
0%   target_radius = 0.04 m
20%  target_radius = 0.07 m
45%  target_radius = 0.11 m
70%  target_radius = 0.16 m
```

如果一开始不稳定，可以先降低动作幅度：

```bash
python train_sac_rope_arm.py --action-scale 0.0015 --force-max 12 --total-timesteps 300000
```

如果已经比较稳定但动作太慢，可以适当增大：

```bash
python train_sac_rope_arm.py --action-scale 0.004 --force-max 20 --total-timesteps 500000
```

## 数据集作用

默认会读取：

```text
dataset/workspace_6rope.csv
```

它用于两个地方：

1. 从数据集附近采样目标点，避免一开始采到明显不可达点。
2. 用 KNN 给初始 commanded tendon lengths 一个 warm start，RL 主要学习修正动作。

如果你想从零训练，可以加：

```bash
python train_sac_rope_arm.py --no-knn-warm-start
```
