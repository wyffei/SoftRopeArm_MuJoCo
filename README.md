# SoftRopeArm_MuJoCo

一个绳驱（tendon-driven）软体机械臂在 MuJoCo 中的完整仿真流水线，从建模到强化学习分两个阶段：

- [`modeling/`](modeling/README.md) —— 机械臂建模流水线：用 `merge.py`/`compute.py` 拼装多节绳驱软体臂的 MuJoCo 模型，做抓取任务和接触力分析验证。
- [`rl/`](rl/README_RL.md) —— 基于 `modeling/` 产出的模型，做工作空间数据采集（穷举绳长组合、记录可达位置）与 SAC 强化学习训练（开环参考控制 + 残差修正到达目标点）。

两个子目录相互独立，各自的依赖、运行方式见各自的 README。
