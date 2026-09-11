# SoftRopeArm_MuJoCo

A complete MuJoCo simulation pipeline for a tendon-driven (rope-driven) soft robotic arm, split into two independent stages:

- [`modeling/`](modeling/README.md) — arm modeling pipeline: assembles a multi-segment tendon-driven soft arm MuJoCo model with `merge.py`/`compute.py`, and validates it through grasping and contact-force analysis.
- [`rl/`](rl/README_RL.md) — reinforcement learning pipeline built on the model produced by `modeling/`: collects a reachable-workspace dataset (exhaustive tendon-length sampling) and trains a SAC agent (open-loop reference control + residual correction) to reach target points.

The two subdirectories are independent — see each one's README for its own dependencies and usage.
