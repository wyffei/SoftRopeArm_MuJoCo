"""Gymnasium environment for six-tendon continuum-arm residual reaching.

Episode logic
-------------
1. Reset MuJoCo and lock the moving plate at the same qpos/ctrl values used by
   the workspace dataset collector.
2. Let the arm settle under gravity with zero rope force.
3. Randomly choose one reachable target row from the dataset.
4. Execute the dataset reference tendon-length command for that row, using the
   same half-cosine ramp logic as the collector.
5. Start the RL episode from the post-reference state. The policy outputs small
   residual increments on top of the current commanded tendon lengths.

This means RL learns the last correction step:
    current state after dataset reference control -> target dataset point.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
import pandas as pd


ROPE_ACTUATORS = (
    "pull_rope1",
    "pull_rope2",
    "pull_rope3",
    "pull_rope_add1",
    "pull_rope_add2",
    "pull_rope_add3",
)
ROPE_TENDONS = (
    "rope1",
    "rope2",
    "rope3",
    "rope_add1",
    "rope_add2",
    "rope_add3",
)
TIP_SITE_NAME = "m20_bottom_center"
TIP_BODY_FALLBACK = "m20_bottom"

PLATE_JOINT_QPOS = {
    "plate_tx": 0.27,
    "plate_ty": -0.16,
}
PLATE_POSITION_CTRLS = {
    "plate_x_ctrl": 0.27,
    "plate_y_ctrl": -0.16,
}

QACC_ABS_LIMIT = 1e6
STATE_ABS_LIMIT = 1e6


@dataclass(slots=True)
class RopePIDConfig:
    kp: float = 60.0
    ki: float = 0.0
    kd: float = 1.5
    integral_min: float = -0.1
    integral_max: float = 0.1
    force_min: float = 0.0
    force_max: float = 15.0
    force_rate_limit: float | None = 150.0
    target_rate_limit: float | None = 1.2
    d_filter_tau: float = 0.03
    deadband: float = 1e-4


@dataclass(slots=True)
class RopeArmEnvConfig:
    root: Path = Path(".")
    xml_path: Path = Path("scene.xml")
    dataset_csv: Path = Path("dataset/workspace_6rope.csv")

    frame_skip: int = 3
    max_episode_steps: int = 120

    # Curriculum target selection: after executing the dataset reference command,
    # accept dataset rows whose residual error ||target - reference_tip|| is within
    # [reference_error_min, reference_error_max]. This avoids training mostly on
    # trivial rows where the reference already reaches the point.
    reference_error_min: float = 0.02
    reference_error_max: float = 0.08
    success_tol: float = 0.035
    success_hold_steps: int = 5

    # Residual action: command_lengths += action * action_scale.
    action_scale: float = 0.0005
    command_rate_limit: float = 0.20
    max_contract: float = 0.60
    max_extend: float = 0.03

    # Reset/reference phase. These mirror the dataset collector default logic.
    passive_settle_steps: int = 500
    reference_steps: int = 500
    reference_ramp_steps: int = 200
    # Minimum hold time after the reference ramp.  After this many steps, the
    # environment will keep holding the same command until the tip/qvel become
    # stable, or until reference_stable_max_steps is reached.
    reference_hold_steps: int = 40
    reference_stable_max_steps: int = 300
    reference_stable_hold_steps: int = 20
    reference_qvel_tol: float = 0.05
    reference_tip_delta_tol: float = 1e-4
    max_reference_retries: int = 40
    accept_nearest_reference_on_retry_failure: bool = True
    skip_reference_if_success: bool = False

    # Keep compatibility with the previous training script. In this environment
    # the dataset row itself is the reference, so KNN warm start is not used.
    use_knn_warm_start: bool = True
    knn_k: int = 1
    use_knn_blend: bool = False

    pid: RopePIDConfig = field(default_factory=RopePIDConfig)

    progress_weight: float = 5.0
    distance_weight: float = 1.0
    action_weight: float = 0.01
    action_smooth_weight: float = 0.02
    force_weight: float = 0.001
    time_penalty: float = 0.002
    success_bonus: float = 8.0
    unstable_penalty: float = 20.0

    debug_done: bool = False
    debug_reset: bool = False

    # Optional cache for the passive-settled MuJoCo state. If set, reset() will
    # load this state instead of running passive_settle_steps every episode.
    passive_cache_path: Path | None = None


class RopeLengthPID:
    """Target tendon length -> one-way tendon motor force."""

    def __init__(self, cfg: RopePIDConfig):
        self.cfg = cfg
        self.integral = 0.0
        self.prev_length: float | None = None
        self.prev_target: float | None = None
        self.dlength_filt = 0.0
        self.prev_force = 0.0

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_length = None
        self.prev_target = None
        self.dlength_filt = 0.0
        self.prev_force = 0.0

    def _rate_limit_target(self, target: float, dt: float) -> float:
        if self.cfg.target_rate_limit is None or self.prev_target is None:
            self.prev_target = float(target)
            return float(target)

        max_step = float(self.cfg.target_rate_limit) * dt
        limited = float(np.clip(target, self.prev_target - max_step, self.prev_target + max_step))
        self.prev_target = limited
        return limited

    def update(self, target_length: float, current_length: float, dt: float) -> float:
        if dt <= 0.0:
            return 0.0

        target_length = self._rate_limit_target(float(target_length), dt)
        error = float(current_length - target_length)
        if abs(error) < self.cfg.deadband:
            error = 0.0

        self.integral += error * dt
        self.integral = float(np.clip(self.integral, self.cfg.integral_min, self.cfg.integral_max))

        if self.prev_length is None:
            raw_dlength = 0.0
        else:
            raw_dlength = (float(current_length) - self.prev_length) / dt

        tau = max(float(self.cfg.d_filter_tau), 1e-8)
        alpha = dt / (tau + dt)
        self.dlength_filt += alpha * (raw_dlength - self.dlength_filt)

        force = (
            self.cfg.kp * error
            + self.cfg.ki * self.integral
            - self.cfg.kd * self.dlength_filt
        )
        force = float(np.clip(force, self.cfg.force_min, self.cfg.force_max))

        if self.cfg.force_rate_limit is not None:
            max_force_step = float(self.cfg.force_rate_limit) * dt
            force = float(np.clip(force, self.prev_force - max_force_step, self.prev_force + max_force_step))
            force = float(np.clip(force, self.cfg.force_min, self.cfg.force_max))

        if force <= self.cfg.force_min and error <= 0.0:
            self.integral -= error * dt
        elif force >= self.cfg.force_max and error > 0.0:
            self.integral -= error * dt
        self.integral = float(np.clip(self.integral, self.cfg.integral_min, self.cfg.integral_max))

        self.prev_length = float(current_length)
        self.prev_force = force
        return force


class RopeLengthController:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, actuator_name: str, tendon_name: str, pid_cfg: RopePIDConfig):
        self.model = model
        self.data = data
        self.actuator_name = actuator_name
        self.tendon_name = tendon_name

        self.actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        self.tendon_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, tendon_name)
        if self.actuator_id < 0:
            raise ValueError(f"找不到 actuator: {actuator_name}")
        if self.tendon_id < 0:
            raise ValueError(f"找不到 tendon: {tendon_name}")

        local_cfg = replace(pid_cfg)
        if model.actuator_ctrllimited[self.actuator_id]:
            ctrl_min, ctrl_max = model.actuator_ctrlrange[self.actuator_id]
            local_cfg.force_min = max(local_cfg.force_min, float(ctrl_min))
            local_cfg.force_max = min(local_cfg.force_max, float(ctrl_max))
        self.pid = RopeLengthPID(local_cfg)

    def reset(self) -> None:
        self.pid.reset()

    def get_pid_state(self) -> dict[str, float | None]:
        return {
            "integral": self.pid.integral,
            "prev_length": self.pid.prev_length,
            "prev_target": self.pid.prev_target,
            "dlength_filt": self.pid.dlength_filt,
            "prev_force": self.pid.prev_force,
        }

    def set_pid_state(self, state: dict[str, float | None]) -> None:
        self.pid.integral = float(state["integral"] or 0.0)
        self.pid.prev_length = None if state["prev_length"] is None else float(state["prev_length"])
        self.pid.prev_target = None if state["prev_target"] is None else float(state["prev_target"])
        self.pid.dlength_filt = float(state["dlength_filt"] or 0.0)
        self.pid.prev_force = float(state["prev_force"] or 0.0)

    def get_length(self) -> float:
        return float(self.data.ten_length[self.tendon_id])

    def step(self, target_length: float) -> tuple[float, float]:
        current_length = self.get_length()
        force = self.pid.update(target_length, current_length, float(self.model.opt.timestep))
        self.data.ctrl[self.actuator_id] = force
        return current_length, force


def half_cosine_ramp(t: int, ramp_steps: int) -> float:
    if ramp_steps <= 1:
        return 1.0
    s = float(np.clip(t / float(ramp_steps - 1), 0.0, 1.0))
    return 0.5 * (1.0 - np.cos(np.pi * s))


class RopeArmReachEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: RopeArmEnvConfig):
        super().__init__()
        self.cfg = cfg
        self.root = Path(cfg.root).resolve()
        self.xml_path = self._resolve_path(cfg.xml_path)
        self.dataset_csv = self._resolve_path(cfg.dataset_csv)

        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)

        self.controllers = [
            RopeLengthController(self.model, self.data, act, tendon, cfg.pid)
            for act, tendon in zip(ROPE_ACTUATORS, ROPE_TENDONS)
        ]

        self.tip_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, TIP_SITE_NAME)
        self.tip_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, TIP_BODY_FALLBACK)
        if self.tip_site_id < 0 and self.tip_body_id < 0:
            raise ValueError(f"找不到末端 site/body: {TIP_SITE_NAME} / {TIP_BODY_FALLBACK}")

        self.plate_qpos_addrs = self._build_plate_joint_qpos_addrs()
        self.plate_ctrl_ids = self._build_plate_ctrl_ids()

        self.dataset_xyz, self.dataset_delta = self._load_dataset(self.dataset_csv)

        mujoco.mj_forward(self.model, self.data)
        self.qpos0 = self.data.qpos.copy()
        self.qvel0 = self.data.qvel.copy()
        self.ctrl0 = np.zeros_like(self.data.ctrl)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        # obs = err(3), tip(3), target(3), tendon_lengths_rel(6), command_rel(6),
        # forces_norm(6), prev_action(6), reference_delta(6), reference_error(3)
        obs_dim = 42
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.target_xyz = np.zeros(3, dtype=np.float64)
        self.target_delta = np.zeros(6, dtype=np.float64)
        self.target_index = -1
        self.passive_lengths_trace = np.zeros((1, 6), dtype=np.float64)
        self.passive_stable_lengths = np.zeros(6, dtype=np.float64)
        self.command_lengths = np.zeros(6, dtype=np.float64)
        self.prev_command_lengths = np.zeros(6, dtype=np.float64)
        self.prev_action = np.zeros(6, dtype=np.float64)
        self.last_forces = np.zeros(6, dtype=np.float64)
        self.reference_tip_xyz = np.zeros(3, dtype=np.float64)
        self.reference_dist = np.inf
        self.last_dist = np.inf
        self.step_count = 0
        self.success_count = 0
        self.np_random = np.random.default_rng()
        self.reference_stable = False
        self.reference_stable_steps = 0
        self.reference_qvel_abs_max = np.inf
        self.reference_tip_delta = np.inf

    def _resolve_path(self, path: Path) -> Path:
        path = Path(path)
        return path if path.is_absolute() else self.root / path

    def _load_dataset(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        if not path.exists():
            raise FileNotFoundError(f"找不到数据集 CSV: {path}")
        df = pd.read_csv(path)
        delta_cols = [f"d{i}" for i in range(1, 7)]
        required = [*delta_cols, "x", "y", "z"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"数据集缺少列: {missing}")
        df = df.dropna(subset=required).reset_index(drop=True)
        if len(df) == 0:
            raise ValueError("数据集为空，或者 x/y/z/d1..d6 全部无效。")
        return df[["x", "y", "z"]].to_numpy(dtype=np.float64), df[delta_cols].to_numpy(dtype=np.float64)

    def _build_plate_joint_qpos_addrs(self) -> dict[int, float]:
        qpos_addrs: dict[int, float] = {}
        for joint_name, value in PLATE_JOINT_QPOS.items():
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                raise ValueError(f"找不到平台 slide joint: {joint_name}")
            qpos_addrs[int(self.model.jnt_qposadr[joint_id])] = float(value)
        return qpos_addrs

    def _build_plate_ctrl_ids(self) -> dict[int, float]:
        ctrl_ids: dict[int, float] = {}
        for actuator_name, value in PLATE_POSITION_CTRLS.items():
            actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
            if actuator_id >= 0:
                ctrl_ids[actuator_id] = float(value)
        return ctrl_ids

    def _set_plate_joint_qpos(self) -> None:
        for qpos_addr, value in self.plate_qpos_addrs.items():
            self.data.qpos[qpos_addr] = value

    def _apply_plate_position_ctrl(self) -> None:
        for actuator_id, value in self.plate_ctrl_ids.items():
            self.data.ctrl[actuator_id] = value

    def _reset_mujoco_state(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.qpos0
        self.data.qvel[:] = self.qvel0
        self.data.ctrl[:] = self.ctrl0
        self._set_plate_joint_qpos()
        self._apply_plate_position_ctrl()
        for controller in self.controllers:
            controller.reset()
        mujoco.mj_forward(self.model, self.data)

    def _get_tip_xyz(self) -> np.ndarray:
        if self.tip_site_id >= 0:
            return self.data.site_xpos[self.tip_site_id].copy()
        return self.data.xpos[self.tip_body_id].copy()

    def _get_tendon_lengths(self) -> np.ndarray:
        return np.asarray([c.get_length() for c in self.controllers], dtype=np.float64)

    def _is_unstable(self) -> bool:
        arrays = (
            self.data.qpos,
            self.data.qvel,
            self.data.qacc,
            self.data.ctrl,
            self._get_tip_xyz(),
            self._get_tendon_lengths(),
        )
        if any(not np.all(np.isfinite(arr)) for arr in arrays):
            return True
        if float(np.max(np.abs(self.data.qacc))) > QACC_ABS_LIMIT:
            return True
        if float(np.max(np.abs(self.data.qpos))) > STATE_ABS_LIMIT:
            return True
        if float(np.max(np.abs(self.data.qvel))) > STATE_ABS_LIMIT:
            return True
        return False

    def _settle_passive(self) -> bool:
        lengths = []
        for _ in range(max(1, self.cfg.passive_settle_steps)):
            self.data.ctrl[:] = 0.0
            self._apply_plate_position_ctrl()
            mujoco.mj_step(self.model, self.data)
            if self._is_unstable():
                return False
            lengths.append(self._get_tendon_lengths())
        self.passive_lengths_trace = np.asarray(lengths, dtype=np.float64)
        self.passive_stable_lengths = self.passive_lengths_trace[-1].copy()
        return True

    def _load_passive_cache(self) -> bool:
        """Load the passive-settled state cached on disk.

        Returns True if a compatible cache was loaded. Returns False if caching is
        disabled, the file does not exist, or the cache does not match this model.
        """
        if self.cfg.passive_cache_path is None:
            return False

        cache_path = self._resolve_path(self.cfg.passive_cache_path)
        if not cache_path.exists():
            return False

        try:
            with np.load(cache_path) as cache:
                qpos = cache["qpos"]
                qvel = cache["qvel"]
                ctrl = cache["ctrl"]
                passive_lengths_trace = cache["passive_lengths_trace"]
                passive_stable_lengths = cache["passive_stable_lengths"]
        except Exception as exc:
            if self.cfg.debug_reset:
                print(f"[PASSIVE CACHE] failed to load {cache_path}: {exc}")
            return False

        if qpos.shape != self.data.qpos.shape or qvel.shape != self.data.qvel.shape or ctrl.shape != self.data.ctrl.shape:
            if self.cfg.debug_reset:
                print(f"[PASSIVE CACHE] ignored incompatible cache: {cache_path}")
            return False

        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self.data.ctrl[:] = ctrl
        self._set_plate_joint_qpos()
        self._apply_plate_position_ctrl()
        mujoco.mj_forward(self.model, self.data)

        self.passive_lengths_trace = np.asarray(passive_lengths_trace, dtype=np.float64)
        self.passive_stable_lengths = np.asarray(passive_stable_lengths, dtype=np.float64)

        for controller in self.controllers:
            controller.reset()

        if self.cfg.debug_reset:
            print(f"[PASSIVE CACHE] loaded from {cache_path}")

        return True

    def _save_passive_cache(self) -> None:
        """Save the passive-settled state to disk for later reset() calls."""
        if self.cfg.passive_cache_path is None:
            return

        cache_path = self._resolve_path(self.cfg.passive_cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        np.savez(
            tmp_path,
            qpos=self.data.qpos.copy(),
            qvel=self.data.qvel.copy(),
            ctrl=self.data.ctrl.copy(),
            passive_lengths_trace=self.passive_lengths_trace.copy(),
            passive_stable_lengths=self.passive_stable_lengths.copy(),
            passive_tip=self._get_tip_xyz().copy(),
        )

        # np.savez appends .npz if the filename does not end with .npz.
        written_path = tmp_path if tmp_path.exists() else tmp_path.with_suffix(tmp_path.suffix + ".npz")
        written_path.replace(cache_path)

        if self.cfg.debug_reset:
            print(f"[PASSIVE CACHE] saved to {cache_path}")

    def _clip_command_lengths(self, command: np.ndarray) -> np.ndarray:
        lo = self.passive_stable_lengths - float(self.cfg.max_contract)
        hi = self.passive_stable_lengths + float(self.cfg.max_extend)
        return np.clip(command, lo, hi)

    def _rate_limit_command_lengths(self, command: np.ndarray) -> np.ndarray:
        if self.cfg.command_rate_limit is None or self.cfg.command_rate_limit <= 0:
            return command
        dt = float(self.model.opt.timestep) * max(1, self.cfg.frame_skip)
        max_step = float(self.cfg.command_rate_limit) * dt
        return np.clip(command, self.command_lengths - max_step, self.command_lengths + max_step)

    def _select_dataset_target_candidate(self, exclude: set[int] | None = None) -> tuple[int, np.ndarray, np.ndarray]:
        """Sample a dataset row before reference rollout.

        The actual curriculum filter is applied *after* running the dataset
        reference command, using reference_dist = ||target - reference_tip||.
        """
        if exclude and len(exclude) < len(self.dataset_xyz):
            while True:
                idx = int(self.np_random.integers(0, len(self.dataset_xyz)))
                if idx not in exclude:
                    break
        else:
            idx = int(self.np_random.integers(0, len(self.dataset_xyz)))
        return idx, self.dataset_xyz[idx].copy(), self.dataset_delta[idx].copy()

    def _reference_error_is_accepted(self, reference_dist: float) -> bool:
        return (
            float(self.cfg.reference_error_min) <= reference_dist <= float(self.cfg.reference_error_max)
        )

    def _reference_error_gap(self, reference_dist: float) -> float:
        """Distance to the accepted residual-error interval; 0 means accepted."""
        lo = float(self.cfg.reference_error_min)
        hi = float(self.cfg.reference_error_max)
        if reference_dist < lo:
            return lo - reference_dist
        if reference_dist > hi:
            return reference_dist - hi
        return 0.0

    def _reference_command_from_delta(self, delta: np.ndarray, t: int) -> np.ndarray:
        base = self.passive_lengths_trace[min(t, len(self.passive_lengths_trace) - 1)]
        command = base + half_cosine_ramp(t, self.cfg.reference_ramp_steps) * delta
        lo = base - float(self.cfg.max_contract)
        hi = base + float(self.cfg.max_extend)
        return np.clip(command, lo, hi)

    def _run_reference_control(self, delta: np.ndarray) -> bool:
        forces = np.zeros(6, dtype=np.float64)
        for controller in self.controllers:
            controller.reset()

        # 1) Ramp to the dataset reference command.
        for t in range(max(1, self.cfg.reference_steps)):
            self.command_lengths = self._reference_command_from_delta(delta, t)
            forces_list: list[float] = []
            self._apply_plate_position_ctrl()
            for controller, target_length in zip(self.controllers, self.command_lengths):
                _, force = controller.step(float(target_length))
                forces_list.append(force)
            mujoco.mj_step(self.model, self.data)
            if self._is_unstable():
                return False
            forces = np.asarray(forces_list, dtype=np.float64)

        # 2) Hold the final reference command.  We do not stop merely because a
        # fixed number of hold steps elapsed; after the minimum hold time, require
        # consecutive stable steps so that zero-action does not mostly measure
        # unfinished settling dynamics.
        stable_count = 0
        total_hold_steps = 0
        max_extra_steps = max(0, int(self.cfg.reference_stable_max_steps))
        min_hold_steps = max(0, int(self.cfg.reference_hold_steps))
        required_stable_steps = max(1, int(self.cfg.reference_stable_hold_steps))
        max_total_hold_steps = min_hold_steps + max_extra_steps

        prev_tip = self._get_tip_xyz()
        self.reference_stable = False
        self.reference_stable_steps = 0
        self.reference_qvel_abs_max = np.inf
        self.reference_tip_delta = np.inf

        for _ in range(max_total_hold_steps):
            self._apply_plate_position_ctrl()
            forces_list = []
            for controller, target_length in zip(self.controllers, self.command_lengths):
                _, force = controller.step(float(target_length))
                forces_list.append(force)
            mujoco.mj_step(self.model, self.data)
            if self._is_unstable():
                return False

            forces = np.asarray(forces_list, dtype=np.float64)
            total_hold_steps += 1

            tip = self._get_tip_xyz()
            tip_delta = float(np.linalg.norm(tip - prev_tip))
            qvel_abs_max = float(np.max(np.abs(self.data.qvel))) if self.data.qvel.size else 0.0
            prev_tip = tip

            self.reference_qvel_abs_max = qvel_abs_max
            self.reference_tip_delta = tip_delta

            if total_hold_steps >= min_hold_steps:
                if qvel_abs_max <= float(self.cfg.reference_qvel_tol) and tip_delta <= float(self.cfg.reference_tip_delta_tol):
                    stable_count += 1
                else:
                    stable_count = 0

                if stable_count >= required_stable_steps:
                    self.reference_stable = True
                    break

        self.reference_stable_steps = total_hold_steps
        self.last_forces = forces.copy()
        return True

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        # 1) Build one passive-settled state.  Then restore this state for each
        # candidate reference rollout. This is much faster than re-settling under
        # gravity for every rejected candidate.
        self._reset_mujoco_state()
        if not self._load_passive_cache():
            if not self._settle_passive():
                raise RuntimeError("reset 失败：被动稳定阶段触发 unstable。")
            self._save_passive_cache()

        passive_tip = self._get_tip_xyz()
        passive_qpos = self.data.qpos.copy()
        passive_qvel = self.data.qvel.copy()
        passive_ctrl = self.data.ctrl.copy()

        tried: set[int] = set()
        best: dict[str, Any] | None = None
        best_gap = np.inf

        for attempt in range(max(1, self.cfg.max_reference_retries)):
            idx, target_xyz, target_delta = self._select_dataset_target_candidate(tried)
            tried.add(idx)

            # Restore passive-settled state before trying this dataset row.
            self.data.qpos[:] = passive_qpos
            self.data.qvel[:] = passive_qvel
            self.data.ctrl[:] = passive_ctrl
            mujoco.mj_forward(self.model, self.data)
            for controller in self.controllers:
                controller.reset()
            self.command_lengths = self.passive_stable_lengths.copy()
            self.last_forces[:] = 0.0

            self.target_index = idx
            self.target_xyz = target_xyz
            self.target_delta = target_delta

            if self.cfg.skip_reference_if_success and np.linalg.norm(passive_tip - target_xyz) <= self.cfg.success_tol:
                reference_ok = True
                self.command_lengths = self.passive_stable_lengths.copy()
            else:
                reference_ok = self._run_reference_control(target_delta)

            if not reference_ok:
                continue

            reference_tip = self._get_tip_xyz()
            reference_dist = float(np.linalg.norm(target_xyz - reference_tip))
            gap = self._reference_error_gap(reference_dist)

            state_snapshot = {
                "idx": idx,
                "target_xyz": target_xyz.copy(),
                "target_delta": target_delta.copy(),
                "reference_tip": reference_tip.copy(),
                "reference_dist": reference_dist,
                "qpos": self.data.qpos.copy(),
                "qvel": self.data.qvel.copy(),
                "ctrl": self.data.ctrl.copy(),
                "command_lengths": self.command_lengths.copy(),
                "last_forces": self.last_forces.copy(),
                "pid_states": [controller.get_pid_state() for controller in self.controllers],
                "reference_stable": bool(self.reference_stable),
                "reference_stable_steps": int(self.reference_stable_steps),
                "reference_qvel_abs_max": float(self.reference_qvel_abs_max),
                "reference_tip_delta": float(self.reference_tip_delta),
                "attempt": attempt + 1,
            }

            if gap < best_gap:
                best_gap = gap
                best = state_snapshot

            if self._reference_error_is_accepted(reference_dist):
                best = state_snapshot
                break

        if best is None:
            raise RuntimeError("reset 失败：数据集参考控制多次触发 unstable，没有可用样本。")

        if best_gap > 0.0 and not self.cfg.accept_nearest_reference_on_retry_failure:
            raise RuntimeError(
                "reset 失败：没有采到满足 reference_error_min/max 的样本；"
                "可增大 max_reference_retries 或放宽 reference_error_min/max。"
            )

        # Restore the accepted/best post-reference state as the RL episode start.
        self.target_index = int(best["idx"])
        self.target_xyz = best["target_xyz"].copy()
        self.target_delta = best["target_delta"].copy()
        self.reference_tip_xyz = best["reference_tip"].copy()
        self.reference_dist = float(best["reference_dist"])
        self.data.qpos[:] = best["qpos"]
        self.data.qvel[:] = best["qvel"]
        self.data.ctrl[:] = best["ctrl"]
        mujoco.mj_forward(self.model, self.data)
        self.command_lengths = self._clip_command_lengths(best["command_lengths"])
        self.prev_command_lengths = self.command_lengths.copy()
        self.last_forces = best["last_forces"].copy()
        for controller, pid_state in zip(self.controllers, best["pid_states"]):
            controller.set_pid_state(pid_state)
        self.reference_stable = bool(best.get("reference_stable", False))
        self.reference_stable_steps = int(best.get("reference_stable_steps", 0))
        self.reference_qvel_abs_max = float(best.get("reference_qvel_abs_max", np.inf))
        self.reference_tip_delta = float(best.get("reference_tip_delta", np.inf))
        self.prev_action = np.zeros(6, dtype=np.float64)
        self.last_dist = self.reference_dist
        self.step_count = 0
        self.success_count = 0

        # Keep the PID memory from the accepted reference rollout.  This makes
        # action=0 mean "continue holding the reference command" rather than
        # restarting the tendon controllers at the beginning of the RL phase.

        obs = self._get_obs()
        info = self._get_info()
        info.update({
            "reset_attempt": int(best["attempt"]),
            "target_index": self.target_index,
            "reference_dist": self.reference_dist,
            "reference_error_min": float(self.cfg.reference_error_min),
            "reference_error_max": float(self.cfg.reference_error_max),
            "reference_error_accepted": bool(self._reference_error_is_accepted(self.reference_dist)),
            "reference_stable": bool(self.reference_stable),
            "reference_stable_steps": int(self.reference_stable_steps),
            "reference_qvel_abs_max": float(self.reference_qvel_abs_max),
            "reference_tip_delta": float(self.reference_tip_delta),
            "passive_tip": passive_tip.astype(np.float64),
            "reference_tip": self.reference_tip_xyz.astype(np.float64),
        })
        if self.cfg.debug_reset:
            print(
                f"[RESET] idx={self.target_index}, attempt={int(best['attempt'])}, "
                f"accepted={self._reference_error_is_accepted(self.reference_dist)}, "
                f"ref_range=[{self.cfg.reference_error_min:.3f}, {self.cfg.reference_error_max:.3f}], "
                f"passive_tip={passive_tip.round(3)}, target={self.target_xyz.round(3)}, "
                f"reference_tip={self.reference_tip_xyz.round(3)}, reference_dist={self.reference_dist:.4f}, "
                f"stable={self.reference_stable}, hold_steps={self.reference_stable_steps}, "
                f"qvel_max={self.reference_qvel_abs_max:.4g}, tip_delta={self.reference_tip_delta:.4g}"
            )
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float64).reshape(6)
        action = np.clip(action, -1.0, 1.0)

        raw_next_command = self.command_lengths + action * float(self.cfg.action_scale)
        self.command_lengths = self._rate_limit_command_lengths(raw_next_command)
        self.command_lengths = self._clip_command_lengths(self.command_lengths)

        unstable = False
        forces = np.zeros(6, dtype=np.float64)

        for _ in range(max(1, self.cfg.frame_skip)):
            self._apply_plate_position_ctrl()
            forces_list: list[float] = []
            for controller, target_length in zip(self.controllers, self.command_lengths):
                _, force = controller.step(float(target_length))
                forces_list.append(force)
            mujoco.mj_step(self.model, self.data)
            forces = np.asarray(forces_list, dtype=np.float64)
            if self._is_unstable():
                unstable = True
                break

        self.last_forces = forces.copy()
        tip_xyz = self._get_tip_xyz()
        dist = float(np.linalg.norm(self.target_xyz - tip_xyz))
        progress = float(self.last_dist - dist)

        reward = 0.0
        reward += self.cfg.progress_weight * progress
        reward -= self.cfg.distance_weight * dist
        reward -= self.cfg.action_weight * float(np.linalg.norm(action))
        reward -= self.cfg.action_smooth_weight * float(np.linalg.norm(action - self.prev_action))
        reward -= self.cfg.force_weight * float(np.linalg.norm(forces))
        reward -= self.cfg.time_penalty

        reached = dist <= self.cfg.success_tol
        if reached:
            self.success_count += 1
        else:
            self.success_count = 0

        terminated = self.success_count >= self.cfg.success_hold_steps
        truncated = self.step_count + 1 >= self.cfg.max_episode_steps

        if terminated:
            reward += self.cfg.success_bonus
        if unstable:
            reward -= self.cfg.unstable_penalty
            terminated = True

        done_reason: str | None = None
        if terminated or truncated:
            if unstable:
                done_reason = "unstable"
            elif self.success_count >= self.cfg.success_hold_steps:
                done_reason = "success"
            elif truncated:
                done_reason = "timeout"
            else:
                done_reason = "unknown"

        self.prev_action = action.copy()
        self.prev_command_lengths = self.command_lengths.copy()
        self.last_dist = dist
        self.step_count += 1

        obs = self._get_obs()
        info = self._get_info()
        info.update({
            "dist": dist,
            "progress": progress,
            "reached": bool(reached),
            "success": bool(reached),
            "success_count": int(self.success_count),
            "unstable": bool(unstable),
            "terminated_success": bool(terminated and not unstable),
            "reward_progress": float(self.cfg.progress_weight * progress),
            "reward_distance": float(-self.cfg.distance_weight * dist),
            "reward_action": float(-self.cfg.action_weight * np.linalg.norm(action)),
            "reward_force": float(-self.cfg.force_weight * np.linalg.norm(forces)),
        })
        if done_reason is not None:
            info["done_reason"] = done_reason
            if self.cfg.debug_done:
                print(
                    f"[DONE] reason={done_reason}, step={self.step_count}, "
                    f"dist={dist:.4f}, ref_dist={self.reference_dist:.4f}, "
                    f"reward={reward:.3f}, target={self.target_xyz.round(3)}, "
                    f"tip={tip_xyz.round(3)}, action={action.round(3)}"
                )

        return obs, float(reward), bool(terminated), bool(truncated), info

    def _get_obs(self) -> np.ndarray:
        tip = self._get_tip_xyz()
        tendon_lengths = self._get_tendon_lengths()
        err = self.target_xyz - tip
        reference_err = self.target_xyz - self.reference_tip_xyz
        tendon_rel = tendon_lengths - self.passive_stable_lengths
        command_rel = self.command_lengths - self.passive_stable_lengths
        forces_norm = self.last_forces / max(float(self.cfg.pid.force_max), 1e-6)
        obs = np.concatenate([
            err,
            tip,
            self.target_xyz,
            tendon_rel,
            command_rel,
            forces_norm,
            self.prev_action,
            self.target_delta,
            reference_err,
        ]).astype(np.float32)
        return obs

    def _get_info(self) -> dict[str, Any]:
        tip = self._get_tip_xyz()
        return {
            "target_index": int(self.target_index),
            "target_xyz": self.target_xyz.copy(),
            "tip_xyz": tip,
            "reference_tip_xyz": self.reference_tip_xyz.copy(),
            "reference_dist": float(self.reference_dist),
            "reference_stable": bool(self.reference_stable),
            "reference_stable_steps": int(self.reference_stable_steps),
            "reference_qvel_abs_max": float(self.reference_qvel_abs_max),
            "reference_tip_delta": float(self.reference_tip_delta),
            "command_lengths": self.command_lengths.copy(),
            "tendon_lengths": self._get_tendon_lengths(),
            "forces": self.last_forces.copy(),
            "time": float(self.data.time),
        }

    def set_curriculum(self, reference_error_min: float, reference_error_max: float, success_tol: float) -> None:
        self.cfg.reference_error_min = float(reference_error_min)
        self.cfg.reference_error_max = float(reference_error_max)
        self.cfg.success_tol = float(success_tol)

    def close(self) -> None:
        pass
