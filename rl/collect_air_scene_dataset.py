from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np


# ============================================================
# 默认配置：按当前场景命名写死，必要时可用命令行覆盖
# ============================================================
DEFAULT_SCENE = "scene.xml"
DEFAULT_OUTPUT = "dataset/workspace_6rope.npz"
QACC_ABS_LIMIT = 1e6
STATE_ABS_LIMIT = 1e6


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

# moving_plate 的两个 slide joint 初始位置。
# 注意：这里设置的是 joint qpos，不是世界坐标。
# 对应 XML：
#   <joint name="plate_tx" type="slide" axis="1 0 0" />
#   <joint name="plate_ty" type="slide" axis="0 1 0" />
PLATE_JOINT_QPOS = {
    "plate_tx": 0.27,
    "plate_ty": -0.16,
}

# plate 的 position actuator 目标值。
# 为了避免平台被拉回 0，这里必须和 PLATE_JOINT_QPOS 保持一致。
PLATE_POSITION_CTRLS = {
    "plate_x_ctrl": 0.27,
    "plate_y_ctrl": -0.16,
}


@dataclass(frozen=True)
class PIDConfig:
    kp: float = 60.0
    ki: float = 0.0
    kd: float = 1.5
    integral_min: float = -0.1
    integral_max: float = 0.1
    force_min: float = 0.0
    force_max: float = 15.0
    force_rate_limit: float | None = 150.0
    d_filter_tau: float = 0.03
    deadband: float = 1e-4


@dataclass(frozen=True)
class CollectConfig:
    scene: str = DEFAULT_SCENE
    output: str = DEFAULT_OUTPUT
    samples: int = 1000
    workers: int = 8
    seed: int = 0

    # 六根绳目标收缩量采样范围，单位 m。
    # 负数表示目标 tendon length 变短；顺序对应 ROPE_TENDONS / ROPE_ACTUATORS。
    delta_min: tuple[float, ...] = (-0.5, -0.5, -0.5, -0.5, -0.5, -0.5)
    delta_max: tuple[float, ...] = (0.02, 0.02, 0.02, 0.02, 0.02, 0.02)

    steps: int = 500
    ramp_steps: int = 200
    hold_steps: int = 120
    max_contract: float = 0.6
    max_extend: float = 0.03

    pid: PIDConfig = PIDConfig()
    strict_plate_joint: bool = True
    strict_plate_ctrl: bool = False


class RopeLengthPID:
    """目标 tendon length -> 单向 tendon motor force。"""

    def __init__(self, cfg: PIDConfig):
        self.cfg = cfg
        self.integral = 0.0
        self.prev_length: float | None = None
        self.dlength_filt = 0.0
        self.prev_force = 0.0

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_length = None
        self.dlength_filt = 0.0
        self.prev_force = 0.0

    def update(self, target_length: float, current_length: float, dt: float) -> float:
        if dt <= 0.0:
            return 0.0

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
            max_step = float(self.cfg.force_rate_limit) * dt
            force = float(np.clip(force, self.prev_force - max_step, self.prev_force + max_step))
            force = float(np.clip(force, self.cfg.force_min, self.cfg.force_max))

        # anti-windup
        if force <= self.cfg.force_min and error <= 0.0:
            self.integral -= error * dt
        elif force >= self.cfg.force_max and error > 0.0:
            self.integral -= error * dt
        self.integral = float(np.clip(self.integral, self.cfg.integral_min, self.cfg.integral_max))

        self.prev_length = float(current_length)
        self.prev_force = force
        return force


class RopeLengthController:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        actuator_name: str,
        tendon_name: str,
        pid_cfg: PIDConfig,
    ):
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

        cfg = PIDConfig(**asdict(pid_cfg))
        if model.actuator_ctrllimited[self.actuator_id]:
            ctrl_min, ctrl_max = model.actuator_ctrlrange[self.actuator_id]
            cfg = PIDConfig(
                **{
                    **asdict(cfg),
                    "force_min": max(cfg.force_min, float(ctrl_min)),
                    "force_max": min(cfg.force_max, float(ctrl_max)),
                }
            )
        self.pid = RopeLengthPID(cfg)

    def reset(self) -> None:
        self.pid.reset()

    def get_length(self) -> float:
        return float(self.data.ten_length[self.tendon_id])

    def step(self, target_length: float) -> tuple[float, float]:
        length = self.get_length()
        force = self.pid.update(target_length, length, float(self.model.opt.timestep))
        self.data.ctrl[self.actuator_id] = force
        return length, force


class AirSceneCollectorEnv:
    """只用于采集数据集的 MuJoCo 环境封装。"""

    def __init__(self, cfg: CollectConfig):
        self.cfg = cfg
        self.model = mujoco.MjModel.from_xml_path(str(Path(cfg.scene)))
        self.data = mujoco.MjData(self.model)

        self.controllers = [
            RopeLengthController(self.model, self.data, act, tendon, cfg.pid)
            for act, tendon in zip(ROPE_ACTUATORS, ROPE_TENDONS)
        ]

        self.tip_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, TIP_SITE_NAME)
        self.tip_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, TIP_BODY_FALLBACK)
        if self.tip_site_id < 0 and self.tip_body_id < 0:
            raise ValueError(f"找不到末端 site/body: {TIP_SITE_NAME} / {TIP_BODY_FALLBACK}")

        self.plate_joint_qpos_addrs = self._build_plate_joint_qpos_addrs(PLATE_JOINT_QPOS)
        self.plate_ctrl_ids = self._build_plate_ctrl_ids(PLATE_POSITION_CTRLS)

        mujoco.mj_forward(self.model, self.data)
        self.qpos0 = self.data.qpos.copy()
        self.qvel0 = self.data.qvel.copy()
        self.ctrl0 = np.zeros_like(self.data.ctrl)
        self.reset()

    def _build_plate_joint_qpos_addrs(self, joint_qpos: dict[str, float]) -> dict[int, float]:
        """获取 plate_tx / plate_ty 的 qpos 地址。"""
        qpos_addrs: dict[int, float] = {}
        for joint_name, value in joint_qpos.items():
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                msg = f"找不到平台 slide joint: {joint_name}"
                if self.cfg.strict_plate_joint:
                    raise ValueError(msg)
                print(f"[WARN] {msg}，跳过该 joint qpos 设置。")
                continue

            qpos_addr = int(self.model.jnt_qposadr[joint_id])
            qpos_addrs[qpos_addr] = float(value)
        return qpos_addrs

    def _build_plate_ctrl_ids(self, plate_ctrls: dict[str, float]) -> dict[int, float]:
        """获取 plate_x_ctrl / plate_y_ctrl 的 actuator id。"""
        ctrl_ids: dict[int, float] = {}
        for actuator_name, value in plate_ctrls.items():
            actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
            if actuator_id < 0:
                msg = f"找不到平台位置控制器 actuator: {actuator_name}"
                if self.cfg.strict_plate_ctrl:
                    raise ValueError(msg)
                print(f"[WARN] {msg}，跳过该 actuator ctrl 保持。")
                continue

            ctrl_ids[actuator_id] = float(value)
        return ctrl_ids

    def set_plate_joint_qpos(self) -> None:
        """
        直接指定 moving_plate 的 slide joint 初始 qpos。

        注意：qpos 是相对于 XML 中 body 初始 pos 的 slide 位移，
        不是平台的世界坐标。
        """
        for qpos_addr, value in self.plate_joint_qpos_addrs.items():
            self.data.qpos[qpos_addr] = value

    def apply_plate_position_ctrls(self) -> None:
        """
        让平台 position actuator 目标值与初始 qpos 一致。

        这样平台不会从指定 qpos 被拉回 ctrl=0 的位置。
        """
        for actuator_id, value in self.plate_ctrl_ids.items():
            self.data.ctrl[actuator_id] = value

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.qpos0
        self.data.qvel[:] = self.qvel0
        self.data.ctrl[:] = self.ctrl0

        self.set_plate_joint_qpos()
        self.apply_plate_position_ctrls()

        for ctrl in self.controllers:
            ctrl.reset()

        mujoco.mj_forward(self.model, self.data)

    def get_tip_pos(self) -> np.ndarray:
        if self.tip_site_id >= 0:
            return self.data.site_xpos[self.tip_site_id].copy()
        return self.data.xpos[self.tip_body_id].copy()

    def get_rope_lengths(self) -> np.ndarray:
        return np.asarray([ctrl.get_length() for ctrl in self.controllers], dtype=float)

    def assert_stable(self, context: str) -> None:
        """MuJoCo 出现 NaN/Inf/巨大加速度时，立刻判定该样本失败。"""
        checks = {
            "qpos": self.data.qpos,
            "qvel": self.data.qvel,
            "qacc": self.data.qacc,
            "ctrl": self.data.ctrl,
            "tip_pos": self.get_tip_pos(),
            "rope_lengths": self.get_rope_lengths(),
        }
        for name, arr in checks.items():
            arr = np.asarray(arr, dtype=float)
            if not np.all(np.isfinite(arr)):
                raise FloatingPointError(f"仿真不稳定：{context} 的 {name} 出现 NaN/Inf")

        if np.max(np.abs(self.data.qacc)) > QACC_ABS_LIMIT:
            raise FloatingPointError(
                f"仿真不稳定：{context} 的 qacc 过大，"
                f"max_abs_qacc={np.max(np.abs(self.data.qacc)):.3e}"
            )

        for name in ("qpos", "qvel"):
            arr = np.asarray(getattr(self.data, name), dtype=float)
            if np.max(np.abs(arr)) > STATE_ABS_LIMIT:
                raise FloatingPointError(
                    f"仿真不稳定：{context} 的 {name} 过大，"
                    f"max_abs_{name}={np.max(np.abs(arr)):.3e}"
                )

    def step_zero_rope_force(self) -> None:
        """六根绳零拉力，但平台位置控制器保持指定目标。"""
        self.data.ctrl[:] = 0.0
        self.apply_plate_position_ctrls()
        mujoco.mj_step(self.model, self.data)
        self.assert_stable("zero_rope_force_step")

    def step_pid(self, target_lengths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """六根绳 PID 控制，平台位置控制器保持指定目标。"""
        lengths, forces = [], []
        for ctrl, target in zip(self.controllers, target_lengths):
            length, force = ctrl.step(float(target))
            lengths.append(length)
            forces.append(force)

        self.apply_plate_position_ctrls()
        mujoco.mj_step(self.model, self.data)
        self.assert_stable("pid_step")
        return np.asarray(lengths), np.asarray(forces)


def half_cosine_ramp(t: int, ramp_steps: int) -> float:
    if ramp_steps <= 1:
        return 1.0
    s = np.clip(t / float(ramp_steps - 1), 0.0, 1.0)
    return float(0.5 * (1.0 - np.cos(np.pi * s)))


def collect_passive_lengths(env: AirSceneCollectorEnv, steps: int) -> np.ndarray:
    """
    在平台 qpos 已指定、平台 ctrl 已锁定的条件下，
    六根绳零拉力，记录重力稳定过程中的被动 tendon length。
    """
    env.reset()
    lengths = []
    for _ in range(steps):
        env.step_zero_rope_force()
        lengths.append(env.get_rope_lengths())
    env.reset()
    return np.asarray(lengths, dtype=float)


def target_lengths_from_delta(
    passive_lengths: np.ndarray,
    delta_goal: np.ndarray,
    t: int,
    cfg: CollectConfig,
) -> np.ndarray:
    base = passive_lengths[min(t, passive_lengths.shape[0] - 1)]
    target = base + half_cosine_ramp(t, cfg.ramp_steps) * np.asarray(delta_goal, dtype=float)
    return np.clip(target, base - cfg.max_contract, base + cfg.max_extend)


def run_delta_sample(cfg: CollectConfig, delta_goal: np.ndarray) -> dict[str, np.ndarray | float]:
    env = AirSceneCollectorEnv(cfg)
    passive_lengths = collect_passive_lengths(env, cfg.steps)

    env.reset()
    tip_trace = []
    for t in range(cfg.steps):
        target_lengths = target_lengths_from_delta(passive_lengths, delta_goal, t, cfg)
        env.step_pid(target_lengths)
        tip_trace.append(env.get_tip_pos())

    tip_trace_arr = np.asarray(tip_trace, dtype=float)
    hold_n = min(cfg.hold_steps, cfg.steps)
    hold_tip = tip_trace_arr[-hold_n:]
    hold_mean = hold_tip.mean(axis=0)
    hold_std = hold_tip.std(axis=0)
    hold_jitter = float(np.mean(np.sum((hold_tip - hold_mean) ** 2, axis=1)))

    return {
        "delta": np.asarray(delta_goal, dtype=float),
        "final_tip": tip_trace_arr[-1],
        "hold_mean_tip": hold_mean,
        "hold_std_tip": hold_std,
        "hold_jitter": hold_jitter,
    }


def validate_delta_bounds(cfg: CollectConfig) -> tuple[np.ndarray, np.ndarray]:
    lo = np.asarray(cfg.delta_min, dtype=float)
    hi = np.asarray(cfg.delta_max, dtype=float)
    expected = rope_count()
    if lo.shape != (expected,) or hi.shape != (expected,):
        raise ValueError(
            f"delta_min / delta_max 必须各有 {expected} 个数，"
            f"当前分别是 {lo.size} / {hi.size} 个。"
        )
    if np.any(lo > hi):
        raise ValueError("delta_min 中不能有元素大于 delta_max。")
    return lo, hi


def sample_deltas(cfg: CollectConfig) -> np.ndarray:
    rng = np.random.default_rng(cfg.seed)
    lo, hi = validate_delta_bounds(cfg)
    return rng.uniform(lo, hi, size=(cfg.samples, rope_count()))


def assemble_dataset(results: list[dict], cfg: CollectConfig, status: str) -> dict:
    return {
        "delta": np.asarray([r["delta"] for r in results], dtype=float),
        "final_tip": np.asarray([r["final_tip"] for r in results], dtype=float),
        "hold_mean_tip": np.asarray([r["hold_mean_tip"] for r in results], dtype=float),
        "hold_std_tip": np.asarray([r["hold_std_tip"] for r in results], dtype=float),
        "hold_jitter": np.asarray([r["hold_jitter"] for r in results], dtype=float),
        "meta": {
            "scene": cfg.scene,
            "samples": cfg.samples,
            "successful_samples": len(results),
            "status": status,
            "seed": cfg.seed,
            "delta_min": cfg.delta_min,
            "delta_max": cfg.delta_max,
            "steps": cfg.steps,
            "ramp_steps": cfg.ramp_steps,
            "hold_steps": cfg.hold_steps,
            "max_contract": cfg.max_contract,
            "max_extend": cfg.max_extend,
            "qacc_abs_limit": QACC_ABS_LIMIT,
            "state_abs_limit": STATE_ABS_LIMIT,
            "rope_actuators": ROPE_ACTUATORS,
            "rope_tendons": ROPE_TENDONS,
            "tip_site_name": TIP_SITE_NAME,
            "tip_body_fallback": TIP_BODY_FALLBACK,
            "plate_joint_qpos": PLATE_JOINT_QPOS,
            "plate_position_ctrls": PLATE_POSITION_CTRLS,
            "pid": asdict(cfg.pid),
        },
    }


def partial_output_path(output_path: str | Path) -> Path:
    output_path = Path(output_path)
    return output_path.with_name(f"{output_path.stem}.partial{output_path.suffix}")


def collect_dataset(cfg: CollectConfig) -> dict:
    deltas = sample_deltas(cfg)
    results: list[dict] = []

    def build_or_raise(status: str) -> dict:
        if not results:
            raise RuntimeError("没有成功采集到任何样本")
        return assemble_dataset(results, cfg, status=status)

    if cfg.workers <= 1:
        try:
            for i, delta in enumerate(deltas, start=1):
                try:
                    results.append(run_delta_sample(cfg, delta))
                except Exception as exc:
                    print(f"[WARN] sample {i} failed and skipped: {exc}")
                if i % 20 == 0 or i == cfg.samples:
                    print(f"collected {i}/{cfg.samples}, valid {len(results)}")
        except KeyboardInterrupt:
            print("\n[WARN] 用户中断采集，准备保存已完成的有效样本。")
            return build_or_raise(status="interrupted")
        return build_or_raise(status="complete")

    pool = ProcessPoolExecutor(max_workers=cfg.workers)
    futures = []
    try:
        futures = [pool.submit(run_delta_sample, cfg, delta) for delta in deltas]
        for i, future in enumerate(as_completed(futures), start=1):
            try:
                results.append(future.result())
            except Exception as exc:
                print(f"[WARN] sample failed and skipped: {exc}")
            if i % 20 == 0 or i == cfg.samples:
                print(f"finished {i}/{cfg.samples}, valid {len(results)}")
    except KeyboardInterrupt:
        print("\n[WARN] 用户中断采集，正在取消未完成任务并保存已完成的有效样本。")
        for future in futures:
            future.cancel()
        pool.shutdown(wait=False, cancel_futures=True)
        return build_or_raise(status="interrupted")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    return build_or_raise(status="complete")

def save_dataset(output_path: str | Path, data: dict) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_path,
        delta=data["delta"],
        final_ee=data["final_tip"],
        hold_mean_ee=data["hold_mean_tip"],
        hold_std_ee=data["hold_std_tip"],
        hold_jitter=data["hold_jitter"],
        meta_json=json.dumps(data["meta"], ensure_ascii=False),
    )

    csv_path = output_path.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(delta_column_names() + ["x", "y", "z", "std_x", "std_y", "std_z", "jitter"])
        for delta, tip, std, jitter in zip(
            data["delta"],
            data["hold_mean_tip"],
            data["hold_std_tip"],
            data["hold_jitter"],
        ):
            writer.writerow([*delta.tolist(), *tip.tolist(), *std.tolist(), float(jitter)])

    print(f"saved npz: {output_path}")
    print(f"saved csv: {csv_path}")


def rope_count() -> int:
    return len(ROPE_TENDONS)


def delta_column_names() -> list[str]:
    return [f"d{i}" for i in range(1, rope_count() + 1)]


def parse_delta_vector(values: list[str], name: str) -> tuple[float, ...]:
    expected = rope_count()
    if len(values) != expected:
        raise argparse.ArgumentTypeError(f"{name} 需要 {expected} 个数，对应 {expected} 根绳")
    return tuple(float(v) for v in values)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="采集六绳软体臂工作空间数据集，平台 qpos 与 position ctrl 锁定一致")
    parser.add_argument("--scene", default=DEFAULT_SCENE, help="MuJoCo 场景 XML 路径")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="输出 .npz 路径；同时生成同名 .csv")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--delta-min", nargs=rope_count(), default=["-0.5"] * rope_count())
    parser.add_argument("--delta-max", nargs=rope_count(), default=["0.02"] * rope_count())
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--ramp-steps", type=int, default=200)
    parser.add_argument("--hold-steps", type=int, default=120)
    parser.add_argument("--kp", type=float, default=60.0)
    parser.add_argument("--kd", type=float, default=1.5)
    parser.add_argument("--force-max", type=float, default=15.0)
    parser.add_argument("--force-rate-limit", type=float, default=150.0, help="小于 0 表示不限制")
    parser.add_argument(
        "--no-strict-plate-joint",
        action="store_true",
        help="找不到 plate_tx / plate_ty 时只警告并继续；默认直接报错",
    )
    parser.add_argument(
        "--strict-plate-ctrl",
        action="store_true",
        help="找不到 plate_x_ctrl / plate_y_ctrl 时直接报错；默认只警告并继续",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    pid = PIDConfig(
        kp=args.kp,
        kd=args.kd,
        force_max=args.force_max,
        force_rate_limit=None if args.force_rate_limit < 0 else args.force_rate_limit,
    )
    cfg = CollectConfig(
        scene=args.scene,
        output=args.output,
        samples=args.samples,
        workers=args.workers,
        seed=args.seed,
        delta_min=parse_delta_vector(args.delta_min, "--delta-min"),
        delta_max=parse_delta_vector(args.delta_max, "--delta-max"),
        steps=args.steps,
        ramp_steps=args.ramp_steps,
        hold_steps=args.hold_steps,
        pid=pid,
        strict_plate_joint=not args.no_strict_plate_joint,
        strict_plate_ctrl=args.strict_plate_ctrl,
    )

    print("collection config:")
    print(json.dumps({**asdict(cfg), "pid": asdict(pid)}, indent=2, ensure_ascii=False))
    print(f"plate joint qpos: {PLATE_JOINT_QPOS}")
    print(f"plate position ctrls: {PLATE_POSITION_CTRLS}")

    data = collect_dataset(cfg)
    output_path = partial_output_path(cfg.output) if data["meta"].get("status") == "interrupted" else Path(cfg.output)
    save_dataset(output_path, data)


if __name__ == "__main__":
    main()
