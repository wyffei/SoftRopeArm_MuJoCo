"""
基于数据集邻近点/加权邻近点的六绳软体机械臂四点顺序到达演示控制，平台位置由 Python 指定加载

放置位置建议：
    new/demo_dataset_four_points_plate_qpos_ctrl_locked_control.py

运行：
    cd new
    python demo_dataset_four_points_control.py

依赖：
    pip install mujoco pandas numpy

说明：
    数据集 workspace_6rope.csv 提供近似映射：
        末端位置 x,y,z  ->  rope 收缩量 d1..d6

    本脚本实现：
        指定 4 个空间点 WAYPOINTS
        机械臂从当前初始末端位置出发，按顺序到达 P1 -> P2 -> P3 -> P4
        每段之间平滑插值
        每到一个点保持一段时间
        通过数据集 KNN 查表得到 d1..d6
        再用 PID 控制 tendon length

    平面/平台位置处理：
        在 Python 里直接设置 moving_plate 的 slide joint 初始 qpos：
            plate_tx = 0.27
            plate_ty = -0.16
        同时让位置控制器 plate_x_ctrl / plate_y_ctrl 保持同样的目标值：
            plate_x_ctrl = 0.27
            plate_y_ctrl = -0.16

        这样 position actuator 不会把平台从指定位置拉回 0。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np
import pandas as pd


# ============================================================
# 你主要改这里
# ============================================================
ROOT = Path(__file__).resolve().parent
XML_PATH = ROOT / "scene.xml"
DATASET_CSV = ROOT / "dataset" / "workspace_6rope.csv"

# 目标点建议选在数据集覆盖范围内。
# 按顺序到达：P1 -> P2 -> P3 -> P4
# 你之后只需要改这里的 4 个点。
WAYPOINTS = np.array(
    [
        [-0.04, -0.2, 0.72],  # P3
        [-0.03, -0.4, 0.71],  # P4
        [-0.04, -0.2, 1.08],  # P1
        [-0.04, -0.4, 1.14],  # P2
    ],
    dtype=float,
)

INITIAL_TO_P1_MOVE_TIME = 2.0  # 红点从初始末端位置平滑移动到 P1 的时间，单位 s
MOVE_TIME = 2.0               # 红点在相邻两个点之间平滑移动的时间，单位 s
FIRST_POINT_HOLD_TIME = 0.1   # 真正到达 P1 后保持时间，单位 s
WAYPOINT_HOLD_TIME = 0.1      # 真正到达 P2/P3/... 后保持时间，单位 s
LOOP_TRAJECTORY = False       # False: 到最后一个点后一直保持；True: P1->P2->... 循环

# 到点判定 + 超时保护：
# move 阶段不再“到时间就切 hold”，而是优先看实际末端 tip 是否到点。
# 如果一直到不了，则超过最大 move 时间后强制进入 hold/切下一个点，避免卡死。
WAYPOINT_REACHED_TOL = 0.08      # 末端距离目标点小于该值，认为到达，单位 m
INITIAL_TO_P1_MAX_MOVE_TIME = 2.0 # 初始末端 -> P1 的最大等待时间，单位 s
WAYPOINT_MAX_MOVE_TIME = 2.0      # Pi -> P(i+1) 的最大等待时间，单位 s

USE_KNN_BLEND = True         # True 更平滑；False 为严格最近邻
K_NEIGHBORS = 1              # 加权邻近点数量
DELTA_SMOOTH_ALPHA = 0.4    # d1..d6 低通滤波，越小越平滑但越慢
PRINT_EVERY = 0.25           # 每隔多少秒打印一次状态
PRINT_VERBOSE_EVERY = 1.0   # 每隔多少秒打印一次详细诊断
SHOW_DEBUG_MARKERS = True   # Viewer 中显示目标点/数据集最近点/实际末端点
DEBUG_MARKER_RADIUS = 0.018

TENDON_NAMES = ["rope1", "rope2", "rope3", "rope_add1", "rope_add2", "rope_add3"]
ACTUATOR_NAMES = ["pull_rope1", "pull_rope2", "pull_rope3", "pull_rope_add1", "pull_rope_add2", "pull_rope_add3"]
TIP_SITE_NAME = "m20_bottom_center"

# ============================================================
# moving_plate 加载位置设置
# ============================================================
PLATE_JOINT_QPOS = {
    "plate_tx": 0.27,
    "plate_ty": -0.16,
}

PLATE_POSITION_CTRL = {
    "plate_x_ctrl": 0.27,
    "plate_y_ctrl": -0.16,
}


# ============================================================
# PID 参数
# ============================================================
@dataclass
class RopePIDConfig:
    kp: float = 60.0
    ki: float = 0.0
    kd: float = 1.5

    integral_min: float = -0.1
    integral_max: float = 0.1

    force_min: float = 0.0
    force_max: float = 15.0

    # 目标绳长变化限速，单位 m/s；None 表示不用
    target_rate_limit: float | None = 1.0

    # D 项低通滤波时间常数
    d_filter_tau: float = 0.035

    # 小误差死区，单位 m
    deadband: float = 1e-3


class RopeLengthPID:
    def __init__(self, cfg: RopePIDConfig):
        self.cfg = cfg
        self.integral = 0.0
        self.prev_length = None
        self.prev_target = None
        self.dlength_filt = 0.0

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_length = None
        self.prev_target = None
        self.dlength_filt = 0.0

    def _rate_limit_target(self, target: float, dt: float) -> float:
        if self.cfg.target_rate_limit is None or self.prev_target is None:
            self.prev_target = target
            return target

        max_step = self.cfg.target_rate_limit * dt
        limited = float(np.clip(target, self.prev_target - max_step, self.prev_target + max_step))
        self.prev_target = limited
        return limited

    def update(self, target_length: float, current_length: float, dt: float) -> float:
        if dt <= 0.0:
            return 0.0

        target_length = self._rate_limit_target(float(target_length), dt)

        # tendon 比目标长 -> 需要拉；比目标短 -> 不拉/少拉
        error = current_length - target_length
        if abs(error) < self.cfg.deadband:
            error = 0.0

        p = self.cfg.kp * error

        self.integral += error * dt
        self.integral = float(np.clip(self.integral, self.cfg.integral_min, self.cfg.integral_max))
        i = self.cfg.ki * self.integral

        if self.prev_length is None:
            raw_dlength = 0.0
        else:
            raw_dlength = (current_length - self.prev_length) / dt

        tau = max(self.cfg.d_filter_tau, 1e-6)
        alpha = dt / (tau + dt)
        self.dlength_filt += alpha * (raw_dlength - self.dlength_filt)
        d = -self.cfg.kd * self.dlength_filt

        u = p + i + d
        force_cmd = float(np.clip(u, self.cfg.force_min, self.cfg.force_max))

        # anti-windup
        if force_cmd <= self.cfg.force_min and error <= 0.0:
            self.integral -= error * dt
        if force_cmd >= self.cfg.force_max and error > 0.0:
            self.integral -= error * dt
        self.integral = float(np.clip(self.integral, self.cfg.integral_min, self.cfg.integral_max))

        self.prev_length = current_length
        return force_cmd


class RopeLengthController:
    def __init__(self, model, data, actuator_name: str, tendon_name: str, cfg: RopePIDConfig):
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

        local_cfg = RopePIDConfig(**asdict(cfg))
        if model.actuator_ctrllimited[self.actuator_id]:
            ctrl_min, ctrl_max = model.actuator_ctrlrange[self.actuator_id]
            local_cfg.force_min = max(local_cfg.force_min, float(ctrl_min))
            local_cfg.force_max = min(local_cfg.force_max, float(ctrl_max))
        self.pid = RopeLengthPID(local_cfg)

    def get_tendon_length(self) -> float:
        return float(self.data.ten_length[self.tendon_id])

    def step(self, target_length: float) -> tuple[float, float]:
        dt = float(self.model.opt.timestep)
        current_length = self.get_tendon_length()
        force_cmd = self.pid.update(target_length, current_length, dt)
        self.data.ctrl[self.actuator_id] = force_cmd
        return current_length, force_cmd


# ============================================================
# 数据集近邻 IK / 查表控制
# ============================================================
class DatasetIK:
    def __init__(self, csv_path: Path, use_knn_blend: bool = True, k: int = 8):
        if not csv_path.exists():
            raise FileNotFoundError(f"找不到数据集: {csv_path}")

        df = pd.read_csv(csv_path)

        delta_cols = [f"d{i}" for i in range(1, len(TENDON_NAMES) + 1)]
        required = ["x", "y", "z", *delta_cols]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"数据集缺少列: {missing}")

        df = df.dropna(subset=required).reset_index(drop=True)
        if len(df) == 0:
            raise ValueError("数据集为空，或者 x/y/z/d1..d6 全部无效。")

        self.xyz = df[["x", "y", "z"]].to_numpy(dtype=float)
        self.delta = df[delta_cols].to_numpy(dtype=float)
        self.use_knn_blend = use_knn_blend
        self.k = int(max(1, min(k, len(df))))

        self.xyz_min = self.xyz.min(axis=0)
        self.xyz_max = self.xyz.max(axis=0)

    def query(self, target_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, bool, int]:
        """
        返回：
            delta_cmd: d1..d6，单位 m，负数表示收缩
            nearest_xyz: 数据集中最近的末端点
            nearest_dist: 最近点距离
            in_range: target 是否在数据集 xyz 包围盒内
            nearest_idx: 最近点在数据集中的行号
        """
        target_xyz = np.asarray(target_xyz, dtype=float).reshape(3)

        diff = self.xyz - target_xyz[None, :]
        dist2 = np.einsum("ij,ij->i", diff, diff)

        nearest_idx = int(np.argmin(dist2))
        nearest_xyz = self.xyz[nearest_idx]
        nearest_dist = float(np.sqrt(dist2[nearest_idx]))

        in_range = bool(np.all(target_xyz >= self.xyz_min) and np.all(target_xyz <= self.xyz_max))

        if not self.use_knn_blend or self.k == 1:
            return self.delta[nearest_idx].copy(), nearest_xyz.copy(), nearest_dist, in_range, nearest_idx

        # k 近邻反距离加权，比单个最近邻更平滑
        idx = np.argpartition(dist2, self.k - 1)[: self.k]
        d = np.sqrt(dist2[idx])
        w = 1.0 / (d + 1e-6)
        w /= w.sum()
        delta_cmd = (self.delta[idx] * w[:, None]).sum(axis=0)
        return delta_cmd, nearest_xyz.copy(), nearest_dist, in_range, nearest_idx


def smoothstep(s: float) -> float:
    s = float(np.clip(s, 0.0, 1.0))
    return s * s * (3.0 - 2.0 * s)


def set_plate_joint_initial_qpos(model, data, joint_qpos: dict[str, float]) -> None:
    """
    直接设置 moving_plate 的 slide joint 初始 qpos。

    注意：这里设置的是 slide joint 位移，不是世界坐标。
    为了避免 position actuator 把平台拉回 0，后续需要让
    plate_x_ctrl / plate_y_ctrl 保持与这里一致的目标值。
    """
    for joint_name, value in joint_qpos.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            print(f"提示：找不到 joint '{joint_name}'，跳过 qpos 设置。")
            continue

        qadr = int(model.jnt_qposadr[joint_id])
        dadr = int(model.jnt_dofadr[joint_id])
        data.qpos[qadr] = float(value)
        data.qvel[dadr] = 0.0
        print(f"设置加载位置: joint '{joint_name}' qpos = {float(value):.4f}")


def build_plate_position_ctrl_targets(model, ctrl_targets: dict[str, float]) -> list[tuple[str, int, float]]:
    """
    查找平台位置控制器，并保存需要保持的 ctrl 目标值。

    如果 XML 中 plate_x_ctrl / plate_y_ctrl 是 position actuator，
    ctrl 就是对应 slide joint 的目标 qpos。
    因此这里应该让 ctrl 与 PLATE_JOINT_QPOS 一致，
    否则平台会被 position actuator 拉回 ctrl 指定的位置。
    """
    targets: list[tuple[str, int, float]] = []

    for actuator_name, ctrl_value in ctrl_targets.items():
        act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        if act_id < 0:
            print(f"提示：场景中没有 actuator '{actuator_name}'，跳过平台 ctrl 保持。")
            continue

        value = float(ctrl_value)
        targets.append((actuator_name, act_id, value))
        print(f"平台位置控制器保持: {actuator_name} ctrl = {value:.4f}")

    return targets


def apply_plate_position_ctrl(data, ctrl_targets: list[tuple[str, int, float]]) -> None:
    """每一步都把平台 position actuator 的目标值保持在指定位置。"""
    for _, act_id, value in ctrl_targets:
        data.ctrl[act_id] = value


class WaypointSequencer:
    """
    基于“实际是否到点”的轨迹状态机。

    和原来的纯时间轨迹不同：
        1. move 阶段：红点会平滑移动到当前目标点；
        2. 红点到目标点后会停在目标点，继续等待机械臂实际末端 tip 到点；
        3. tip_err <= WAYPOINT_REACHED_TOL 时，才进入 hold；
        4. 如果超过最大 move 时间还没到点，则触发 timeout，强制进入 hold/下一个点。
    """

    def __init__(self, initial_tip_xyz: np.ndarray):
        self.points = np.asarray(WAYPOINTS, dtype=float)
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError("WAYPOINTS 必须是形状为 (N, 3) 的数组。")
        if len(self.points) < 1:
            raise ValueError("WAYPOINTS 至少需要 1 个点。")

        self.target_index = 0
        self.phase = "move"
        self.segment_start_time = 0.0
        self.phase_start_time = 0.0
        self.segment_start_pos = np.asarray(initial_tip_xyz, dtype=float).reshape(3).copy()
        self.segment_target_pos = self.points[0].copy()
        self.is_initial_segment = True
        self.last_event = "start"

    def _move_ramp_time(self) -> float:
        return INITIAL_TO_P1_MOVE_TIME if self.is_initial_segment else MOVE_TIME

    def _move_timeout(self) -> float:
        return INITIAL_TO_P1_MAX_MOVE_TIME if self.is_initial_segment else WAYPOINT_MAX_MOVE_TIME

    def _hold_time(self) -> float:
        return FIRST_POINT_HOLD_TIME if self.target_index == 0 else WAYPOINT_HOLD_TIME

    def _start_next_move(self, sim_t: float, current_tip_xyz: np.ndarray) -> None:
        if self.target_index >= len(self.points) - 1:
            if LOOP_TRAJECTORY:
                self.target_index = 0
            else:
                # 非循环模式：最后一个点 hold 完以后，继续保持最后一个点。
                self.phase = "hold"
                self.phase_start_time = sim_t
                return
        else:
            self.target_index += 1

        self.phase = "move"
        self.segment_start_time = sim_t
        self.phase_start_time = sim_t
        self.segment_start_pos = np.asarray(current_tip_xyz, dtype=float).reshape(3).copy()
        self.segment_target_pos = self.points[self.target_index].copy()
        self.is_initial_segment = False

    def update(self, sim_t: float, current_tip_xyz: np.ndarray) -> tuple[np.ndarray, int, str, str | None, float, float, float]:
        """
        返回：
            target_xyz: 当前红点/查表目标
            target_index: 当前目标点编号，0 表示 P1
            phase: move 或 hold
            event: 状态切换信息；无切换时为 None
            move_elapsed: 当前 move 已用时间
            hold_elapsed: 当前 hold 已用时间
            waypoint_err: 实际末端到当前 waypoint 的距离
        """
        current_tip_xyz = np.asarray(current_tip_xyz, dtype=float).reshape(3)
        waypoint_xyz = self.points[self.target_index]
        waypoint_err = float(np.linalg.norm(current_tip_xyz - waypoint_xyz))
        event: str | None = None

        if self.phase == "move":
            move_elapsed = float(sim_t - self.segment_start_time)
            ramp_time = max(self._move_ramp_time(), 1e-6)
            s = smoothstep(move_elapsed / ramp_time)
            target_xyz = (1.0 - s) * self.segment_start_pos + s * self.segment_target_pos

            reached = waypoint_err <= WAYPOINT_REACHED_TOL
            timed_out = move_elapsed >= self._move_timeout()
            if reached or timed_out:
                self.phase = "hold"
                self.phase_start_time = sim_t
                target_xyz = waypoint_xyz.copy()
                reason = "reached" if reached else "timeout"
                event = (
                    f"P{self.target_index + 1} enter hold by {reason}: "
                    f"wp_err={waypoint_err:.4f}, move_elapsed={move_elapsed:.2f}s"
                )

            return target_xyz, self.target_index, self.phase, event, move_elapsed, 0.0, waypoint_err

        # hold 阶段：目标保持在当前 waypoint。
        hold_elapsed = float(sim_t - self.phase_start_time)
        hold_time = self._hold_time()
        target_xyz = waypoint_xyz.copy()

        if hold_elapsed >= hold_time:
            old_index = self.target_index
            self._start_next_move(sim_t, current_tip_xyz)
            if self.phase == "move":
                event = f"P{old_index + 1} hold finished, switch to P{self.target_index + 1} move"
            else:
                event = f"P{old_index + 1} hold finished, keep final waypoint"

        return target_xyz, self.target_index, self.phase, event, 0.0, hold_elapsed, waypoint_err

def print_waypoints() -> None:
    print("四个目标点：")
    for i, p in enumerate(WAYPOINTS, start=1):
        print(f"  P{i}: [{p[0]: .4f}, {p[1]: .4f}, {p[2]: .4f}]")



def add_debug_sphere(scene, pos: np.ndarray, radius: float, rgba: np.ndarray) -> None:
    """在 MuJoCo viewer.user_scn 中添加一个调试小球。"""
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, 0.0, 0.0], dtype=float),
        np.asarray(pos, dtype=float).reshape(3),
        np.eye(3).reshape(-1),
        np.asarray(rgba, dtype=float).reshape(4),
    )
    scene.ngeom += 1


def update_debug_markers(viewer, target_xyz: np.ndarray, nearest_xyz: np.ndarray, tip_xyz: np.ndarray) -> None:
    """
    Viewer 额外显示三类点：
        红色：当前期望 target
        绿色：数据集中最近点 nearest
        蓝色：实际末端 tip
    """
    if not SHOW_DEBUG_MARKERS:
        return
    scene = viewer.user_scn
    scene.ngeom = 0
    add_debug_sphere(scene, target_xyz, DEBUG_MARKER_RADIUS, np.array([1.0, 0.05, 0.05, 0.85]))
    add_debug_sphere(scene, nearest_xyz, DEBUG_MARKER_RADIUS * 0.85, np.array([0.05, 1.0, 0.05, 0.85]))
    add_debug_sphere(scene, tip_xyz, DEBUG_MARKER_RADIUS * 0.70, np.array([0.05, 0.25, 1.0, 0.85]))


def main() -> None:
    print(f"加载模型: {XML_PATH}")
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    # 直接指定 moving_plate 的加载位置。
    # 同时让 position actuator 的 ctrl 目标值与 qpos 一致，防止平台被拉回 0。
    set_plate_joint_initial_qpos(model, data, PLATE_JOINT_QPOS)
    plate_ctrl_targets = build_plate_position_ctrl_targets(model, PLATE_POSITION_CTRL)

    print(f"加载数据集: {DATASET_CSV}")
    ik = DatasetIK(DATASET_CSV, use_knn_blend=USE_KNN_BLEND, k=K_NEIGHBORS)
    print("数据集 xyz 范围:")
    print(f"  min = {ik.xyz_min}")
    print(f"  max = {ik.xyz_max}")
    print_waypoints()

    tip_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, TIP_SITE_NAME)
    if tip_site_id < 0:
        raise ValueError(f"找不到 site: {TIP_SITE_NAME}")

    cfg = RopePIDConfig()
    controllers = [
        RopeLengthController(model, data, act, ten, cfg)
        for act, ten in zip(ACTUATOR_NAMES, TENDON_NAMES)
    ]

    # 让平台位置控制器保持在与加载 qpos 相同的目标值。
    # 这里拿到的 initial_lengths 对应的是“plate_tx/plate_ty 已指定 + ctrl 同位置保持”下的初始绳长。
    apply_plate_position_ctrl(data, plate_ctrl_targets)
    mujoco.mj_forward(model, data)
    initial_tip_xyz = data.site_xpos[tip_site_id].copy()
    initial_lengths = np.array([c.get_tendon_length() for c in controllers], dtype=float)
    print(f"初始末端位置 initial_tip_xyz = {initial_tip_xyz}")
    print(f"初始 tendon lengths = {initial_lengths}")

    follower = WaypointSequencer(initial_tip_xyz)

    delta_filtered = np.zeros(len(TENDON_NAMES), dtype=float)
    last_print_t = -1e9
    last_verbose_t = -1e9
    last_target_index = -1

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 2.2
        viewer.cam.azimuth = 120
        viewer.cam.elevation = -20
        viewer.cam.lookat[:] = np.array([0.0, 0.0, 0.75])

        wall_start = time.time()
        while viewer.is_running():
            sim_t = float(data.time)

            # 平台位置控制器始终保持在指定加载位置，避免被拉回 0。
            apply_plate_position_ctrl(data, plate_ctrl_targets)

            # 轨迹状态机：优先根据实际末端是否到点决定是否进入 hold，
            # 同时保留最大 move 时间，避免机械臂一直不到点时卡死。
            tip_before_step = data.site_xpos[tip_site_id].copy()
            target_xyz, target_index, phase, switch_event, move_elapsed, hold_elapsed, waypoint_err = follower.update(
                sim_t, tip_before_step
            )
            delta_raw, nearest_xyz, nearest_dist, in_range, nearest_idx = ik.query(target_xyz)

            # 低通滤波，减少近邻切换造成的跳变。
            delta_filtered = (1.0 - DELTA_SMOOTH_ALPHA) * delta_filtered + DELTA_SMOOTH_ALPHA * delta_raw

            # 数据集里的 d 是相对初始绳长的变化；负数表示收缩。
            target_lengths = initial_lengths + delta_filtered

            # tendon range 保护。
            for i, c in enumerate(controllers):
                if model.tendon_limited[c.tendon_id]:
                    lo, hi = model.tendon_range[c.tendon_id]
                    target_lengths[i] = np.clip(target_lengths[i], lo, hi)

            forces = []
            current_lengths = []
            for c, target_length in zip(controllers, target_lengths):
                current_length, force = c.step(float(target_length))
                current_lengths.append(current_length)
                forces.append(force)

            mujoco.mj_step(model, data)
            tip_xyz = data.site_xpos[tip_site_id].copy()
            update_debug_markers(viewer, target_xyz, nearest_xyz, tip_xyz)
            viewer.sync()

            if switch_event is not None:
                print(f"状态切换: {switch_event}, target={target_xyz.round(4)}")

            if target_index != last_target_index:
                print(f"当前目标: P{target_index + 1}, phase={phase}, target={target_xyz.round(4)}")
                last_target_index = target_index

            if sim_t - last_print_t >= PRINT_EVERY:
                tip_err = float(np.linalg.norm(tip_xyz - target_xyz))
                nearest_tip_err = float(np.linalg.norm(tip_xyz - nearest_xyz))
                active = [name for name, force in zip(TENDON_NAMES, forces) if force > 1e-3]
                print(
                    f"t={sim_t:6.2f} | "
                    f"P{target_index + 1} {phase:4s} | "
                    f"target={target_xyz.round(3)} | "
                    f"tip={tip_xyz.round(3)} | "
                    f"tip_err={tip_err:.4f} | "
                    f"wp_err={waypoint_err:.4f} | "
                    f"move_t={move_elapsed:.2f} | "
                    f"hold_t={hold_elapsed:.2f} | "
                    f"nn_idx={nearest_idx} | "
                    f"nn={nearest_xyz.round(3)} | "
                    f"nn_err={nearest_dist:.4f} | "
                    f"tip_to_nn={nearest_tip_err:.4f} | "
                    f"in_range={in_range} | "
                    f"active={active}"
                )
                last_print_t = sim_t

            if sim_t - last_verbose_t >= PRINT_VERBOSE_EVERY:
                current_lengths = np.asarray(current_lengths, dtype=float)
                forces_arr = np.asarray(forces, dtype=float)
                length_err = current_lengths - target_lengths
                print(
                    "  诊断:\n"
                    f"    delta_raw      = {delta_raw.round(4)}\n"
                    f"    delta_filtered = {delta_filtered.round(4)}\n"
                    f"    init_lengths   = {initial_lengths.round(4)}\n"
                    f"    target_lengths = {target_lengths.round(4)}\n"
                    f"    current_lengths= {current_lengths.round(4)}\n"
                    f"    length_err     = {length_err.round(4)}  # >0 会拉，<=0 通常不拉\n"
                    f"    force          = {forces_arr.round(3)}"
                )
                last_verbose_t = sim_t

            # 尽量按真实时间播放。
            # elapsed = time.time() - wall_start
            # sleep_time = data.time - elapsed
            # if sleep_time > 0:
            #     time.sleep(min(sleep_time, 0.01))


if __name__ == "__main__":
    main()
