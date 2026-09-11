"""
Generate interactive HTML visualization for soft-arm workspace.

This script creates a Plotly HTML file:
- Browser-based interactive 3D view: drag to rotate, scroll to zoom.
- Workspace samples from CSV x/y/z.
- Environment objects: floor, tray/table board, screen, moved plate.
- Corrected moved plate and m1_top nominal center using PLATE_POSITION_CTRL.

Install in Ubuntu venv:
    python -m pip install pandas numpy plotly

Usage:
    python Interactive_workspace_visualization_html.py \
        --csv workspace_6rope.csv \
        --output interactive_workspace_moved_plate_m1top.html

Then open:
    xdg-open interactive_workspace_moved_plate_m1top.html
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go


DELTA_COLUMNS = [f"d{i}" for i in range(1, 7)]

REQUIRED_COLUMNS = [
    *DELTA_COLUMNS,
    "x", "y", "z",
    "std_x", "std_y", "std_z",
    "jitter",
]


PLATE_POSITION_CTRL = {
    "plate_x_ctrl": 0.27,
    "plate_y_ctrl": -0.16,
}


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV missing required columns: {missing}\n"
            f"Existing columns: {list(df.columns)}"
        )

    df = df.dropna(subset=["x", "y", "z", *DELTA_COLUMNS]).reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("Dataset is empty after removing NaN rows.")

    return df


def make_box_mesh(
    name: str,
    center: np.ndarray,
    half_size: np.ndarray,
    opacity: float = 0.25,
) -> go.Mesh3d:
    """
    Create a transparent box mesh.

    Note:
        MuJoCo box geom uses half-size:
            full size = 2 * size
    """
    cx, cy, cz = center
    sx, sy, sz = half_size

    vertices = np.array([
        [cx - sx, cy - sy, cz - sz],
        [cx + sx, cy - sy, cz - sz],
        [cx + sx, cy + sy, cz - sz],
        [cx - sx, cy + sy, cz - sz],
        [cx - sx, cy - sy, cz + sz],
        [cx + sx, cy - sy, cz + sz],
        [cx + sx, cy + sy, cz + sz],
        [cx - sx, cy + sy, cz + sz],
    ])

    faces = np.array([
        [0, 1, 2], [0, 2, 3],
        [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4],
        [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6],
        [3, 0, 4], [3, 4, 7],
    ])

    return go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        name=name,
        opacity=opacity,
        hovertemplate=(
            f"<b>{name}</b><br>"
            f"center=({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}) m<br>"
            f"half-size=({half_size[0]:.4f}, {half_size[1]:.4f}, {half_size[2]:.4f}) m"
            "<extra></extra>"
        ),
        showscale=False,
    )


def make_box_wire(
    name: str,
    center: np.ndarray,
    half_size: np.ndarray,
    width: float = 4,
) -> go.Scatter3d:
    """
    Create a box wireframe.
    """
    cx, cy, cz = center
    sx, sy, sz = half_size

    vertices = np.array([
        [cx - sx, cy - sy, cz - sz],
        [cx + sx, cy - sy, cz - sz],
        [cx + sx, cy + sy, cz - sz],
        [cx - sx, cy + sy, cz - sz],
        [cx - sx, cy - sy, cz + sz],
        [cx + sx, cy - sy, cz + sz],
        [cx + sx, cy + sy, cz + sz],
        [cx - sx, cy + sy, cz + sz],
    ])

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    x, y, z = [], [], []
    for a, b in edges:
        x += [vertices[a, 0], vertices[b, 0], None]
        y += [vertices[a, 1], vertices[b, 1], None]
        z += [vertices[a, 2], vertices[b, 2], None]

    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="lines",
        name=f"{name} edge",
        line=dict(width=width),
        hoverinfo="skip",
        showlegend=False,
    )


def add_box(
    fig: go.Figure,
    name: str,
    center: np.ndarray,
    half_size: np.ndarray,
    opacity: float,
) -> None:
    fig.add_trace(make_box_mesh(name, center, half_size, opacity=opacity))
    fig.add_trace(make_box_wire(name, center, half_size))


def build_figure(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    # ============================================================
    # 1. Environment objects from scene.xml
    # ============================================================

    # <geom name="screen" type="box" pos="0 -0.3 1.1" size="0.01 0.2 0.1" />
    screen_center = np.array([0.0, -0.3, 1.1])
    screen_half_size = np.array([0.01, 0.2, 0.1])

    # <geom name="tray" type="box" pos="0 -0.3 0.7" size="0.15 0.2 0.01" />
    tray_center = np.array([0.0, -0.3, 0.7])
    tray_half_size = np.array([0.15, 0.2, 0.01])

    # <geom name="extra_floor" type="box" pos="0 0 -0.03" size="4.2 2.8 0.01" />
    floor_center = np.array([0.0, 0.0, -0.03])
    floor_half_size = np.array([4.2, 2.8, 0.01])

    # ============================================================
    # 2. Moved plate position
    # ============================================================

    # scene.xml:
    # <body name="moving_plate" pos="-0.4 -0.4 0">
    moving_plate_body_pos_initial = np.array([-0.4, -0.4, 0.0])

    # actual slide-joint controlled offset:
    moving_plate_body_pos_moved = moving_plate_body_pos_initial + np.array([
        PLATE_POSITION_CTRL["plate_x_ctrl"],
        PLATE_POSITION_CTRL["plate_y_ctrl"],
        0.0,
    ])

    # scene.xml:
    # <geom name="plate" type="box" pos="0 -0.05 0.80" size="0.1 0.1 0.01" />
    plate_local_pos = np.array([0.0, -0.05, 0.80])
    plate_center_moved = moving_plate_body_pos_moved + plate_local_pos
    plate_half_size = np.array([0.1, 0.1, 0.01])

    # ============================================================
    # 3. m1_top moved nominal center
    # ============================================================

    # module_body.xml:
    # <body name="module_mount" pos="0 0.3 0.1">
    #   <body name="m1_top" pos="0 -0.4 0.82" ...>
    module_mount_pos = np.array([0.0, 0.3, 0.1])
    m1_top_local_pos = np.array([0.0, -0.4, 0.82])

    m1_top_nominal_center_moved = (
        moving_plate_body_pos_moved
        + module_mount_pos
        + m1_top_local_pos
    )

    # ============================================================
    # 4. Draw environment
    # ============================================================

    add_box(fig, "floor", floor_center, floor_half_size, opacity=0.08)
    add_box(fig, "tray / table board", tray_center, tray_half_size, opacity=0.30)
    add_box(fig, "screen", screen_center, screen_half_size, opacity=0.26)
    add_box(fig, "moving plate moved", plate_center_moved, plate_half_size, opacity=0.38)

    # Mark moved plate center.
    fig.add_trace(go.Scatter3d(
        x=[plate_center_moved[0]],
        y=[plate_center_moved[1]],
        z=[plate_center_moved[2]],
        mode="markers+text",
        name="plate moved center",
        marker=dict(size=8, symbol="square"),
        text=["plate<br>moved center"],
        textposition="bottom center",
        hovertemplate=(
            "<b>plate moved center</b><br>"
            "x=%{x:.4f} m<br>"
            "y=%{y:.4f} m<br>"
            "z=%{z:.4f} m"
            "<extra></extra>"
        ),
    ))

    # Mark moved m1_top nominal center.
    fig.add_trace(go.Scatter3d(
        x=[m1_top_nominal_center_moved[0]],
        y=[m1_top_nominal_center_moved[1]],
        z=[m1_top_nominal_center_moved[2]],
        mode="markers+text",
        name="m1_top moved nominal center",
        marker=dict(size=11, symbol="diamond"),
        text=["m1_top<br>moved nominal center"],
        textposition="top center",
        hovertemplate=(
            "<b>m1_top moved nominal center</b><br>"
            "x=%{x:.4f} m<br>"
            "y=%{y:.4f} m<br>"
            "z=%{z:.4f} m"
            "<extra></extra>"
        ),
    ))

    # Reference line between plate and m1_top.
    fig.add_trace(go.Scatter3d(
        x=[plate_center_moved[0], m1_top_nominal_center_moved[0]],
        y=[plate_center_moved[1], m1_top_nominal_center_moved[1]],
        z=[plate_center_moved[2], m1_top_nominal_center_moved[2]],
        mode="lines",
        name="plate to m1_top reference",
        line=dict(width=5, dash="dash"),
        hoverinfo="skip",
    ))

    # ============================================================
    # 5. Workspace point cloud
    # ============================================================

    fig.add_trace(go.Scatter3d(
        x=df["x"],
        y=df["y"],
        z=df["z"],
        mode="markers",
        name="workspace samples / measured centers",
        marker=dict(
            size=4,
            color=df["z"],
            colorscale="Viridis",
            opacity=0.82,
            colorbar=dict(title="z / m"),
        ),
        customdata=df[[*DELTA_COLUMNS, "jitter"]].to_numpy(),
        hovertemplate=(
            "<b>workspace sample</b><br>"
            "x=%{x:.4f} m<br>"
            "y=%{y:.4f} m<br>"
            "z=%{z:.4f} m<br>"
            "d1=%{customdata[0]:.4f} m<br>"
            "d2=%{customdata[1]:.4f} m<br>"
            "d3=%{customdata[2]:.4f} m<br>"
            "d4=%{customdata[3]:.4f} m<br>"
            "d5=%{customdata[4]:.4f} m<br>"
            "d6=%{customdata[5]:.4f} m<br>"
            "jitter=%{customdata[6]:.3e} m"
            "<extra></extra>"
        ),
    ))

    workspace_mean_center = df[["x", "y", "z"]].mean().to_numpy()

    fig.add_trace(go.Scatter3d(
        x=[workspace_mean_center[0]],
        y=[workspace_mean_center[1]],
        z=[workspace_mean_center[2]],
        mode="markers+text",
        name="workspace mean center",
        marker=dict(size=10, symbol="circle"),
        text=["workspace<br>mean"],
        textposition="top center",
        hovertemplate=(
            "<b>workspace mean center</b><br>"
            "x=%{x:.4f} m<br>"
            "y=%{y:.4f} m<br>"
            "z=%{z:.4f} m"
            "<extra></extra>"
        ),
    ))

    # Workspace bounding box.
    min_pt = df[["x", "y", "z"]].min().to_numpy()
    max_pt = df[["x", "y", "z"]].max().to_numpy()

    bbox_center = (min_pt + max_pt) / 2.0
    bbox_half_size = (max_pt - min_pt) / 2.0

    fig.add_trace(make_box_wire(
        "workspace bounding box",
        bbox_center,
        bbox_half_size,
        width=3,
    ))

    # ============================================================
    # 6. Path sites from scene.xml
    # ============================================================

    path_points = np.array([
        [-0.01, -0.2, 1.1],
        [-0.01, -0.4, 1.1],
        [-0.05, -0.2, 0.7],
        [-0.05, -0.4, 0.7],
    ])

    fig.add_trace(go.Scatter3d(
        x=path_points[:, 0],
        y=path_points[:, 1],
        z=path_points[:, 2],
        mode="markers+text",
        name="path sites",
        marker=dict(size=6),
        text=["path1", "path2", "path3", "path4"],
        textposition="top center",
        hovertemplate=(
            "%{text}<br>"
            "x=%{x:.4f} m<br>"
            "y=%{y:.4f} m<br>"
            "z=%{z:.4f} m"
            "<extra></extra>"
        ),
    ))

    # ============================================================
    # 7. Equal-axis range
    # ============================================================

    all_points = np.vstack([
        df[["x", "y", "z"]].to_numpy(),
        screen_center,
        tray_center,
        plate_center_moved,
        floor_center,
        m1_top_nominal_center_moved,
        path_points,
    ])

    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)

    center = (mins + maxs) / 2.0
    span = float((maxs - mins).max())

    if span <= 0:
        span = 1.0

    pad = span * 0.08

    axis_ranges = [
        [center[i] - span / 2.0 - pad, center[i] + span / 2.0 + pad]
        for i in range(3)
    ]

    fig.update_layout(
        title=(
            "Interactive 3D Workspace with Moved Plate and m1_top Center<br>"
            "<sup>Drag to rotate, scroll to zoom; hover to inspect coordinates.</sup>"
        ),
        scene=dict(
            xaxis=dict(title="x / m", range=axis_ranges[0], showspikes=False),
            yaxis=dict(title="y / m", range=axis_ranges[1], showspikes=False),
            zaxis=dict(title="z / m", range=axis_ranges[2], showspikes=False),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.45, y=-1.65, z=1.15)),
        ),
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=0, r=0, b=0, t=75),
        height=850,
        annotations=[
            dict(
                text=(
                    f"moving_plate_body_pos_moved = "
                    f"[{moving_plate_body_pos_moved[0]:.3f}, "
                    f"{moving_plate_body_pos_moved[1]:.3f}, "
                    f"{moving_plate_body_pos_moved[2]:.3f}] m<br>"
                    f"plate_center_moved = "
                    f"[{plate_center_moved[0]:.3f}, "
                    f"{plate_center_moved[1]:.3f}, "
                    f"{plate_center_moved[2]:.3f}] m<br>"
                    f"m1_top_moved_nominal_center = "
                    f"[{m1_top_nominal_center_moved[0]:.3f}, "
                    f"{m1_top_nominal_center_moved[1]:.3f}, "
                    f"{m1_top_nominal_center_moved[2]:.3f}] m<br>"
                    f"workspace_mean = "
                    f"[{workspace_mean_center[0]:.3f}, "
                    f"{workspace_mean_center[1]:.3f}, "
                    f"{workspace_mean_center[2]:.3f}] m"
                ),
                x=0.01,
                y=0.01,
                xref="paper",
                yref="paper",
                showarrow=False,
                align="left",
                bgcolor="rgba(255,255,255,0.78)",
                bordercolor="rgba(0,0,0,0.25)",
                borderwidth=1,
            )
        ],
    )

    return fig, {
        "moving_plate_body_pos_moved": moving_plate_body_pos_moved,
        "plate_center_moved": plate_center_moved,
        "m1_top_nominal_center_moved": m1_top_nominal_center_moved,
        "workspace_mean_center": workspace_mean_center,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=str,
        default="dataset/workspace_6rope.csv",
        help="Path to workspace CSV.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="interactive_workspace_moved_plate_m1top.html",
        help="Output interactive HTML file.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Embed Plotly JS into HTML. Larger file, but works without internet.",
    )
    args = parser.parse_args()

    df = load_dataset(args.csv)
    fig, info = build_figure(df)

    include_plotlyjs = True if args.offline else "cdn"

    fig.write_html(
        args.output,
        include_plotlyjs=include_plotlyjs,
        full_html=True,
    )



if __name__ == "__main__":
    main()
