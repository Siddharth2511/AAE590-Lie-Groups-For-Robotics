#!/usr/bin/env python3
"""Generate report-ready animated filter-comparison GIFs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from lie_projection import make_square_camera_rig, project_state
from run_comparison import FIGURE_EIGHT_PERIOD_S, Scenario, make_measurement, run_one, trajectory


COLORS = {
    "truth": "#111111",
    "camera": "#ff9f1c",
    "MEKF": "#2ca02c",
    "ItEKF": "#1f77b4",
    "MEKF-gated": "#9467bd",
    "ItEKF-gated": "#ff7f0e",
    "PF": "#d62728",
}


def default_scenarios() -> dict[str, Scenario]:
    return {
        "nominal": Scenario(
            "nominal",
            pixel_sigma=8.0,
            init_bias=np.array([0.7, -0.5, 0.25]),
            init_vel_bias=np.array([-0.2, 0.2, 0.0]),
            init_pos_std=0.8,
            init_vel_std=0.45,
        ),
        "poor_init": Scenario(
            "poor_init",
            pixel_sigma=8.0,
            init_bias=np.array([2.5, -2.0, 0.9]),
            init_vel_bias=np.array([-0.6, 0.4, 0.0]),
            init_pos_std=1.7,
            init_vel_std=0.8,
        ),
        "high_noise": Scenario(
            "high_noise",
            pixel_sigma=22.0,
            init_bias=np.array([0.9, -0.7, 0.25]),
            init_vel_bias=np.array([-0.2, 0.2, 0.0]),
            init_pos_std=1.0,
            init_vel_std=0.55,
        ),
        "dropout_outlier": Scenario(
            "dropout_outlier",
            pixel_sigma=12.0,
            init_bias=np.array([1.0, -0.8, 0.35]),
            init_vel_bias=np.array([-0.3, 0.2, 0.0]),
            init_pos_std=1.2,
            init_vel_std=0.6,
            dropout_prob=0.18,
            outlier_prob=0.06,
        ),
    }


def camera_centers_xy():
    centers = []
    for cam in make_square_camera_rig():
        center = -cam.T_cw.R.T @ cam.T_cw.t
        centers.append(center[:2])
    return np.asarray(centers)


def camera_title(idx, cam):
    center = cam.center_w if cam.center_w is not None else -cam.T_cw.R.T @ cam.T_cw.t
    yaw = np.degrees(cam.yaw_rad) if cam.yaw_rad is not None else np.nan
    return (
        f"camera{idx}: K=({cam.K[0,0]:.0f},{cam.K[1,1]:.0f},{cam.K[0,2]:.0f},{cam.K[1,2]:.0f})\n"
        f"C=({center[0]:+.1f},{center[1]:+.1f},{center[2]:+.1f}), yaw={yaw:+.0f} deg"
    )


def make_animation(
    scenario: Scenario,
    out_path: Path,
    duration: float,
    dt: float,
    particles: int,
    seed: int,
    fps: int,
    stride: int,
):
    t = np.arange(0.0, duration + 1e-12, dt)
    pos, vel = trajectory(t)
    true_state = np.column_stack([pos, vel])
    results = run_one(scenario, seed, t, true_state, particles)

    frame_ids = np.arange(0, len(t), stride)
    if frame_ids[-1] != len(t) - 1:
        frame_ids = np.append(frame_ids, len(t) - 1)

    fig, (ax_scene, ax_err) = plt.subplots(
        1,
        2,
        figsize=(10.8, 5.4),
        gridspec_kw={"width_ratios": [1.1, 1.0]},
    )
    fig.patch.set_facecolor("#f8f8f5")
    fig.suptitle(f"ROS-style filter simulation: {scenario.name}", fontsize=14)

    cam_xy = camera_centers_xy()
    ax_scene.scatter(cam_xy[:, 0], cam_xy[:, 1], s=70, marker="s", color=COLORS["camera"], label="cameras")
    for idx, xy in enumerate(cam_xy):
        ax_scene.text(xy[0] + 0.12, xy[1] + 0.12, f"cam{idx}", fontsize=8, color="#5c3b00")

    ax_scene.plot(pos[:, 0], pos[:, 1], "--", color="#777777", linewidth=1.2, label="truth path")
    truth_tail, = ax_scene.plot([], [], color=COLORS["truth"], linewidth=2.5, label="truth")
    truth_dot, = ax_scene.plot([], [], "o", color=COLORS["truth"], markersize=7)

    filter_tails = {}
    filter_dots = {}
    for name, data in results.items():
        color = COLORS[name]
        est = data["estimates"]
        ax_scene.plot(est[:, 0], est[:, 1], color=color, linewidth=1.0, alpha=0.18)
        filter_tails[name], = ax_scene.plot([], [], color=color, linewidth=2.2, label=name)
        filter_dots[name], = ax_scene.plot([], [], "o", color=color, markersize=6)

    ray_lines = []
    for _ in range(len(cam_xy)):
        line, = ax_scene.plot([], [], color=COLORS["camera"], linewidth=0.8, alpha=0.32)
        ray_lines.append(line)

    ax_scene.set_aspect("equal", adjustable="box")
    ax_scene.set_xlim(-7.2, 7.2)
    ax_scene.set_ylim(-7.2, 7.2)
    ax_scene.set_xlabel("x [m]")
    ax_scene.set_ylabel("y [m]")
    ax_scene.grid(True, alpha=0.28)
    ax_scene.legend(loc="upper right", fontsize=8)

    err_lines = {}
    err_dots = {}
    max_err = 0.0
    for name, data in results.items():
        max_err = max(max_err, float(np.max(data["errors"])))
    max_err = max(0.45, 1.1 * max_err)

    for name, data in results.items():
        color = COLORS[name]
        ax_err.plot(t, data["errors"], color=color, alpha=0.18, linewidth=1.0)
        err_lines[name], = ax_err.plot([], [], color=color, linewidth=2.0, label=name)
        err_dots[name], = ax_err.plot([], [], "o", color=color, markersize=5)
    time_line = ax_err.axvline(0.0, color="#333333", linestyle=":", linewidth=1.2)
    time_text = ax_err.text(0.03, 0.94, "", transform=ax_err.transAxes, fontsize=10, va="top")
    ax_err.set_xlim(float(t[0]), float(t[-1]))
    ax_err.set_ylim(0.0, max_err)
    ax_err.set_xlabel("time [s]")
    ax_err.set_ylabel("position error [m]")
    ax_err.grid(True, alpha=0.28)
    ax_err.legend(loc="upper right", fontsize=8)

    def update(frame_number: int):
        k = int(frame_ids[frame_number])
        trail_start = max(0, k - int(1.6 / dt))
        truth_tail.set_data(pos[trail_start : k + 1, 0], pos[trail_start : k + 1, 1])
        truth_dot.set_data([pos[k, 0]], [pos[k, 1]])

        for line, cam in zip(ray_lines, cam_xy):
            line.set_data([cam[0], pos[k, 0]], [cam[1], pos[k, 1]])

        for name, data in results.items():
            est = data["estimates"]
            filter_tails[name].set_data(est[trail_start : k + 1, 0], est[trail_start : k + 1, 1])
            filter_dots[name].set_data([est[k, 0]], [est[k, 1]])
            err_lines[name].set_data(t[: k + 1], data["errors"][: k + 1])
            err_dots[name].set_data([t[k]], [data["errors"][k]])

        time_line.set_xdata([t[k], t[k]])
        err_text = ", ".join(f"{name}: {results[name]['errors'][k]:.2f} m" for name in results)
        time_text.set_text(f"t = {t[k]:.2f} s\n{err_text}")
        return []

    animation = FuncAnimation(fig, update, frames=len(frame_ids), interval=1000 / fps, blit=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(out_path, writer=PillowWriter(fps=fps), dpi=105)
    plt.close(fig)


def make_zoomed_top_view_animation(
    out_path: Path,
    duration: float,
    dt: float,
    fps: int,
    stride: int,
):
    """Animate the figure-eight clearly while preserving the same camera rig."""
    t = np.arange(0.0, duration + 1e-12, dt)
    pos, _ = trajectory(t)
    cameras = make_square_camera_rig()
    cam_xy = camera_centers_xy()

    frame_ids = np.arange(0, len(t), stride)
    if frame_ids[-1] != len(t) - 1:
        frame_ids = np.append(frame_ids, len(t) - 1)

    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    fig.patch.set_facecolor("#f8f8f5")
    ax.set_facecolor("#f1f2ee")
    ax.plot(pos[:, 0], pos[:, 1], color="#555555", linestyle="--", linewidth=1.6, label="complete figure-eight")
    trail, = ax.plot([], [], color="#111111", linewidth=3.0, label="drone path")
    dot, = ax.plot([], [], "o", color="#111111", markersize=8)
    time_text = ax.text(0.03, 0.95, "", transform=ax.transAxes, fontsize=10, va="top")

    for idx, (xy, cam) in enumerate(zip(cam_xy, cameras)):
        edge = np.array([np.clip(xy[0], -2.9, 2.9), np.clip(xy[1], -2.15, 2.15)])
        ax.annotate(
            f"cam{idx}",
            xy=edge,
            xytext=(edge[0] * 0.83, edge[1] * 0.83),
            arrowprops={"arrowstyle": "->", "color": COLORS["camera"], "lw": 1.6},
            fontsize=8,
            color="#7a4a00",
            ha="center",
            va="center",
        )
        center = cam.center_w if cam.center_w is not None else -cam.T_cw.R.T @ cam.T_cw.t
        ax.text(
            edge[0],
            edge[1] - 0.13,
            f"C=({center[0]:+.0f},{center[1]:+.0f})",
            fontsize=6.8,
            color="#7a4a00",
            ha="center",
        )

    ray_lines = []
    for _ in range(4):
        line, = ax.plot([], [], color=COLORS["camera"], linewidth=0.8, alpha=0.28)
        ray_lines.append(line)

    ax.set_xlim(-2.9, 2.9)
    ax.set_ylim(-2.15, 2.15)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Zoomed top view: one complete figure-eight, same four-camera rig")
    ax.grid(True, alpha=0.32)
    ax.legend(loc="lower right", fontsize=8)

    def update(frame_number: int):
        k = int(frame_ids[frame_number])
        trail_start = max(0, k - int(2.2 / dt))
        trail.set_data(pos[trail_start : k + 1, 0], pos[trail_start : k + 1, 1])
        dot.set_data([pos[k, 0]], [pos[k, 1]])
        for line, xy in zip(ray_lines, cam_xy):
            edge = np.array([np.clip(xy[0], -2.9, 2.9), np.clip(xy[1], -2.15, 2.15)])
            line.set_data([edge[0], pos[k, 0]], [edge[1], pos[k, 1]])
        time_text.set_text(f"t = {t[k]:.2f} s / {duration:.1f} s")
        return []

    animation = FuncAnimation(fig, update, frames=len(frame_ids), interval=1000 / fps, blit=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(out_path, writer=PillowWriter(fps=fps), dpi=115)
    plt.close(fig)


def draw_bbox(ax, center, color, label, linestyle="-", alpha=1.0):
    u, v = float(center[0]), float(center[1])
    size = 44.0
    rect = plt.Rectangle(
        (u - size / 2.0, v - size / 2.0),
        size,
        size,
        fill=False,
        linewidth=2.0,
        edgecolor=color,
        linestyle=linestyle,
        alpha=alpha,
    )
    ax.add_patch(rect)
    ax.plot([u], [v], marker="+", color=color, markersize=8, markeredgewidth=1.8, alpha=alpha)
    if label:
        ax.text(
            u + 28,
            v - 26,
            label,
            color=color,
            fontsize=7,
            weight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.5},
        )
    return rect


def make_camera_feed_animation(
    scenario: Scenario,
    out_path: Path,
    duration: float,
    dt: float,
    seed: int,
    fps: int,
    stride: int,
):
    """Animate the four image-plane measurement streams for dropout/outliers."""
    t = np.arange(0.0, duration + 1e-12, dt)
    pos, vel = trajectory(t)
    true_state = np.column_stack([pos, vel])
    cameras = make_square_camera_rig()
    rng_meas = np.random.default_rng(seed + 10_000)
    measurements = [make_measurement(cameras, p, scenario, rng_meas) for p in true_state[:, :3]]
    truths = [project_state(cameras, np.hstack([p, np.zeros(3)])) for p in true_state[:, :3]]

    frame_ids = np.arange(0, len(t), stride)
    if frame_ids[-1] != len(t) - 1:
        frame_ids = np.append(frame_ids, len(t) - 1)

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.2))
    axes = axes.ravel()
    fig.patch.set_facecolor("#f8f8f5")
    fig.suptitle("Camera-feed view of dropout and outlier measurements", fontsize=14)

    width, height = cameras[0].width, cameras[0].height
    bg_y = np.linspace(0.85, 0.55, height).reshape(height, 1)
    bg = np.dstack(
        [
            0.60 * bg_y + 0.20,
            0.72 * bg_y + 0.16,
            0.92 * bg_y + 0.06,
        ]
    )
    bg = np.repeat(bg, width, axis=1)

    for idx, ax in enumerate(axes):
        ax.imshow(bg, extent=[0, width, height, 0], interpolation="nearest")
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.set_title(camera_title(idx, cameras[idx]), fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
            spine.set_edgecolor("#444444")

    status_text = fig.text(0.5, 0.025, "", ha="center", fontsize=10)

    def update(frame_number: int):
        k = int(frame_ids[frame_number])
        y, mask = measurements[k]
        y_true, true_mask = truths[k]
        for idx, ax in enumerate(axes):
            ax.clear()
            ax.imshow(bg, extent=[0, width, height, 0], interpolation="nearest")
            ax.set_title(camera_title(idx, cameras[idx]), fontsize=8)
            u_idx = 2 * idx
            v_idx = u_idx + 1
            truth_ok = bool(true_mask[u_idx] and true_mask[v_idx])
            meas_ok = bool(mask[u_idx] and mask[v_idx] and np.isfinite(y[u_idx]) and np.isfinite(y[v_idx]))
            if truth_ok:
                draw_bbox(ax, (y_true[u_idx], y_true[v_idx]), "#111111", "true", linestyle="--", alpha=0.72)

            if meas_ok:
                err = float(np.linalg.norm(y[u_idx : v_idx + 1] - y_true[u_idx : v_idx + 1])) if truth_ok else np.inf
                is_outlier = err > max(45.0, 3.0 * scenario.pixel_sigma)
                color = "#d62728" if is_outlier else "#2ca02c"
                label = "outlier bbox" if is_outlier else "bbox"
                draw_bbox(ax, (y[u_idx], y[v_idx]), color, label)
                if is_outlier:
                    ax.plot([y[u_idx]], [y[v_idx]], marker="*", color="#ff9f1c", markersize=10)
            else:
                ax.text(
                    width / 2.0,
                    height / 2.0,
                    "DROPOUT",
                    color="#d62728",
                    fontsize=16,
                    weight="bold",
                    ha="center",
                    va="center",
                    bbox={"facecolor": "white", "edgecolor": "#d62728", "alpha": 0.82, "pad": 4.0},
                )

            ax.set_xlim(0, width)
            ax.set_ylim(height, 0)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(1.0)
                spine.set_edgecolor("#444444")

        valid_count = sum(bool(mask[2 * i] and mask[2 * i + 1]) for i in range(4))
        status_text.set_text(
            f"t = {t[k]:.2f} s | valid camera detections: {valid_count}/4 | "
            "black dashed = true projection, green = usable bbox, red = false/outlier bbox"
        )
        return []

    animation = FuncAnimation(fig, update, frames=len(frame_ids), interval=1000 / fps, blit=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(out_path, writer=PillowWriter(fps=fps), dpi=105)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("lg_filter_comparison/results/figures"))
    parser.add_argument("--duration", type=float, default=FIGURE_EIGHT_PERIOD_S)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--particles", type=int, default=450)
    parser.add_argument("--seed", type=int, default=590)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--camera-feed", action="store_true", help="Also generate a camera-feed dropout/outlier GIF.")
    parser.add_argument("--zoomed-top-view", action="store_true", help="Generate a clean zoomed top-view figure-eight GIF.")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["nominal", "poor_init", "high_noise", "dropout_outlier"],
        choices=sorted(default_scenarios().keys()),
    )
    args = parser.parse_args()

    scenarios = default_scenarios()
    for name in args.scenarios:
        out_path = args.out / f"animated_{name}.gif"
        print(f"writing {out_path}")
        make_animation(
            scenarios[name],
            out_path,
            duration=args.duration,
            dt=args.dt,
            particles=args.particles,
            seed=args.seed,
            fps=args.fps,
            stride=args.stride,
        )
    if args.camera_feed:
        scenario = scenarios["dropout_outlier"]
        out_path = args.out / "animated_camera_dropout_outlier.gif"
        print(f"writing {out_path}")
        make_camera_feed_animation(
            scenario,
            out_path,
            duration=args.duration,
            dt=args.dt,
            seed=args.seed,
            fps=args.fps,
            stride=args.stride,
        )
    if args.zoomed_top_view:
        out_path = args.out / "animated_zoomed_figure_eight.gif"
        print(f"writing {out_path}")
        make_zoomed_top_view_animation(
            out_path,
            duration=args.duration,
            dt=args.dt,
            fps=args.fps,
            stride=args.stride,
        )


if __name__ == "__main__":
    main()
