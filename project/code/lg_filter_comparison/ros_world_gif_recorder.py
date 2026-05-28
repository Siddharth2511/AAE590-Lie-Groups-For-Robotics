#!/usr/bin/env python3
"""Record a composite ROS/Gazebo GIF for the filter comparison.

This script is intended to be run while the Purdue Gazebo simulation is active.
It subscribes to the four Gazebo camera feeds plus `/drone1/pose`, synthesizes
controlled bbox measurements from the projected target position, runs ungated
EKFs, gated EKFs, and PF, and writes a single GIF with:

  * a top-down world panel,
  * all four rendered camera feeds,
  * bbox dropout/outlier visualization,
  * sigma/rate/runtime/error telemetry.
"""

from __future__ import annotations

import argparse
import math
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from PIL import Image
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image as RosImage

from filters import BootstrapPF, ErrorStateEKF, GatedErrorStateEKF, GatedIteratedEKF, IteratedEKF
from lie_projection import make_square_camera_rig, project_points_lie, project_state
from run_comparison import FIGURE_EIGHT_PERIOD_S, Scenario, initial_conditions


FILTER_CLASSES = [ErrorStateEKF, IteratedEKF, GatedErrorStateEKF, GatedIteratedEKF, BootstrapPF]
FILTER_COLORS = {
    "MEKF": (40, 190, 70),
    "ItEKF": (235, 150, 35),
    "MEKF-gated": (170, 80, 210),
    "ItEKF-gated": (30, 170, 220),
    "PF": (60, 125, 240),
}
TRUTH_COLOR = (25, 25, 25)
MEAS_COLOR = (30, 190, 70)
OUTLIER_COLOR = (40, 40, 230)
DROPOUT_COLOR = (30, 30, 230)
DRONE_BBOX_DIMS_M = np.array([0.8, 0.8, 0.35], dtype=float)


def default_scenario() -> Scenario:
    return Scenario(
        "dropout_outlier",
        pixel_sigma=12.0,
        init_bias=np.array([1.0, -0.8, 0.35]),
        init_vel_bias=np.array([-0.3, 0.2, 0.0]),
        init_pos_std=1.2,
        init_vel_std=0.6,
        dropout_prob=0.18,
        outlier_prob=0.06,
    )


def pose_to_state(msg: PoseStamped) -> np.ndarray:
    p = msg.pose.position
    return np.array([p.x, p.y, p.z, 0.0, 0.0, 0.0], dtype=float)


def draw_text(img, text, org, scale=0.45, color=(40, 40, 40), thickness=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_text_shadow(img, text, org, scale=0.38, color=(255, 255, 255), thickness=1):
    x, y = org
    draw_text(img, text, (x + 1, y + 1), scale, (20, 20, 20), thickness + 1)
    draw_text(img, text, org, scale, color, thickness)


def draw_label(img, text, org, color):
    x, y = org
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    cv2.rectangle(img, (x - 4, y - h - 6), (x + w + 4, y + 4), (255, 255, 255), -1)
    cv2.rectangle(img, (x - 4, y - h - 6), (x + w + 4, y + 4), color, 1)
    draw_text(img, text, (x, y), scale=0.42, color=color, thickness=1)


def camera_caption_lines(cam_idx: int, cam) -> list[str]:
    fx, fy = cam.K[0, 0], cam.K[1, 1]
    cx, cy = cam.K[0, 2], cam.K[1, 2]
    t_cw = np.asarray(cam.T_cw.t, dtype=float)
    R_cw = np.asarray(cam.T_cw.R, dtype=float)
    t_wc = cam.center_w if cam.center_w is not None else -R_cw.T @ t_cw
    row_text = "; ".join(
        "[" + ",".join(f"{value:+.1f}" for value in row) + "]" for row in R_cw
    )
    return [
        f"camera_{cam_idx} / 30 Hz   K^({cam_idx})=[{fx:.0f},{fy:.0f},{cx:.0f},{cy:.0f}]",
        f"Tcw^({cam_idx})=[Rcw^({cam_idx}) t_cw^({cam_idx}); 0 1]",
        f"t_wc^({cam_idx})=({t_wc[0]:+.2f},{t_wc[1]:+.2f},{t_wc[2]:+.2f}) m",
        f"Rcw^({cam_idx}) rows: {row_text}",
        f"t_cw^({cam_idx})=-Rcw^({cam_idx})t_wc^({cam_idx})=({t_cw[0]:+.2f},{t_cw[1]:+.2f},{t_cw[2]:+.2f}) m",
    ]


def clamp_box(box, w, h):
    x0, y0, x1, y1 = [float(v) for v in box]
    x0 = np.clip(x0, 0.0, w - 1.0)
    x1 = np.clip(x1, 0.0, w - 1.0)
    y0 = np.clip(y0, 0.0, h - 1.0)
    y1 = np.clip(y1, 0.0, h - 1.0)
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return np.array([x0, y0, x1, y1], dtype=float)


def box_center(box):
    x0, y0, x1, y1 = box
    return np.array([(x0 + x1) * 0.5, (y0 + y1) * 0.5], dtype=float)


def shift_box(box, shift, w, h):
    shifted = np.array(box, dtype=float) + np.array([shift[0], shift[1], shift[0], shift[1]], dtype=float)
    return clamp_box(shifted, w, h)


def draw_bbox(img, box, color, label=None, dashed=False):
    h, w = img.shape[:2]
    if box is None:
        return
    x0, y0, x1, y1 = clamp_box(box, w, h)
    if not np.all(np.isfinite([x0, y0, x1, y1])):
        return
    x0, y0, x1, y1 = map(int, [x0, y0, x1, y1])
    if dashed:
        for x in range(x0, x1, 10):
            cv2.line(img, (x, y0), (min(x + 5, x1), y0), color, 2)
            cv2.line(img, (x, y1), (min(x + 5, x1), y1), color, 2)
        for y in range(y0, y1, 10):
            cv2.line(img, (x0, y), (x0, min(y + 5, y1)), color, 2)
            cv2.line(img, (x1, y), (x1, min(y + 5, y1)), color, 2)
    else:
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
    u, v = box_center((x0, y0, x1, y1))
    cv2.drawMarker(img, (int(u), int(v)), color, markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
    if label:
        draw_label(img, label, (x0 + 2, max(18, y0 - 4)), color)


class RosWorldGifRecorder(Node):
    def __init__(self, args):
        super().__init__(
            "three_filter_gif_recorder",
            parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)],
            automatically_declare_parameters_from_overrides=True,
        )
        self.args = args
        self.scenario = Scenario(
            "dropout_outlier",
            pixel_sigma=float(args.pixel_sigma),
            init_bias=np.array(args.init_bias, dtype=float),
            init_vel_bias=np.array(args.init_vel_bias, dtype=float),
            init_pos_std=float(args.init_pos_std),
            init_vel_std=float(args.init_vel_std),
            dropout_prob=float(args.dropout_prob),
            outlier_prob=float(args.outlier_prob),
        )
        self.cameras = make_square_camera_rig()
        self.bridge = CvBridge()
        self.rng = np.random.default_rng(args.seed)
        self.latest_images: dict[int, np.ndarray] = {}
        self.image_stamps: dict[int, float] = {}
        self.truth_msg: PoseStamped | None = None
        self.filters = None
        path_len = max(12, int(math.ceil(float(args.record_s) * float(args.filter_rate_hz))) + 8)
        self.filter_paths = {cls.name: deque(maxlen=path_len) for cls in FILTER_CLASSES}
        self.truth_path = deque(maxlen=path_len)
        self.errors = {cls.name: [] for cls in FILTER_CLASSES}
        self.runtimes_ms = {cls.name: [] for cls in FILTER_CLASSES}
        self.frames = []
        self.start_time_ros: float | None = None
        self.first_truth_xy: np.ndarray | None = None
        self.loop_closure_xy = float("nan")
        self.last_truth_state: np.ndarray | None = None
        self.latest_measurement = None
        self.latest_measurement_status = None
        self.frame_counter = 0
        self.shutting_down = False

        qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        for idx in range(4):
            self.create_subscription(RosImage, f"/camera_{idx}/image", lambda m, i=idx: self.image_cb(m, i), qos)
        self.create_subscription(PoseStamped, "/drone1/pose", self.truth_cb, 10)

        self.update_timer = self.create_timer(1.0 / args.filter_rate_hz, self.update_cb)
        self.get_logger().info(
            f"Recording composite GIF: waiting for /drone1/pose and /camera_*/image topics. "
            f"After {args.warmup_s:.1f}s warmup, recording {args.record_s:.1f}s."
        )

    def image_cb(self, msg: RosImage, idx: int):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.latest_images[idx] = frame
        self.image_stamps[idx] = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec

    def truth_cb(self, msg: PoseStamped):
        self.truth_msg = msg

    def init_filters(self, truth_state: np.ndarray):
        mean0, cov0 = initial_conditions(truth_state, self.scenario)
        filters = {}
        for idx, cls in enumerate(FILTER_CLASSES):
            rng = np.random.default_rng(self.args.seed + 1000 * (idx + 1))
            if cls is BootstrapPF:
                filt = cls(
                    self.cameras,
                    1.0 / self.args.filter_rate_hz,
                    self.scenario.pixel_sigma,
                    rng,
                    n_particles=self.args.particles,
                )
            else:
                filt = cls(self.cameras, 1.0 / self.args.filter_rate_hz, self.scenario.pixel_sigma, rng)
            filt.initialize(mean0, cov0)
            filters[filt.name] = filt
        self.filters = filters

    def project_drone_boxes(self, truth_state: np.ndarray):
        center = np.asarray(truth_state[:3], dtype=float)
        half = 0.5 * DRONE_BBOX_DIMS_M
        corners = []
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    corners.append(center + np.array([sx * half[0], sy * half[1], sz * half[2]], dtype=float))
        z, valid = project_points_lie(self.cameras, np.asarray(corners))
        boxes = []
        for cam_idx, cam in enumerate(self.cameras):
            pts = z[cam_idx, valid[cam_idx]]
            if len(pts) < 4:
                boxes.append(None)
                continue
            x0, y0 = np.min(pts, axis=0)
            x1, y1 = np.max(pts, axis=0)
            box = clamp_box([x0, y0, x1, y1], cam.width, cam.height)
            cx, cy = box_center(box)
            min_w, min_h = 24.0, 20.0
            if box[2] - box[0] < min_w:
                box[0], box[2] = cx - min_w / 2.0, cx + min_w / 2.0
            if box[3] - box[1] < min_h:
                box[1], box[3] = cy - min_h / 2.0, cy + min_h / 2.0
            boxes.append(clamp_box(box, cam.width, cam.height))
        return boxes

    def make_measurement(self, truth_state: np.ndarray):
        y_true, mask = project_state(self.cameras, truth_state)
        y = np.array(y_true, copy=True)
        mask = np.array(mask, copy=True)
        status = ["ok" for _ in range(4)]
        true_boxes = self.project_drone_boxes(truth_state)
        meas_boxes = [None for _ in range(4)]

        for cam_idx in range(4):
            cam = self.cameras[cam_idx]
            rows = [2 * cam_idx, 2 * cam_idx + 1]
            if not (mask[rows[0]] and mask[rows[1]]) or true_boxes[cam_idx] is None:
                status[cam_idx] = "not_visible"
                continue
            if self.rng.uniform() < self.scenario.dropout_prob:
                mask[rows] = False
                y[rows] = np.nan
                status[cam_idx] = "dropout"
                continue
            center_noise = self.rng.normal(0.0, self.scenario.pixel_sigma, size=2)
            y[rows] += center_noise
            meas_boxes[cam_idx] = shift_box(true_boxes[cam_idx], center_noise, cam.width, cam.height)
            if self.rng.uniform() < self.scenario.outlier_prob:
                shift = self.rng.normal(0.0, 105.0, size=2)
                if np.linalg.norm(shift) < 65.0:
                    shift += 85.0 * shift / max(np.linalg.norm(shift), 1e-6)
                y[rows] += shift
                y[rows[0]] = np.clip(y[rows[0]], 0.0, cam.width - 1.0)
                y[rows[1]] = np.clip(y[rows[1]], 0.0, cam.height - 1.0)
                meas_boxes[cam_idx] = shift_box(true_boxes[cam_idx], center_noise + shift, cam.width, cam.height)
                status[cam_idx] = "outlier"
        return y, mask, y_true, status, true_boxes, meas_boxes

    def update_cb(self):
        if self.shutting_down:
            return
        if self.truth_msg is None:
            return
        now_msg = self.get_clock().now()
        now_s = now_msg.nanoseconds * 1e-9
        if self.start_time_ros is None:
            self.start_time_ros = now_s
        elapsed = now_s - self.start_time_ros
        if elapsed < self.args.warmup_s:
            return
        record_elapsed = elapsed - self.args.warmup_s

        truth_state = pose_to_state(self.truth_msg)
        if self.filters is None:
            self.init_filters(truth_state)
            self.first_truth_xy = truth_state[:2].copy()

        if self.last_truth_state is not None:
            truth_state[3:6] = (truth_state[:3] - self.last_truth_state[:3]) * self.args.filter_rate_hz
        self.last_truth_state = truth_state.copy()
        if self.first_truth_xy is None:
            self.first_truth_xy = truth_state[:2].copy()
        self.loop_closure_xy = float(np.linalg.norm(truth_state[:2] - self.first_truth_xy))

        y, mask, y_true, status, true_boxes, meas_boxes = self.make_measurement(truth_state)
        self.latest_measurement = (y, mask, y_true, true_boxes, meas_boxes)
        self.latest_measurement_status = status
        self.truth_path.append(truth_state[:3].copy())

        for name, filt in self.filters.items():
            est, runtime = filt.step(y, mask)
            self.filter_paths[name].append(est[:3].copy())
            self.errors[name].append(float(np.linalg.norm(est[:3] - truth_state[:3])))
            self.runtimes_ms[name].append(float(runtime * 1e3))

        if self.frame_counter % max(1, round(self.args.filter_rate_hz / self.args.gif_fps)) == 0:
            self.frames.append(self.compose_frame(record_elapsed))
            if self.frames and (len(self.frames) % 10) == 0:
                self.get_logger().info(f"Captured {len(self.frames)} GIF frames.")
        self.frame_counter += 1

        if record_elapsed >= self.args.record_s:
            self.save_and_shutdown()

    def world_to_panel(self, p: np.ndarray, origin=(40, 70), size=430, lim=12.0):
        x = origin[0] + int((p[0] + lim) / (2 * lim) * size)
        y = origin[1] + int((lim - p[1]) / (2 * lim) * size)
        return x, y

    def compose_world_panel(self, w=510, h=560):
        img = np.full((h, w, 3), (238, 239, 234), dtype=np.uint8)
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), (70, 70, 70), 1)
        draw_text(img, "Purdue Gazebo World / Top-Down Filter Tracks", (22, 30), 0.55, (30, 30, 30), 2)
        draw_text(
            img,
            f"dropout/outlier run, {self.args.record_s:.1f}s XY figure-eight, cameras at +/-10 m",
            (22, 53),
            0.42,
            (70, 70, 70),
            1,
        )

        origin = (40, 78)
        size = 430
        lim = 12.0
        cv2.rectangle(img, origin, (origin[0] + size, origin[1] + size), (210, 210, 205), -1)
        cv2.rectangle(img, origin, (origin[0] + size, origin[1] + size), (90, 90, 90), 1)
        for k in range(-10, 11, 5):
            x0, y0 = self.world_to_panel(np.array([k, -lim, 0.0]), origin, size, lim)
            x1, y1 = self.world_to_panel(np.array([k, lim, 0.0]), origin, size, lim)
            cv2.line(img, (x0, y0), (x1, y1), (225, 225, 220), 1)
            x0, y0 = self.world_to_panel(np.array([-lim, k, 0.0]), origin, size, lim)
            x1, y1 = self.world_to_panel(np.array([lim, k, 0.0]), origin, size, lim)
            cv2.line(img, (x0, y0), (x1, y1), (225, 225, 220), 1)

        camera_xy = [(10, 10), (-10, 10), (-10, -10), (10, -10)]
        for idx, (x, y) in enumerate(camera_xy):
            px, py = self.world_to_panel(np.array([x, y, 0.0]), origin, size, lim)
            cv2.rectangle(img, (px - 7, py - 7), (px + 7, py + 7), (30, 150, 230), -1)
            draw_text(img, f"cam{idx}", (px + 8, py - 8), 0.36, (20, 95, 155), 1)

        def draw_path(points, color, thickness=2):
            if len(points) < 2:
                return
            pts = [self.world_to_panel(p, origin, size, lim) for p in points]
            for p0, p1 in zip(pts[:-1], pts[1:]):
                cv2.line(img, p0, p1, color, thickness)

        draw_path(list(self.truth_path), TRUTH_COLOR, 3)
        for name, points in self.filter_paths.items():
            draw_path(list(points), FILTER_COLORS[name], 2)

        if len(self.truth_path) >= 2:
            sx, sy = self.world_to_panel(self.truth_path[0], origin, size, lim)
            cv2.circle(img, (sx, sy), 5, (250, 250, 250), -1)
            cv2.circle(img, (sx, sy), 5, TRUTH_COLOR, 1)
            draw_label(img, "start", (sx + 9, sy - 7), TRUTH_COLOR)
        if self.truth_path:
            px, py = self.world_to_panel(self.truth_path[-1], origin, size, lim)
            cv2.circle(img, (px, py), 6, TRUTH_COLOR, -1)
            draw_label(img, "truth", (px + 9, py - 7), TRUTH_COLOR)
        for name, points in self.filter_paths.items():
            if points:
                px, py = self.world_to_panel(points[-1], origin, size, lim)
                cv2.circle(img, (px, py), 5, FILTER_COLORS[name], -1)

        legend_y = 512
        legend = [("truth", TRUTH_COLOR)] + list(FILTER_COLORS.items())
        x = 25
        for label, color in legend:
            if x > w - 140:
                x = 25
                legend_y += 24
            cv2.line(img, (x, legend_y), (x + 20, legend_y), color, 3)
            draw_text(img, label, (x + 26, legend_y + 5), 0.38, (35, 35, 35), 1)
            x += 78 if len(label) < 8 else 104
        loop_text = (
            "XY loop closure: warming"
            if not np.isfinite(self.loop_closure_xy)
            else f"XY loop closure to first frame: {self.loop_closure_xy:.3f} m"
        )
        draw_text(img, loop_text, (22, 73), 0.38, (55, 55, 55), 1)
        return img

    def draw_camera_caption(self, panel: np.ndarray, cam_idx: int, y0: int):
        cv2.rectangle(panel, (0, y0), (panel.shape[1] - 1, panel.shape[0] - 1), (252, 252, 248), -1)
        cv2.line(panel, (0, y0), (panel.shape[1] - 1, y0), (90, 90, 90), 1)
        for line_idx, line in enumerate(camera_caption_lines(cam_idx, self.cameras[cam_idx])):
            draw_text(panel, line, (5, y0 + 13 + 14 * line_idx), 0.26, (35, 35, 35), 1)

    def compose_camera_panel(self, cam_idx: int, w=310, image_h=181, caption_h=82):
        panel_h = image_h + caption_h
        panel = np.full((panel_h, w, 3), (252, 252, 248), dtype=np.uint8)
        frame = self.latest_images.get(cam_idx)
        if frame is None:
            panel[:image_h, :] = (40, 40, 40)
            draw_text(panel, f"/camera_{cam_idx}/image waiting", (28, image_h // 2), 0.48, (230, 230, 230), 1)
            self.draw_camera_caption(panel, cam_idx, image_h)
            return panel
        img = cv2.resize(frame, (w, image_h), interpolation=cv2.INTER_AREA)
        meas = self.latest_measurement
        status = self.latest_measurement_status
        if meas is None or status is None:
            panel[:image_h, :] = img
            self.draw_camera_caption(panel, cam_idx, image_h)
            return panel
        y, mask, y_true, true_boxes, meas_boxes = meas
        scale_x = w / self.cameras[cam_idx].width
        scale_y = image_h / self.cameras[cam_idx].height
        rows = [2 * cam_idx, 2 * cam_idx + 1]

        def scale_box(box):
            if box is None:
                return None
            return np.array([box[0] * scale_x, box[1] * scale_y, box[2] * scale_x, box[3] * scale_y], dtype=float)

        if np.isfinite(y_true[rows]).all() and true_boxes[cam_idx] is not None:
            draw_bbox(img, scale_box(true_boxes[cam_idx]), TRUTH_COLOR, "truth bbox", dashed=True)

        st = status[cam_idx]
        if st in ("dropout", "not_visible"):
            cv2.rectangle(img, (0, 0), (w - 1, image_h - 1), DROPOUT_COLOR, 3)
            draw_label(img, "DROPOUT" if st == "dropout" else "not visible", (12, 24), DROPOUT_COLOR)
        elif mask[rows[0]] and mask[rows[1]] and np.isfinite(y[rows]).all():
            color = OUTLIER_COLOR if "outlier" in st else MEAS_COLOR
            label = "false bbox" if "outlier" in st else "bbox"
            draw_bbox(img, scale_box(meas_boxes[cam_idx]), color, label, dashed=False)

        panel[:image_h, :] = img
        self.draw_camera_caption(panel, cam_idx, image_h)
        return panel

    def compose_telemetry_panel(self, w=500, h=300, elapsed=0.0):
        img = np.full((h, w, 3), (248, 248, 244), dtype=np.uint8)
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), (80, 80, 80), 1)
        draw_text(img, "Measurement / Filter Telemetry", (16, 26), 0.55, (35, 35, 35), 2)
        lines = [
            f"record t = {elapsed:5.2f} / {self.args.record_s:.1f} s",
            f"XY loop closure = {self.loop_closure_xy:.3f} m",
            f"pixel sigma = {self.scenario.pixel_sigma:.1f} px",
            f"dropout prob = {self.scenario.dropout_prob:.2f}",
            f"outlier prob = {self.scenario.outlier_prob:.2f}",
            f"camera feeds = 30 Hz, filter = {self.args.filter_rate_hz:.0f} Hz",
            f"PF particles = {self.args.particles}",
            "outliers = synthetic false bboxes",
            "EKF gate: chi-square gamma=11.83",
            "feed subtitles: K, Rcw, and t_cw",
        ]
        y0 = 52
        for line in lines:
            draw_text(img, line, (18, y0), 0.35, (45, 45, 45), 1)
            y0 += 20

        x0 = 292
        y = 58
        draw_text(img, "error / runtime", (x0, 44), 0.38, (35, 35, 35), 1)
        for name in [cls.name for cls in FILTER_CLASSES]:
            err = self.errors[name][-1] if self.errors[name] else float("nan")
            rt = float(np.mean(self.runtimes_ms[name])) if self.runtimes_ms[name] else float("nan")
            color = FILTER_COLORS[name]
            cv2.line(img, (x0, y - 5), (x0 + 20, y - 5), color, 3)
            draw_text(img, f"{name}: {err:4.2f} m / {rt:4.2f} ms", (x0 + 28, y), 0.31, (35, 35, 35), 1)
            y += 22

        meas_status = self.latest_measurement_status or []
        if meas_status:
            draw_text(img, "camera status:", (x0, y + 6), 0.36, (35, 35, 35), 1)
            y += 26
            for idx, st in enumerate(meas_status):
                color = MEAS_COLOR if st == "ok" else (OUTLIER_COLOR if "outlier" in st else DROPOUT_COLOR)
                draw_text(img, f"cam{idx}: {st}", (x0, y), 0.34, color, 1)
                y += 18
        return img

    def compose_frame(self, elapsed: float):
        canvas = np.full((900, 1600, 3), (232, 232, 226), dtype=np.uint8)
        world = self.compose_world_panel()
        canvas[20 : 20 + world.shape[0], 20 : 20 + world.shape[1]] = world

        cam_positions = [(550, 20), (880, 20), (550, 290), (880, 290)]
        for idx, (x, y) in enumerate(cam_positions):
            panel = self.compose_camera_panel(idx)
            canvas[y : y + panel.shape[0], x : x + panel.shape[1]] = panel

        telem = self.compose_telemetry_panel(elapsed=elapsed)
        lower_y = 575
        canvas[lower_y : lower_y + telem.shape[0], 550 : 550 + telem.shape[1]] = telem

        # Compact error sparkline panel.
        spark = np.full((300, 500, 3), (248, 248, 244), dtype=np.uint8)
        cv2.rectangle(spark, (0, 0), (spark.shape[1] - 1, spark.shape[0] - 1), (80, 80, 80), 1)
        draw_text(spark, "Live position error", (18, 26), 0.55, (35, 35, 35), 2)
        max_err = max([max(vals[-120:] or [0.1]) for vals in self.errors.values()] + [0.5])
        max_err = max(0.5, 1.15 * max_err)
        plot_x0, plot_y0, plot_w, plot_h = 42, 58, 425, 210
        cv2.rectangle(spark, (plot_x0, plot_y0), (plot_x0 + plot_w, plot_y0 + plot_h), (225, 225, 220), 1)
        for name, vals in self.errors.items():
            tail = vals[-120:]
            if len(tail) < 2:
                continue
            pts = []
            for i, err in enumerate(tail):
                x = plot_x0 + int(i / max(len(tail) - 1, 1) * plot_w)
                y = plot_y0 + plot_h - int(min(err, max_err) / max_err * plot_h)
                pts.append((x, y))
            for p0, p1 in zip(pts[:-1], pts[1:]):
                cv2.line(spark, p0, p1, FILTER_COLORS[name], 2)
        draw_text(spark, f"0 m", (8, plot_y0 + plot_h), 0.35, (70, 70, 70), 1)
        draw_text(spark, f"{max_err:.1f} m", (2, plot_y0 + 8), 0.35, (70, 70, 70), 1)
        canvas[lower_y : lower_y + spark.shape[0], 1070 : 1070 + spark.shape[1]] = spark

        draw_text(canvas, "Composite ROS/Gazebo GIF: Purdue world + four calibrated camera feeds + gated/ungated filters", (28, 892), 0.55, (35, 35, 35), 2)
        return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    def save_and_shutdown(self):
        if self.shutting_down:
            return
        self.shutting_down = True
        out = Path(self.args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        if not self.frames:
            self.get_logger().error("No frames captured; not writing GIF.")
        else:
            pil_frames = [Image.fromarray(frame) for frame in self.frames]
            pil_frames[0].save(
                out,
                save_all=True,
                append_images=pil_frames[1:],
                duration=int(1000 / self.args.gif_fps),
                loop=0,
                optimize=True,
            )
            self.get_logger().info(f"Wrote {out} ({len(self.frames)} frames).")
        rclpy.shutdown()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="lg_filter_comparison/results/figures/ros_purdue_filter_comparison.gif")
    parser.add_argument(
        "--record-s",
        type=float,
        default=FIGURE_EIGHT_PERIOD_S,
        help="Seconds to record after warmup. The default is one full XY figure-eight period.",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=None,
        help="Deprecated total duration. If supplied, record_s is set to duration_s - warmup_s.",
    )
    parser.add_argument("--warmup-s", type=float, default=1.5)
    parser.add_argument("--gif-fps", type=float, default=10.0)
    parser.add_argument("--filter-rate-hz", type=float, default=20.0)
    parser.add_argument("--particles", type=int, default=450)
    parser.add_argument("--seed", type=int, default=590)
    parser.add_argument("--pixel-sigma", type=float, default=12.0)
    parser.add_argument("--dropout-prob", type=float, default=0.18)
    parser.add_argument("--outlier-prob", type=float, default=0.06)
    parser.add_argument("--init-bias", nargs=3, type=float, default=[1.0, -0.8, 0.35])
    parser.add_argument("--init-vel-bias", nargs=3, type=float, default=[-0.3, 0.2, 0.0])
    parser.add_argument("--init-pos-std", type=float, default=1.2)
    parser.add_argument("--init-vel-std", type=float, default=0.6)
    args = parser.parse_args()
    if args.duration_s is not None:
        args.record_s = max(0.1, float(args.duration_s) - float(args.warmup_s))
    return args


def main():
    args = parse_args()
    rclpy.init()
    node = RosWorldGifRecorder(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.save_and_shutdown()
    finally:
        if rclpy.ok():
            rclpy.shutdown()
        node.destroy_node()


if __name__ == "__main__":
    main()
