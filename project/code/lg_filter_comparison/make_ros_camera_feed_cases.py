#!/usr/bin/env python3
"""Generate camera-feed GIFs from the recorded Purdue ROS/Gazebo composite.

The live composite GIF contains the real Gazebo building views.  This script
uses those rendered camera panels as the background, removes the previous bbox
overlays as best as possible, and redraws case-specific measurements so the
camera-feed visuals match the Monte Carlo scenarios.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageSequence

from lie_projection import make_square_camera_rig, project_points_lie
from run_comparison import FIGURE_EIGHT_PERIOD_S, trajectory


SOURCE_GIF = Path("results/figures/ros_purdue_filter_comparison.gif")
CAMERA_POSITIONS = [(550, 20), (880, 20), (550, 290), (880, 290)]
PANEL_W = 310
IMAGE_H = 181
CAPTION_H = 96
OUT_W = 675
OUT_H = 2 * (IMAGE_H + CAPTION_H) + 26
OUT_POSITIONS = [(15, 12), (345, 12), (15, 12 + IMAGE_H + CAPTION_H + 7), (345, 12 + IMAGE_H + CAPTION_H + 7)]
DRONE_BBOX_DIMS_M = np.array([0.8, 0.8, 0.35], dtype=float)


@dataclass(frozen=True)
class FeedCase:
    name: str
    pixel_sigma: float
    dropout_prob: float = 0.0


CASES = {
    "nominal": FeedCase("nominal", pixel_sigma=8.0),
    "high_noise": FeedCase("high_noise", pixel_sigma=22.0),
    "dropout": FeedCase("dropout", pixel_sigma=12.0, dropout_prob=0.18),
}


def draw_text(img: np.ndarray, text: str, org: tuple[int, int], scale: float, color: tuple[int, int, int], thickness: int = 1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_label(img: np.ndarray, text: str, org: tuple[int, int], color: tuple[int, int, int], scale: float = 0.39):
    x, y = org
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    cv2.rectangle(img, (x - 4, y - h - 6), (x + w + 4, y + 4), (255, 255, 255), -1)
    cv2.rectangle(img, (x - 4, y - h - 6), (x + w + 4, y + 4), color, 1)
    draw_text(img, text, (x, y), scale, color, 1)


def clamp_box(box: np.ndarray, w: int, h: int) -> np.ndarray:
    x0, y0, x1, y1 = [float(v) for v in box]
    x0 = float(np.clip(x0, 0.0, w - 1.0))
    x1 = float(np.clip(x1, 0.0, w - 1.0))
    y0 = float(np.clip(y0, 0.0, h - 1.0))
    y1 = float(np.clip(y1, 0.0, h - 1.0))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return np.array([x0, y0, x1, y1], dtype=float)


def scale_box(box: np.ndarray, cam) -> np.ndarray:
    return np.array(
        [
            box[0] * PANEL_W / cam.width,
            box[1] * IMAGE_H / cam.height,
            box[2] * PANEL_W / cam.width,
            box[3] * IMAGE_H / cam.height,
        ],
        dtype=float,
    )


def shift_box(box: np.ndarray, shift: np.ndarray, cam) -> np.ndarray:
    shifted = np.array(box, dtype=float) + np.array([shift[0], shift[1], shift[0], shift[1]], dtype=float)
    return clamp_box(shifted, cam.width, cam.height)


def box_center(box: np.ndarray) -> np.ndarray:
    return np.array([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5], dtype=float)


def draw_bbox(img: np.ndarray, box: np.ndarray | None, color: tuple[int, int, int], label: str, dashed: bool = False):
    if box is None:
        return
    x0, y0, x1, y1 = clamp_box(box, img.shape[1], img.shape[0]).astype(int)
    if dashed:
        for x in range(x0, x1, 10):
            cv2.line(img, (x, y0), (min(x + 5, x1), y0), color, 2)
            cv2.line(img, (x, y1), (min(x + 5, x1), y1), color, 2)
        for y in range(y0, y1, 10):
            cv2.line(img, (x0, y), (x0, min(y + 5, y1)), color, 2)
            cv2.line(img, (x1, y), (x1, min(y + 5, y1)), color, 2)
    else:
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
    u, v = box_center(np.array([x0, y0, x1, y1], dtype=float))
    cv2.drawMarker(img, (int(u), int(v)), color, markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
    draw_label(img, label, (x0 + 2, max(18, y0 - 4)), color)


def camera_caption_lines(cam_idx: int, cam, case: FeedCase) -> list[str]:
    fx, fy = cam.K[0, 0], cam.K[1, 1]
    cx, cy = cam.K[0, 2], cam.K[1, 2]
    t_cw = np.asarray(cam.T_cw.t, dtype=float)
    R_cw = np.asarray(cam.T_cw.R, dtype=float)
    t_wc = cam.center_w if cam.center_w is not None else -R_cw.T @ t_cw
    row_text = "; ".join("[" + ",".join(f"{value:+.1f}" for value in row) + "]" for row in R_cw)
    return [
        f"camera_{cam_idx} / 30 Hz | {case.name} | sigma_z={case.pixel_sigma:.0f} px",
        f"K^({cam_idx})=[{fx:.0f},{fy:.0f},{cx:.0f},{cy:.0f}]",
        f"Tcw^({cam_idx})=[Rcw^({cam_idx}) t_cw^({cam_idx}); 0 1]",
        f"t_wc^({cam_idx})=({t_wc[0]:+.2f},{t_wc[1]:+.2f},{t_wc[2]:+.2f}) m",
        f"Rcw^({cam_idx}) rows: {row_text}",
        f"t_cw^({cam_idx})=-Rcw^({cam_idx})t_wc^({cam_idx})=({t_cw[0]:+.2f},{t_cw[1]:+.2f},{t_cw[2]:+.2f}) m",
    ]


def draw_caption(panel: np.ndarray, cam_idx: int, cam, case: FeedCase):
    y0 = IMAGE_H
    cv2.rectangle(panel, (0, y0), (panel.shape[1] - 1, panel.shape[0] - 1), (252, 252, 248), -1)
    cv2.line(panel, (0, y0), (panel.shape[1] - 1, y0), (90, 90, 90), 1)
    for line_idx, line in enumerate(camera_caption_lines(cam_idx, cam, case)):
        draw_text(panel, line, (5, y0 + 13 + 14 * line_idx), 0.24, (35, 35, 35), 1)


def project_drone_boxes(cameras, p_w: np.ndarray) -> list[np.ndarray | None]:
    half = 0.5 * DRONE_BBOX_DIMS_M
    corners = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                corners.append(p_w + np.array([sx * half[0], sy * half[1], sz * half[2]], dtype=float))
    z, valid = project_points_lie(cameras, np.asarray(corners))
    boxes = []
    for cam_idx, cam in enumerate(cameras):
        pts = z[cam_idx, valid[cam_idx]]
        if len(pts) < 4:
            boxes.append(None)
            continue
        x0, y0 = np.min(pts, axis=0)
        x1, y1 = np.max(pts, axis=0)
        box = clamp_box(np.array([x0, y0, x1, y1], dtype=float), cam.width, cam.height)
        cx, cy = box_center(box)
        min_w, min_h = 24.0, 20.0
        if box[2] - box[0] < min_w:
            box[0], box[2] = cx - min_w / 2.0, cx + min_w / 2.0
        if box[3] - box[1] < min_h:
            box[1], box[3] = cy - min_h / 2.0, cy + min_h / 2.0
        boxes.append(clamp_box(box, cam.width, cam.height))
    return boxes


def local_cleanup_mask(img: np.ndarray, true_box_scaled: np.ndarray | None) -> np.ndarray:
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    red = (r > 150) & (g < 120) & (b < 120)
    mask[red] = 255
    red_dilated = cv2.dilate((red.astype(np.uint8) * 255), np.ones((23, 23), np.uint8))
    bright = (r > 220) & (g > 220) & (b > 220)
    dark = (r < 55) & (g < 55) & (b < 55)
    mask[((bright | dark) & (red_dilated > 0))] = 255

    if true_box_scaled is not None:
        x0, y0, x1, y1 = true_box_scaled.astype(int)
        lx0 = max(0, x0 - 150)
        ly0 = max(0, y0 - 90)
        lx1 = min(img.shape[1] - 1, x1 + 175)
        ly1 = min(img.shape[0] - 1, y1 + 80)
        local = np.zeros(img.shape[:2], dtype=bool)
        local[ly0 : ly1 + 1, lx0 : lx1 + 1] = True
        green = (g > 140) & (r < 130) & (b < 130)
        colored = (green | red) & local
        mask[colored] = 255

        colored_dilated = cv2.dilate((colored.astype(np.uint8) * 255), np.ones((21, 21), np.uint8)) > 0
        bright_local = bright & local
        bright_dilated = cv2.dilate((bright_local.astype(np.uint8) * 255), np.ones((9, 9), np.uint8)) > 0
        mask[bright_local | (dark & bright_dilated) | ((bright | dark) & colored_dilated)] = 255

    if np.count_nonzero(red[:45, :120]) > 10:
        mask[:45, :120] = np.maximum(mask[:45, :120], 255)

    return cv2.dilate(mask, np.ones((3, 3), np.uint8))


def clean_image_area(img: np.ndarray, true_box_scaled: np.ndarray | None) -> np.ndarray:
    mask = local_cleanup_mask(img, true_box_scaled)
    if np.count_nonzero(mask) == 0:
        return img.copy()
    return cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)


def measurements_for_case(case: FeedCase, cameras, positions: np.ndarray, seed: int):
    rng = np.random.default_rng(seed + 72_000 + int(case.pixel_sigma * 10))
    all_status = []
    all_true_boxes = []
    all_meas_boxes = []
    for p_w in positions:
        true_boxes = project_drone_boxes(cameras, p_w)
        status = []
        meas_boxes = []
        for cam_idx, cam in enumerate(cameras):
            box = true_boxes[cam_idx]
            if box is None:
                status.append("not_visible")
                meas_boxes.append(None)
                continue
            if rng.uniform() < case.dropout_prob:
                status.append("dropout")
                meas_boxes.append(None)
                continue
            center_noise = rng.normal(0.0, case.pixel_sigma, size=2)
            meas_box = shift_box(box, center_noise, cam)
            status.append("ok")
            meas_boxes.append(meas_box)
        all_status.append(status)
        all_true_boxes.append(true_boxes)
        all_meas_boxes.append(meas_boxes)
    return all_true_boxes, all_meas_boxes, all_status


def compose_case_frames(source_gif: Path, case: FeedCase, seed: int) -> tuple[list[Image.Image], list[int]]:
    source = Image.open(source_gif)
    source_frames = [frame.convert("RGB") for frame in ImageSequence.Iterator(source)]
    durations = [frame.info.get("duration", source.info.get("duration", 100)) for frame in ImageSequence.Iterator(source)]
    cameras = make_square_camera_rig()
    times = np.linspace(0.0, FIGURE_EIGHT_PERIOD_S, len(source_frames))
    positions, _ = trajectory(times)
    true_boxes, meas_boxes, statuses = measurements_for_case(case, cameras, positions, seed)
    output_frames = []

    for frame_idx, frame in enumerate(source_frames):
        frame_arr = np.asarray(frame)
        canvas = np.full((OUT_H, OUT_W, 3), (232, 232, 226), dtype=np.uint8)
        for cam_idx, ((sx, sy), (dx, dy)) in enumerate(zip(CAMERA_POSITIONS, OUT_POSITIONS)):
            panel_src = frame_arr[sy : sy + IMAGE_H + 82, sx : sx + PANEL_W].copy()
            panel = np.full((IMAGE_H + CAPTION_H, PANEL_W, 3), (252, 252, 248), dtype=np.uint8)
            image = panel_src[:IMAGE_H, :PANEL_W].copy()
            cam = cameras[cam_idx]
            true_box = true_boxes[frame_idx][cam_idx]
            true_scaled = scale_box(true_box, cam) if true_box is not None else None
            image = clean_image_area(image, true_scaled)

            status = statuses[frame_idx][cam_idx]
            if true_scaled is not None:
                draw_bbox(image, true_scaled, (25, 25, 25), "truth bbox", dashed=True)
            if status == "dropout":
                cv2.rectangle(image, (0, 0), (PANEL_W - 1, IMAGE_H - 1), (220, 30, 30), 3)
                draw_label(image, f"DROPOUT | sigma_z={case.pixel_sigma:.0f}px", (12, 25), (220, 30, 30), 0.35)
            elif status == "not_visible":
                cv2.rectangle(image, (0, 0), (PANEL_W - 1, IMAGE_H - 1), (220, 30, 30), 3)
                draw_label(image, "not visible", (12, 25), (220, 30, 30), 0.35)
            else:
                meas_scaled = scale_box(meas_boxes[frame_idx][cam_idx], cam)
                draw_bbox(image, meas_scaled, (35, 180, 45), "bbox", dashed=False)

            draw_label(image, f"{case.name} | sigma_z={case.pixel_sigma:.0f}px", (8, IMAGE_H - 10), (45, 45, 45), 0.34)
            panel[:IMAGE_H, :] = image
            draw_caption(panel, cam_idx, cam, case)
            canvas[dy : dy + panel.shape[0], dx : dx + panel.shape[1]] = panel
        output_frames.append(Image.fromarray(canvas))
    return output_frames, durations


def write_gif(frames: list[Image.Image], durations: list[int], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE_GIF)
    parser.add_argument("--out", type=Path, default=Path("results/figures"))
    parser.add_argument("--seed", type=int, default=590)
    parser.add_argument("--cases", nargs="+", default=["nominal", "high_noise", "dropout"], choices=sorted(CASES))
    args = parser.parse_args()

    for case_name in args.cases:
        case = CASES[case_name]
        frames, durations = compose_case_frames(args.source, case, args.seed)
        out_path = args.out / f"ros_camera_feeds_{case_name}.gif"
        print(f"writing {out_path}")
        write_gif(frames, durations, out_path)


if __name__ == "__main__":
    main()
