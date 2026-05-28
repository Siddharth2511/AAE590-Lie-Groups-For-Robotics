#!/usr/bin/env python3
"""Create a narrative GIF for the PURT hardware-calibration hypothesis."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import cv2
import numpy as np
import rosbag2_py
import yaml
from cv_bridge import CvBridge
from PIL import Image, ImageDraw, ImageFont
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image as RosImage


PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
QTM_TRAJ = RESULTS_DIR / "hardware_qtm_run1_assets" / "drone_trajectory.csv"
ABLATION_CSV = RESULTS_DIR / "hardware_calibration_ablation.csv"
BAG_ROOT = Path("/home/siddharth/up_bags/2026-04-24")

CANVAS_W = 1600
CANVAS_H = 900
BG = (247, 248, 248)
INK = (28, 31, 36)
MUTED = (92, 99, 112)
BLUE = (39, 105, 173)
GREEN = (35, 140, 82)
RED = (190, 51, 45)
ORANGE = (231, 125, 35)
PURPLE = (116, 78, 170)

PURT_CAM_CENTERS_W = {
    "UP1 / cam2": np.array([-13.654, 8.705, 0.759], dtype=float),
    "UP2 / cam4": np.array([14.201, -9.387, 0.926], dtype=float),
    "UP3 / cam1": np.array([-3.886, -5.616, 0.371], dtype=float),
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    names = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


FONT_16 = font(16)
FONT_18 = font(18)
FONT_20 = font(20)
FONT_24 = font(24)
FONT_26B = font(26, True)
FONT_30B = font(30, True)
FONT_36B = font(36, True)


def draw_round_rect(draw: ImageDraw.ImageDraw, box, radius=10, fill=(255, 255, 255), outline=(210, 214, 220), width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def draw_text(draw: ImageDraw.ImageDraw, xy, text: str, fill=INK, fnt=FONT_18, anchor=None):
    draw.text(xy, text, fill=fill, font=fnt, anchor=anchor)


def load_bag_metadata(cam: str) -> tuple[int, int]:
    path = BAG_ROOT / f"{cam.lower()}_2026-04-24-run1" / "metadata.yaml"
    info = yaml.safe_load(path.read_text(encoding="utf-8"))["rosbag2_bagfile_information"]
    start = int(info["starting_time"]["nanoseconds_since_epoch"])
    duration = int(info["duration"]["nanoseconds"])
    return start, start + duration


def extract_rosbag_frames(target_rel_s: list[float]) -> dict[str, list[np.ndarray]]:
    cams = ["UP1", "UP2", "UP3"]
    starts = {}
    ends = {}
    for cam in cams:
        starts[cam], ends[cam] = load_bag_metadata(cam)
    common_start = max(starts.values())
    bridge = CvBridge()
    out: dict[str, list[np.ndarray]] = {}

    for cam in cams:
        bag_dir = BAG_ROOT / f"{cam.lower()}_2026-04-24-run1"
        topic = f"/{cam}/image_raw"
        targets_ns = [common_start + int(t * 1e9) for t in target_rel_s]
        frames: list[np.ndarray | None] = [None] * len(targets_ns)
        reader = rosbag2_py.SequentialReader()
        reader.open(
            rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id="sqlite3"),
            rosbag2_py.ConverterOptions("", ""),
        )
        idx = 0
        last = None
        while reader.has_next() and idx < len(targets_ns):
            topic_name, data, stamp_ns = reader.read_next()
            if topic_name != topic:
                continue
            if stamp_ns >= targets_ns[idx]:
                msg = deserialize_message(data, RosImage)
                frame_bgr = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                last = frame_rgb
                frames[idx] = frame_rgb
                idx += 1
        for j in range(len(frames)):
            if frames[j] is None:
                frames[j] = last if last is not None else np.zeros((480, 640, 3), dtype=np.uint8)
        out[cam] = [np.asarray(f, dtype=np.uint8) for f in frames]
    return out


def load_qtm_path() -> np.ndarray:
    rows = []
    with QTM_TRAJ.open("r", encoding="utf-8") as fs:
        reader = csv.DictReader(fs)
        for row in reader:
            rows.append([float(row["t"]), float(row["x"]), float(row["y"]), float(row["z"])])
    return np.asarray(rows, dtype=float)


def load_ablation_rows() -> list[dict]:
    rows = []
    with ABLATION_CSV.open("r", encoding="utf-8", newline="") as fs:
        for row in csv.DictReader(fs):
            rows.append(
                {
                    "label": row["label"],
                    "pf": float(row["pf_rmse_mean_m"]),
                    "tri": float(row["tri_rmse_mean_m"]),
                    "pix": float(row["pixel_residual_p90_px"]),
                }
            )
    order = ["Correct", "camera0.yaml only", "15 deg, 0.35 m", "15 deg + camera0"]
    return [next(r for r in rows if r["label"] == label) for label in order]


def path_to_screen(points_xy: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    pad = 26
    mn = points_xy.min(axis=0)
    mx = points_xy.max(axis=0)
    for c in PURT_CAM_CENTERS_W.values():
        mn = np.minimum(mn, c[:2])
        mx = np.maximum(mx, c[:2])
    span = np.maximum(mx - mn, 1e-6)
    sx = (x1 - x0 - 2 * pad) / span[0]
    sy = (y1 - y0 - 2 * pad) / span[1]
    s = min(sx, sy)
    center = 0.5 * (mn + mx)
    px_center = np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0])
    screen = np.empty_like(points_xy)
    screen[:, 0] = px_center[0] + s * (points_xy[:, 0] - center[0])
    screen[:, 1] = px_center[1] - s * (points_xy[:, 1] - center[1])
    return screen


def draw_path_panel(draw: ImageDraw.ImageDraw, qtm: np.ndarray, frame_idx: int, frame_count: int, box):
    draw_round_rect(draw, box, radius=14)
    x0, y0, x1, y1 = box
    draw_text(draw, (x0 + 18, y0 + 14), "QTM motion replay", fnt=FONT_24, fill=INK)
    draw_text(draw, (x0 + 18, y0 + 44), "actual PURT run-1 path, not figure-eight", fnt=FONT_16, fill=MUTED)

    plot_box = (x0 + 24, y0 + 78, x1 - 24, y1 - 24)
    pts = path_to_screen(qtm[:, 1:3], plot_box)
    for gx in np.linspace(plot_box[0], plot_box[2], 5):
        draw.line([(gx, plot_box[1]), (gx, plot_box[3])], fill=(229, 232, 236), width=1)
    for gy in np.linspace(plot_box[1], plot_box[3], 5):
        draw.line([(plot_box[0], gy), (plot_box[2], gy)], fill=(229, 232, 236), width=1)

    trail_end = int((frame_idx / max(frame_count - 1, 1)) * (len(pts) - 1))
    trail_end = max(2, trail_end)
    draw.line([tuple(p) for p in pts[:trail_end: max(1, trail_end // 600)]], fill=(60, 100, 170), width=3)
    draw.line([tuple(p) for p in pts[trail_end:: max(1, (len(pts) - trail_end) // 300 or 1)]], fill=(175, 184, 198), width=1)
    p_now = pts[trail_end]
    draw.ellipse((p_now[0] - 7, p_now[1] - 7, p_now[0] + 7, p_now[1] + 7), fill=RED, outline=(255, 255, 255), width=2)

    cam_points = path_to_screen(np.array([c[:2] for c in PURT_CAM_CENTERS_W.values()]), plot_box)
    for (name, _), p in zip(PURT_CAM_CENTERS_W.items(), cam_points):
        draw.polygon([(p[0], p[1] - 8), (p[0] - 8, p[1] + 7), (p[0] + 8, p[1] + 7)], fill=GREEN)
        draw_text(draw, (p[0] + 8, p[1] - 12), name.split()[0], fnt=FONT_16, fill=INK)

    t_now = qtm[trail_end, 0]
    time_box = (plot_box[2] - 150, plot_box[1] + 10, plot_box[2] - 10, plot_box[1] + 42)
    draw_round_rect(draw, time_box, radius=8, fill=(232, 241, 252), outline=(198, 215, 235), width=1)
    draw_text(draw, (plot_box[2] - 22, plot_box[1] + 16), f"QTM t = {t_now:05.1f} s", fnt=FONT_16, fill=BLUE, anchor="ra")


def draw_bag_panel(canvas: Image.Image, frames: dict[str, list[np.ndarray]], frame_idx: int, target_rel: list[float], box):
    draw = ImageDraw.Draw(canvas)
    draw_round_rect(draw, box, radius=14)
    x0, y0, x1, y1 = box
    draw_text(draw, (x0 + 18, y0 + 14), "Real ROS bag snippets", fnt=FONT_24)
    draw_text(draw, (x0 + 18, y0 + 44), "/UP1, /UP2, /UP3 image_raw from PURT 2026-04-24", fnt=FONT_16, fill=MUTED)

    feed_w = x1 - x0 - 36
    feed_h = 170
    for i, cam in enumerate(["UP1", "UP2", "UP3"]):
        fy = y0 + 78 + i * (feed_h + 16)
        frame = Image.fromarray(frames[cam][frame_idx]).resize((feed_w, feed_h))
        canvas.paste(frame, (x0 + 18, fy))
        d2 = ImageDraw.Draw(canvas)
        d2.rectangle((x0 + 18, fy, x0 + 18 + feed_w, fy + 28), fill=(0, 0, 0))
        d2.text((x0 + 26, fy + 5), f"{cam}  real bag feed  |  excerpt t={target_rel[frame_idx]:04.1f}s", fill=(255, 255, 255), font=FONT_16)
        d2.rectangle((x0 + 18, fy, x0 + 18 + feed_w, fy + feed_h), outline=(215, 219, 225), width=1)


def draw_pipeline_panel(draw: ImageDraw.ImageDraw, phase: float, box):
    draw_round_rect(draw, box, radius=14)
    x0, y0, x1, y1 = box
    draw_text(draw, (x0 + 18, y0 + 14), "Narrative", fnt=FONT_24)
    steps = [
        ("ROS bags", "real image streams"),
        ("QTM path", "PURT motion replay"),
        ("Sim ablation", "perturb K and T_cw"),
        ("PF metric", "does error become hardware-scale?"),
    ]
    active = min(int(phase * len(steps)), len(steps) - 1)
    bx = x0 + 24
    by = y0 + 72
    bw = x1 - x0 - 48
    bh = 68
    for i, (title, desc) in enumerate(steps):
        yy = by + i * 88
        fill = (232, 241, 252) if i == active else (255, 255, 255)
        outline = BLUE if i == active else (210, 214, 220)
        draw_round_rect(draw, (bx, yy, bx + bw, yy + bh), radius=12, fill=fill, outline=outline, width=2 if i == active else 1)
        draw_text(draw, (bx + 18, yy + 12), title, fnt=FONT_20, fill=INK)
        draw_text(draw, (bx + 18, yy + 38), desc, fnt=FONT_16, fill=MUTED)
        if i < len(steps) - 1:
            cx = bx + bw / 2
            draw.line([(cx, yy + bh + 6), (cx, yy + 82)], fill=(145, 153, 166), width=2)
            draw.polygon([(cx, yy + 86), (cx - 5, yy + 76), (cx + 5, yy + 76)], fill=(145, 153, 166))


def draw_results_panel(draw: ImageDraw.ImageDraw, rows: list[dict], phase: float, box):
    draw_round_rect(draw, box, radius=14)
    x0, y0, x1, y1 = box
    draw_text(draw, (x0 + 18, y0 + 14), "Hypothesis test", fnt=FONT_24)
    draw_text(draw, (x0 + 18, y0 + 44), "wrong intrinsics/extrinsics reproduce multi-meter PF error", fnt=FONT_16, fill=MUTED)
    chart = (x0 + 55, y0 + 105, x1 - 30, y1 - 115)
    draw.line([(chart[0], chart[3]), (chart[2], chart[3])], fill=(80, 86, 96), width=2)
    draw.line([(chart[0], chart[1]), (chart[0], chart[3])], fill=(80, 86, 96), width=2)
    max_y = 10.0
    y5 = chart[3] - (5.0 / max_y) * (chart[3] - chart[1])
    draw.line([(chart[0], y5), (chart[2], y5)], fill=RED, width=2)
    draw_text(draw, (chart[2] - 4, y5 - 22), "5 m hardware-scale", fnt=FONT_16, fill=RED, anchor="ra")
    visible = max(1, int(math.ceil(phase * len(rows))))
    bar_gap = 18
    bar_w = (chart[2] - chart[0] - bar_gap * (len(rows) + 1)) / len(rows)
    for i, row in enumerate(rows):
        bx = chart[0] + bar_gap + i * (bar_w + bar_gap)
        h = min(row["pf"], max_y) / max_y * (chart[3] - chart[1])
        top = chart[3] - h
        color = GREEN if row["pf"] < 2.0 else ORANGE if row["pf"] < 5.0 else RED
        if i < visible:
            draw.rectangle((bx, top, bx + bar_w, chart[3]), fill=color)
            draw_text(draw, (bx + bar_w / 2, top - 24), f"{row['pf']:.1f} m", fnt=FONT_16, fill=INK, anchor="ma")
        label = row["label"].replace("camera0.yaml only", "bad K").replace("15 deg, 0.35 m", "bad T").replace("15 deg + camera0", "bad K+T")
        draw_text(draw, (bx + bar_w / 2, chart[3] + 12), label, fnt=FONT_16, fill=INK, anchor="ma")
    draw_text(draw, (chart[0] - 12, chart[1]), "10", fnt=FONT_16, fill=MUTED, anchor="ra")
    draw_text(draw, (chart[0] - 12, chart[3] - 4), "0", fnt=FONT_16, fill=MUTED, anchor="ra")
    draw_text(draw, (x0 + 18, y1 - 82), "Takeaway:", fnt=FONT_20, fill=INK)
    draw_text(draw, (x0 + 18, y1 - 54), "Wrong calibration alone reproduces multi-meter error.", fnt=FONT_18, fill=RED)


def make_frame(frames, qtm, rows, frame_idx: int, frame_count: int, target_rel: list[float]) -> Image.Image:
    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
    draw = ImageDraw.Draw(canvas)
    draw_text(draw, (36, 24), "Hardware Data In Simulation: Calibration Hypothesis", fnt=FONT_36B, fill=INK)
    draw_text(
        draw,
        (36, 68),
        "Real ROS bags + QTM PURT motion are replayed; wrong K/T_cw is injected to explain poor hardware PF results.",
        fnt=FONT_20,
        fill=MUTED,
    )

    phase = frame_idx / max(frame_count - 1, 1)
    draw_bag_panel(canvas, frames, frame_idx, target_rel, (28, 110, 525, 780))
    draw_pipeline_panel(draw, phase, (550, 110, 945, 500))
    draw_path_panel(draw, qtm, frame_idx, frame_count, (550, 520, 945, 875))
    draw_results_panel(draw, rows, phase, (970, 110, 1572, 875))
    return canvas


def make_pipeline_png(path: Path) -> None:
    img = Image.new("RGB", (1500, 620), BG)
    draw = ImageDraw.Draw(img)
    draw_text(draw, (42, 35), "Hardware-to-Simulation Hypothesis Test", fnt=FONT_36B)
    boxes = [
        ((65, 150, 335, 300), "ROS bags", "UP camera streams\nreal PURT feeds"),
        ((425, 150, 695, 300), "QTM data", "drone path\ncamera body poses"),
        ((785, 150, 1055, 300), "Simulator replay", "clean bbox pixels\nfrom PURT motion"),
        ((1145, 150, 1415, 300), "Calibration test", "correct model\nvs bad K,Tcw"),
    ]
    for i, (box, title, desc) in enumerate(boxes):
        fill = (232, 241, 252) if i in (2, 3) else (255, 255, 255)
        draw_round_rect(draw, box, radius=18, fill=fill, outline=BLUE if i in (2, 3) else (198, 205, 214), width=3 if i in (2, 3) else 2)
        draw_text(draw, (box[0] + 22, box[1] + 24), title, fnt=FONT_26B)
        for j, line in enumerate(desc.split("\n")):
            draw_text(draw, (box[0] + 22, box[1] + 70 + 28 * j), line, fnt=FONT_20, fill=MUTED)
        if i < len(boxes) - 1:
            next_box = boxes[i + 1][0]
            x = box[2] + 28
            x2 = next_box[0] - 28
            y = (box[1] + box[3]) / 2
            draw.line([(x, y), (x2 - 18, y)], fill=(100, 108, 122), width=5)
            draw.polygon([(x2, y), (x2 - 22, y - 13), (x2 - 22, y + 13)], fill=(100, 108, 122))
    draw_text(draw, (70, 390), "Question", fnt=FONT_26B, fill=INK)
    draw_text(draw, (70, 430), "Can the camera calibration mistakes alone reproduce the 3-6 m hardware failure?", fnt=FONT_22 if "FONT_22" in globals() else FONT_20, fill=MUTED)
    draw_text(draw, (70, 505), "Answer from ablation", fnt=FONT_26B, fill=INK)
    draw_text(draw, (70, 545), "Yes: camera0.yaml and marker-frame optical extrinsic errors push PF RMSE into multi-meter range.", fnt=FONT_20, fill=RED)
    img.save(path)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    frame_count = 36
    target_rel = np.linspace(0.0, 12.0, frame_count).tolist()
    bag_frames = extract_rosbag_frames(target_rel)
    qtm = load_qtm_path()
    rows = load_ablation_rows()

    frames = [make_frame(bag_frames, qtm, rows, i, frame_count, target_rel) for i in range(frame_count)]
    gif_path = FIGURES_DIR / "hardware_hypothesis_story.gif"
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=140,
        loop=0,
        optimize=True,
    )
    make_pipeline_png(FIGURES_DIR / "hardware_hypothesis_pipeline.png")
    print(f"wrote {gif_path}")
    print(f"wrote {FIGURES_DIR / 'hardware_hypothesis_pipeline.png'}")


if __name__ == "__main__":
    main()
