#!/usr/bin/env python3
"""Simulate PURT hardware calibration failures in the camera-projection PF.

The real 2026-04-24 hardware bags produced multi-meter PF error.  This script
tests one concrete hypothesis in simulation: the image measurements themselves
can be good, but if the filter uses a wrong optical camera calibration
(`camera0.yaml`) and/or a marker-body pose as if it were the optical camera
pose, triangulation and PF updates can move to the wrong 3D location.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from filters import BootstrapPF, robust_triangulate
from lie_projection import Camera, SE3, exp_so3, look_at_R_wc, project_state


PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

VISNET_INFO_DIR = Path("/home/siddharth/visnet_ws/src/visnet/camera_info")
ACTIVE_INTRINSICS_PATH = VISNET_INFO_DIR / "up1.yaml"
BAD_INTRINSICS_PATH = VISNET_INFO_DIR / "camera0.yaml"
HARDWARE_DOC_PATH = Path("/home/siddharth/visnet_ws/docs/hardware_real_data_evaluation.md")
MARKER_PHOTO_PATH = Path("/home/siddharth/Desktop/AAE 590 LGM/Project/cam image.jpeg")
QTM_MAPPING_PATH = Path("/home/siddharth/Desktop/AAE 590 LGM/Project/qtm_up_mapping_run1.png")
QTM_RUN1_ASSETS_DIR = RESULTS_DIR / "hardware_qtm_run1_assets"
QTM_RUN1_TRAJECTORY_PATH = QTM_RUN1_ASSETS_DIR / "drone_trajectory.csv"
QTM_RUN1_BODIES_PATH = QTM_RUN1_ASSETS_DIR / "qtm_bodies.json"

PURT_CAM_CENTERS_W = {
    "UP1/cam2": np.array([-13.654, 8.705, 0.759], dtype=float),
    "UP2/cam4": np.array([14.201, -9.387, 0.926], dtype=float),
    "UP3/cam1": np.array([-3.886, -5.616, 0.371], dtype=float),
}
TRAJECTORY_CENTER_W = np.array([-3.0, -2.0, 1.18], dtype=float)
STATE_BOUNDS_MIN = np.array([-12.0, -10.0, 0.0], dtype=float)
STATE_BOUNDS_MAX = np.array([6.0, 5.0, 2.1], dtype=float)
MAX_SPEED_MPS = 5.0


@dataclass(frozen=True)
class CalibrationCase:
    name: str
    label: str
    use_bad_intrinsics: bool
    rot_deg: float
    trans_m: float
    note: str


def load_camera_yaml(path: Path) -> tuple[np.ndarray, int, int]:
    with path.open("r", encoding="utf-8") as fs:
        data = yaml.safe_load(fs)
    K = np.array(data["camera_matrix"]["data"], dtype=float).reshape(3, 3)
    return K, int(data["image_width"]), int(data["image_height"])


def make_purt_camera_rig(K: np.ndarray, width: int, height: int, look_at_w: np.ndarray | None = None) -> list[Camera]:
    if look_at_w is None:
        look_at_w = TRAJECTORY_CENTER_W
    cameras: list[Camera] = []
    for name, center_w in PURT_CAM_CENTERS_W.items():
        R_wc = look_at_R_wc(center_w, look_at_w)
        R_cw = R_wc.T
        t_cw = -R_cw @ center_w
        cameras.append(Camera(name, SE3(R_cw, t_cw), K.copy(), width, height, center_w.copy(), None))
    return cameras


def clone_with_intrinsics(cameras: list[Camera], K: np.ndarray, width: int, height: int) -> list[Camera]:
    return [
        Camera(cam.name, cam.T_cw, K.copy(), width, height, cam.center_w.copy(), cam.yaw_rad)
        for cam in cameras
    ]


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def apply_marker_frame_error(cameras: list[Camera], rot_deg: float, trans_m: float) -> list[Camera]:
    """Perturb optical extrinsics as if marker-frame-to-optical-frame is wrong.

    The translation offsets are expressed in each camera's optical frame before
    being mapped into world coordinates.  The rotation errors are camera-frame
    right-multiplicative errors on R_wc.
    """
    if rot_deg == 0.0 and trans_m == 0.0:
        return cameras

    rot_axes = [
        unit(np.array([0.55, -0.20, 0.81])),
        unit(np.array([-0.35, 0.72, 0.60])),
        unit(np.array([0.78, 0.41, -0.47])),
    ]
    trans_axes_cam = [
        unit(np.array([0.75, -0.52, -0.40])),
        unit(np.array([-0.62, -0.25, -0.74])),
        unit(np.array([0.28, -0.87, 0.42])),
    ]
    angle_scales = [1.0, 0.85, 1.15]
    trans_scales = [1.0, 0.75, 1.20]

    out: list[Camera] = []
    for idx, cam in enumerate(cameras):
        R_wc = cam.T_cw.R.T
        center_w = -cam.T_cw.R.T @ cam.T_cw.t
        delta_R = exp_so3(np.deg2rad(rot_deg * angle_scales[idx]) * rot_axes[idx])
        R_wc_bad = R_wc @ delta_R
        center_bad_w = center_w + R_wc @ (trans_m * trans_scales[idx] * trans_axes_cam[idx])
        R_cw_bad = R_wc_bad.T
        t_cw_bad = -R_cw_bad @ center_bad_w
        out.append(Camera(cam.name, SE3(R_cw_bad, t_cw_bad), cam.K.copy(), cam.width, cam.height, center_bad_w, None))
    return out


def purt_trajectory(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hardware-style indoor trajectory inside the documented PF bounds."""
    omega = 2.0 * math.pi / 18.0
    x = TRAJECTORY_CENTER_W[0] + 2.7 * np.sin(omega * t)
    y = TRAJECTORY_CENTER_W[1] + 1.8 * np.sin(2.0 * omega * t + 0.55)
    z = TRAJECTORY_CENTER_W[2] + 0.22 * np.sin(0.7 * omega * t + 0.2)
    vx = 2.7 * omega * np.cos(omega * t)
    vy = 1.8 * 2.0 * omega * np.cos(2.0 * omega * t + 0.55)
    vz = 0.22 * 0.7 * omega * np.cos(0.7 * omega * t + 0.2)
    return np.column_stack([x, y, z]), np.column_stack([vx, vy, vz])


def smooth_positions(p: np.ndarray, window: int = 5) -> np.ndarray:
    if window <= 1:
        return p
    pad = window // 2
    padded = np.pad(p, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    out = np.empty_like(p)
    for axis in range(3):
        out[:, axis] = np.convolve(padded[:, axis], kernel, mode="valid")
    return out


def load_qtm_run1_trajectory(output_rate_hz: float) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load the actual PURT run-1 QTM drone path and resample it uniformly."""
    if not QTM_RUN1_TRAJECTORY_PATH.exists():
        dt = 1.0 / output_rate_hz
        t = np.arange(0.0, 36.0, dt)
        p_w, v_w = purt_trajectory(t)
        return t, np.column_stack([p_w, v_w]), {
            "trajectory_source": "fallback_synthetic_purt_style",
            "trajectory_csv": "",
            "duration_s": float(t[-1] - t[0]),
            "raw_samples": int(len(t)),
            "resampled_samples": int(len(t)),
            "qtm_file": "/home/siddharth/up_bags/2026-04-24/20260424ardupilot_run1.qtm",
        }

    rows = []
    with QTM_RUN1_TRAJECTORY_PATH.open("r", encoding="utf-8") as fs:
        reader = csv.DictReader(fs)
        for row in reader:
            try:
                rows.append([float(row["t"]), float(row["x"]), float(row["y"]), float(row["z"])])
            except (KeyError, ValueError):
                continue
    if len(rows) < 3:
        raise RuntimeError(f"Not enough QTM trajectory samples in {QTM_RUN1_TRAJECTORY_PATH}")

    raw = np.asarray(rows, dtype=float)
    raw = raw[np.argsort(raw[:, 0])]
    keep = np.concatenate([[True], np.diff(raw[:, 0]) > 1e-9])
    raw = raw[keep]
    dt = 1.0 / output_rate_hz
    t = np.arange(raw[0, 0], raw[-1, 0] + 0.5 * dt, dt)
    p_w = np.empty((len(t), 3), dtype=float)
    for axis in range(3):
        p_w[:, axis] = np.interp(t, raw[:, 0], raw[:, axis + 1])
    p_w = smooth_positions(p_w, window=5)
    v_w = np.gradient(p_w, dt, axis=0)
    v_w = np.clip(v_w, -MAX_SPEED_MPS, MAX_SPEED_MPS)

    meta = {
        "trajectory_source": "qtm_run1_drone_trajectory_csv",
        "trajectory_csv": str(QTM_RUN1_TRAJECTORY_PATH),
        "duration_s": float(t[-1] - t[0]),
        "raw_samples": int(len(raw)),
        "resampled_samples": int(len(t)),
        "qtm_file": "/home/siddharth/up_bags/2026-04-24/20260424ardupilot_run1.qtm",
    }
    if QTM_RUN1_BODIES_PATH.exists():
        with QTM_RUN1_BODIES_PATH.open("r", encoding="utf-8") as fs:
            qtm_meta = json.load(fs)
        meta["trajectory_source"] = qtm_meta.get("drone_trajectory_source", meta["trajectory_source"])
        meta["qtm_file"] = qtm_meta.get("qtm_file", meta["qtm_file"])

    return t, np.column_stack([p_w, v_w]), meta


def make_measurements(
    true_cameras: list[Camera],
    positions_w: np.ndarray,
    noise_sigma_px: float,
    rng: np.random.Generator,
) -> list[tuple[np.ndarray, np.ndarray]]:
    measurements = []
    for p_w in positions_w:
        y, mask = project_state(true_cameras, np.hstack([p_w, np.zeros(3)]))
        y = np.array(y, copy=True)
        mask = np.array(mask, copy=True)
        for cam_idx in range(len(true_cameras)):
            rows = np.array([2 * cam_idx, 2 * cam_idx + 1])
            if np.all(mask[rows]):
                y[rows] += rng.normal(0.0, noise_sigma_px, size=2)
        measurements.append((y, mask))
    return measurements


def triangulation_series(
    cameras: list[Camera],
    measurements: list[tuple[np.ndarray, np.ndarray]],
    truth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    estimates = np.full((len(measurements), 3), np.nan, dtype=float)
    valid = np.zeros(len(measurements), dtype=bool)
    for idx, (y, mask) in enumerate(measurements):
        p = robust_triangulate(cameras, y, mask)
        if p is None:
            continue
        estimates[idx] = p
        valid[idx] = True
    errors = np.full(len(measurements), np.nan, dtype=float)
    errors[valid] = np.linalg.norm(estimates[valid] - truth[valid, :3], axis=1)
    valid_percent = 100.0 * float(np.mean(valid))
    return estimates, errors, valid_percent


def bootstrap_initial_state(
    cameras: list[Camera],
    measurements: list[tuple[np.ndarray, np.ndarray]],
    dt: float,
    fallback_state: np.ndarray,
    window: int = 18,
) -> tuple[np.ndarray, np.ndarray]:
    anchors = []
    anchor_indices = []
    for idx, (y, mask) in enumerate(measurements[: max(window * 2, window + 1)]):
        p = robust_triangulate(cameras, y, mask)
        if p is None or not np.all(np.isfinite(p)):
            continue
        anchors.append(p)
        anchor_indices.append(idx)
        if len(anchors) >= window:
            break

    if len(anchors) >= 2:
        anchors_arr = np.asarray(anchors, dtype=float)
        p0 = np.median(anchors_arr, axis=0)
        t_span = max((anchor_indices[-1] - anchor_indices[0]) * dt, dt)
        v0 = np.clip((anchors_arr[-1] - anchors_arr[0]) / t_span, -5.0, 5.0)
        mean = np.hstack([p0, v0])
    else:
        mean = fallback_state.copy()
        mean[:3] += np.array([1.8, -1.2, 0.25])
        mean[3:] = 0.0

    mean[:3] = np.clip(mean[:3], STATE_BOUNDS_MIN, STATE_BOUNDS_MAX)
    mean[3:] = np.clip(mean[3:], -MAX_SPEED_MPS, MAX_SPEED_MPS)
    cov = np.diag([0.9, 0.9, 0.45, 0.55, 0.55, 0.25]) ** 2
    return mean, cov


def enforce_hardware_bounds(pf: BootstrapPF) -> np.ndarray:
    pf.particles[:, :3] = np.clip(pf.particles[:, :3], STATE_BOUNDS_MIN, STATE_BOUNDS_MAX)
    pf.particles[:, 3:6] = np.clip(pf.particles[:, 3:6], -MAX_SPEED_MPS, MAX_SPEED_MPS)
    pf.estimate = np.average(pf.particles, axis=0, weights=pf.weights)
    return pf.estimate.copy()


def run_pf_case(
    cameras: list[Camera],
    measurements: list[tuple[np.ndarray, np.ndarray]],
    truth: np.ndarray,
    dt: float,
    filter_sigma_px: float,
    seed: int,
    n_particles: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    mean0, cov0 = bootstrap_initial_state(cameras, measurements, dt, truth[0])
    pf = BootstrapPF(cameras, dt, filter_sigma_px, rng, n_particles=n_particles)
    pf.initialize(mean0, cov0)
    enforce_hardware_bounds(pf)
    estimates = []
    runtimes = []
    for y, mask in measurements:
        est, runtime = pf.step(y, mask)
        est = enforce_hardware_bounds(pf)
        estimates.append(est)
        runtimes.append(runtime)
    estimates_arr = np.asarray(estimates)
    errors = np.linalg.norm(estimates_arr[:, :3] - truth[:, :3], axis=1)
    return estimates_arr, errors, np.asarray(runtimes)


def projection_residual_stats(
    cameras: list[Camera],
    measurements: list[tuple[np.ndarray, np.ndarray]],
    truth: np.ndarray,
) -> tuple[float, float, float]:
    residuals = []
    for x, (y, mask) in zip(truth, measurements):
        yhat, hmask = project_state(cameras, x)
        use = mask & hmask & np.isfinite(y) & np.isfinite(yhat)
        for cam_idx in range(len(cameras)):
            rows = np.array([2 * cam_idx, 2 * cam_idx + 1])
            if np.all(use[rows]):
                residuals.append(float(np.linalg.norm(y[rows] - yhat[rows])))
    if not residuals:
        return float("nan"), float("nan"), 0.0
    arr = np.asarray(residuals, dtype=float)
    return float(np.median(arr)), float(np.percentile(arr, 90)), 100.0 * len(arr) / (len(measurements) * len(cameras))


def summarize_case_runs(case_runs: list[dict]) -> dict:
    rmse = np.array([run["pf_rmse_m"] for run in case_runs], dtype=float)
    mean_err = np.array([run["pf_mean_error_m"] for run in case_runs], dtype=float)
    p90_err = np.array([run["pf_p90_error_m"] for run in case_runs], dtype=float)
    tri_rmse = np.array([run["tri_rmse_m"] for run in case_runs], dtype=float)
    tri_valid = np.array([run["tri_valid_percent"] for run in case_runs], dtype=float)
    pix_med = np.array([run["pixel_residual_median_px"] for run in case_runs], dtype=float)
    pix_p90 = np.array([run["pixel_residual_p90_px"] for run in case_runs], dtype=float)
    pix_valid = np.array([run["pixel_common_valid_percent"] for run in case_runs], dtype=float)
    runtime = np.array([run["pf_runtime_ms"] for run in case_runs], dtype=float)
    return {
        "pf_rmse_mean_m": float(np.nanmean(rmse)),
        "pf_rmse_std_m": float(np.nanstd(rmse)),
        "pf_mean_error_m": float(np.nanmean(mean_err)),
        "pf_p90_error_m": float(np.nanmean(p90_err)),
        "tri_rmse_mean_m": float(np.nanmean(tri_rmse)),
        "tri_valid_percent": float(np.nanmean(tri_valid)),
        "pixel_residual_median_px": float(np.nanmean(pix_med)),
        "pixel_residual_p90_px": float(np.nanmean(pix_p90)),
        "pixel_common_valid_percent": float(np.nanmean(pix_valid)),
        "pf_runtime_ms": float(np.nanmean(runtime)),
    }


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fs:
        writer = csv.DictWriter(fs, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_ablation(rows: list[dict], representative: dict[str, dict], truth: np.ndarray, look_at_w: np.ndarray) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    labels = [r["label"] for r in rows]
    x = np.arange(len(rows))

    plt.figure(figsize=(12.5, 5.3))
    plt.bar(x - 0.18, [r["pf_rmse_mean_m"] for r in rows], width=0.36, label="PF RMSE")
    plt.bar(x + 0.18, [r["tri_rmse_mean_m"] for r in rows], width=0.36, label="Triangulation RMSE")
    plt.axhline(5.0, color="#b22222", linestyle="--", linewidth=1.4, label="5 m hardware-scale error")
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("3D error [m]")
    plt.title("PURT calibration ablation: wrong intrinsics/extrinsics reproduce multi-meter error")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "hardware_calibration_ablation_rmse.png", dpi=180)
    plt.close()

    plt.figure(figsize=(12.5, 4.8))
    plt.bar(x, [r["pixel_residual_p90_px"] for r in rows], color="#5b7dbb")
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("P90 reprojection residual [px]")
    plt.title("Pixel residual at the true 3D point under the assumed calibration")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "hardware_calibration_pixel_residual.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8.0, 5.4))
    for name, c in PURT_CAM_CENTERS_W.items():
        plt.scatter(c[0], c[1], s=90)
        plt.text(c[0] + 0.25, c[1] + 0.25, name)
        plt.plot([c[0], look_at_w[0]], [c[1], look_at_w[1]], color="#555555", alpha=0.45)
    plt.plot(truth[:, 0], truth[:, 1], "k--", linewidth=2.0, label="simulated drone path")
    plt.scatter([0.0], [0.0], marker="+", s=120, color="k", label="QTM origin")
    plt.axis("equal")
    plt.xlabel("QTM x [m]")
    plt.ylabel("QTM y [m]")
    plt.title("Simulated PURT camera layout from run-1 mapping")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "hardware_purt_camera_layout.png", dpi=180)
    plt.close()

    if MARKER_PHOTO_PATH.exists():
        shutil.copyfile(MARKER_PHOTO_PATH, FIGURES_DIR / "hardware_active_marker_camera.jpeg")
    if QTM_MAPPING_PATH.exists():
        shutil.copyfile(QTM_MAPPING_PATH, FIGURES_DIR / "hardware_qtm_up_mapping_run1.png")


def format_intrinsics(K: np.ndarray, width: int, height: int) -> str:
    return (
        f"`{width}x{height}`, fx `{K[0,0]:.3f}`, fy `{K[1,1]:.3f}`, "
        f"cx `{K[0,2]:.3f}`, cy `{K[1,2]:.3f}`"
    )


def write_hardware_report(
    rows: list[dict],
    active_info: tuple[np.ndarray, int, int],
    bad_info: tuple[np.ndarray, int, int],
    trajectory_meta: dict,
    look_at_w: np.ndarray,
) -> None:
    K_good, w_good, h_good = active_info
    K_bad, w_bad, h_bad = bad_info

    table = [
        "| Case | Assumed intrinsics | Extrinsic error | PF RMSE [m] | PF P90 [m] | Triangulation RMSE [m] | True-point P90 residual [px] | Step time [ms] |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        intr = "`camera0.yaml`" if r["use_bad_intrinsics"] else "`up*.yaml`"
        extr = "correct" if r["rot_deg"] == 0 and r["trans_m"] == 0 else f"{r['rot_deg']:.0f} deg, {r['trans_m']:.2f} m"
        table.append(
            f"| {r['label']} | {intr} | {extr} | {r['pf_rmse_mean_m']:.3f} +/- {r['pf_rmse_std_m']:.3f} | "
            f"{r['pf_p90_error_m']:.3f} | {r['tri_rmse_mean_m']:.3f} | "
            f"{r['pixel_residual_p90_px']:.1f} | {r['pf_runtime_ms']:.3f} |"
        )

    marker_like = next(r for r in rows if r["name"] == "marker_like_plus_camera0")
    clean = next(r for r in rows if r["name"] == "correct")
    camera0_only = next(r for r in rows if r["name"] == "camera0_intrinsics_only")
    marker_only = next(r for r in rows if r["name"] == "marker_like_extrinsics")

    lie_group_pf_extension = r"""## Proposed Extension: Body-Frame-Acceleration Lie-Group PF

The current implemented PF in this report is a position/velocity bootstrap PF.
It does not estimate drone attitude.  A more Lie-group-centered extension is to
let each particle carry a latent inertial-navigation state while still reporting
only the drone trajectory:

$$
X_k^{(i)} =
\left(R_{wb,k}^{(i)}, v_{w,k}^{(i)}, p_{w,k}^{(i)}\right)
\in SE_2(3).
$$

Equivalently, one can write the particle state in block form as

$$
X_k^{(i)} =
\begin{bmatrix}
R_{wb,k}^{(i)} & v_{w,k}^{(i)} & p_{w,k}^{(i)}\\
0_{1\times 3} & 1 & 0\\
0_{1\times 3} & 0 & 1
\end{bmatrix}.
$$

Here \(R_{wb,k}^{(i)}\in SO(3)\) is a latent attitude hypothesis,
\(v_{w,k}^{(i)}\) is world-frame velocity, and \(p_{w,k}^{(i)}\) is the
world-frame position used for camera projection and for the final trajectory
metric.  The attitude is included only so that body-frame IMU readings can be
used correctly in prediction.

Let the body-frame IMU input be

$$
u_k=(\omega_{m,k}, a_{m,k}),
$$

where \(\omega_{m,k}\) is gyroscope angular velocity and \(a_{m,k}\) is the
accelerometer specific-force reading.  With sampled body-frame process noise
\(\eta_{\omega,k}^{(i)}\) and \(\eta_{a,k}^{(i)}\), and no additional IMU
state terms, the particle prediction is

$$
R_{wb,k+1}^{(i)}
=
R_{wb,k}^{(i)}
\operatorname{Exp}
\left(
\left(\omega_{m,k}+\eta_{\omega,k}^{(i)}\right)^\wedge \Delta t
\right).
$$

Equivalently, in code this should be implemented as

$$
R_{wb,k+1}^{(i)}
=
R_{wb,k}^{(i)}
\operatorname{Exp}_{SO(3)}
\left(
\left(\omega_{m,k}+\eta_{\omega,k}^{(i)}\right)\Delta t
\right).
$$

The accelerometer is mapped from body frame to world frame through the particle
attitude:

$$
a_{w,k}^{(i)}
=
R_{wb,k}^{(i)}
\left(a_{m,k}+\eta_{a,k}^{(i)}\right)
+g_w.
$$

If \(a_{m,k}\) is already gravity-compensated acceleration instead of specific
force, the \(g_w\) term is omitted.  The Euclidean velocity and position blocks
then propagate as

$$
v_{w,k+1}^{(i)}
=
v_{w,k}^{(i)}
+\Delta t\,a_{w,k}^{(i)},
$$

$$
p_{w,k+1}^{(i)}
=
p_{w,k}^{(i)}
+\Delta t\,v_{w,k}^{(i)}
+\frac{1}{2}\Delta t^2 a_{w,k}^{(i)}.
$$

The camera measurement update remains the same projection likelihood used in the
current PF:

$$
\hat z_{j,k}^{(i)}
=
\pi\left(K_j,T_{j,cw}\bar p_{w,k}^{(i)}\right),
\qquad
w_k^{(i)}
\propto
w_{k|k-1}^{(i)}
\prod_{j\in\mathcal V_k}
\mathcal N\left(z_{j,k};\hat z_{j,k}^{(i)},R_j\right).
$$

After normalization, resampling copies the entire latent state
\((R_{wb}^{(i)},v_w^{(i)},p_w^{(i)})\).  The reported trajectory estimate is
still only the weighted position mean:

$$
\hat p_{w,k}
=
\sum_{i=1}^{N_p} w_k^{(i)}p_{w,k}^{(i)}.
$$

This is the main conceptual point: the PF can carry latent attitude for
body-frame acceleration prediction without changing the output task into full
pose estimation.

### Left-Invariant Error: Diagnostic, Not The PF Estimate

For this particle filter, a left-invariant error is not needed to perform the
filter update.  The PF represents uncertainty directly with samples, so it does
not propagate a Kalman covariance or linearized error dynamics.  The
left-invariant error is still useful for explaining particle spread on the
group.

Given a weighted group mean \(\bar X_k\), define the particle error

$$
\eta_k^{(i)}=\bar X_k^{-1}X_k^{(i)}.
$$

The tangent-space error is

$$
\epsilon_k^{(i)}
=
\log\left(\bar X_k^{-1}X_k^{(i)}\right)^\vee.
$$

For \(SE_2(3)\), this error contains attitude, velocity, and position error
coordinates:

$$
\epsilon_k^{(i)}
\approx
\begin{bmatrix}
\delta\theta_k^{(i)}\\
\delta v_{b,k}^{(i)}\\
\delta p_{b,k}^{(i)}
\end{bmatrix}.
$$

The \(b\) subscript means the error is expressed in the body frame of the
reference mean.  It does not mean the filter estimates body-frame position or
body-frame velocity; the stored particle states remain \(p_w^{(i)}\) and
\(v_w^{(i)}\).

The weighted Lie-group mean is locally defined as

$$
\bar X_k
=
\arg\min_Y
\frac{1}{2}
\sum_i w_k^{(i)}
\left\|
\log\left(Y^{-1}X_k^{(i)}\right)^\vee
\right\|^2.
$$

Let

$$
\epsilon_i=
\log\left(\bar X_k^{-1}X_k^{(i)}\right)^\vee.
$$

Perturb the mean by

$$
Y=\bar X_k\operatorname{Exp}(\delta^\wedge).
$$

Then

$$
Y^{-1}X_k^{(i)}
=
\operatorname{Exp}(-\delta^\wedge)\bar X_k^{-1}X_k^{(i)}.
$$

Using the first-order Baker-Campbell-Hausdorff approximation near the mean,

$$
\log\left(Y^{-1}X_k^{(i)}\right)^\vee
\approx
\epsilon_i-\delta.
$$

So the local cost becomes

$$
J(\delta)
\approx
\frac{1}{2}
\sum_i w_k^{(i)}
\left\|\epsilon_i-\delta\right\|^2.
$$

At the optimum, the derivative at \(\delta=0\) must vanish:

$$
\left.
\frac{\partial J}{\partial \delta}
\right|_{\delta=0}
=
-\sum_i w_k^{(i)}\epsilon_i
=0.
$$

Therefore

$$
\sum_i w_k^{(i)}
\log\left(\bar X_k^{-1}X_k^{(i)}\right)^\vee
\approx 0.
$$

This is the Lie-group analogue of the Euclidean fact that weighted errors around
the mean sum to zero.  In this proposed PF, it should be presented as a
diagnostic or uncertainty-description tool, not as the state estimate itself.
"""

    report = fr"""# Hardware Calibration Hypothesis Simulation

Generated by `lg_filter_comparison/hardware_calibration_ablation.py`.

## Question

The PURT hardware run on `2026-04-24` produced very large raw PF errors.  The
hypothesis tested here is:

1. the active-marker rigid bodies gave a stable QTM body pose, but that body
   pose was not the optical camera pose used by the projection model; and
2. the old `camera0.yaml` intrinsics were used for `640x480` UP camera images,
   which put the principal point and focal lengths in the wrong pixel frame.

This is a simulator ablation.  The image measurements are generated from a
clean PURT camera rig following the actual extracted PURT run-1 QTM motion,
then the filter is deliberately run with the wrong calibration.  That isolates
calibration error from detector, ROS timing, and QTM parsing.

## Hardware Inputs Mirrored In Simulation

- Real data root: `/home/siddharth/up_bags/2026-04-24`
- Hardware evaluation note: `{HARDWARE_DOC_PATH}`
- Confirmed stream/body mapping: `UP1 -> cam2`, `UP2 -> cam4`, `UP3 -> cam1`
- Active UP intrinsics: {format_intrinsics(K_good, w_good, h_good)}
- Erroneous `camera0.yaml`: {format_intrinsics(K_bad, w_bad, h_bad)}
- PF bounds copied from the hardware tracker: `x in [-12, 6] m`,
  `y in [-10, 5] m`, `z in [0, 2.1] m`, speed clipped to `5 m/s`
- Drone trajectory: actual PURT run-1 QTM path from
  `{trajectory_meta.get("trajectory_csv", "")}`
- QTM trajectory source: `{trajectory_meta.get("trajectory_source", "unknown")}`,
  `{trajectory_meta.get("raw_samples", 0)}` raw samples, resampled to
  `{trajectory_meta.get("resampled_samples", 0)}` simulation steps over
  `{trajectory_meta.get("duration_s", 0.0):.2f} s`
- Synthetic optical axes in this ablation are aimed at the trajectory median:
  `({look_at_w[0]:+.2f}, {look_at_w[1]:+.2f}, {look_at_w[2]:+.2f}) m`

The key bug in `camera0.yaml` is the pixel frame.  It is a `1600x1200`
calibration:

```text
K_bad = [[{K_bad[0,0]:.5f}, 0, {K_bad[0,2]:.5f}],
         [0, {K_bad[1,1]:.5f}, {K_bad[1,2]:.5f}],
         [0, 0, 1]]
```

but the UP bag images are `640x480`.  The active `up*.yaml` files correctly use

```text
K_up = [[{K_good[0,0]:.5f}, 0, {K_good[0,2]:.5f}],
        [0, {K_good[1,1]:.5f}, {K_good[1,2]:.5f}],
        [0, 0, 1]]
```

## Lie-Group Projection Math For The Ablation

The true synthetic camera measurement is

$$
z_{{j,k}}=\pi\left(K_j, T_{{j,cw}}\bar p_{{w,k}}\right)+\nu_{{j,k}},
$$

where the camera extrinsic is the Lie-group element

$$
T_{{j,cw}}=(R_{{j,cw}},t_{{j,cw}})\in SE(3),
$$

and it maps world points into camera coordinates as

$$
p_{{j,c,k}} = R_{{j,cw}}p_{{w,k}} + t_{{j,cw}}.
$$

The pinhole projection is

$$
u=f_x X_c/Z_c+c_x,\qquad v=f_y Y_c/Z_c+c_y.
$$

The wrong-extrinsic cases use the same physical camera layout for measurement
generation, but the filter assumes

$$
\hat R_{{j,wc}} = R_{{j,wc}}\operatorname{{Exp}}(\delta\phi_j),
\qquad
\hat t_{{j,wc}} = t_{{j,wc}} + R_{{j,wc}}\delta c_j.
$$

Then

$$
\hat R_{{j,cw}} = \hat R_{{j,wc}}^T,\qquad
\hat t_{{j,cw}} = -\hat R_{{j,wc}}^T\hat t_{{j,wc}}.
$$

This models the practical hardware mistake: treating an active marker body
frame as if it were the camera optical frame, or using a weak hand-eye transform
between those two frames.

## Results

{chr(10).join(table)}

The clean case is the control: with the correct `up*.yaml` intrinsics and the
correct optical extrinsics, PF RMSE is `{clean['pf_rmse_mean_m']:.3f} m`.

Using `camera0.yaml` alone raises the PF RMSE to
`{camera0_only['pf_rmse_mean_m']:.3f} m`.  A marker-like optical-frame error
alone raises it to `{marker_only['pf_rmse_mean_m']:.3f} m`.  Combining the same
marker-like extrinsic error with `camera0.yaml` raises it to
`{marker_like['pf_rmse_mean_m']:.3f} m`, which is the same order as the
multi-meter hardware errors.

![Hardware calibration RMSE ablation](figures/hardware_calibration_ablation_rmse.png)

![Hardware calibration pixel residual](figures/hardware_calibration_pixel_residual.png)

![Simulated PURT camera layout](figures/hardware_purt_camera_layout.png)

![Run-1 QTM to UP mapping](figures/hardware_qtm_up_mapping_run1.png)

![Active marker camera setup](figures/hardware_active_marker_camera.jpeg)

## Interpretation

The simulation supports your hypothesis.  The result is not saying the hardware
run was definitely caused only by calibration; the real run can still include
time offset, detector quality, and QTM marker visibility effects.  But it does
show that the calibration failure mode is sufficient: a `camera0.yaml` pixel
frame mismatch plus plausible marker-frame-to-optical-frame errors can produce
the observed 3-6 m class of raw PF error even when the simulated measurements
are otherwise clean.

The direct triangulation row is important because it removes PF tuning from the
argument.  If the same 2D points are triangulated through the wrong camera
model, the recovered 3D point is already wrong by meters.  The PF then starts
from and repeatedly gets pulled by that wrong geometry.

{lie_group_pf_extension}

## Filter Initialization In This Simulation

For the hardware-style PF ablation, the filter is initialized from the first
few camera measurements, not from the true state.  The code triangulates early
multi-camera bbox centers, takes the median of the early 3D anchors as the
initial position, and estimates initial velocity by a finite difference between
the first and last accepted anchors:

$$
\hat p_0 = \operatorname{{median}}(\tilde p_1,\ldots,\tilde p_{{N_b}}),
\qquad
\hat v_0 = \frac{{\tilde p_{{N_b}}-\tilde p_1}}{{t_{{N_b}}-t_1}}.
$$

This mirrors the hardware tracker bootstrap idea.  The particles are then drawn
around `[\hat p_0,\hat v_0]` using a broad covariance.  If the assumed
calibration is wrong, this bootstrap is also wrong, which is exactly the
failure mode we are testing.

The ground-truth velocity in the simulator is obtained from the QTM trajectory
by resampling position at a fixed rate and differentiating it numerically:

$$
v_k \approx \frac{{p_{{k+1}}-p_{{k-1}}}}{{2\Delta t}}.
$$

That velocity is used only as simulator truth and for error evaluation.  The PF
does not receive this true velocity.  It only receives 2D image measurements
and its own bootstrap/prediction model.

## Recommendation

For the next hardware pass, the calibration checklist should be:

- Use only the active `up1.yaml`, `up2.yaml`, and `up3.yaml` intrinsics for the
  `640x480` streams.
- Treat QTM active-marker poses as marker-body poses, not optical camera poses.
- Calibrate a fixed `T_marker,optical` transform for each physical camera, or
  fit optical `R_cw` from synchronized image detections and QTM drone positions.
- I will figure out a better way to get camera extrinsics, either with a proper
  hand-eye calibration between the active-marker body and optical frame, or by
  fitting optical extrinsics from QTM drone positions and image detections.
- Record a short checkerboard/AprilTag-style validation sequence for each UP
  camera so the intrinsic scaling and distortion model can be checked before
  the drone flight.
- Before PF, run a reprojection sanity check: project QTM drone positions into
  each camera and verify that the projected point lands on the detected drone.
  If the residual is hundreds of pixels, do not run the filter yet.
- Use a single global QTM/video offset, then verify it with image-space
  residuals over time.  Different best offsets per camera mean timing and
  calibration are still coupled.
- Start with a simple flight segment where the drone is visible in at least two
  cameras for several seconds; use that segment to validate camera
  triangulation before asking the PF to survive dropouts.
- Save the calibrated `camera_poses.json`, the exact camera YAMLs, and the QTM
  offset next to the bag outputs so the run can be reproduced exactly.
"""

    out_path = RESULTS_DIR / "hardware_calibration_ablation.md"
    out_path.write_text(report, encoding="utf-8")

    main_report = RESULTS_DIR / "report.md"
    if main_report.exists():
        marker_start = "<!-- hardware_calibration_ablation:start -->"
        marker_end = "<!-- hardware_calibration_ablation:end -->"
        section = f"\n\n{marker_start}\n{report}\n{marker_end}\n"
        text = main_report.read_text(encoding="utf-8")
        if marker_start in text and marker_end in text:
            before = text.split(marker_start)[0]
            after = text.split(marker_end, 1)[1]
            text = before + section.lstrip("\n") + after
        else:
            text = text.rstrip() + section
        main_report.write_text(text, encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    active_info = load_camera_yaml(ACTIVE_INTRINSICS_PATH)
    bad_info = load_camera_yaml(BAD_INTRINSICS_PATH)
    K_good, width_good, height_good = active_info
    K_bad, width_bad, height_bad = bad_info

    output_rate_hz = 20.0
    t, truth, trajectory_meta = load_qtm_run1_trajectory(output_rate_hz)
    p_w = truth[:, :3]
    look_at_w = np.median(p_w, axis=0)
    true_cameras = make_purt_camera_rig(K_good, width_good, height_good, look_at_w=look_at_w)
    cases = [
        CalibrationCase(
            "correct",
            "Correct",
            False,
            0.0,
            0.0,
            "Correct UP intrinsics and optical camera extrinsics.",
        ),
        CalibrationCase(
            "camera0_intrinsics_only",
            "camera0.yaml only",
            True,
            0.0,
            0.0,
            "Use old 1600x1200 camera0.yaml on 640x480 detections.",
        ),
        CalibrationCase(
            "mild_extrinsics",
            "5 deg, 0.10 m",
            False,
            5.0,
            0.10,
            "Small marker-to-optical-frame error.",
        ),
        CalibrationCase(
            "marker_like_extrinsics",
            "15 deg, 0.35 m",
            False,
            15.0,
            0.35,
            "Marker-bar/body pose treated approximately as optical pose.",
        ),
        CalibrationCase(
            "severe_extrinsics",
            "25 deg, 0.70 m",
            False,
            25.0,
            0.70,
            "Large marker-frame/optical-frame calibration error.",
        ),
        CalibrationCase(
            "marker_like_plus_camera0",
            "15 deg + camera0",
            True,
            15.0,
            0.35,
            "Marker-like extrinsic error plus wrong camera0.yaml intrinsics.",
        ),
        CalibrationCase(
            "severe_plus_camera0",
            "25 deg + camera0",
            True,
            25.0,
            0.70,
            "Severe extrinsic error plus wrong camera0.yaml intrinsics.",
        ),
    ]

    dt = float(np.median(np.diff(t)))
    measurement_noise_px = 8.0
    filter_sigma_px = 18.0
    n_particles = 450
    n_mc = 6

    rows = []
    representative: dict[str, dict] = {}

    for case_idx, case in enumerate(cases):
        assumed_K, assumed_w, assumed_h = (K_bad, width_bad, height_bad) if case.use_bad_intrinsics else (K_good, width_good, height_good)
        assumed = clone_with_intrinsics(true_cameras, assumed_K, assumed_w, assumed_h)
        assumed = apply_marker_frame_error(assumed, case.rot_deg, case.trans_m)
        case_runs = []

        for mc in range(n_mc):
            rng_meas = np.random.default_rng(20260424 + 1000 * case_idx + mc)
            measurements = make_measurements(true_cameras, p_w, measurement_noise_px, rng_meas)
            tri_est, tri_errors, tri_valid_percent = triangulation_series(assumed, measurements, truth)
            tri_rmse = math.sqrt(float(np.nanmean(tri_errors * tri_errors)))
            pix_med, pix_p90, pix_valid = projection_residual_stats(assumed, measurements, truth)
            estimates, pf_errors, runtimes = run_pf_case(
                assumed,
                measurements,
                truth,
                dt,
                filter_sigma_px,
                seed=9000 + 1000 * case_idx + mc,
                n_particles=n_particles,
            )
            case_runs.append(
                {
                    "pf_rmse_m": math.sqrt(float(np.mean(pf_errors * pf_errors))),
                    "pf_mean_error_m": float(np.mean(pf_errors)),
                    "pf_p90_error_m": float(np.percentile(pf_errors, 90)),
                    "tri_rmse_m": tri_rmse,
                    "tri_valid_percent": tri_valid_percent,
                    "pixel_residual_median_px": pix_med,
                    "pixel_residual_p90_px": pix_p90,
                    "pixel_common_valid_percent": pix_valid,
                    "pf_runtime_ms": float(np.mean(runtimes) * 1e3),
                }
            )
            if mc == 0:
                representative[case.name] = {
                    "label": case.label,
                    "estimates": estimates,
                    "pf_errors": pf_errors,
                    "tri_estimates": tri_est,
                    "tri_errors": tri_errors,
                }

        summary = summarize_case_runs(case_runs)
        rows.append(
            {
                "name": case.name,
                "label": case.label,
                "use_bad_intrinsics": case.use_bad_intrinsics,
                "rot_deg": case.rot_deg,
                "trans_m": case.trans_m,
                "measurement_noise_px": measurement_noise_px,
                "filter_sigma_px": filter_sigma_px,
                "particles": n_particles,
                "monte_carlo_runs": n_mc,
                "trajectory_source": trajectory_meta.get("trajectory_source", ""),
                "trajectory_duration_s": trajectory_meta.get("duration_s", float(t[-1] - t[0])),
                "note": case.note,
                **summary,
            }
        )

    write_csv(rows, RESULTS_DIR / "hardware_calibration_ablation.csv")
    plot_ablation(rows, representative, truth, look_at_w)
    write_hardware_report(rows, active_info, bad_info, trajectory_meta, look_at_w)

    print("Hardware calibration ablation complete")
    for r in rows:
        print(
            f"{r['label']:<20s} PF RMSE={r['pf_rmse_mean_m']:.3f} m, "
            f"tri={r['tri_rmse_mean_m']:.3f} m, pixP90={r['pixel_residual_p90_px']:.1f} px"
        )
    print(f"Wrote {RESULTS_DIR / 'hardware_calibration_ablation.csv'}")
    print(f"Wrote {RESULTS_DIR / 'hardware_calibration_ablation.md'}")


if __name__ == "__main__":
    main()
