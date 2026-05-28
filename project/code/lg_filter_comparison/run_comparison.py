#!/usr/bin/env python3
"""Run the Lie-group projection filter comparison and write a report."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from filters import BootstrapPF, ErrorStateEKF, GatedErrorStateEKF, GatedIteratedEKF, IteratedEKF
from lie_projection import make_square_camera_rig, project_state


FILTER_CLASSES = [ErrorStateEKF, IteratedEKF, GatedErrorStateEKF, GatedIteratedEKF, BootstrapPF]
FIGURE_EIGHT_PERIOD_S = 12.0


@dataclass(frozen=True)
class Scenario:
    name: str
    pixel_sigma: float
    init_bias: np.ndarray
    init_vel_bias: np.ndarray
    init_pos_std: float
    init_vel_std: float
    dropout_prob: float = 0.0


def trajectory(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A complete, compact figure-eight that stays visible in all cameras."""
    amp_x = 2.0
    amp_y = 1.2
    amp_z = 0.25
    altitude = 2.8
    omega = 2.0 * math.pi / FIGURE_EIGHT_PERIOD_S
    phase = 0.6
    vertical_scale = 0.35

    x = amp_x * np.sin(omega * t)
    y = amp_y * np.sin(2.0 * omega * t + phase)
    z = altitude + amp_z * np.sin(vertical_scale * omega * t)
    vx = amp_x * omega * np.cos(omega * t)
    vy = 2.0 * amp_y * omega * np.cos(2.0 * omega * t + phase)
    vz = amp_z * vertical_scale * omega * np.cos(vertical_scale * omega * t)
    return np.column_stack([x, y, z]), np.column_stack([vx, vy, vz])


def make_measurement(cameras, p_w: np.ndarray, scenario: Scenario, rng: np.random.Generator):
    y, mask = project_state(cameras, np.hstack([p_w, np.zeros(3)]))
    y = np.array(y, copy=True)
    mask = np.array(mask, copy=True)
    for cam_idx, cam in enumerate(cameras):
        rows = np.array([2 * cam_idx, 2 * cam_idx + 1])
        if not np.all(mask[rows]):
            continue
        if rng.uniform() < scenario.dropout_prob:
            mask[rows] = False
            y[rows] = np.nan
            continue
        y[rows] += rng.normal(0.0, scenario.pixel_sigma, size=2)
    return y, mask


def initial_conditions(true_state0: np.ndarray, scenario: Scenario):
    mean = true_state0.copy()
    mean[:3] += scenario.init_bias
    mean[3:] += scenario.init_vel_bias
    cov = np.diag(
        [
            scenario.init_pos_std,
            scenario.init_pos_std,
            0.6 * scenario.init_pos_std,
            scenario.init_vel_std,
            scenario.init_vel_std,
            0.7 * scenario.init_vel_std,
        ]
    ) ** 2
    return mean, cov


def run_one(
    scenario: Scenario,
    seed: int,
    t: np.ndarray,
    true_state: np.ndarray,
    n_particles: int,
):
    cameras = make_square_camera_rig()
    rng_meas = np.random.default_rng(seed + 10_000)
    measurements = [make_measurement(cameras, p, scenario, rng_meas) for p in true_state[:, :3]]
    mean0, cov0 = initial_conditions(true_state[0], scenario)
    results = {}

    for idx, cls in enumerate(FILTER_CLASSES):
        rng = np.random.default_rng(seed + 1_000 * (idx + 1))
        if cls is BootstrapPF:
            filt = cls(cameras, t[1] - t[0], scenario.pixel_sigma, rng, n_particles=n_particles)
        else:
            filt = cls(cameras, t[1] - t[0], scenario.pixel_sigma, rng)
        filt.initialize(mean0, cov0)
        estimates = []
        runtimes = []
        for y, mask in measurements:
            est, dt_run = filt.step(y, mask)
            estimates.append(est)
            runtimes.append(dt_run)
        estimates = np.asarray(estimates)
        errors = np.linalg.norm(estimates[:, :3] - true_state[:, :3], axis=1)
        results[filt.name] = {
            "estimates": estimates,
            "errors": errors,
            "runtimes": np.asarray(runtimes),
        }
    return results


def summarize(all_results):
    rows = []
    for scenario_name, scenario_runs in all_results.items():
        by_filter = {cls.name: [] for cls in FILTER_CLASSES}
        for run in scenario_runs:
            for name, data in run.items():
                by_filter[name].append(data)
        for name, runs in by_filter.items():
            rmse = np.array([math.sqrt(float(np.mean(r["errors"] ** 2))) for r in runs])
            mean_err = np.array([float(np.mean(r["errors"])) for r in runs])
            p90 = np.array([float(np.percentile(r["errors"], 90)) for r in runs])
            final = np.array([float(r["errors"][-1]) for r in runs])
            runtime_ms = np.array([float(np.mean(r["runtimes"]) * 1e3) for r in runs])
            rows.append(
                {
                    "scenario": scenario_name,
                    "filter": name,
                    "rmse_mean_m": float(np.mean(rmse)),
                    "rmse_std_m": float(np.std(rmse)),
                    "mean_error_m": float(np.mean(mean_err)),
                    "p90_error_m": float(np.mean(p90)),
                    "final_error_m": float(np.mean(final)),
                    "runtime_mean_ms": float(np.mean(runtime_ms)),
                    "runtime_std_ms": float(np.std(runtime_ms)),
                }
            )
    return rows


def write_csv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(all_results, summary_rows, t: np.ndarray, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        "MEKF": "#222222",
        "ItEKF": "#2ca02c",
        "MEKF-gated": "#9467bd",
        "ItEKF-gated": "#ff7f0e",
        "PF": "#1f77b4",
    }

    first_scenario = next(iter(all_results))
    first_run = all_results[first_scenario][0]
    plt.figure(figsize=(7, 4.5))
    for name, data in first_run.items():
        est = data["estimates"]
        plt.plot(est[:, 0], est[:, 1], label=name, color=colors.get(name))
    p, _ = trajectory(t)
    plt.plot(p[:, 0], p[:, 1], "k--", label="truth")
    plt.axis("equal")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title(f"Top-view trajectory, {first_scenario}")
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "trajectory_top_view.png", dpi=180)
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, (scenario_name, runs) in zip(axes, all_results.items()):
        for name in first_run:
            err = np.array([run[name]["errors"] for run in runs])
            ax.plot(t, np.mean(err, axis=0), label=name, color=colors.get(name))
        ax.set_title(scenario_name)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("time [s]")
        ax.set_ylabel("position error [m]")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(FILTER_CLASSES))
    fig.suptitle("Mean position error over Monte Carlo runs", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_dir / "time_error_comparison.png", dpi=180)
    plt.close(fig)

    scenarios = list(all_results.keys())
    filters = [cls.name for cls in FILTER_CLASSES]
    x = np.arange(len(scenarios))
    width = min(0.16, 0.82 / max(len(filters), 1))
    plt.figure(figsize=(10, 4.8))
    for i, name in enumerate(filters):
        vals = [
            next(r for r in summary_rows if r["scenario"] == s and r["filter"] == name)["rmse_mean_m"]
            for s in scenarios
        ]
        stds = [
            next(r for r in summary_rows if r["scenario"] == s and r["filter"] == name)["rmse_std_m"]
            for s in scenarios
        ]
        offset = (i - (len(filters) - 1) / 2.0) * width
        plt.bar(x + offset, vals, width, yerr=stds, capsize=3, label=name, color=colors.get(name))
    plt.xticks(x, scenarios, rotation=15, ha="right")
    plt.ylabel("RMSE [m]")
    plt.title("Monte Carlo RMSE by scenario")
    plt.grid(axis="y", alpha=0.3)
    plt.legend(ncol=len(filters))
    plt.tight_layout()
    plt.savefig(out_dir / "rmse_bar.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4.6))
    vals = [
        np.mean([r["runtime_mean_ms"] for r in summary_rows if r["filter"] == name])
        for name in filters
    ]
    plt.bar(filters, vals, color=[colors.get(f) for f in filters])
    plt.ylabel("mean step time [ms]")
    plt.title("Mean per-step runtime")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "runtime_bar.png", dpi=180)
    plt.close()


def write_report(path: Path, rows, scenarios, config):
    fig_dir = "figures"
    table_lines = [
        "| Scenario | Filter | RMSE mean [m] | RMSE std [m] | Mean error [m] | P90 [m] | Final [m] | Step time [ms] |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        table_lines.append(
            f"| {r['scenario']} | {r['filter']} | {r['rmse_mean_m']:.4f} | {r['rmse_std_m']:.4f} | "
            f"{r['mean_error_m']:.4f} | {r['p90_error_m']:.4f} | {r['final_error_m']:.4f} | {r['runtime_mean_ms']:.3f} |"
        )

    runtime_lines = [
        "| Filter | Mean Step Time [ms] | Std Across Scenarios [ms] |",
        "|---|---:|---:|",
    ]
    for name in [cls.name for cls in FILTER_CLASSES]:
        vals = np.array([r["runtime_mean_ms"] for r in rows if r["filter"] == name], dtype=float)
        runtime_lines.append(f"| {name} | {np.mean(vals):.3f} | {np.std(vals):.3f} |")

    scenario_text = "\n".join(
        f"- `{s.name}`: pixel sigma `{s.pixel_sigma}` px, init position bias `{np.round(s.init_bias, 2).tolist()}`, "
        f"dropout `{s.dropout_prob}`."
        for s in scenarios
    )
    camera_lines = [
        "| Camera | C_w [m] | yaw [deg] | fx | fy | cx | cy | image |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, cam in enumerate(make_square_camera_rig()):
        center = cam.center_w if cam.center_w is not None else -cam.T_cw.R.T @ cam.T_cw.t
        yaw = math.degrees(cam.yaw_rad) if cam.yaw_rad is not None else float("nan")
        camera_lines.append(
            f"| camera_{idx} | ({center[0]:+.2f}, {center[1]:+.2f}, {center[2]:+.2f}) | "
            f"{yaw:+.1f} | {cam.K[0,0]:.1f} | {cam.K[1,1]:.1f} | "
            f"{cam.K[0,2]:.1f} | {cam.K[1,2]:.1f} | {cam.width}x{cam.height} |"
        )
    particle_sweep_section = ""
    particle_sweep_path = path.parent / "pf_particle_sweep.csv"
    if particle_sweep_path.exists():
        with particle_sweep_path.open(newline="") as f:
            sweep_rows = list(csv.DictReader(f))
        if sweep_rows:
            numeric_sweep = [
                {
                    "particles": int(float(r["particles"])),
                    "rmse_mean_m": float(r["rmse_mean_m"]),
                    "p90_error_m": float(r["p90_error_m"]),
                    "runtime_mean_ms": float(r["runtime_mean_ms"]),
                }
                for r in sweep_rows
            ]
            best_pf = min(numeric_sweep, key=lambda r: r["rmse_mean_m"])
            pf_450 = next((r for r in numeric_sweep if r["particles"] == 450), None)
            gated_rows = [
                r
                for r in rows
                if r["scenario"] == "dropout" and r["filter"] in ("MEKF-gated", "ItEKF-gated")
            ]
            best_gated = min(gated_rows, key=lambda r: r["rmse_mean_m"]) if gated_rows else None
            sweep_lines = [
                "| Particles | PF RMSE mean [m] | RMSE std [m] | P90 [m] | Step time [ms] |",
                "|---:|---:|---:|---:|---:|",
            ]
            for r in sweep_rows:
                sweep_lines.append(
                    f"| {int(float(r['particles']))} | {float(r['rmse_mean_m']):.4f} | "
                    f"{float(r['rmse_std_m']):.4f} | {float(r['p90_error_m']):.4f} | "
                    f"{float(r['runtime_mean_ms']):.3f} |"
                )
            particle_sweep_section = "\n".join(
                [
                    "## PF Particle-Count Sensitivity",
                    "",
                    "Increasing PF particles usually helps by reducing sampling error and",
                    "particle impoverishment, but the gain is sublinear while computation grows",
                    "roughly linearly. The sweep below uses the same dropout scenario.",
                    "The main comparison keeps `PF` at 450 particles as the practical baseline;",
                    "the sweep shows the stronger high-particle PF behavior separately.",
                    "",
                    *sweep_lines,
                    "",
                    (
                        f"Best tested PF: `{best_pf['particles']}` particles, "
                        f"`{best_pf['rmse_mean_m']:.4f} m` RMSE, "
                        f"`{best_pf['runtime_mean_ms']:.3f} ms` per step."
                    ),
                    (
                        f"PF-450 baseline: `{pf_450['rmse_mean_m']:.4f} m` RMSE, "
                        f"`{pf_450['runtime_mean_ms']:.3f} ms` per step."
                        if pf_450
                        else ""
                    ),
                    (
                        f"Best gated EKF in the same scenario: `{best_gated['filter']}`, "
                        f"`{best_gated['rmse_mean_m']:.4f} m` RMSE, "
                        f"`{best_gated['runtime_mean_ms']:.3f} ms` per step."
                        if best_gated
                        else ""
                    ),
                    "",
                    "Interpretation: more particles improve the PF, but in this mostly unimodal",
                    "camera-geometry experiment they do not by themselves beat a well-gated EKF.",
                    "The PF advantage should be presented in robustness terms: sampled posterior",
                    "representation, non-Gaussian uncertainty, and tolerance to ambiguous",
                    "candidate sets. We should not claim PF wins merely by choosing a huge particle",
                    "count unless the Monte Carlo table actually supports that claim.",
                    "",
                    f"![PF particle-count sweep]({fig_dir}/pf_particle_sweep.png)",
                    "",
                ]
            )
    non_outlier_sweep_section = ""
    non_outlier_sweep_path = path.parent / "pf_non_outlier_sweep.csv"
    if non_outlier_sweep_path.exists():
        with non_outlier_sweep_path.open(newline="") as f:
            non_outlier_rows = list(csv.DictReader(f))
        if non_outlier_rows:
            non_outlier_lines = [
                "| Scenario | Particles | PF RMSE mean [m] | RMSE std [m] | P90 [m] | Step time [ms] | Best EKF RMSE [m] |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
            for r in non_outlier_rows:
                scenario_name = r["scenario"]
                best_ekf = min(
                    [
                        row
                        for row in rows
                        if row["scenario"] == scenario_name and row["filter"] != "PF"
                    ],
                    key=lambda row: row["rmse_mean_m"],
                )
                non_outlier_lines.append(
                    f"| {scenario_name} | {int(float(r['particles']))} | {float(r['rmse_mean_m']):.4f} | "
                    f"{float(r['rmse_std_m']):.4f} | {float(r['p90_error_m']):.4f} | "
                    f"{float(r['runtime_mean_ms']):.3f} | {best_ekf['rmse_mean_m']:.4f} |"
                )
            non_outlier_sweep_section = "\n".join(
                [
                    "## PF Particle Count In Non-Dropout Cases",
                    "",
                    "The same particle-count question was checked for the clean nominal,",
                    "poor-initialization, and high-pixel-noise cases. Here the measurements are",
                    "still single-target and mostly unimodal, so the EKF approximation is very",
                    "competitive. More particles reduce PF sampling error, but the PF remains",
                    "behind the best EKF in these clean cases.",
                    "",
                    *non_outlier_lines,
                    "",
                    "This is why the report presents PF as the robust sampled estimator rather",
                    "than claiming it is universally more accurate. The PF is the main research",
                    "object, but the clean-case baseline should show the true cost of using a",
                    "sampling method where a Gaussian filter is already well matched.",
                    "",
                ]
            )
    ambiguous_candidate_section = ""
    ambiguous_candidate_path = path.parent / "ambiguous_candidate_summary.csv"
    if ambiguous_candidate_path.exists():
        with ambiguous_candidate_path.open(newline="") as f:
            ambiguous_rows = list(csv.DictReader(f))
        if ambiguous_rows:
            ambiguous_lines = [
                "| Filter | RMSE mean [m] | RMSE std [m] | Mean error [m] | P90 [m] | Final [m] | Step time [ms] |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
            for r in ambiguous_rows:
                ambiguous_lines.append(
                    f"| {r['filter']} | {float(r['rmse_mean_m']):.4f} | {float(r['rmse_std_m']):.4f} | "
                    f"{float(r['mean_error_m']):.4f} | {float(r['p90_error_m']):.4f} | "
                    f"{float(r['final_error_m']):.4f} | {float(r['runtime_mean_ms']):.3f} |"
                )
            best_ambiguous = min(ambiguous_rows, key=lambda r: float(r["rmse_mean_m"]))
            ambiguous_candidate_section = "\n".join(
                [
                    "## Ambiguous Candidate-Set Case Where PF Is Preferable",
                    "",
                    "This experiment keeps the same trajectory, camera rig, and position-only",
                    "state, but changes the detector output. Each visible camera can return a",
                    "set of candidate bbox centers: one true-looking candidate and one",
                    "independent false candidate. The false candidate has the higher detector",
                    "score with probability `0.70`, so a single-association EKF is often fed",
                    "the wrong bbox. This is not an oracle advantage for the PF: the PF receives",
                    "the whole unlabeled candidate set, not the true association.",
                    "",
                    "For camera `j`, the detector candidate set at time `k` is",
                    "",
                    "$$",
                    "Z_{j,k}=\\{z_{j,k}^{(m)}\\}_{m=1}^{M_{j,k}}.",
                    "$$",
                    "",
                    "The EKF-style filters consume the top-scored association",
                    "",
                    "$$",
                    "\\tilde z_{j,k}=z_{j,k}^{(m^*)},\\quad m^*=\\arg\\max_m s_{j,k}^{(m)}.",
                    "$$",
                    "",
                    "The PF instead uses a candidate-mixture likelihood per camera:",
                    "",
                    "$$",
                    "L_{j,k}(x_k^{(i)})=\\epsilon+\\sum_{m=1}^{M_{j,k}}\\alpha_m",
                    "\\exp\\left[-\\frac{1}{2}\\left\\|\\frac{z_{j,k}^{(m)}-h_j(x_k^{(i)})}{\\sigma_z}\\right\\|^2\\right].",
                    "$$",
                    "",
                    "The particle weights are updated as",
                    "",
                    "$$",
                    "\\omega_k^{(i)} \\propto \\omega_{k|k-1}^{(i)}\\prod_j L_{j,k}(x_k^{(i)}).",
                    "$$",
                    "",
                    "Because the false candidates are independent across cameras, they usually",
                    "do not correspond to a single physically consistent 3D point. The true",
                    "hypothesis is the one that remains jointly plausible across the camera rig,",
                    "which is exactly what the PF's multi-hypothesis likelihood can exploit.",
                    "",
                    *ambiguous_lines,
                    "",
                    (
                        f"Best filter in this case: `{best_ambiguous['filter']}` with "
                        f"`{float(best_ambiguous['rmse_mean_m']):.4f} m` RMSE."
                    ),
                    "",
                    "This comparison is against the current EKF machinery. A more sophisticated",
                    "multi-hypothesis EKF, JPDA, or MHT-style tracker could also consume the",
                    "candidate set, but that is a different estimator architecture.",
                    "",
                    f"![Ambiguous candidate mean error]({fig_dir}/ambiguous_candidate_error.png)",
                    "",
                    f"![Ambiguous candidate RMSE]({fig_dir}/ambiguous_candidate_rmse.png)",
                    "",
                ]
            )
    report = r"""# Lie-Group Projection and Filter Comparison for Multi-Camera UAV Tracking

Generated by `lg_filter_comparison/run_comparison.py`.

## Executive Summary

This experiment compares four implemented estimator families under a shared
multi-camera UAV tracking simulation:

- **MEKF**: an error-state EKF over position and velocity. It is not a
  full body-frame pose-error EKF; because the current target state is
  `R^6`, its retraction reduces to addition.
- **ItEKF**: an Iterated EKF. It uses the same prediction model as MEKF, but
  relinearizes the nonlinear camera projection measurement several times inside
  each update.
- **MEKF-gated** and **ItEKF-gated**: the same EKF updates with per-camera
  Mahalanobis innovation gating. In a pure dropout case, missing camera feeds
  are skipped by both gated and ungated EKFs; gating only changes updates for
  measurements that actually arrive.
- **PF**: a bootstrap particle filter with likelihoods computed from
  multi-camera reprojection error using a sampled posterior.

The important modeling decision is that the **estimated drone state is not a
full pose**. It is

```text
x = [p, v] = [x, y, z, vx, vy, vz] in R^6.
```

The Lie-group component is the **measurement model**: each camera pose is an
element of `SE(3)`, and projection is written as a map from world coordinates to
camera coordinates to pixels.

## Problem Statement

Given a fixed calibrated camera network and noisy image-space detections of a
single UAV, estimate the UAV 3D position over time. The filters receive only
2D bounding-box centers from each camera. They do not receive the true 3D
position except for evaluation.

The evaluation asks:

1. Can Lie-group notation cleanly represent the camera projection geometry?
2. How do the ungated EKFs, gated EKFs, and PF compare under nominal,
   poor-initialization, high-noise, and dropout conditions?
3. Which pieces benefit from Lie groups and which pieces remain Euclidean
   because the target attitude is not estimated?

## Simulation Pipeline

The ROS/Gazebo scene corresponding to this simulation is the Purdue world
(`purdue.sdf`) with four fixed camera feeds and one `x500` target spawned as
`drone1`. The GIF and Monte Carlo experiment intentionally use a single target.
The Monte Carlo code abstracts the rendered feeds into calibrated camera
projections and bbox-center measurements.

1. A compact figure-eight target trajectory is generated. The angular rate is
chosen so the default simulation completes one full figure-eight cycle:

$$
p_x(t)=a_x \\sin(\\omega t),\\quad
p_y(t)=a_y \\sin(2\\omega t + \\phi),\\quad
p_z(t)=h + a_z \\sin(c_z \\omega t),\\quad
\\omega = 2\\pi / 12.
$$

Velocity is computed analytically and used as the ground-truth `v(t)`.

2. Four cameras are placed around the workspace:

```text
camera0 = ( r,  r, h_c)
camera1 = (-r,  r, h_c)
camera2 = (-r, -r, h_c)
camera3 = ( r, -r, h_c)
```

The Gazebo cameras are yaw-only cameras whose model/link +X axis points toward
the workspace. In the projection code they are converted into the usual optical
camera convention:

```text
camera x = image right, camera y = image down, camera z = forward.
```

Camera extrinsics/intrinsics used by the Monte Carlo code and the ROS GIF:

{chr(10).join(camera_lines)}

3. For camera `j`, the world-to-camera pose is represented as

$$
T_{{j,cw}} =
\\begin{{bmatrix}}
R_{{j,cw}} & t_{{j,cw}} \\\\
0 & 1
\\end{{bmatrix}} \\in SE(3).
$$

4. A world point is transformed and projected:

$$
\\bar p_{{j,c,k}} = T_{{j,cw}} \\bar p_{{w,k}},\\quad
p_{{j,c,k}} = [X_{{j,c,k}},Y_{{j,c,k}},Z_{{j,c,k}}]^\\top.
$$

$$
\\pi(K_j,p_{{j,c,k}})=
\\begin{{bmatrix}}
f_{{x,j}} X_{{j,c,k}}/Z_{{j,c,k}} + c_{{x,j}} \\\\
f_{{y,j}} Y_{{j,c,k}}/Z_{{j,c,k}} + c_{{y,j}}
\\end{{bmatrix}}.
$$

The measurement model for camera `j` at time `k` is therefore:

$$
z_{{j,k}} = h_j(x_k) + n_{{j,k}}
    = \\pi(K_j, T_{{j,cw}} \\bar p_{{w,k}}) + n_{{j,k}}.
$$

5. Pixel noise and optional dropout are applied to the projected detections.
Dropout means the camera produced no usable bbox for that frame, so the
measurement mask for that camera is false.

6. All filter variants consume the same measurement sequence in each Monte Carlo
run.

## Lie-Group Interpretation

Only quantities that naturally live on manifolds are represented as Lie-group
objects:

- `R_cw in SO(3)` for camera rotation,
- `T_cw in SE(3)` for camera pose,
- measurement residuals are evaluated after mapping through the group action.

The compact camera block displayed under each camera feed is

$$
K^{{(j)}} =
\begin{{bmatrix}}
f_x^{{(j)}} & 0 & c_x^{{(j)}} \\\\
0 & f_y^{{(j)}} & c_y^{{(j)}} \\\\
0 & 0 & 1
\end{{bmatrix}},
$$

$$
T_{{cw}}^{{(j)}} =
\begin{{bmatrix}}
R_{{cw}}^{{(j)}} & t_{{cw}}^{{(j)}} \\\\
0 & 1
\end{{bmatrix}}
\in SE(3),
$$

where each camera has its own superscripted rotation and translation terms:

$$
R_{{cw}}^{{(j)}} =
\begin{{bmatrix}}
r_{{11}}^{{(j)}} & r_{{12}}^{{(j)}} & r_{{13}}^{{(j)}} \\\\
r_{{21}}^{{(j)}} & r_{{22}}^{{(j)}} & r_{{23}}^{{(j)}} \\\\
r_{{31}}^{{(j)}} & r_{{32}}^{{(j)}} & r_{{33}}^{{(j)}}
\end{{bmatrix}},
\quad
t_{{cw}}^{{(j)}} =
\begin{{bmatrix}}
t_x^{{(j)}} & t_y^{{(j)}} & t_z^{{(j)}}
\end{{bmatrix}}^\top.
$$

The GIF also lists the camera center as `t_wc^(j)`, where

$$
t_{{cw}}^{{(j)}} = -R_{{cw}}^{{(j)}}t_{{wc}}^{{(j)}}.
$$

For the GIF caption, the vectorized display is therefore:

$$
\operatorname{{vec}}(T_{{cw}}^{{(j)}}) =
\begin{{bmatrix}}
r_{{11}}^{{(j)}} & r_{{12}}^{{(j)}} & \cdots & r_{{33}}^{{(j)}} &
t_x^{{(j)}} & t_y^{{(j)}} & t_z^{{(j)}}
\end{{bmatrix}}^\top.
$$

If we embed the same fixed camera extrinsic in `SE_2(3)`, the navigation
velocity slot is zero:

$$
X_{{cw}}^{{(j)}} =
\begin{{bmatrix}}
R_{{cw}}^{{(j)}} & 0 & t_{{cw}}^{{(j)}} \\\\
0 & 1 & 0 \\\\
0 & 0 & 1
\end{{bmatrix}}.
$$

Because the target state is currently `R^6`, the prediction model remains:

$$
p_{{k+1}} = p_k + \\Delta t\\,v_k + w_p,\\quad
v_{{k+1}} = v_k + w_v.
$$

This can be written as a translation-group update, but since the translation
group is Abelian it is numerically the same as addition. A true full-pose
filter would instead use, for example,

$$
T_{{k+1}} = T_k \\operatorname{{Exp}}(\\Delta t\\,\\xi_k).
$$

That extension is outside this position-only comparison.

## Filter Math

### Full Lie-Group Pose Filter Form, Not Used Here

If this were a pose-estimation problem, the filter state could live directly on
`SE(3)`:

$$
X_k =
\begin{{bmatrix}}
R_k & p_k \\\\
0 & 1
\end{{bmatrix}}\in SE(3),
\quad
X_{{k+1}} = X_k\operatorname{{Exp}}(\xi_k^\wedge\Delta t).
$$

A left-invariant error for a full-pose Lie-group construction
would be

$$
\eta_k = \hat X_k^{-1}X_k,\quad
\delta\xi_k = \operatorname{{Log}}(\eta_k)^\vee.
$$

After a measurement update,

$$
\delta\xi_k = K r_k,\quad
\hat X_k^+ = \hat X_k^-\operatorname{{Exp}}(\delta\xi_k^\wedge).
$$

For a pose-plus-velocity navigation state, we would more naturally use
`SE_2(3)`:

$$
\chi_k =
\begin{{bmatrix}}
R_k & v_k & p_k \\\\
0 & 1 & 0 \\\\
0 & 0 & 1
\end{{bmatrix}}\in SE_2(3),
\quad
\eta_k=\hat\chi_k^{-1}\chi_k,\quad
\delta\xi_k=\operatorname{{Log}}(\eta_k)^\vee.
$$

Those expressions are mathematically useful for the report because they show
what a true body-frame/invariant pose filter would look like. They are not the
implemented state model here: our drone orientation is not being estimated, so
the active state is only

$$
x_k =
\begin{{bmatrix}}p_{w,k} \\\\ v_{w,k}\end{{bmatrix}}\in R^6.
$$

The only Lie-group operation used in the implemented comparison is the camera
projection through `T_cw^(j) in SE(3)`.

### MEKF / Error-State EKF

Prediction:

$$
\\hat x_k^- = F\\hat x_{{k-1}}^+,\quad
P_k^- = F P_{{k-1}}^+ F^\\top + Q,
$$

where

$$
F =
\\begin{{bmatrix}}
I & \\Delta t I \\\\
0 & I
\\end{{bmatrix}}.
$$

At time `k`, all valid camera measurements are stacked into one vector:

$$
z_k =
\\operatorname{{stack}}_{{j\\in\\mathcal V_k}} z_{{j,k}},
\\qquad
h(\\hat x_k^-)=
\\operatorname{{stack}}_{{j\\in\\mathcal V_k}} h_j(\\hat x_k^-).
$$

Here `\\mathcal V_k` is the set of cameras that have valid measurements at time
`k`. The EKF measurement Jacobian is

$$
H_k =
\\left.\\frac{{\\partial h}}{{\\partial x}}\\right|_{{\\hat x_k^-}}.
$$

For a single camera `j`, the corresponding two-row block is

$$
H_{{j,k}} =
\\begin{{bmatrix}}
J_{{\\pi,j,k}}R_{{j,cw}} & 0_{{2\\times 3}}
\\end{{bmatrix}},
$$

where

$$
J_{{\\pi,j,k}} =
\\begin{{bmatrix}}
f_{{x,j}}/Z_{{j,c,k}} & 0 & -f_{{x,j}}X_{{j,c,k}}/Z_{{j,c,k}}^2 \\\\
0 & f_{{y,j}}/Z_{{j,c,k}} & -f_{{y,j}}Y_{{j,c,k}}/Z_{{j,c,k}}^2
\\end{{bmatrix}}.
$$

This `H_{{j,k}}` maps a small state perturbation
`delta x = [delta p_w, delta v_w]` into pixel motion. The right block is zero
because the camera measurement directly depends on position, not velocity; the
velocity still matters through prediction.

The MEKF measurement update is then:

$$
r_k = z_k - h(\\hat x_k^-),\\quad
S_k = H_k P_k^- H_k^\\top + R_k,
$$

$$
K_k = P_k^- H_k^\\top S_k^{{-1}},\\quad
\\delta x_k = K_k r_k,
$$

$$
\\hat x_k^+ = \\hat x_k^- \\oplus \\delta x_k.
$$

Here `oplus` is addition because the estimated state is `R^6`.

### Iterated EKF (ItEKF)

The ItEKF uses the same prediction as the MEKF. The difference is only in the
measurement correction.  Instead of linearizing the camera projection once at
the predicted state, it repeatedly relinearizes around the current corrected
guess.

Start each measurement update from the predicted state:

$$
\\delta x_{{k,0}} = 0,\\qquad
x_{{k,\\ell}} = \\hat x_k^- + \\delta x_{{k,\\ell}}.
$$

At iteration `\\ell`, evaluate the nonlinear projection and its Jacobian at
`x_{{k,\\ell}}`:

$$
\\hat z_{{k,\\ell}} = h(x_{{k,\\ell}}),\\qquad
H_{{k,\\ell}} =
\\left.\\frac{{\\partial h}}{{\\partial x}}\\right|_{{x_{{k,\\ell}}}}.
$$

The iterated EKF residual is written relative to the predicted state:

$$
r_{{k,\\ell}} =
z_k - \\hat z_{{k,\\ell}} + H_{{k,\\ell}}\\delta x_{{k,\\ell}}.
$$

Then the Kalman system is solved using the predicted covariance:

$$
S_{{k,\\ell}} = H_{{k,\\ell}} P_k^- H_{{k,\\ell}}^\\top + R_k,
\\qquad
K_{{k,\\ell}} = P_k^- H_{{k,\\ell}}^\\top S_{{k,\\ell}}^{{-1}},
$$

$$
\\delta x_{{k,\\ell+1}}=K_{{k,\\ell}} r_{{k,\\ell}}.
$$

After a small fixed number of iterations, or once
`\\|\\delta x_{k,\\ell+1}-\\delta x_{k,\\ell}\\|` is small, the update is
committed:

$$
\\hat x_k^+ = \\hat x_k^- + \\delta x_{{k,L}}.
$$

The covariance is updated with the final Jacobian and gain:

$$
P_k^+ =
(I-K_{{k,L}}H_{{k,L}})P_k^-(I-K_{{k,L}}H_{{k,L}})^\\top
+ K_{{k,L}} R_k K_{{k,L}}^\\top.
$$

This is an Iterated EKF, not an invariant EKF. Because the estimated state is
`R^6`, the retraction is ordinary addition. A true pose-state invariant filter
would instead update a group-valued state with an exponential-map correction.

### Gated MEKF / Gated ItEKF

The gated variants first test each camera's 2D innovation before including it in
the EKF update:

$$
d_{{j,k}}^2 =
r_{{j,k}}^\top
\left(H_{{j,k}} P_k^- H_{{j,k}}^\top + R_j\right)^{-1}
r_{{j,k}}.
$$

For camera `j`, the bbox center is accepted only if

$$
d_{{j,k}}^2 \le \gamma,
$$

where `gamma = 11.83`, approximately the 99.7 percent chi-square threshold for
a two-dimensional pixel measurement. This is not oracle cleaning: it uses only
the predicted measurement, covariance, and actual observed innovation.

For pure dropout, no residual exists for the missing camera. Therefore MEKF,
ItEKF, gated MEKF, and gated ItEKF all skip that camera update. Gating is kept
in the implementation and tables for completeness, but it should not be
presented as the mechanism that fixes missing camera feeds.

### Bootstrap Particle Filter

Particles are predicted by:

$$
x_k^{{(i)}} = f(x_{{k-1}}^{{(i)}}) + q_k^{{(i)}}.
$$

Each particle is projected into every camera:

$$
\\hat z_{{j,k}}^{{(i)}} =
h_j(x_k^{{(i)}})
= \\pi(K_j, T_{{j,cw}}\\bar p_{{w,k}}^{{(i)}}).
$$

Weights are assigned from pixel reprojection error. Each available camera
contributes a Gaussian likelihood for ordinary pixel noise, with a small
likelihood floor for numerical robustness:

$$
\\omega_k^{{(i)}} \\propto
\\exp\\left(
\\sum_j
\\log\\left[
\\exp\\left(-\\frac{{1}}{{2}}
\\left\\|\\frac{{z_{{j,k}}-\\hat z_{{j,k}}^{{(i)}}}}{{\\sigma_z}}\\right\\|^2
\\right)
+\\epsilon
\\right]
\\right).
$$

The state estimate is computed before resampling, and particles are
systematically resampled when the effective sample size is low.

## Code Map

- `lie_projection.py`: `SE3`, `SO(3)`/`SE(3)` helpers, four-camera rig, Lie-group projection, and analytic projection Jacobian.
- `filters.py`: ungated MEKF/ItEKF, gated MEKF/ItEKF, and robust PF implementations.
- `ambiguous_candidate_case.py`: candidate-set detector ambiguity case where the PF uses a mixture likelihood.
- `make_ros_camera_feed_cases.py`: report-ready ROS/Gazebo camera-feed GIF crops for nominal, high-noise, and dropout measurement streams.
- `run_comparison.py`: scenario definitions, Monte Carlo runner, plots, CSV table, and this report.

## Scenarios

{scenario_text}

## Dropout Measurement Handling

Dropout means the camera gives no usable bbox for that frame. In code, the
measurement mask for that camera is set false, so the filter simply skips that
camera update and relies on prediction plus any remaining cameras. Increasing
the dropout rate makes the problem less observable and increases drift, but it
does not give the filter a bad measurement to reject.

That is why gated and ungated EKFs are expected to be close in the dropout-only
case. Gating is a measurement-validation step; when a camera measurement is
missing, there is no innovation to validate. The PF handles dropout in the same
basic way: unavailable camera likelihoods are omitted, and the particle set is
propagated using the process model plus the cameras that still see the target.

The separate ambiguous candidate-set experiment is not called dropout. It is a
multi-hypothesis measurement update: the filter receives multiple plausible
bbox candidates, and the PF can keep multiple hypotheses alive through a
mixture likelihood instead of committing to one association immediately.

Simulation configuration:

```json
{json.dumps(config, indent=2)}
```

## Filter Initialization In Software Simulation

The main Monte Carlo software simulation initializes all filters from the same
controlled prior, not from camera triangulation.  At the first time step, the
truth state is

$$
x_0 = [p_0, v_0].
$$

The simulation then applies a prescribed initial bias and covariance:

$$
\hat x_0 = x_0 + b_0,\qquad P_0=\operatorname{diag}(\sigma_{p,x}^2,\ldots,\sigma_{v,z}^2).
$$

The EKF-style filters receive `(\hat x_0, P_0)`.  The PF receives the same
prior by sampling particles:

$$
x_0^{(n)} \sim \mathcal N(\hat x_0,P_0),\qquad w_0^{(n)} = 1/N.
$$

So in the software Monte Carlo comparison, the initial velocity is part of the
same controlled prior: the truth velocity is computed analytically from the
figure-eight trajectory, then the configured velocity bias/noise is applied.
This is done for fairness, so MEKF, ItEKF, gated EKFs, and PF all start with
the same amount of initialization error.

The hardware-style ablation later in this report uses a different bootstrap:
early camera bbox centers are triangulated into rough 3D anchors, initial
position is the median of those anchors, and initial velocity is estimated by
finite difference between early anchors.  That version deliberately tests how a
bad camera model can corrupt the PF before the first update.

## Results

{chr(10).join(table_lines)}

## Computation Time

The table above includes the per-scenario mean update time. Averaged across
scenarios, the per-step computation time is:

{chr(10).join(runtime_lines)}

{particle_sweep_section}

{non_outlier_sweep_section}

{ambiguous_candidate_section}

## Figures

![Top-view trajectory]({fig_dir}/trajectory_top_view.png)

![Mean time error]({fig_dir}/time_error_comparison.png)

![RMSE bar chart]({fig_dir}/rmse_bar.png)

![Runtime bar chart]({fig_dir}/runtime_bar.png)

## Animated Simulation GIFs

The live ROS/Gazebo composite below is generated from the Purdue world camera
feeds using `ros_world_gif_recorder.py`. It is a qualitative visual demo, while
the numerical comparison above comes from the controlled Monte Carlo pipeline.
The recorder uses Gazebo `/clock`, and the top-left panel reports the XY loop
closure after one 12-second simulated figure-eight period:

![Live ROS/Gazebo Purdue-world composite]({fig_dir}/ros_purdue_filter_comparison.gif)

The clean zoomed top view below uses the same camera rig, but crops around the
target motion so the complete figure-eight is obvious. This is the clearest
animation to use when explaining the simulation trajectory:

![Zoomed figure-eight top view]({fig_dir}/animated_zoomed_figure_eight.gif)

The cropped ROS/Gazebo camera-feed views below keep the buildings visible while
showing the true bbox and measured bbox directly in the four camera streams.
The bottom subtitles list the case name, pixel measurement noise `sigma_z`,
`K^{(i)}`, `R_{cw}^{(i)}`, and `t_{cw}^{(i)}`.

Nominal measurements use `sigma_z = 8 px` and no artificial dropout:

![ROS/Gazebo camera-feed nominal view]({fig_dir}/ros_camera_feeds_nominal.gif)

High-noise measurements use `sigma_z = 22 px` and no artificial dropout:

![ROS/Gazebo camera-feed high-noise view]({fig_dir}/ros_camera_feeds_high_noise.gif)

The dropout visual uses `sigma_z = 12 px` and dropout probability `0.18`:

![ROS/Gazebo camera-feed dropout view]({fig_dir}/ros_camera_feeds_dropout.gif)

## Hardware Story Visuals

The hardware-calibration hypothesis visual below combines a short real ROS-bag
camera-feed excerpt from `/UP1`, `/UP2`, and `/UP3`, the actual extracted PURT
QTM path replayed in simulation, and the calibration-ablation PF RMSE bars:

<img src="{fig_dir}/hardware_hypothesis_story.gif" alt="Hardware hypothesis story GIF" width="100%">

The static version is included for PDF/print contexts where GIF playback is not
available:

<img src="{fig_dir}/hardware_hypothesis_pipeline.png" alt="Hardware hypothesis pipeline" width="100%">

Direct files: [`hardware_hypothesis_story.gif`]({fig_dir}/hardware_hypothesis_story.gif)
and [`hardware_hypothesis_pipeline.png`]({fig_dir}/hardware_hypothesis_pipeline.png).

## Analytic Projection Jacobian

The experiment uses analytic NumPy Jacobians. The projection Jacobian is short,
explicit, and cheap to evaluate:

$$
\frac{\partial u}{\partial p_w}
=
\begin{bmatrix}
f_x/Z_c & 0 & -f_xX_c/Z_c^2
\end{bmatrix}R_{cw},
$$

$$
\frac{\partial v}{\partial p_w}
=
\begin{bmatrix}
0 & f_y/Z_c & -f_yY_c/Z_c^2
\end{bmatrix}R_{cw}.
$$

For this Monte Carlo particle-filter comparison, vectorized NumPy is the more
direct runtime choice.

## Fairness Note On Dropout

The dropout scenario includes both ungated and gated EKF variants. This is kept
for transparency, but the interpretation is simple: if a camera feed is missing,
both variants skip that camera update. Therefore the gated rows should not be
sold as a dropout fix; they are mainly a consistency check that the gating logic
does not change behavior when there is no measurement to validate.

The live ROS/Gazebo GIF should not be used as the metric source. It is a visual
storyboard with synthetic bboxes drawn onto rendered camera feeds. The metric
source is the Monte Carlo table, because it controls the measurement noise,
dropout, seeds, and ground-truth alignment exactly.

## Interpretation

The comparison should be read with one important caveat: since the current
state is position and velocity, the ItEKF and MEKF differ only in the nonlinear
measurement update. The ItEKF repeatedly relinearizes the projection model,
while the MEKF linearizes once per update. The PF differs more strongly because
it maintains a sampled distribution and uses robust likelihood weighting plus
resampling.

Particle methods usually tolerate poor initialization and multimodal candidate
sets better because they maintain a distribution instead of a single Gaussian
estimate. In pure dropout, the PF advantage is less dramatic because every
filter simply uses fewer cameras at that update. The EKF-style filters are
typically faster per step because they propagate only a mean and covariance.

The absolute PF error in this report is higher than the earlier timestamp-fixed
ROS PF result because the assumptions are different. Here, the nominal pixel
noise is `8 px`, and the dropout scenario injects missing camera detections. A
quick sanity sweep with the same corrected camera geometry
gives PF errors near `10 cm` for `4 px` image noise and near `6 cm` for `2 px`
image noise, which is consistent with the earlier `10 cm` and `4.8 cm` range.
So the difference is mainly measurement-noise/stress-test configuration, not a
new timestamp regression.

## Future Work

The current implementation establishes the Lie-group projection model and a
fair comparison among MEKF, iterated EKF, gated EKFs, and PF. The next steps are:

- **Implement FPF on Lie groups**. This should be a future estimator, not a
  current result. The natural direction is to implement a Feedback Particle
  Filter whose innovation feedback is defined on the appropriate Lie group,
  then compare constant-gain, kernel, and Galerkin/Poisson-equation gain
  approximations against the current bootstrap PF.
- **Implement an equivariant filter**. If the project expands from position-only
  tracking to pose or pose-plus-velocity estimation on `SE(3)` or `SE_2(3)`,
  an equivariant filter would preserve the symmetry of the dynamics and
  measurement model more faithfully than the current Euclidean error-state
  approximation.
- **Accelerate high-particle PF with JAX/CasADi-style tooling**. The projection
  and likelihood evaluation are embarrassingly parallel across particles and
  cameras. JAX is the most promising path for `vmap`/`jit` and GPU execution
  when using thousands of particles. CasADi can still be useful for automatic
  differentiation and code generation of projection/Jacobian pieces, although
  the immediate speed win for PF is likely vectorized GPU evaluation.
- **Study genuinely multimodal and multitarget scenes**. The PF is most
  valuable when the posterior is non-Gaussian: multiple similar drones,
  ambiguous labels, or crossing trajectories. Future
  work should extend the candidate-set experiment into multi-target tracking
  with explicit data association, label uncertainty, and possibly JPDA/MHT-style
  baselines.
- **Improve hardware data capture and storage**. The current hardware workflow
  records raw ROS image bags, which are heavy in memory and disk bandwidth.
  Future runs should use encoded video transport or compressed image topics,
  store the exact camera YAMLs and calibrated `camera_poses.json` with the bag,
  and support streaming evaluation instead of loading large raw image logs.
- **Close the hardware calibration loop**. A successful hardware run needs
  reliable `T_marker,optical` hand-eye calibration, a single global QTM/video
  time offset, and a pre-filter reprojection check. The practical rule should
  be: if QTM drone positions do not project near the detected drone in each
  camera, do not run PF yet.

## Conclusion

The Lie-group formulation is valuable and mathematically appropriate in the
measurement model:

$$
z_{{j,k}} = \\pi(K_j, T_{{j,cw}}\\bar p_{{w,k}}) + n_{{j,k}}.
$$

It makes the camera/world frame convention explicit. It does not by itself make
the current position-only PF faster. Speed comes primarily from vectorizing
projection and likelihood calculations. If the project later estimates the full
drone pose, then the state should move to `SE(3)` or `SE_2(3)`, and the
Lie-group formulation will become central to the prediction and correction
steps as well.
"""
    report = report.replace("{{", "{").replace("}}", "}")
    report = report.replace("\\\\", "\\")
    report = report.replace("{scenario_text}", scenario_text)
    report = report.replace("{json.dumps(config, indent=2)}", json.dumps(config, indent=2))
    report = report.replace("{chr(10).join(table_lines)}", "\n".join(table_lines))
    report = report.replace("{chr(10).join(runtime_lines)}", "\n".join(runtime_lines))
    report = report.replace("{chr(10).join(camera_lines)}", "\n".join(camera_lines))
    report = report.replace("{particle_sweep_section}", particle_sweep_section)
    report = report.replace("{non_outlier_sweep_section}", non_outlier_sweep_section)
    report = report.replace("{ambiguous_candidate_section}", ambiguous_candidate_section)
    report = report.replace("{fig_dir}", fig_dir)
    path.write_text(report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=12)
    parser.add_argument("--particles", type=int, default=450)
    parser.add_argument("--duration", type=float, default=FIGURE_EIGHT_PERIOD_S)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=590)
    parser.add_argument("--out", type=Path, default=Path("lg_filter_comparison/results"))
    args = parser.parse_args()

    scenarios = [
        Scenario(
            "nominal",
            pixel_sigma=8.0,
            init_bias=np.array([0.7, -0.5, 0.25]),
            init_vel_bias=np.array([-0.2, 0.2, 0.0]),
            init_pos_std=0.8,
            init_vel_std=0.45,
        ),
        Scenario(
            "poor_init",
            pixel_sigma=8.0,
            init_bias=np.array([2.5, -2.0, 0.9]),
            init_vel_bias=np.array([-0.6, 0.4, 0.0]),
            init_pos_std=1.7,
            init_vel_std=0.8,
        ),
        Scenario(
            "high_noise",
            pixel_sigma=22.0,
            init_bias=np.array([0.9, -0.7, 0.25]),
            init_vel_bias=np.array([-0.2, 0.2, 0.0]),
            init_pos_std=1.0,
            init_vel_std=0.55,
        ),
        Scenario(
            "dropout",
            pixel_sigma=12.0,
            init_bias=np.array([1.0, -0.8, 0.35]),
            init_vel_bias=np.array([-0.3, 0.2, 0.0]),
            init_pos_std=1.2,
            init_vel_std=0.6,
            dropout_prob=0.18,
        ),
    ]

    t = np.arange(0.0, args.duration + 1e-12, args.dt)
    pos, vel = trajectory(t)
    true_state = np.column_stack([pos, vel])
    all_results = {}
    for scenario in scenarios:
        runs = []
        print(f"scenario={scenario.name}")
        for k in range(args.runs):
            print(f"  run {k + 1}/{args.runs}")
            runs.append(run_one(scenario, args.seed + 100 * k, t, true_state, args.particles))
        all_results[scenario.name] = runs

    rows = summarize(all_results)
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.out / "summary.csv")
    plot_results(all_results, rows, t, args.out / "figures")
    config = {
        "runs": args.runs,
        "particles": args.particles,
        "duration_s": args.duration,
        "dt_s": args.dt,
        "seed": args.seed,
        "filters": [cls.name for cls in FILTER_CLASSES],
    }
    (args.out / "config.json").write_text(json.dumps(config, indent=2))
    write_report(args.out / "report.md", rows, scenarios, config)
    print(f"wrote {args.out / 'summary.csv'}")
    print(f"wrote {args.out / 'report.md'}")


if __name__ == "__main__":
    main()
