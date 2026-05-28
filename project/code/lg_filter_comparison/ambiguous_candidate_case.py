#!/usr/bin/env python3
"""Ambiguous detector-candidate experiment where PF is the natural estimator.

The regular comparison gives every filter one bbox center per visible camera.
This script simulates a detector that returns a candidate set per camera:
one true-looking detection plus an independent false candidate.  The EKFs are
fed the top-scored single association, while the PF consumes the whole candidate
set with a mixture likelihood.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from filters import (
    ErrorStateEKF,
    GatedErrorStateEKF,
    GatedIteratedEKF,
    IteratedEKF,
    normalize_weights,
    systematic_resample,
)
from lie_projection import make_square_camera_rig, project_points_lie, project_state
from run_comparison import FIGURE_EIGHT_PERIOD_S, Scenario, initial_conditions, trajectory


FILTER_CLASSES = [ErrorStateEKF, IteratedEKF, GatedErrorStateEKF, GatedIteratedEKF]


@dataclass(frozen=True)
class AmbiguousCandidateConfig:
    pixel_sigma: float = 8.0
    clutter_prob: float = 0.80
    false_preferred_prob: float = 0.70
    dropout_prob: float = 0.03
    clutter_offset_min_px: float = 25.0
    clutter_offset_max_px: float = 75.0


class CandidateSetPF:
    """Bootstrap PF with a per-camera mixture likelihood over bbox candidates."""

    name = "PF-candidates"

    def __init__(self, cameras, dt: float, pixel_sigma: float, rng: np.random.Generator, n_particles: int):
        self.cameras = cameras
        self.dt = float(dt)
        self.pixel_sigma = float(pixel_sigma)
        self.rng = rng
        self.n_particles = int(n_particles)

    def initialize(self, mean: np.ndarray, cov: np.ndarray):
        self.particles = self.rng.multivariate_normal(mean, cov, size=self.n_particles)
        self.weights = np.ones(self.n_particles) / self.n_particles
        self.process_std = np.array([0.025, 0.025, 0.015, 0.20, 0.20, 0.10])
        self.estimate = np.average(self.particles, axis=0, weights=self.weights)

    def predict(self):
        self.particles[:, :3] += self.dt * self.particles[:, 3:6]
        self.particles += self.rng.normal(0.0, self.process_std, size=self.particles.shape)

    def update(self, candidates_by_camera: list[list[np.ndarray]]):
        z, valid = project_points_lie(self.cameras, self.particles[:, :3])
        pred = z.transpose(1, 0, 2)
        sigma = max(self.pixel_sigma, 1e-12)
        outlier_floor = 1e-4
        loss_cap = 40.0
        logw = np.zeros(self.n_particles, dtype=float)
        counts = np.zeros(self.n_particles, dtype=int)
        used_any_camera = False

        for cam_idx, candidates in enumerate(candidates_by_camera):
            if not candidates:
                continue
            used_any_camera = True
            cam_valid = valid[cam_idx] & np.all(np.isfinite(pred[:, cam_idx, :]), axis=1)
            like = np.full(self.n_particles, outlier_floor, dtype=float)
            for candidate in candidates:
                diff = (pred[:, cam_idx, :] - candidate.reshape(1, 2)) / sigma
                d2 = np.sum(diff * diff, axis=1)
                like[cam_valid] += np.exp(-0.5 * np.minimum(d2[cam_valid], loss_cap)) / len(candidates)
            logw += np.log(like)
            counts += cam_valid.astype(int)

        if not used_any_camera:
            return
        logw[counts < 1] = -1e12
        logw -= np.max(logw)
        self.weights = normalize_weights(np.exp(logw))
        self.estimate = np.average(self.particles, axis=0, weights=self.weights)
        ess = 1.0 / np.sum(self.weights * self.weights)
        if ess < 0.55 * self.n_particles:
            idx = systematic_resample(self.weights, self.rng)
            self.particles = self.particles[idx]
            self.weights = np.ones(self.n_particles) / self.n_particles

    def step(self, candidates_by_camera: list[list[np.ndarray]]):
        tic = perf_counter()
        self.predict()
        self.update(candidates_by_camera)
        return self.estimate.copy(), perf_counter() - tic


def ambiguous_scenario(config: AmbiguousCandidateConfig) -> Scenario:
    return Scenario(
        "ambiguous_candidates",
        pixel_sigma=config.pixel_sigma,
        init_bias=np.array([0.7, -0.5, 0.25]),
        init_vel_bias=np.array([-0.2, 0.2, 0.0]),
        init_pos_std=0.8,
        init_vel_std=0.45,
    )


def make_ambiguous_measurement(cameras, p_w: np.ndarray, scenario: Scenario, config: AmbiguousCandidateConfig, rng):
    y_true, mask_true = project_state(cameras, np.hstack([p_w, np.zeros(3)]))
    y_single = np.full_like(y_true, np.nan, dtype=float)
    mask_single = np.zeros_like(mask_true, dtype=bool)
    candidates_by_camera: list[list[np.ndarray]] = []

    for cam_idx, cam in enumerate(cameras):
        rows = np.array([2 * cam_idx, 2 * cam_idx + 1])
        candidates: list[np.ndarray] = []
        scored_candidates: list[tuple[float, np.ndarray]] = []
        if np.all(mask_true[rows]) and rng.uniform() > config.dropout_prob:
            true_candidate = y_true[rows] + rng.normal(0.0, scenario.pixel_sigma, size=2)
            true_candidate[0] = np.clip(true_candidate[0], 0.0, cam.width - 1.0)
            true_candidate[1] = np.clip(true_candidate[1], 0.0, cam.height - 1.0)
            true_score = rng.normal(0.75, 0.06)
            candidates.append(true_candidate)
            scored_candidates.append((true_score, true_candidate))

            if rng.uniform() < config.clutter_prob:
                angle = rng.uniform(0.0, 2.0 * math.pi)
                radius = rng.uniform(config.clutter_offset_min_px, config.clutter_offset_max_px)
                false_candidate = true_candidate + radius * np.array([math.cos(angle), math.sin(angle)])
                false_candidate += rng.normal(0.0, 5.0, size=2)
                false_candidate[0] = np.clip(false_candidate[0], 0.0, cam.width - 1.0)
                false_candidate[1] = np.clip(false_candidate[1], 0.0, cam.height - 1.0)
                if rng.uniform() < config.false_preferred_prob:
                    false_score = true_score + rng.uniform(0.02, 0.18)
                else:
                    false_score = true_score - rng.uniform(0.02, 0.18)
                candidates.append(false_candidate)
                scored_candidates.append((false_score, false_candidate))

            best_candidate = max(scored_candidates, key=lambda item: item[0])[1]
            y_single[rows] = best_candidate
            mask_single[rows] = True
        candidates_by_camera.append(candidates)

    return y_single, mask_single, candidates_by_camera


def run_one(seed: int, particles: int, config: AmbiguousCandidateConfig, t: np.ndarray, true_state: np.ndarray):
    cameras = make_square_camera_rig()
    scenario = ambiguous_scenario(config)
    rng_meas = np.random.default_rng(seed + 10_000)
    measurements = [
        make_ambiguous_measurement(cameras, p, scenario, config, rng_meas)
        for p in true_state[:, :3]
    ]
    mean0, cov0 = initial_conditions(true_state[0], scenario)
    results = {}

    for idx, cls in enumerate(FILTER_CLASSES):
        rng = np.random.default_rng(seed + 1_000 * (idx + 1))
        filt = cls(cameras, t[1] - t[0], scenario.pixel_sigma, rng)
        filt.initialize(mean0, cov0)
        estimates = []
        runtimes = []
        for y_single, mask_single, _ in measurements:
            est, dt_run = filt.step(y_single, mask_single)
            estimates.append(est)
            runtimes.append(dt_run)
        estimates = np.asarray(estimates)
        errors = np.linalg.norm(estimates[:, :3] - true_state[:, :3], axis=1)
        results[filt.name] = {
            "estimates": estimates,
            "errors": errors,
            "runtimes": np.asarray(runtimes),
        }

    rng = np.random.default_rng(seed + 5_000)
    pf = CandidateSetPF(cameras, t[1] - t[0], scenario.pixel_sigma, rng, n_particles=particles)
    pf.initialize(mean0, cov0)
    estimates = []
    runtimes = []
    for _, _, candidates_by_camera in measurements:
        est, dt_run = pf.step(candidates_by_camera)
        estimates.append(est)
        runtimes.append(dt_run)
    estimates = np.asarray(estimates)
    errors = np.linalg.norm(estimates[:, :3] - true_state[:, :3], axis=1)
    results[pf.name] = {
        "estimates": estimates,
        "errors": errors,
        "runtimes": np.asarray(runtimes),
    }
    return results


def summarize(runs: list[dict]):
    names = [cls.name for cls in FILTER_CLASSES] + [CandidateSetPF.name]
    rows = []
    for name in names:
        rmse = np.array([math.sqrt(float(np.mean(run[name]["errors"] ** 2))) for run in runs])
        mean_err = np.array([float(np.mean(run[name]["errors"])) for run in runs])
        p90 = np.array([float(np.percentile(run[name]["errors"], 90)) for run in runs])
        final = np.array([float(run[name]["errors"][-1]) for run in runs])
        runtime_ms = np.array([float(np.mean(run[name]["runtimes"]) * 1e3) for run in runs])
        rows.append(
            {
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


def plot_results(runs: list[dict], rows: list[dict], t: np.ndarray, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        "MEKF": "#222222",
        "ItEKF": "#2ca02c",
        "MEKF-gated": "#9467bd",
        "ItEKF-gated": "#ff7f0e",
        "PF-candidates": "#1f77b4",
    }
    plt.figure(figsize=(8, 4.8))
    for name in colors:
        err = np.array([run[name]["errors"] for run in runs])
        plt.plot(t, np.mean(err, axis=0), label=name, color=colors[name])
    plt.grid(True, alpha=0.3)
    plt.xlabel("time [s]")
    plt.ylabel("position error [m]")
    plt.title("Ambiguous candidate-set case: mean position error")
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "ambiguous_candidate_error.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7.2, 4.2))
    names = [row["filter"] for row in rows]
    vals = [row["rmse_mean_m"] for row in rows]
    stds = [row["rmse_std_m"] for row in rows]
    plt.bar(names, vals, yerr=stds, capsize=3, color=[colors[name] for name in names])
    plt.xticks(rotation=15, ha="right")
    plt.ylabel("RMSE [m]")
    plt.title("Ambiguous candidate-set RMSE")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "ambiguous_candidate_rmse.png", dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=12)
    parser.add_argument("--particles", type=int, default=450)
    parser.add_argument("--duration", type=float, default=FIGURE_EIGHT_PERIOD_S)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=590)
    parser.add_argument("--out", type=Path, default=Path("lg_filter_comparison/results"))
    args = parser.parse_args()

    config = AmbiguousCandidateConfig()
    t = np.arange(0.0, args.duration + 1e-12, args.dt)
    pos, vel = trajectory(t)
    true_state = np.column_stack([pos, vel])

    runs = []
    for k in range(args.runs):
        print(f"ambiguous candidate run {k + 1}/{args.runs}")
        runs.append(run_one(args.seed + 100 * k, args.particles, config, t, true_state))
    rows = summarize(runs)
    write_csv(rows, args.out / "ambiguous_candidate_summary.csv")
    plot_results(runs, rows, t, args.out / "figures")
    print(f"wrote {args.out / 'ambiguous_candidate_summary.csv'}")
    print(f"wrote {args.out / 'figures' / 'ambiguous_candidate_error.png'}")
    print(f"wrote {args.out / 'figures' / 'ambiguous_candidate_rmse.png'}")


if __name__ == "__main__":
    main()
