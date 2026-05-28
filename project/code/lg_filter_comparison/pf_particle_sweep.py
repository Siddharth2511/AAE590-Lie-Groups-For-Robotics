#!/usr/bin/env python3
"""Run a focused PF particle-count sensitivity study."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from filters import BootstrapPF
from lie_projection import make_square_camera_rig
from run_comparison import FIGURE_EIGHT_PERIOD_S, Scenario, initial_conditions, make_measurement, trajectory


def dropout_outlier_scenario() -> Scenario:
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


def run_pf_once(
    scenario: Scenario,
    particles: int,
    seed: int,
    t: np.ndarray,
    true_state: np.ndarray,
):
    cameras = make_square_camera_rig()
    rng_meas = np.random.default_rng(seed + 10_000)
    measurements = [make_measurement(cameras, p, scenario, rng_meas) for p in true_state[:, :3]]
    mean0, cov0 = initial_conditions(true_state[0], scenario)
    rng = np.random.default_rng(seed + 5_000)
    filt = BootstrapPF(cameras, t[1] - t[0], scenario.pixel_sigma, rng, n_particles=particles)
    filt.initialize(mean0, cov0)

    estimates = []
    runtimes = []
    for y, mask in measurements:
        est, dt_run = filt.step(y, mask)
        estimates.append(est)
        runtimes.append(dt_run)

    estimates = np.asarray(estimates)
    errors = np.linalg.norm(estimates[:, :3] - true_state[:, :3], axis=1)
    return errors, np.asarray(runtimes)


def summarize_particle_count(
    scenario: Scenario,
    particles: int,
    runs: int,
    seed: int,
    t: np.ndarray,
    true_state: np.ndarray,
):
    rmse = []
    mean_err = []
    p90 = []
    final = []
    runtime_ms = []
    for k in range(runs):
        errors, runtimes = run_pf_once(scenario, particles, seed + 100 * k, t, true_state)
        rmse.append(math.sqrt(float(np.mean(errors * errors))))
        mean_err.append(float(np.mean(errors)))
        p90.append(float(np.percentile(errors, 90)))
        final.append(float(errors[-1]))
        runtime_ms.append(float(np.mean(runtimes) * 1e3))
    return {
        "particles": particles,
        "runs": runs,
        "rmse_mean_m": float(np.mean(rmse)),
        "rmse_std_m": float(np.std(rmse)),
        "mean_error_m": float(np.mean(mean_err)),
        "p90_error_m": float(np.mean(p90)),
        "final_error_m": float(np.mean(final)),
        "runtime_mean_ms": float(np.mean(runtime_ms)),
        "runtime_std_ms": float(np.std(runtime_ms)),
    }


def write_csv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_sweep(rows, out_path: Path):
    particles = np.array([r["particles"] for r in rows], dtype=float)
    rmse = np.array([r["rmse_mean_m"] for r in rows], dtype=float)
    rmse_std = np.array([r["rmse_std_m"] for r in rows], dtype=float)
    runtime = np.array([r["runtime_mean_ms"] for r in rows], dtype=float)
    runtime_std = np.array([r["runtime_std_ms"] for r in rows], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    fig.patch.set_facecolor("#f8f8f5")
    axes[0].errorbar(particles, rmse, yerr=rmse_std, marker="o", linewidth=2.0, capsize=3)
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("particles")
    axes[0].set_ylabel("PF RMSE [m]")
    axes[0].set_title("Accuracy")
    axes[0].grid(True, alpha=0.3)

    axes[1].errorbar(particles, runtime, yerr=runtime_std, marker="o", color="#d62728", linewidth=2.0, capsize=3)
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("particles")
    axes[1].set_ylabel("step time [ms]")
    axes[1].set_title("Runtime")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("PF particle-count sensitivity, dropout/outlier scenario")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=12)
    parser.add_argument("--particles", nargs="+", type=int, default=[225, 450, 900, 1800])
    parser.add_argument("--duration", type=float, default=FIGURE_EIGHT_PERIOD_S)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=590)
    parser.add_argument("--out", type=Path, default=Path("lg_filter_comparison/results"))
    args = parser.parse_args()

    scenario = dropout_outlier_scenario()
    t = np.arange(0.0, args.duration + 1e-12, args.dt)
    pos, vel = trajectory(t)
    true_state = np.column_stack([pos, vel])

    rows = []
    for particles in args.particles:
        print(f"particles={particles}")
        rows.append(summarize_particle_count(scenario, particles, args.runs, args.seed, t, true_state))

    write_csv(rows, args.out / "pf_particle_sweep.csv")
    plot_sweep(rows, args.out / "figures" / "pf_particle_sweep.png")
    print(f"wrote {args.out / 'pf_particle_sweep.csv'}")
    print(f"wrote {args.out / 'figures' / 'pf_particle_sweep.png'}")


if __name__ == "__main__":
    main()
