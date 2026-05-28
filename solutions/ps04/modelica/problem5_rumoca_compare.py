from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    import rumoca
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "The `rumoca` package is not installed. Install it first with "
        "`pip install rumoca`."
    ) from exc


MODEL_PATH = Path(__file__).with_name("Unicycle.mo")
MODEL_NAME = "Unicycle"
T_END = 35.0
DT = 0.1
V_NOM = 2.0
OMEGA_TURN = 0.4
T_LOBE = 2.0 * np.pi / OMEGA_TURN
T_FIG8 = 2.0 * T_LOBE


def simulate_rumoca() -> dict[str, np.ndarray]:
    result = json.loads(
        rumoca.simulate_file(
            str(MODEL_PATH),
            model_name=MODEL_NAME,
            t_end=T_END,
            dt=DT,
        )
    )
    names = ["time", *result["payload"]["names"]]
    arrays = [np.array(series, dtype=float) for series in result["payload"]["allData"]]
    return dict(zip(names, arrays, strict=True))


def omega_ref_continuous(t: float) -> float:
    if t < T_LOBE:
        return OMEGA_TURN
    if t < T_FIG8:
        return -OMEGA_TURN
    return OMEGA_TURN


def omega_ref_problem1_grid(t: float) -> float:
    tau = np.mod(t, T_FIG8)
    return OMEGA_TURN if tau < T_LOBE else -OMEGA_TURN


def arc_step(x: float, y: float, theta: float, v: float, omega: float, dt: float) -> tuple[float, float, float]:
    if abs(omega) < 1e-12:
        return x + v * dt * np.cos(theta), y + v * dt * np.sin(theta), theta
    return (
        x + (v / omega) * (np.sin(theta + omega * dt) - np.sin(theta)),
        y - (v / omega) * (np.cos(theta + omega * dt) - np.cos(theta)),
        theta + omega * dt,
    )


def continuous_reference(times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = []
    ys = []
    thetas = []
    for t in times:
        x = 0.0
        y = 0.0
        theta = 0.0
        remaining = float(t)

        seg = min(max(remaining, 0.0), T_LOBE)
        x, y, theta = arc_step(x, y, theta, V_NOM, OMEGA_TURN, seg)
        remaining -= seg

        if remaining > 0.0:
            seg = min(remaining, T_FIG8 - T_LOBE)
            x, y, theta = arc_step(x, y, theta, V_NOM, -OMEGA_TURN, seg)
            remaining -= seg

        if remaining > 0.0:
            x, y, theta = arc_step(x, y, theta, V_NOM, OMEGA_TURN, remaining)

        xs.append(x)
        ys.append(y)
        thetas.append(theta)

    return np.array(xs), np.array(ys), np.array(thetas)


def problem1_grid_reference(times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = 0.0
    y = 0.0
    theta = 0.0
    xs = [x]
    ys = [y]
    thetas = [theta]

    for t in times[:-1]:
        x, y, theta = arc_step(x, y, theta, V_NOM, omega_ref_problem1_grid(float(t)), DT)
        xs.append(x)
        ys.append(y)
        thetas.append(theta)

    return np.array(xs), np.array(ys), np.array(thetas)


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))


def main() -> None:
    sim = simulate_rumoca()
    t = sim["time"]
    x_m = sim["x"]
    y_m = sim["y"]
    theta_m = sim["theta"]

    x_c, y_c, theta_c = continuous_reference(t)
    x_p, y_p, theta_p = problem1_grid_reference(t)

    err_cont = np.sqrt((x_m - x_c) ** 2 + (y_m - y_c) ** 2)
    err_grid = np.sqrt((x_m - x_p) ** 2 + (y_m - y_p) ** 2)
    theta_err_cont = np.abs(wrap_angle(theta_m - theta_c))
    theta_err_grid = np.abs(wrap_angle(theta_m - theta_p))

    print(f"Rumoca vs continuous reference max position error: {np.max(err_cont):.6e} m")
    print(f"Rumoca vs continuous reference max heading error:  {np.max(theta_err_cont):.6e} rad")
    print(f"Rumoca vs Problem 1 grid reference max position error: {np.max(err_grid):.6e} m")
    print(f"Rumoca vs Problem 1 grid reference max heading error:  {np.max(theta_err_grid):.6e} rad")
    print(
        "Note: the larger Problem 1 mismatch is expected if your Python reference "
        "switches turn direction only on the dt = 0.1 grid instead of at the exact "
        f"event times t = {T_LOBE:.6f} s and t = {T_FIG8:.6f} s."
    )

    plt.figure(figsize=(8, 7))
    plt.plot(x_m, y_m, label="Rumoca / Modelica", linewidth=2.5)
    plt.plot(x_c, y_c, "--", label="Continuous exact reference", linewidth=2.0)
    plt.plot(x_p, y_p, ":", label="Problem 1 grid reference", linewidth=2.0)

    sample_idx = np.linspace(0, len(t) - 1, 10, dtype=int)
    heading_len = 0.7
    plt.quiver(
        x_m[sample_idx],
        y_m[sample_idx],
        heading_len * np.cos(theta_m[sample_idx]),
        heading_len * np.sin(theta_m[sample_idx]),
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.004,
        color="black",
        alpha=0.8,
    )

    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Problem 5: Unicycle Figure-Eight with Rumoca")
    plt.legend()
    plt.tight_layout()
    plt.savefig("problem5_rumoca_overlay.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
