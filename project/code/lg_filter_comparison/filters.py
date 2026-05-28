"""Filter variants for the Lie-group projection comparison experiment."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from lie_projection import Camera, project_points_lie, project_state, projection_jacobian


def normalize_weights(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    w = np.clip(w, 0.0, None)
    s = float(np.sum(w))
    if not np.isfinite(s) or s <= 0.0:
        return np.ones_like(w) / len(w)
    return w / s


def systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(weights)
    positions = (np.arange(n) + rng.uniform()) / n
    indices = np.zeros(n, dtype=int)
    cumsum = np.cumsum(weights)
    i = 0
    j = 0
    while i < n:
        if positions[i] < cumsum[j]:
            indices[i] = j
            i += 1
        else:
            j += 1
    return indices


def safe_solve(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A) @ B


def robust_triangulate(cameras: list[Camera], y: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    """Robustly triangulate one 3D point from the available camera pixels."""
    rays = []
    for cam_idx, cam in enumerate(cameras):
        u_idx = 2 * cam_idx
        v_idx = u_idx + 1
        if not (mask[u_idx] and mask[v_idx]):
            continue
        u = float(y[u_idx])
        v = float(y[v_idx])
        if not (np.isfinite(u) and np.isfinite(v)):
            continue
        ray_cam = np.linalg.inv(cam.K) @ np.array([u, v, 1.0], dtype=float)
        ray_cam = ray_cam / np.linalg.norm(ray_cam)
        ray_world = cam.T_cw.R.T @ ray_cam
        ray_world = ray_world / np.linalg.norm(ray_world)
        cam_center_world = -cam.T_cw.R.T @ cam.T_cw.t
        rays.append((cam_center_world, ray_world))

    if len(rays) < 2:
        return None

    midpoints = []
    for i in range(len(rays)):
        oi, ui = rays[i]
        for j in range(i + 1, len(rays)):
            oj, uj = rays[j]
            w0 = oi - oj
            a = float(ui @ ui)
            b = float(ui @ uj)
            c = float(uj @ uj)
            d = float(ui @ w0)
            e = float(uj @ w0)
            den = a * c - b * b
            if abs(den) < 1e-10:
                continue
            si = (b * e - c * d) / den
            sj = (a * e - b * d) / den
            midpoints.append(0.5 * (oi + si * ui + oj + sj * uj))

    if not midpoints:
        return None
    anchor = np.median(np.asarray(midpoints), axis=0)
    if not np.all(np.isfinite(anchor)):
        return None
    return anchor


@dataclass
class FilterResult:
    name: str
    estimates: np.ndarray
    errors: np.ndarray
    runtimes: np.ndarray


class BaseFilter:
    name = "base"

    def __init__(self, cameras: list[Camera], dt: float, pixel_sigma: float, rng: np.random.Generator):
        self.cameras = cameras
        self.dt = float(dt)
        self.pixel_sigma = float(pixel_sigma)
        self.rng = rng

    def initialize(self, mean: np.ndarray, cov: np.ndarray):
        raise NotImplementedError

    def step(self, y: np.ndarray, mask: np.ndarray) -> np.ndarray:
        tic = perf_counter()
        est = self._step_impl(y, mask)
        return est, perf_counter() - tic

    def _step_impl(self, y: np.ndarray, mask: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ErrorStateEKF(BaseFilter):
    """Multiplicative/error-state EKF on the translation subgroup.

    Retraction: x = x_hat + delta_x.  For the current position-only target this
    is exactly the Abelian Lie group R^6, so the multiplicative wording is about
    the error-state construction rather than attitude estimation.
    """

    name = "MEKF"
    gate_chi2: float | None = None

    def initialize(self, mean: np.ndarray, cov: np.ndarray):
        self.x = np.asarray(mean, dtype=float).reshape(6)
        self.P = np.asarray(cov, dtype=float).reshape(6, 6)
        self.Q = np.diag([0.015, 0.015, 0.008, 0.18, 0.18, 0.08]) ** 2

    def select_measurements(
        self,
        y: np.ndarray,
        mask: np.ndarray,
        yhat: np.ndarray,
        H: np.ndarray,
        hmask: np.ndarray,
        P: np.ndarray,
    ) -> np.ndarray:
        use = mask & hmask & np.isfinite(y) & np.isfinite(yhat)
        if self.gate_chi2 is None:
            return use

        gated = np.zeros_like(use, dtype=bool)
        for cam_idx in range(len(self.cameras)):
            rows = np.array([2 * cam_idx, 2 * cam_idx + 1])
            if not np.all(use[rows]):
                continue
            H_i = H[rows]
            r_i = y[rows] - yhat[rows]
            R_i = (self.pixel_sigma ** 2) * np.eye(2)
            S_i = H_i @ P @ H_i.T + R_i
            d2 = float(r_i.T @ safe_solve(S_i, r_i))
            if np.isfinite(d2) and d2 <= self.gate_chi2:
                gated[rows] = True
        return gated

    def predict(self):
        F = np.eye(6)
        F[0:3, 3:6] = self.dt * np.eye(3)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

    def update_once(self, y: np.ndarray, mask: np.ndarray, joseph: bool = True):
        yhat, H, hmask = projection_jacobian(self.cameras, self.x)
        use = self.select_measurements(y, mask, yhat, H, hmask, self.P)
        if np.count_nonzero(use) < 2:
            return
        H = H[use]
        r = y[use] - yhat[use]
        R = (self.pixel_sigma ** 2) * np.eye(len(r))
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ safe_solve(S, np.eye(S.shape[0]))
        dx = K @ r
        self.x = self.x + dx
        I = np.eye(6)
        if joseph:
            IKH = I - K @ H
            self.P = IKH @ self.P @ IKH.T + K @ R @ K.T
        else:
            self.P = (I - K @ H) @ self.P
        self.P = 0.5 * (self.P + self.P.T)

    def _step_impl(self, y: np.ndarray, mask: np.ndarray) -> np.ndarray:
        self.predict()
        self.update_once(y, mask)
        return self.x.copy()


class IteratedEKF(ErrorStateEKF):
    """Iterated EKF with repeated measurement relinearization.

    The state is translation and velocity in R^6, so this is not an invariant
    EKF on SE(3).  It is an iterated EKF measurement update: after prediction,
    the nonlinear SE(3)-projection measurement model is relinearized several
    times before committing the correction.
    """

    name = "ItEKF"

    def update_once(self, y: np.ndarray, mask: np.ndarray, joseph: bool = True):
        x_pred = self.x.copy()
        P_pred = self.P.copy()
        dx_total = np.zeros(6)
        for _ in range(3):
            yhat, H, hmask = projection_jacobian(self.cameras, x_pred + dx_total)
            use = self.select_measurements(y, mask, yhat, H, hmask, P_pred)
            if np.count_nonzero(use) < 2:
                return
            H_use = H[use]
            # Iterated EKF residual relative to the predicted state.
            r = y[use] - yhat[use] + H_use @ dx_total
            R = (self.pixel_sigma ** 2) * np.eye(len(r))
            S = H_use @ P_pred @ H_use.T + R
            K = P_pred @ H_use.T @ safe_solve(S, np.eye(S.shape[0]))
            dx_new = K @ r
            if np.linalg.norm(dx_new - dx_total) < 1e-5:
                dx_total = dx_new
                break
            dx_total = dx_new
        self.x = x_pred + dx_total
        I = np.eye(6)
        IKH = I - K @ H_use
        self.P = IKH @ P_pred @ IKH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)


class GatedErrorStateEKF(ErrorStateEKF):
    """MEKF with per-camera Mahalanobis innovation gating."""

    name = "MEKF-gated"
    gate_chi2 = 11.83


class GatedIteratedEKF(IteratedEKF):
    """Iterated EKF update with the same per-camera innovation gate."""

    name = "ItEKF-gated"
    gate_chi2 = 11.83


class BootstrapPF(BaseFilter):
    name = "PF"

    def __init__(
        self,
        cameras: list[Camera],
        dt: float,
        pixel_sigma: float,
        rng: np.random.Generator,
        n_particles: int = 600,
    ):
        super().__init__(cameras, dt, pixel_sigma, rng)
        self.n_particles = int(n_particles)

    def initialize(self, mean: np.ndarray, cov: np.ndarray):
        self.particles = self.rng.multivariate_normal(mean, cov, size=self.n_particles)
        self.weights = np.ones(self.n_particles) / self.n_particles
        self.process_std = np.array([0.025, 0.025, 0.015, 0.20, 0.20, 0.10])
        self.estimate = np.average(self.particles, axis=0, weights=self.weights)

    def predict(self):
        self.particles[:, :3] += self.dt * self.particles[:, 3:6]
        self.particles += self.rng.normal(0.0, self.process_std, size=self.particles.shape)

    def update(self, y: np.ndarray, mask: np.ndarray):
        z, valid = project_points_lie(self.cameras, self.particles[:, :3])
        pred = z.transpose(1, 0, 2)
        logw = np.zeros(self.n_particles, dtype=float)
        counts = np.zeros(self.n_particles, dtype=int)
        sigma = max(self.pixel_sigma, 1e-12)
        outlier_floor = 0.015
        loss_cap = 25.0
        used_any_camera = False

        for cam_idx, cam in enumerate(self.cameras):
            rows = np.array([2 * cam_idx, 2 * cam_idx + 1])
            if not (mask[rows[0]] and mask[rows[1]] and np.all(np.isfinite(y[rows]))):
                continue
            used_any_camera = True
            cam_valid = valid[cam_idx] & np.all(np.isfinite(pred[:, cam_idx, :]), axis=1)
            diff = (pred[:, cam_idx, :] - y[rows].reshape(1, 2)) / sigma
            d2 = np.sum(diff * diff, axis=1)
            like = np.full(self.n_particles, outlier_floor, dtype=float)
            like[cam_valid] += np.exp(-0.5 * np.minimum(d2[cam_valid], loss_cap))
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

    def _step_impl(self, y: np.ndarray, mask: np.ndarray) -> np.ndarray:
        self.predict()
        self.update(y, mask)
        return self.estimate.copy()
