"""Lie-group camera projection utilities for multi-camera UAV tracking.

The estimated target state in this project is position and velocity, not full
attitude.  The Lie-group object is therefore the camera pose, represented as an
element of SE(3), and the target position is transformed through this pose
before applying the pinhole projection.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def hat_so3(w: np.ndarray) -> np.ndarray:
    """Map a vector in R^3 to a skew-symmetric matrix in so(3)."""
    wx, wy, wz = np.asarray(w, dtype=float).reshape(3)
    return np.array(
        [
            [0.0, -wz, wy],
            [wz, 0.0, -wx],
            [-wy, wx, 0.0],
        ],
        dtype=float,
    )


def exp_so3(w: np.ndarray) -> np.ndarray:
    """Rodrigues exponential map from so(3) vector coordinates to SO(3)."""
    w = np.asarray(w, dtype=float).reshape(3)
    theta = float(np.linalg.norm(w))
    W = hat_so3(w)
    if theta < 1e-12:
        return np.eye(3) + W
    a = np.sin(theta) / theta
    b = (1.0 - np.cos(theta)) / (theta * theta)
    return np.eye(3) + a * W + b * (W @ W)


def hat_se3(xi: np.ndarray) -> np.ndarray:
    """Map xi=(rho, phi) to se(3) matrix coordinates."""
    xi = np.asarray(xi, dtype=float).reshape(6)
    rho = xi[:3]
    phi = xi[3:]
    X = np.zeros((4, 4), dtype=float)
    X[:3, :3] = hat_so3(phi)
    X[:3, 3] = rho
    return X


@dataclass(frozen=True)
class SE3:
    """Minimal SE(3) container.

    R and t represent the transform from world coordinates to camera
    coordinates:

        p_c = R_cw p_w + t_cw.
    """

    R: np.ndarray
    t: np.ndarray

    @property
    def matrix(self) -> np.ndarray:
        T = np.eye(4, dtype=float)
        T[:3, :3] = self.R
        T[:3, 3] = self.t
        return T

    @property
    def inv(self) -> "SE3":
        R_inv = self.R.T
        return SE3(R_inv, -R_inv @ self.t)

    def transform_points(self, points_w: np.ndarray) -> np.ndarray:
        points_w = np.asarray(points_w, dtype=float)
        return points_w @ self.R.T + self.t.reshape(1, 3)


@dataclass(frozen=True)
class Camera:
    name: str
    T_cw: SE3
    K: np.ndarray
    width: int
    height: int
    center_w: np.ndarray | None = None
    yaw_rad: float | None = None


def look_at_R_wc(camera_pos: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
    """Camera-to-world rotation with optical z-axis looking at target."""
    camera_pos = np.asarray(camera_pos, dtype=float).reshape(3)
    target_pos = np.asarray(target_pos, dtype=float).reshape(3)
    z_axis = target_pos - camera_pos
    z_axis = z_axis / np.linalg.norm(z_axis)
    world_up = np.array([0.0, 0.0, 1.0])
    x_axis = np.cross(world_up, z_axis)
    if np.linalg.norm(x_axis) < 1e-9:
        world_up = np.array([0.0, 1.0, 0.0])
        x_axis = np.cross(world_up, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    return np.column_stack([x_axis, y_axis, z_axis])


def gazebo_optical_R_wc(yaw_rad: float) -> np.ndarray:
    """Camera-to-world rotation matching the yaw-only Gazebo camera sensors.

    The Purdue SDF cameras are model/link cameras whose optical axis follows the
    link +X direction.  For pinhole projection we expose them in the usual
    optical convention: camera x is image-right, camera y is image-down, and
    camera z is forward.
    """
    c = float(np.cos(yaw_rad))
    s = float(np.sin(yaw_rad))
    forward_w = np.array([c, s, 0.0], dtype=float)
    link_y_w = np.array([-s, c, 0.0], dtype=float)
    right_w = -link_y_w
    down_w = np.array([0.0, 0.0, -1.0], dtype=float)
    return np.column_stack([right_w, down_w, forward_w])


def make_square_camera_rig(
    radius: float = 10.0,
    height: float = 10.0,
    look_at: np.ndarray | None = None,
    image_size: tuple[int, int] = (640, 480),
    focal_px: float = 343.0,
) -> list[Camera]:
    """Create the same four-camera square geometry used in the Gazebo sim."""
    width, height_px = image_size
    K = np.array(
        [
            [focal_px, 0.0, width / 2.0],
            [0.0, focal_px, height_px / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    layout = {
        "camera0": (np.array([radius, radius, height], dtype=float), -2.35),
        "camera1": (np.array([-radius, radius, height], dtype=float), -0.78),
        "camera2": (np.array([-radius, -radius, height], dtype=float), 0.78),
        "camera3": (np.array([radius, -radius, height], dtype=float), 2.35),
    }
    link_offset = np.array([0.1, 0.05, 0.05], dtype=float)
    cameras: list[Camera] = []
    for name, (model_pos_w, yaw_rad) in layout.items():
        c = np.cos(yaw_rad)
        s = np.sin(yaw_rad)
        R_w_model = np.array(
            [
                [c, -s, 0.0],
                [s, c, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        c_w = model_pos_w + R_w_model @ link_offset
        if look_at is None:
            R_wc = gazebo_optical_R_wc(yaw_rad)
        else:
            R_wc = look_at_R_wc(c_w, look_at)
        R_cw = R_wc.T
        t_cw = -R_cw @ c_w
        cameras.append(Camera(name, SE3(R_cw, t_cw), K.copy(), width, height_px, c_w, yaw_rad))
    return cameras


def project_points_lie(cameras: list[Camera], points_w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project world points through SE(3) camera poses.

    Returns
    -------
    z:
        Array with shape (C, N, 2), containing pixel coordinates. Invalid
        projections are set to NaN.
    valid:
        Boolean array with shape (C, N).
    """
    points_w = np.asarray(points_w, dtype=float)
    if points_w.ndim == 1:
        points_w = points_w.reshape(1, 3)
    C = len(cameras)
    N = points_w.shape[0]
    z = np.full((C, N, 2), np.nan, dtype=float)
    valid = np.zeros((C, N), dtype=bool)
    for ci, cam in enumerate(cameras):
        pc = cam.T_cw.transform_points(points_w)
        X = pc[:, 0]
        Y = pc[:, 1]
        Z = pc[:, 2]
        good_depth = Z > 1e-9
        u = cam.K[0, 0] * X / Z + cam.K[0, 2]
        v = cam.K[1, 1] * Y / Z + cam.K[1, 2]
        in_image = (
            good_depth
            & np.isfinite(u)
            & np.isfinite(v)
            & (u >= 0.0)
            & (u < cam.width)
            & (v >= 0.0)
            & (v < cam.height)
        )
        z[ci, :, 0] = u
        z[ci, :, 1] = v
        z[ci, ~in_image, :] = np.nan
        valid[ci] = in_image
    return z, valid


def stack_measurement(z_cam: np.ndarray, valid_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flatten camera measurements and return valid row mask."""
    values: list[float] = []
    mask: list[bool] = []
    for ci in range(z_cam.shape[0]):
        ok = bool(valid_cam[ci])
        values.extend([z_cam[ci, 0], z_cam[ci, 1]])
        mask.extend([ok, ok])
    return np.asarray(values, dtype=float), np.asarray(mask, dtype=bool)


def project_state(cameras: list[Camera], x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z, valid = project_points_lie(cameras, np.asarray(x[:3]).reshape(1, 3))
    return stack_measurement(z[:, 0, :], valid[:, 0])


def projection_jacobian(cameras: list[Camera], x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Analytic measurement Jacobian d pi(T_cw p_w) / d [p, v]."""
    x = np.asarray(x, dtype=float).reshape(6)
    H_rows: list[np.ndarray] = []
    z_rows: list[float] = []
    mask: list[bool] = []
    p_w = x[:3]
    for cam in cameras:
        pc = cam.T_cw.R @ p_w + cam.T_cw.t
        X, Y, Z = pc
        valid = (
            Z > 1e-9
            and np.isfinite(X)
            and np.isfinite(Y)
            and np.isfinite(Z)
        )
        if valid:
            u = cam.K[0, 0] * X / Z + cam.K[0, 2]
            v = cam.K[1, 1] * Y / Z + cam.K[1, 2]
            valid = 0.0 <= u < cam.width and 0.0 <= v < cam.height
        else:
            u = np.nan
            v = np.nan
        if valid:
            J_cam = np.array(
                [
                    [cam.K[0, 0] / Z, 0.0, -cam.K[0, 0] * X / (Z * Z)],
                    [0.0, cam.K[1, 1] / Z, -cam.K[1, 1] * Y / (Z * Z)],
                ],
                dtype=float,
            )
            J_pos = J_cam @ cam.T_cw.R
        else:
            J_pos = np.zeros((2, 3), dtype=float)
        H = np.zeros((2, 6), dtype=float)
        H[:, :3] = J_pos
        H_rows.extend([H[0], H[1]])
        z_rows.extend([u, v])
        mask.extend([valid, valid])
    return np.asarray(z_rows, dtype=float), np.asarray(H_rows, dtype=float), np.asarray(mask, dtype=bool)
