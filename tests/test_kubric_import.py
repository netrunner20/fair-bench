"""The Kubric camera conversion must reproduce Kubric's own projection."""

import numpy as np

from fairbench.geometry import Camera
from fairbench.kubric_import import BLENDER_TO_OPENCV, camera_from_kubric, quat_to_matrix


def _matrix_to_quat(R: np.ndarray) -> np.ndarray:
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2
    y = np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2
    z = np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2
    x = np.copysign(x, R[2, 1] - R[1, 2])
    y = np.copysign(y, R[0, 2] - R[2, 0])
    z = np.copysign(z, R[1, 0] - R[0, 1])
    return np.array([w, x, y, z])


def _kubric_project(point, focal_length, sensor_width, W, H, position, quat):
    """Kubric's PerspectiveCamera.project_point (normalized image coords in [0, 1])."""
    fx = focal_length / sensor_width
    fy = focal_length / (sensor_width * H / W)  # sensor_fit horizontal
    K = np.array([[fx, 0, -0.5], [0, -fy, -0.5], [0, 0, -1]])
    M = np.eye(4)
    M[:3, :3] = quat_to_matrix(quat)
    M[:3, 3] = position
    p = K @ (np.linalg.inv(M) @ np.append(point, 1.0))[:3]
    return p[:2] / p[2]


def test_camera_from_kubric_matches_kubric_projection():
    W, H = 1280, 720
    cam = Camera.look_at(np.array([-4.5, -4.3, 1.7]), np.array([0.2, 0.0, 0.8]), fov_x_deg=50, width=W, height=H)
    q = _matrix_to_quat(cam.R @ BLENDER_TO_OPENCV)
    sensor_width = 36.0
    focal_length = cam.focal / W * sensor_width
    meta = {"focal_length": focal_length, "sensor_width": sensor_width,
            "positions": [cam.C.tolist()] * 3, "quaternions": [q.tolist()] * 3}
    got, info = camera_from_kubric(meta, W, H)
    assert info["camera_translation_over_clip_m"] == 0.0
    pts = np.array([[0.3, 0.2, 1.1], [-1.5, 0.5, 0.2], [2.0, -0.5, 1.6]])
    ours = got.project(pts)
    theirs = np.array([_kubric_project(p, focal_length, sensor_width, W, H, cam.C, q) for p in pts]) * [W, H]
    assert np.allclose(ours, theirs, atol=1e-6)
    assert np.allclose(ours, cam.project(pts), atol=1e-6)
