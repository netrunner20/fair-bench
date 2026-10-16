import numpy as np

from fairbench.geometry import (
    Camera, Plane, consensus_point, intersect_rays_plane, motion_plane, point_to_ray_distance,
    sphere_center_from_centroid,
)


def _cam(eye=(0.0, -6.0, 1.0), target=(0.0, 0.0, 0.8), w=1280, h=720):
    return Camera.look_at(np.array(eye), np.array(target), fov_x_deg=55, width=w, height=h)


def test_project_and_rays_roundtrip():
    cam = _cam()
    X = np.array([[0.3, 0.2, 1.1], [-1.5, 0.5, 0.2], [2.0, -0.5, 1.6]])
    uv = cam.project(X)
    d = cam.rays(uv)
    assert np.allclose(point_to_ray_distance(X, cam.C, d), 0, atol=1e-9)


def test_ray_plane_intersection():
    cam = _cam()
    plane = Plane(point=[0, 0.4, 0], normal=[0, 1, 0])
    X = np.array([[0.1, 0.4, 0.9], [-1.2, 0.4, 0.3]])
    Y = intersect_rays_plane(cam.C, cam.rays(cam.project(X)), plane)
    assert np.allclose(X, Y, atol=1e-9)


def test_motion_plane_line_is_fronto_parallel():
    cam = _cam()
    line = np.stack([np.linspace(-2, 2, 50), np.full(50, 0.45), np.full(50, 0.1)], 1)
    p = motion_plane(line, cam)
    # normal is horizontal-ish and points along the viewing direction
    assert abs(p.normal @ cam.optical_axis) > 0.95


def test_consensus_point_exact_and_noisy():
    X = np.array([[0.2, 0.1, 0.9]])
    cams = [_cam(eye=e) for e in [(-4.5, -4.3, 1.7), (4.9, -4.3, 1.7), (0.2, -4.0, 5.9)]]
    dirs = np.stack([c.rays(c.project(X)) for c in cams], axis=1)
    Xh, e = consensus_point(np.stack([c.C for c in cams]), dirs)
    assert np.allclose(Xh, X, atol=1e-9) and e[0] < 1e-9


def _sphere_coverage_centroid(cam: Camera, X: np.ndarray, r: float, ss: int = 8) -> np.ndarray:
    uv = cam.project(X[None])[0]
    rad = cam.focal * r / cam.depth(X[None])[0]
    m = int(2 * rad) + 4
    u0, v0 = int(uv[0]) - m, int(uv[1]) - m
    n = 2 * m
    g = (np.arange(n * ss) + 0.5) / ss
    uu, vv = np.meshgrid(u0 + g, v0 + g)
    d = cam.rays(np.stack([uu, vv], -1))
    oc = cam.C - X
    b = d @ oc
    c = oc @ oc - r * r
    hit = (b * b - c) >= 0
    cov = hit.reshape(n, ss, n, ss).mean(axis=(1, 3))
    pu, pv = np.meshgrid(u0 + np.arange(n) + 0.5, v0 + np.arange(n) + 0.5)
    return np.array([np.sum(cov * pu), np.sum(cov * pv)]) / cov.sum()


def test_sphere_centroid_correction_is_exact_in_closed_form():
    cam = _cam(eye=(0.0, -3.0, 1.0))
    r = 0.15
    X = np.array([1.1, 0.0, 1.6])  # far off-axis so the bias is visible
    true_uv = cam.project(X[None])[0]
    cen = _sphere_coverage_centroid(cam, X, r)
    raw_err = np.linalg.norm(cen - true_uv)
    corr = sphere_center_from_centroid(cen[None], cam, np.array([cam.depth(X[None])[0]]), r)[0]
    corr_err = np.linalg.norm(corr - true_uv)
    assert raw_err > 0.5  # the bias is real (pixels)
    assert corr_err < 0.05  # and the closed form removes it
