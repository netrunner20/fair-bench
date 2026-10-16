"""Camera geometry used by every metric.

Conventions (used everywhere in this package):

* World frame: right-handed, ``up`` given by the manifest (Kubric/Blender is z-up).
* Camera frame: OpenCV pinhole convention -- x to the right, y down, z forward
  (the direction the camera looks).
* ``Camera.R`` is the camera-to-world rotation: its columns are the camera's x, y and z
  axes written in world coordinates. ``Camera.C`` is the camera center in world
  coordinates. A world point X has camera coordinates ``R.T @ (X - C)``.
* Pixel coordinates (u, v) have their origin at the top-left corner of the image,
  u to the right and v down, in pixels of the *reference* resolution (1280x720).

With this convention the line of sight through pixel (u, v) has world direction
``d ∝ R K^-1 [u, v, 1]^T``, which is the formula in the proposal's VIMS appendix.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Camera:
    """A static pinhole camera."""

    K: np.ndarray  # (3, 3) intrinsics in pixels
    R: np.ndarray  # (3, 3) camera-to-world rotation
    C: np.ndarray  # (3,) camera center in world coordinates
    width: int = 1280
    height: int = 720
    _K_inv: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.K = np.asarray(self.K, dtype=float).reshape(3, 3)
        self.R = np.asarray(self.R, dtype=float).reshape(3, 3)
        self.C = np.asarray(self.C, dtype=float).reshape(3)
        self._K_inv = np.linalg.inv(self.K)
        if not np.allclose(self.R.T @ self.R, np.eye(3), atol=1e-6):
            raise ValueError("Camera.R is not a rotation matrix")

    # ------------------------------------------------------------------ basics
    @property
    def focal(self) -> float:
        """Mean focal length in pixels (fx and fy are equal for square pixels)."""
        return float(0.5 * (self.K[0, 0] + self.K[1, 1]))

    @property
    def principal_point(self) -> np.ndarray:
        return self.K[:2, 2].copy()

    @property
    def optical_axis(self) -> np.ndarray:
        """World direction the camera looks along (camera +z)."""
        return self.R[:, 2].copy()

    @property
    def diagonal_px(self) -> float:
        return float(np.hypot(self.width, self.height))

    def to_camera(self, X: np.ndarray) -> np.ndarray:
        """World points (..., 3) -> camera coordinates (..., 3)."""
        X = np.asarray(X, dtype=float)
        return (X - self.C) @ self.R  # row-vector form of R^T (X - C)

    def depth(self, X: np.ndarray) -> np.ndarray:
        """Depth along the optical axis (camera z) of world points (..., 3)."""
        return self.to_camera(X)[..., 2]

    def project(self, X: np.ndarray) -> np.ndarray:
        """World points (..., 3) -> pixel coordinates (..., 2)."""
        Xc = self.to_camera(X)
        uvw = Xc @ self.K.T
        return uvw[..., :2] / uvw[..., 2:3]

    def rays(self, uv: np.ndarray) -> np.ndarray:
        """Unit world directions (..., 3) of the lines of sight through pixels (..., 2)."""
        uv = np.asarray(uv, dtype=float)
        ones = np.ones(uv.shape[:-1] + (1,))
        pix = np.concatenate([uv, ones], axis=-1)
        d_cam = pix @ self._K_inv.T
        d_world = d_cam @ self.R.T
        return d_world / np.linalg.norm(d_world, axis=-1, keepdims=True)

    def normalized(self, uv: np.ndarray) -> np.ndarray:
        """Pixels (..., 2) -> normalized image coordinates (x/z, y/z) (..., 2)."""
        uv = np.asarray(uv, dtype=float)
        ones = np.ones(uv.shape[:-1] + (1,))
        p = np.concatenate([uv, ones], axis=-1) @ self._K_inv.T
        return p[..., :2] / p[..., 2:3]

    def from_normalized(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=float)
        ones = np.ones(xy.shape[:-1] + (1,))
        p = np.concatenate([xy, ones], axis=-1) @ self.K.T
        return p[..., :2] / p[..., 2:3]

    def scaled(self, width: int, height: int) -> "Camera":
        """The same camera for an image resized to (width, height)."""
        sx, sy = width / self.width, height / self.height
        S = np.diag([sx, sy, 1.0])
        return Camera(K=S @ self.K, R=self.R, C=self.C, width=width, height=height)

    def to_dict(self) -> dict:
        return {
            "K": self.K.tolist(),
            "R_c2w": self.R.tolist(),
            "C": self.C.tolist(),
            "width": int(self.width),
            "height": int(self.height),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Camera":
        R = d.get("R_c2w")
        if R is None and "R_w2c" in d:
            R = np.asarray(d["R_w2c"], dtype=float).T
        if R is None:
            raise KeyError("camera needs 'R_c2w' (camera-to-world) or 'R_w2c'")
        return cls(K=d["K"], R=R, C=d["C"], width=d.get("width", 1280), height=d.get("height", 720))

    @classmethod
    def look_at(
        cls,
        eye: np.ndarray,
        target: np.ndarray,
        up: np.ndarray = (0.0, 0.0, 1.0),
        fov_x_deg: float = 50.0,
        width: int = 1280,
        height: int = 720,
    ) -> "Camera":
        """Build a camera at ``eye`` looking at ``target`` with horizontal field of view."""
        eye = np.asarray(eye, dtype=float)
        target = np.asarray(target, dtype=float)
        up = np.asarray(up, dtype=float)
        z = target - eye
        z /= np.linalg.norm(z)
        x = np.cross(z, up)
        if np.linalg.norm(x) < 1e-9:  # looking straight along up: pick any x
            x = np.cross(z, np.array([1.0, 0.0, 0.0]))
        x /= np.linalg.norm(x)
        y = np.cross(z, x)  # camera y points down in the image
        R = np.stack([x, y, z], axis=1)
        f = 0.5 * width / np.tan(np.deg2rad(fov_x_deg) / 2)
        K = np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]])
        return cls(K=K, R=R, C=eye, width=width, height=height)


# ---------------------------------------------------------------------- planes
@dataclass
class Plane:
    point: np.ndarray  # (3,)
    normal: np.ndarray  # (3,) unit

    def __post_init__(self) -> None:
        self.point = np.asarray(self.point, dtype=float).reshape(3)
        n = np.asarray(self.normal, dtype=float).reshape(3)
        self.normal = n / np.linalg.norm(n)

    def to_dict(self) -> dict:
        return {"point": self.point.tolist(), "normal": self.normal.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "Plane":
        return cls(point=d["point"], normal=d["normal"])


def intersect_rays_plane(C: np.ndarray, d: np.ndarray, plane: Plane) -> np.ndarray:
    """Intersect lines of sight ``C + s d`` (d: (..., 3)) with a plane.

    Returns (..., 3); NaN where the ray is (nearly) parallel to the plane or the
    intersection lies behind the camera.
    """
    d = np.asarray(d, dtype=float)
    denom = d @ plane.normal
    num = float((plane.point - C) @ plane.normal)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = num / denom
    out = C + s[..., None] * d
    bad = (np.abs(denom) < 1e-6) | (s <= 0)
    out[bad] = np.nan
    return out


def motion_plane(points: np.ndarray, camera: Camera, line_tol: float = 0.01) -> Plane:
    """Plane that contains a (planar) ground-truth trajectory.

    If the trajectory spans a plane (projectile with horizontal motion), that plane is
    returned. If it is (close to) a straight line -- a vertical drop or a straight roll --
    every plane through the line contains it; we return the one whose normal is closest
    to the camera's optical axis, i.e. the most fronto-parallel one, because it makes
    the back-projection of tracked pixels best conditioned.
    """
    P = np.asarray(points, dtype=float)
    P = P[np.all(np.isfinite(P), axis=1)]
    c = P.mean(axis=0)
    U, S, Vt = np.linalg.svd(P - c, full_matrices=False)
    spread = S / max(S[0], 1e-12)
    if len(spread) >= 3 and spread[1] > line_tol:
        # spans a plane (best-fit plane if the motion is not exactly planar)
        return Plane(point=c, normal=Vt[2])
    # line-like: direction = first principal axis
    u = Vt[0]
    a = camera.optical_axis
    n = a - (a @ u) * u
    if np.linalg.norm(n) < 1e-6:  # line along the optical axis: degenerate, use any
        n = np.cross(u, [0.0, 0.0, 1.0])
    return Plane(point=c, normal=n)


# ----------------------------------------------------------------- lines of sight
def point_to_ray_distance(X: np.ndarray, C: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Distance from points X (..., 3) to lines through C with unit direction d (..., 3)."""
    v = np.asarray(X, dtype=float) - C
    along = np.sum(v * d, axis=-1, keepdims=True)
    return np.linalg.norm(v - along * d, axis=-1)


def consensus_point(centers: np.ndarray, dirs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares point closest to K lines, for T frames at once.

    centers: (K, 3) camera centers; dirs: (T, K, 3) unit directions.
    Returns (X_hat (T, 3), e (T,)) where e is the root-mean-square distance from X_hat
    to the K lines (the proposal's e_t).
    """
    dirs = np.asarray(dirs, dtype=float)
    T, K, _ = dirs.shape
    I = np.eye(3)
    P = I[None, None] - dirs[..., :, None] * dirs[..., None, :]  # (T, K, 3, 3)
    A = P.sum(axis=1)  # (T, 3, 3)
    b = np.einsum("tkij,kj->ti", P, centers)  # (T, 3)
    X = np.full((T, 3), np.nan)
    ok = np.all(np.isfinite(dirs), axis=(1, 2))
    if ok.any():
        X[ok] = np.linalg.solve(A[ok], b[ok][..., None])[..., 0]
    resid = np.einsum("tkij,tkj->tki", P, X[:, None, :] - centers[None])  # (T, K, 3)
    e = np.sqrt(np.mean(np.sum(resid**2, axis=-1), axis=-1))
    e[~ok] = np.nan
    return X, e


# ---------------------------------------------------------------- sphere centers
def sphere_center_from_centroid(uv: np.ndarray, camera: Camera, depth: np.ndarray, radius: float) -> np.ndarray:
    """Correct the image centroid of a projected sphere to the projection of its center.

    A sphere's outline projects to an ellipse whose center is pushed away from the
    principal point. In normalized image coordinates the ellipse center p0 and the
    projected sphere center pc satisfy exactly

        p0 = pc / (1 - (r / Z)^2)

    where r is the sphere radius and Z the depth of its center along the optical axis
    (derivation in docs/METRICS.md). The correction is second order in r/Z, so an
    approximate depth (the ground-truth depth of the reference ball) is enough.

    uv: (..., 2) centroids in pixels; depth: (...,) or scalar Z in meters.
    """
    uv = np.asarray(uv, dtype=float)
    Z = np.asarray(depth, dtype=float)
    k = 1.0 - (radius / Z) ** 2
    xy = camera.normalized(uv) * np.asarray(k)[..., None]
    return camera.from_normalized(xy)


def projected_radius_px(camera: Camera, X: np.ndarray, radius: float) -> np.ndarray:
    """Approximate radius in pixels of a sphere of ``radius`` at world point(s) X."""
    Z = camera.depth(X)
    return camera.focal * radius / np.sqrt(np.maximum(Z**2 - radius**2, 1e-12))


def angle_between_deg(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))
