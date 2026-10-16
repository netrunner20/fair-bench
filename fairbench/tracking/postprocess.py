"""Clean-up applied to every track before any metric sees it."""

from __future__ import annotations

import numpy as np

from ..geometry import Camera, sphere_center_from_centroid


def remove_single_frame_jumps(uv: np.ndarray, thresh_px: float) -> tuple[np.ndarray, int]:
    """Replace isolated one-frame jumps by the midpoint of their neighbours.

    A frame is a jump when it sits more than ``thresh_px`` from the midpoint of its two
    neighbours while those neighbours are close to each other. Real motion, including a
    bounce, moves far less than ``thresh_px`` (a few ball radii) between frames.
    Returns the cleaned track and the number of frames replaced.
    """
    out = np.array(uv, dtype=float, copy=True)
    n = 0
    for t in range(1, len(uv) - 1):
        a, b, c = uv[t - 1], uv[t], uv[t + 1]
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b)) and np.all(np.isfinite(c))):
            continue
        mid = 0.5 * (a + c)
        if np.linalg.norm(b - mid) > thresh_px and np.linalg.norm(a - c) < thresh_px:
            out[t] = mid
            n += 1
    return out, n


def resample_track(times_src: np.ndarray, values: np.ndarray, times_dst: np.ndarray, max_gap_s: float,
                   end_tolerance_s: float = 0.0) -> np.ndarray:
    """Linear interpolation of a (T, D) track (NaN = missing) onto ``times_dst``.

    Only interpolates between two valid samples at most ``max_gap_s`` apart, so a ball that
    is lost for several frames stays missing instead of being filled in.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    ok = np.all(np.isfinite(values), axis=1)
    ts, vs = np.asarray(times_src)[ok], values[ok]
    out = np.full((len(times_dst), values.shape[1]), np.nan)
    if len(ts) == 0:
        return out
    idx = np.searchsorted(ts, times_dst)
    for k, (tau, i) in enumerate(zip(times_dst, idx)):
        if i < len(ts) and abs(ts[i] - tau) < 1e-9:
            out[k] = vs[i]
        elif 0 < i < len(ts):
            t0, t1 = ts[i - 1], ts[i]
            if t1 - t0 <= max_gap_s + 1e-9:
                w = (tau - t0) / (t1 - t0)
                out[k] = (1 - w) * vs[i - 1] + w * vs[i]
        elif i == len(ts) and tau - ts[-1] <= end_tolerance_s + 1e-9:
            out[k] = vs[-1]
        elif i == 0 and ts[0] - tau <= end_tolerance_s + 1e-9:
            out[k] = vs[0]
    return out


def correct_sphere_centers(uv: np.ndarray, camera: Camera, depth: np.ndarray, radius: float) -> np.ndarray:
    """Mask centroid -> projection of the ball center (closed form, see geometry)."""
    out = np.array(uv, dtype=float, copy=True)
    ok = np.all(np.isfinite(out), axis=1) & np.isfinite(depth)
    if ok.any():
        out[ok] = sphere_center_from_centroid(out[ok], camera, depth[ok], radius)
    return out
