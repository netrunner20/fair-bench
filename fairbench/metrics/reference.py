"""Reference error: how far the generated ball is from the simulator's ball.

Front-camera groups (proposal): pixel error x true depth / focal length, in meters.
Camera-angle group (proposal): distance from the true 3D position to the generated line
of sight of that view. Both are reported for every clip; ``mode`` picks the primary one.
"""

from __future__ import annotations

import numpy as np

from ..geometry import Camera, point_to_ray_distance
from .common import mean_after_median, penalize_missing


def reference_frames(uv_gen: np.ndarray, uv_gt: np.ndarray, X_gt: np.ndarray, cam: Camera, depth: np.ndarray,
                     visible: np.ndarray, mode: str, penalty_m: float | None) -> tuple[np.ndarray, np.ndarray]:
    if mode == "ray":
        d = point_to_ray_distance(X_gt, cam.C, cam.rays(uv_gen))
    elif mode == "depth_scaled":
        d = np.linalg.norm(uv_gen - uv_gt, axis=1) * depth / cam.focal
    else:
        raise ValueError(f"unknown reference-error mode {mode!r}")
    lost = visible & ~np.all(np.isfinite(uv_gen), axis=1)
    d[~visible] = np.nan
    if penalty_m is not None:
        d = penalize_missing(d, lost, penalty_m)
    return d, lost


def reference_error(uv_gen, uv_gt, X_gt, cam, depth, visible, eval_frames, mode="depth_scaled",
                    penalty_m=None, median_k=5) -> dict:
    d, lost = reference_frames(uv_gen, uv_gt, X_gt, cam, depth, visible, mode, penalty_m)
    n_vis = int(np.sum(visible & eval_frames))
    return {
        "err_m": mean_after_median(d, eval_frames, median_k),
        "final_err_m": float(d[np.where(eval_frames & visible)[0][-1]]) if n_vis else float("nan"),
        "lost_frac": float(np.sum(lost & eval_frames) / max(n_vis, 1)),
        "per_frame_m": d,
    }
