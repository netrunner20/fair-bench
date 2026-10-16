"""Intervention robustness for the color and background groups.

The camera is the same for the three variants, so tracked ball positions are compared
frame by frame. Pixel distances are converted to meters at the ball's true depth
(pixel distance x depth / focal length), the unit used by the reference error, which
makes groups with different depths comparable.

Group distance D = mean over the three variant pairs, as in the proposal:
(r, g), (r, b), (g, b)  or  (1, 2), (1, 3), (2, 3).
"""

from __future__ import annotations

from itertools import product

import numpy as np

from ..manifest import iter_pairs
from .common import mean_after_median, penalize_missing


def pair_distance_frames(uv_a: np.ndarray, uv_b: np.ndarray, depth: np.ndarray, focal: float,
                         visible: np.ndarray, penalty_m: float | None) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame distance in meters (NaN where not compared) and the 'missing' mask."""
    d_px = np.linalg.norm(uv_a - uv_b, axis=1)
    d_m = d_px * depth / focal
    lost = visible & ~(np.all(np.isfinite(uv_a), axis=1) & np.all(np.isfinite(uv_b), axis=1))
    d_m[~visible] = np.nan
    if penalty_m is not None:
        d_m = penalize_missing(d_m, lost, penalty_m)
    return d_m, lost


def pair_distance(uv_a, uv_b, depth, focal, visible, eval_frames, penalty_m=None, median_k=5) -> dict:
    d, lost = pair_distance_frames(uv_a, uv_b, depth, focal, visible, penalty_m)
    D = mean_after_median(d, eval_frames, median_k)
    n_vis = int(np.sum(visible & eval_frames))
    return {"D_m": D, "lost_frac": float(np.sum(lost & eval_frames) / max(n_vis, 1)), "per_frame_m": d}


def group_distance(tracks: dict[str, list[np.ndarray]], variants: tuple[str, ...], depth: np.ndarray,
                   focal: float, visible: np.ndarray, eval_frames: np.ndarray, penalty_m: float | None,
                   median_k: int = 5) -> dict:
    """Mean distance over variant pairs, averaging over every combination of samples.

    tracks[variant] = list of (T, 2) tracks, one per generated sample. With K samples per
    clip each variant pair contributes K*K sample pairs (independent generations).
    Also returns the within-variant (seed) distance: two samples of the SAME input --
    the floor that sampling randomness alone produces.
    """
    pair_D = {}
    lost = []
    for a, b in iter_pairs(variants):
        vals = []
        for ta, tb in product(tracks.get(a, []), tracks.get(b, [])):
            r = pair_distance(ta, tb, depth, focal, visible, eval_frames, penalty_m, median_k)
            vals.append(r["D_m"])
            lost.append(r["lost_frac"])
        pair_D[f"{a}-{b}"] = float(np.nanmean(vals)) if vals else float("nan")
    seed = []
    for v in variants:
        ts = tracks.get(v, [])
        for i in range(len(ts)):
            for j in range(i + 1, len(ts)):
                seed.append(pair_distance(ts[i], ts[j], depth, focal, visible, eval_frames, penalty_m, median_k)["D_m"])
    D = float(np.nanmean(list(pair_D.values()))) if pair_D else float("nan")
    return {
        "D_m": D,
        "pairs": pair_D,
        "seed_D_m": float(np.nanmean(seed)) if seed else float("nan"),
        "lost_frac": float(np.mean(lost)) if lost else float("nan"),
    }
