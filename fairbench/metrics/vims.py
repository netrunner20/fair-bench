"""View-Invariant Motion Similarity (VIMS) -- the proposal's camera-angle metric.

If the three camera-angle generations show the same 3D motion, the three lines of sight
to the tracked ball meet at one point in every frame. VIMS uses only the tracked ball
centers (no size or depth estimate):

    d_kt  ∝ R_k K_k^-1 [u_kt, v_kt, 1]^T              line of sight of view k at frame t
    X_t   = (Σ_k P_kt)^-1 Σ_k P_kt C_k,  P_kt = I - d_kt d_kt^T   (least-squares point)
    e_t   = RMS distance (m) from X_t to the three lines
    E     = mean over generated frames of the 5-frame running median of e_t
    VIMS  = exp(-max(E - E_ceil, 0) / (E_ref - E_ceil))

E_ceil = E from tracker noise alone (0.3 px) and E_ref = E when one view lags by one
frame, both computed from the scene's ground truth.
"""

from __future__ import annotations

from itertools import product

import numpy as np

from ..geometry import Camera, consensus_point
from .common import exp_score, mean_after_median, penalize_missing


def vims_frames(uvs: list[np.ndarray], cams: list[Camera], visible: np.ndarray,
                penalty_m: float | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame inconsistency e_t (m), consensus points X_t, and the 'missing' mask."""
    dirs = np.stack([cam.rays(uv) for uv, cam in zip(uvs, cams)], axis=1)  # (T, K, 3)
    centers = np.stack([cam.C for cam in cams])
    X, e = consensus_point(centers, dirs)
    found = np.all(np.isfinite(dirs), axis=(1, 2))
    lost = visible & ~found
    e[~visible] = np.nan
    if penalty_m is not None:
        e = penalize_missing(e, lost, penalty_m)
    return e, X, lost


def vims_E(uvs, cams, visible, eval_frames, penalty_m=None, median_k=5) -> dict:
    e, X, lost = vims_frames(uvs, cams, visible, penalty_m)
    E = mean_after_median(e, eval_frames, median_k)
    n_vis = int(np.sum(visible & eval_frames))
    return {"E_m": E, "lost_frac": float(np.sum(lost & eval_frames) / max(n_vis, 1)), "e_t": e, "X": X}


def vims_score(E: float, E_ceil: float, E_ref: float) -> float:
    return exp_score(E, E_ceil, E_ref)


def deviating_view(uvs, cams, visible, eval_frames, median_k=5) -> int:
    """Which view disagrees most: the view whose removal lowers E the most (2-view E)."""
    best, arg = np.inf, -1
    for k in range(len(uvs)):
        keep = [i for i in range(len(uvs)) if i != k]
        # with two lines, e_t is half their distance; still comparable across k
        E = vims_E([uvs[i] for i in keep], [cams[i] for i in keep], visible, eval_frames, None, median_k)["E_m"]
        if E < best:
            best, arg = E, k
    return arg


def group_vims(tracks: dict[str, list[np.ndarray]], variants: tuple[str, ...], cams: dict[str, Camera],
               visible: np.ndarray, eval_frames: np.ndarray, E_ceil: float, E_ref: float,
               penalty_m: float | None, median_k: int = 5, max_triplets: int = 125) -> dict:
    """VIMS over every combination of samples (one per view); returns mean E and VIMS."""
    lists = [tracks.get(v, []) for v in variants]
    if any(len(L) == 0 for L in lists):
        return {"E_m": float("nan"), "VIMS": 0.0, "n_triplets": 0, "lost_frac": float("nan")}
    combos = list(product(*[range(len(L)) for L in lists]))
    if len(combos) > max_triplets:
        rng = np.random.default_rng(0)
        combos = [combos[i] for i in rng.choice(len(combos), max_triplets, replace=False)]
    Es, Ss, lost = [], [], []
    for idx in combos:
        uvs = [lists[k][i] for k, i in enumerate(idx)]
        r = vims_E(uvs, [cams[v] for v in variants], visible, eval_frames, penalty_m, median_k)
        Es.append(r["E_m"])
        Ss.append(vims_score(r["E_m"], E_ceil, E_ref))
        lost.append(r["lost_frac"])
    return {
        "E_m": float(np.nanmean(Es)),
        "VIMS": float(np.mean(Ss)),
        "VIMS_of_mean_E": vims_score(float(np.nanmean(Es)), E_ceil, E_ref),
        "n_triplets": len(combos),
        "lost_frac": float(np.mean(lost)),
    }
