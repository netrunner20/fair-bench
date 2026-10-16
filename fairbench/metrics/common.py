"""Small helpers shared by the metrics."""

from __future__ import annotations

import numpy as np


def running_median(x: np.ndarray, k: int = 5) -> np.ndarray:
    """Centered running median that ignores NaNs (window shrinks at the ends).

    With k = 5 a glitch lasting 1 or 2 frames is removed while any disagreement lasting
    3 frames or more survives -- the filter the proposal uses for VIMS.
    """
    x = np.asarray(x, dtype=float)
    if k <= 1:
        return x.copy()
    h = k // 2
    out = np.full_like(x, np.nan)
    for i in range(len(x)):
        w = x[max(0, i - h): i + h + 1]
        w = w[np.isfinite(w)]
        if len(w):
            out[i] = np.median(w)
    return out


def exp_score(E: float, E_ceil: float, E_ref: float) -> float:
    """VIMS-style 0-1 score: 1 at the noise floor, e^-1 at the reference error."""
    if not np.isfinite(E):
        return 0.0
    span = max(E_ref - E_ceil, 1e-12)
    return float(np.exp(-max(E - E_ceil, 0.0) / span))


def penalize_missing(d: np.ndarray, missing: np.ndarray, penalty: float) -> np.ndarray:
    """Set per-frame distances to ``penalty`` where the ball should be visible but was not found."""
    d = np.array(d, dtype=float, copy=True)
    d[missing] = penalty
    return d


def mean_after_median(d: np.ndarray, frames: np.ndarray, k: int = 5) -> float:
    """Mean of the running-median-filtered per-frame values over the selected frames."""
    dm = running_median(d, k)
    sel = dm[frames]
    sel = sel[np.isfinite(sel)]
    return float(np.mean(sel)) if len(sel) else float("nan")
