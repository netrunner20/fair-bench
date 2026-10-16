"""Scene-specific constants for turning distances into 0-1 scores, from ground truth only.

For every distance metric we compute, from the simulator's own trajectory,
  * a floor ("ceil" in the proposal's notation): the value tracker noise alone produces
    (Gaussian noise of ``noise_px`` per pixel axis on every track), and
  * a reference: the value when one track lags by one frame.
The 0-1 score is then exp(-max(x - floor, 0) / (reference - floor)): 1 at the noise floor,
e^-1 (0.37) at a one-frame lag. For VIMS this is exactly the proposal's definition; for
the color/background distance and the reference error it is the same construction applied
to those metrics (an open choice, see docs/DECISIONS.md).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..manifest import Dataset
from .reference import reference_error
from .trajectory import pair_distance
from .vims import vims_E


def _lag(uv: np.ndarray, k: int = 1) -> np.ndarray:
    out = uv.copy()
    out[k:] = uv[:-k]
    return out


def calibrate(ds: Dataset, noise_px: float = 0.3, n_mc: int = 50, seed: int = 0, median_k: int = 5,
              full_visible: float = 0.99) -> dict:
    rng = np.random.default_rng(seed)
    ev = np.arange(ds.n_gen + 1) >= 1
    out: dict = {"noise_px": noise_px, "n_mc": n_mc, "groups": {}, "clips": {}}
    for gid, clips in ds.groups().items():
        if gid.endswith("/angle"):
            cams = [c.camera for c in clips]
            gts = [ds.gt_pixels(c) for c in clips]
            vis = np.logical_and.reduce([ds.gt_visible(c, full_visible) for c in clips])
            E_exact = vims_E(gts, cams, vis, ev, None, median_k)["E_m"]
            E_noise = np.mean([
                vims_E([g + rng.normal(0, noise_px, g.shape) for g in gts], cams, vis, ev, None, median_k)["E_m"]
                for _ in range(n_mc)
            ])
            E_lag = np.mean([
                vims_E([_lag(g) if i == k else g for i, g in enumerate(gts)], cams, vis, ev, None, median_k)["E_m"]
                for k in range(len(gts))
            ])
            out["groups"][gid] = {"metric": "vims", "E_exact_m": E_exact, "ceil_m": float(E_noise),
                                  "ref_m": float(E_lag), "penalty_m": float(10 * E_lag)}
        else:
            c0 = clips[0]
            gt = ds.gt_pixels(c0)
            depth = ds.gt_depth(c0)
            vis = np.logical_and.reduce([ds.gt_visible(c, full_visible) for c in clips])
            f = c0.camera.focal
            D_noise = np.mean([
                pair_distance(gt + rng.normal(0, noise_px, gt.shape), gt + rng.normal(0, noise_px, gt.shape),
                              depth, f, vis, ev, None, median_k)["D_m"]
                for _ in range(n_mc)
            ])
            D_lag = pair_distance(gt, _lag(gt), depth, f, vis, ev, None, median_k)["D_m"]
            out["groups"][gid] = {"metric": "trajectory", "ceil_m": float(D_noise), "ref_m": float(D_lag),
                                  "penalty_m": float(10 * D_lag)}
    for c in ds.clips.values():
        gt = ds.gt_pixels(c)
        X = ds.gt_world(c)
        depth = ds.gt_depth(c)
        vis = ds.gt_visible(c, full_visible)
        mode = "ray" if c.intervention == "angle" else "depth_scaled"
        e_noise = np.mean([
            reference_error(gt + rng.normal(0, noise_px, gt.shape), gt, X, c.camera, depth, vis, ev, mode, None,
                            median_k)["err_m"]
            for _ in range(n_mc)
        ])
        e_lag = reference_error(_lag(gt), gt, X, c.camera, depth, vis, ev, mode, None, median_k)["err_m"]
        out["clips"][c.id] = {"mode": mode, "ceil_m": float(e_noise), "ref_m": float(e_lag),
                              "penalty_m": float(10 * e_lag)}
    return out


def load_or_calibrate(ds: Dataset, cache: Path | None = None, **kw) -> dict:
    cache = cache or (ds.root / "calibration.json")
    if cache.exists():
        cal = json.loads(cache.read_text())
        if cal.get("noise_px") == kw.get("noise_px", 0.3):
            return cal
    cal = calibrate(ds, **kw)
    cache.write_text(json.dumps(cal, indent=2))
    return cal
