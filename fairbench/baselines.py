"""Trajectory-level baselines: reference points for every table in the paper.

They skip video generation and tracking and produce ball tracks directly, so they show
where each metric sits for known behaviours:

* ``oracle``             the simulator's continuation (metric ceilings).
* ``freeze``             the ball stops where the input ends. Perfectly robust to every
                         intervention, physically wrong: robustness cannot stand alone.
* ``constant_velocity``  the last input velocity, no gravity, no floor.
* ``physics_from_input`` reads speed and bounciness off the 2 s input and simulates
                         forward with g. Shows the future really is fixed by the input:
                         a ten-line extrapolator should come close to the oracle.

``noise_px`` adds tracker-like noise so baseline numbers are comparable with tracked videos.
"""

from __future__ import annotations

import numpy as np

from .manifest import MOTION_KIND, Dataset
from .metrics.plausibility import detect_contacts
from .score import TrackRecord

BASELINES = ("oracle", "freeze", "constant_velocity", "physics_from_input")


def _split(X: np.ndarray, up: np.ndarray):
    h = X @ up
    return X - np.outer(h, up), h


def _flight_fit(t: np.ndarray, h: np.ndarray, g: float, t_ref: float) -> tuple[float, float]:
    """Height h0 and upward speed w at t_ref of a gravity-constrained parabola."""
    tau = t - t_ref
    A = np.stack([np.ones_like(tau), tau], 1)
    c0, w = np.linalg.lstsq(A, h + 0.5 * g * tau**2, rcond=None)[0]
    return float(c0), float(w)


def _contact_speed(h0: float, w: float, g: float, contact_h: float) -> float:
    """Upward speed of a flight as it passes the contact height (energy is constant in flight)."""
    return float(np.sqrt(max(w * w + 2 * g * (h0 - contact_h), 0.0)))


def _last_flight_state(ds: Dataset, P: np.ndarray, contact_h: float):
    """State at the last input frame from gravity-constrained fits of the input's flights."""
    up, g = ds.up, ds.g
    t = np.arange(ds.input_frames) / ds.fps
    t_last = t[-1]
    Xh, h = _split(P[: ds.input_frames], up)
    minima = detect_contacts(h, contact_h, 0.03)
    start = (minima[-1] + 1) if minima else 0
    seg = slice(start, ds.input_frames)
    h0, w0 = _flight_fit(t[seg], h[seg], g, t_last)
    tau = t[seg] - t_last
    cx = np.linalg.lstsq(np.stack([np.ones_like(tau), tau], 1), Xh[seg], rcond=None)[0]  # X = X0 + v t
    e = None
    if len(minima) >= 2:
        prev = slice(minima[-2] + 1, minima[-1])
        hp, wp = _flight_fit(t[prev], h[prev], g, t_last)
        v_prev = _contact_speed(hp, wp, g, contact_h)
        v_last = _contact_speed(h0, w0, g, contact_h)
        if v_prev > 1e-3:
            e = float(np.clip(v_last / v_prev, 0.0, 1.0))  # successive post-contact speeds
    return cx[0], cx[1], h0, w0, e


def _simulate(ds: Dataset, X0h, vh, h0, w0, e, contact_h, n):
    """Ballistic flight with floor contacts from the last conditioning frame."""
    up, g = ds.up, ds.g
    out = np.zeros((n + 1, 3))
    t_all = np.arange(n + 1) / ds.fps
    tc, hc, wc, resting = 0.0, h0, w0, False
    k = 0
    while k <= n:
        if resting or e is None:
            hh = np.full(n + 1 - k, contact_h) if resting else hc + wc * (t_all[k:] - tc) - 0.5 * g * (t_all[k:] - tc) ** 2
            out[k:] = X0h + np.outer(t_all[k:], vh) + np.outer(np.maximum(hh, contact_h), up)
            break
        disc = wc**2 + 2 * g * max(hc - contact_h, 0)
        tau = (wc + np.sqrt(disc)) / g
        while k <= n and t_all[k] < tc + tau:
            dt = t_all[k] - tc
            out[k] = X0h + vh * t_all[k] + (hc + wc * dt - 0.5 * g * dt * dt) * up
            k += 1
        w_down = wc - g * tau
        tc, hc, wc = tc + tau, contact_h, -e * w_down
        if wc < 0.05:
            resting = True
    return out


def baseline_world(ds: Dataset, motion: str, name: str) -> np.ndarray:
    """World trajectory (n_gen+1, 3) on the eval grid for one baseline."""
    P = ds.motions[motion].positions
    n = ds.n_gen
    last = ds.input_frames - 1
    t = np.arange(n + 1) / ds.fps
    if name == "oracle":
        return P[last:].copy()
    if name == "freeze":
        return np.repeat(P[last][None], n + 1, axis=0)
    if name == "constant_velocity":
        v = (P[last] - P[last - 3]) / (3 / ds.fps)
        return P[last] + np.outer(t, v)
    if name == "physics_from_input":
        ch = ds.floor_contact_height()
        if MOTION_KIND.get(motion) == "rolling":
            v = (P[last] - P[last - 12]) / (12 / ds.fps)
            return P[last] + np.outer(t, v)
        X0h, vh, h0, w0, e = _last_flight_state(ds, P, ch)
        if e is None:
            e = float((ds.motions[motion].events or {}).get("restitution", 0.8))
        return _simulate(ds, X0h, vh, h0, w0, e, ch, n)
    raise ValueError(f"unknown baseline {name!r}")


def baseline_tracks(ds: Dataset, name: str, noise_px: float = 0.0, samples: int = 1,
                    seed: int = 0) -> dict[str, list[TrackRecord]]:
    rng = np.random.default_rng(seed)
    cache: dict[str, np.ndarray] = {}
    out = {}
    for cid, c in sorted(ds.clips.items()):
        if c.motion not in cache:
            cache[c.motion] = baseline_world(ds, c.motion, name)
        uv = c.camera.project(cache[c.motion])
        recs = []
        for k in range(samples):
            u = uv + (rng.normal(0, noise_px, uv.shape) if noise_px > 0 else 0.0)
            recs.append(TrackRecord(cid, k, u, np.ones(len(u)), {"source": f"baseline:{name}", "noise_px": noise_px}))
        out[cid] = recs
    return out
