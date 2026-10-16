"""Physical plausibility of one generated trajectory, borrowed from Morpheus.

Morpheus (Tragoudaras et al., arXiv 2504.02918; code: github.com/physics-from-video/Morpheus)
scores a generated object track against the laws of its setup, never against a
reference video:

* Dynamical score: fit the trajectory with a model that obeys the equations of motion;
  S_dyn = max(1 - NMSE, 0), NMSE = squared fit error / variance of the trajectory.
* Physical Invariance score: in sliding windows, measure how constant the quantities
  the setup conserves stay (energy per mass, vertical acceleration = g, horizontal
  velocity); per window S = 1 / (1 + std / |mean|)  (or 1 / (1 + std) when the mean is
  near zero), best window per quantity, mean over quantities.
* Videos where the object vanishes, duplicates or never moves are discarded (score 0).

Adaptations for FAIR-Bench (each one is a switch or documented in docs/METRICS.md):

1. Units. Morpheus has no calibration; our cameras are known, so each tracked pixel is
   lifted onto the ground-truth motion plane and everything is in meters and seconds,
   with the scene's g.
2. Bounces. A thrown or bouncing ball hits the floor several times in the 4 s
   continuation, so the trajectory is split at floor contacts and the equations of motion
   ("constant gravitational acceleration in flight") are applied per flight; windows for
   the invariance score stay inside one flight.
3. Fit. ``method="pinn"`` trains a Morpheus-style PINN per segment (soft physics
   constraint, lambda = 1). ``method="lsq"`` imposes the same equations exactly (closed-form
   least squares, no lambda); it is fast and deterministic and is always reported too.
4. Extra diagnostic (not in Morpheus): energy gained at a floor contact, which neither
   score can see because each flight gets its own initial velocity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import savgol_filter

from ..geometry import Camera, Plane, intersect_rays_plane


@dataclass
class PlausibilityConfig:
    method: str = "pinn"  # primary dynamical-score method: "pinn" | "lsq"
    pinn_epochs: int = 20000
    pinn_lr: float = 1e-3
    pinn_hidden: int = 20
    pinn_lambda: float = 1.0
    pinn_patience: int = 1000
    pinn_phys_points: int = 50
    contact_tol_m: float = 0.03
    min_segment_frames: int = 4
    velocity_window: int = 5
    invariance_alpha: float = 1.0
    window_frac: float = 0.25
    invariance_reduce: str = "max"  # "max" (Morpheus: best window) | "mean" (over segments)
    discard_lost_frac: float = 0.2
    discard_multi_frac: float = 0.1
    discard_still_frac: float = 0.1  # never gets this share as far as the GT ball = 'never moves'
    energy_gain_tol: float = 0.10  # relative energy gain at a contact counted as a violation
    seed: int = 0
    extra: dict = field(default_factory=dict)


# ------------------------------------------------------------------ lifting to meters
def plane_basis(plane: Plane, up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(e_s, e_h): horizontal in-plane direction and vertical direction."""
    e_s = np.cross(up, plane.normal)
    if np.linalg.norm(e_s) < 1e-6:  # horizontal plane: any horizontal axis
        e_s = np.cross(up, [1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.cross(up, [0.0, 1.0, 0.0])
    e_s /= np.linalg.norm(e_s)
    return e_s, np.asarray(up, dtype=float)


def lift_to_plane(uv: np.ndarray, cam: Camera, plane: Plane, up: np.ndarray) -> np.ndarray:
    """Tracked pixels (T, 2) -> in-plane coordinates (T, 2): s (horizontal), h (height)."""
    X = intersect_rays_plane(cam.C, cam.rays(uv), plane)
    e_s, e_h = plane_basis(plane, up)
    return np.stack([X @ e_s, X @ e_h], axis=1)


# ----------------------------------------------------------------------- segmentation
def _fill_small_gaps(x: np.ndarray, max_gap: int = 2) -> np.ndarray:
    x = np.array(x, dtype=float, copy=True)
    ok = np.isfinite(x)
    if ok.sum() < 2:
        return x
    idx = np.arange(len(x))
    filled = np.interp(idx, idx[ok], x[ok])
    # only fill runs of NaN no longer than max_gap that are bounded on both sides
    run_start = None
    for i in range(len(x) + 1):
        missing = i < len(x) and not ok[i]
        if missing and run_start is None:
            run_start = i
        elif not missing and run_start is not None:
            if run_start > 0 and i < len(x) and i - run_start <= max_gap:
                x[run_start:i] = filled[run_start:i]
            run_start = None
    return x


def detect_contacts(h: np.ndarray, contact_h: float, tol: float) -> list[int]:
    """Frames of floor contacts: sharp local minima of the height near the floor.

    The sampled minimum can sit above the true contact by up to half a frame of fall, so
    "near" allows tol + 0.6 x the one-frame drop next to it. Dips smaller than tol/2 per
    frame (noise, tiny final hops) are ignored.
    """
    ok = np.isfinite(h)
    out: list[int] = []
    for t in range(1, len(h) - 1):
        if not (ok[t - 1] and ok[t] and ok[t + 1]):
            continue
        a, b, c = h[t - 1], h[t], h[t + 1]
        if b <= a and b <= c and (b < a or b < c):
            drop = max(a - b, c - b)
            if drop > 0.5 * tol and b - contact_h <= tol + 0.6 * drop:
                if not out or t - out[-1] >= 2:
                    out.append(t)
    return out


@dataclass
class Segment:
    start: int  # inclusive eval-grid index
    stop: int  # exclusive
    kind: str  # "flight" | "roll"


def segment_ballistic(h: np.ndarray, contact_h: float, tol: float, min_len: int) -> tuple[list[Segment], list[int]]:
    """Split a height track at floor contacts and at the start/end of resting phases.

    A contact is a sharp local minimum of the height near the floor. The sampled minimum
    can sit above the true contact by up to half a frame of fall, so "near" allows
    ``tol + 0.6 x`` the one-frame drop next to it. A minimum high above the floor is NOT a
    contact (a mid-air reversal is a violation, and stays inside one flight to be
    penalized). Dips smaller than tol/2 per frame (noise, tiny final hops) are not split;
    those frames end up in "roll" segments (height constant).
    Returns (segments, contact frame indices).
    """
    hf = _fill_small_gaps(h)
    n = len(hf)
    ok = np.isfinite(hf)
    contacts = detect_contacts(hf, contact_h, tol)
    # resting phases: runs of >= 4 frames on the floor
    rest = ok & (hf - contact_h <= tol)
    bounds = {0, n, *contacts}
    t = 0
    while t < n:
        if rest[t]:
            u = t
            while u < n and rest[u]:
                u += 1
            if u - t >= 4:
                bounds.update({t, u})
            t = u
        else:
            t += 1
    bounds = sorted(bounds)
    segs = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        a2 = a + 1 if a in contacts else a  # leave the contact frame out of both flights
        if b - a2 < min_len:
            continue
        seg = hf[a2:b]
        seg = seg[np.isfinite(seg)]
        if len(seg) < min_len:
            continue
        kind = "roll" if np.mean(seg - contact_h <= 1.5 * tol) >= 0.8 else "flight"
        segs.append(Segment(a2, b, kind))
    return segs, contacts


# --------------------------------------------------------------------- dynamical: LSQ
def _lsq_segment(t: np.ndarray, s: np.ndarray, h: np.ndarray, kind: str, g: float) -> tuple[np.ndarray, np.ndarray]:
    """Best fit obeying the equations of motion exactly (g fixed)."""
    tt = t - t.mean()
    if kind == "flight":
        A = np.stack([np.ones_like(tt), tt], axis=1)
        cs = np.linalg.lstsq(A, s, rcond=None)[0]
        ch = np.linalg.lstsq(A, h + 0.5 * g * tt**2, rcond=None)[0]
        return A @ cs, A @ ch - 0.5 * g * tt**2
    # roll: height constant, speed may change at a constant rate (friction)
    A = np.stack([np.ones_like(tt), tt, 0.5 * tt**2], axis=1)
    cs = np.linalg.lstsq(A, s, rcond=None)[0]
    return A @ cs, np.full_like(h, np.mean(h))


# -------------------------------------------------------------------- dynamical: PINN
def _pinn_segments(segs: list[tuple[np.ndarray, np.ndarray, np.ndarray, str]], g: float,
                   cfg: PlausibilityConfig) -> list[tuple[np.ndarray, np.ndarray]]:
    """Morpheus-style PINN fit of every segment of one track (soft physics constraint).

    Each segment has its own network (1 -> H -> H -> 2, tanh), its own Adam state and its
    own early stopping, exactly as if it were trained alone; the networks are only batched
    into one set of tensors so that a track with ten segments costs about one training run.
    (Adam is element-wise and the loss is a sum over segments, so the per-segment updates
    are unchanged.) Inputs per segment: (t, s, h, kind) with kind "flight" or "roll".
    """
    import torch

    S, H, P = len(segs), cfg.pinn_hidden, cfg.pinn_phys_points
    torch.manual_seed(cfg.seed)  # every segment starts from the same initial network
    proto = torch.nn.Sequential(torch.nn.Linear(1, H), torch.nn.Tanh(), torch.nn.Linear(H, H), torch.nn.Tanh(),
                                torch.nn.Linear(H, 2))
    rep = lambda w: torch.nn.Parameter(w.detach().clone().unsqueeze(0).repeat(S, *([1] * w.dim())))
    W1, b1 = rep(proto[0].weight.T), rep(proto[0].bias)  # (S,1,H), (S,H)
    W2, b2 = rep(proto[2].weight.T), rep(proto[2].bias)  # (S,H,H), (S,H)
    W3, b3 = rep(proto[4].weight.T), rep(proto[4].bias)  # (S,H,2), (S,2)
    a_roll = torch.nn.Parameter(torch.zeros(S, 1, 1))  # learnable along-track acceleration (Morpheus slide)

    def net(x):  # x: (S, n, 1)
        z = torch.tanh(torch.bmm(x, W1) + b1[:, None])
        z = torch.tanh(torch.bmm(z, W2) + b2[:, None])
        return torch.bmm(z, W3) + b3[:, None]

    n_max = max(len(t) for t, *_ in segs)
    td = torch.zeros(S, n_max, 1)
    yd = torch.zeros(S, n_max, 2)
    md = torch.zeros(S, n_max, 1)
    tp = torch.zeros(S, P, 1)
    roll = torch.zeros(S, 1, 1, dtype=torch.bool)
    offs = []
    for i, (t, s, h, kind) in enumerate(segs):
        t0 = float(t.min())
        T = max(float(t.max() - t0), 1e-3)
        s0, h0 = float(np.mean(s)), float(np.mean(h))
        offs.append((s0, h0))
        n = len(t)
        td[i, :n, 0] = torch.tensor(t - t0, dtype=torch.float32)
        yd[i, :n] = torch.tensor(np.stack([s - s0, h - h0], 1), dtype=torch.float32)
        md[i, :n] = 1.0
        tp[i, :, 0] = torch.linspace(0.0, T, P)
        roll[i] = kind == "roll"
    n_i = md.sum(dim=(1, 2))  # points per segment
    opt = torch.optim.Adam([W1, b1, W2, b2, W3, b3, a_roll], lr=cfg.pinn_lr)
    best = torch.full((S,), float("inf"), dtype=torch.float64)
    bad = torch.zeros(S, dtype=torch.long)
    done = torch.zeros(S, dtype=torch.bool)
    preds: list = [None] * S
    for _ in range(cfg.pinn_epochs):
        opt.zero_grad()
        tq = tp.clone().requires_grad_(True)
        y = net(tq)
        ys, yh = y[..., :1], y[..., 1:]
        ds = torch.autograd.grad(ys, tq, torch.ones_like(ys), create_graph=True)[0]
        dh = torch.autograd.grad(yh, tq, torch.ones_like(yh), create_graph=True)[0]
        dds = torch.autograd.grad(ds, tq, torch.ones_like(ds), create_graph=True)[0]
        ddh = torch.autograd.grad(dh, tq, torch.ones_like(dh), create_graph=True)[0]
        res = torch.where(roll, (dds - a_roll) ** 2 + ddh**2, dds**2 + (ddh + g) ** 2)
        phys = res.mean(dim=(1, 2))
        data = (((net(td) - yd) ** 2) * md).sum(dim=(1, 2)) / (2.0 * n_i)
        loss = data + cfg.pinn_lambda * phys  # (S,)
        loss.sum().backward()
        opt.step()
        lv = loss.detach().double()
        improved = lv < best - 1e-9
        best = torch.where(improved, lv, best)
        bad = torch.where(improved, torch.zeros_like(bad), bad + 1)
        stop = (~done) & (bad >= cfg.pinn_patience)
        if stop.any():
            with torch.no_grad():
                out = net(td)
            for i in torch.nonzero(stop).ravel().tolist():
                preds[i] = out[i].numpy().copy()
            done |= stop
            if bool(done.all()):
                break
    if not bool(done.all()):
        with torch.no_grad():
            out = net(td)
        for i in range(S):
            if preds[i] is None:
                preds[i] = out[i].numpy().copy()
    result = []
    for (t, *_), (s0, h0), p in zip(segs, offs, preds):
        n = len(t)
        result.append((p[:n, 0] + s0, p[:n, 1] + h0))
    return result


def dynamical_score(t, s, h, segments: list[Segment], g: float, method: str, cfg: PlausibilityConfig) -> dict:
    if method not in ("lsq", "pinn"):
        raise ValueError(f"unknown dynamical-score method {method!r}")
    data = []
    for seg in segments:
        sl = slice(seg.start, seg.stop)
        ok = np.isfinite(s[sl]) & np.isfinite(h[sl])
        if ok.sum() >= 3:
            data.append((t[sl][ok], s[sl][ok], h[sl][ok], seg.kind))
    if method == "lsq":
        fits = [_lsq_segment(ts, ss, hh, kind, g) for ts, ss, hh, kind in data]
    else:
        fits = _pinn_segments(data, g, cfg) if data else []
    res, used = [], []
    for (ts, ss, hh, _), (ps, ph) in zip(data, fits):
        res.append(np.stack([ss - ps, hh - ph], 1))
        used.append(np.stack([ss, hh], 1))
    if not res:
        return {"S_dyn": 0.0, "NMSE": float("nan"), "n_frames": 0}
    R = np.concatenate(res)
    Y = np.concatenate(used)
    var = np.sum((Y - Y.mean(axis=0)) ** 2)
    nmse = float(np.sum(R**2) / max(var, 1e-12))
    return {"S_dyn": float(max(1.0 - nmse, 0.0)), "NMSE": nmse, "n_frames": int(len(R))}


# --------------------------------------------------------------- invariance (Morpheus)
def morpheus_velocity(x: np.ndarray, t: np.ndarray, window: int = 5) -> np.ndarray:
    """Morpheus' velocity estimate: 0.7 x local linear regression + 0.3 x central difference,
    then a Savitzky-Golay filter (order 3)."""
    window = window + 1 if window % 2 == 0 else window
    n = len(x)
    vc = np.zeros(n)
    vc[1:-1] = (x[2:] - x[:-2]) / (t[2:] - t[:-2])
    vc[0] = (x[1] - x[0]) / (t[1] - t[0])
    vc[-1] = (x[-1] - x[-2]) / (t[-1] - t[-2])
    vr = np.zeros(n)
    hw = window // 2
    for i in range(n):
        a, b = max(0, i - hw), min(n, i + hw + 1)
        A = np.stack([t[a:b] - t[i], np.ones(b - a)], 1)
        vr[i] = np.linalg.lstsq(A, x[a:b], rcond=None)[0][0]
    v = 0.7 * vr + 0.3 * vc
    if n >= window:
        v = savgol_filter(v, window, 3)
    return v


def morpheus_constancy(values: np.ndarray, alpha: float = 1.0) -> float:
    """Morpheus' conservation score for one window of an invariant's time series."""
    m, sd = float(np.mean(values)), float(np.std(values))
    if abs(m) >= 10 * abs(sd) or abs(m) >= 10:
        return 1.0 / (1.0 + alpha * sd / (abs(m) + 1e-10))
    return 1.0 / (1.0 + alpha * sd)


def invariance_score(t, s, h, segments: list[Segment], g: float, contact_h: float, n_total: int,
                     cfg: PlausibilityConfig) -> dict:
    L = max(5, int(cfg.window_frac * n_total))
    per_inv: dict[str, list[float]] = {"energy": [], "accel": [], "momentum": []}
    seg_means: dict[str, list[float]] = {"energy": [], "accel": [], "momentum": []}
    energies = []
    for seg in segments:
        idx = np.arange(seg.start, seg.stop)
        ok = np.isfinite(s[idx]) & np.isfinite(h[idx])
        idx = idx[ok]
        if len(idx) < 5:
            continue
        # contiguous runs only (velocities across gaps are meaningless)
        for run in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1):
            if len(run) < 5:
                continue
            tt, ss, hh = t[run], s[run], h[run]
            vs = morpheus_velocity(ss, tt, cfg.velocity_window)
            vh = morpheus_velocity(hh, tt, cfg.velocity_window)
            ah = np.gradient(vh, tt)
            as_ = np.gradient(vs, tt)
            E = 0.5 * (vs**2 + vh**2) + g * (hh - contact_h)
            if seg.kind == "flight":
                series = {"energy": E, "accel": ah, "momentum": vs}
            else:
                series = {"energy": E, "accel": as_, "momentum": vs}
            energies.append((run[0], run[-1], float(np.median(E))))
            # drop the 2 frames next to each end (velocity smoothing reaches across contacts)
            core = slice(2, len(run) - 2) if len(run) >= 9 else slice(0, len(run))
            w = min(L, len(run[core]))
            step = max(w // 2, 1)
            for name, x in series.items():
                xc = x[core]
                if len(xc) < 3:
                    continue
                scores = [morpheus_constancy(xc[a:a + w], cfg.invariance_alpha) for a in range(0, len(xc) - w + 1, step)]
                per_inv[name] += scores
                seg_means[name].append(float(np.mean(scores)))
    if not any(per_inv.values()):
        return {"S_inv": 0.0, "parts": {}, "energy_gain_violations": 0, "n_contacts_checked": 0}
    if cfg.invariance_reduce == "mean":
        parts = {k: float(np.mean(v)) for k, v in seg_means.items() if v}
    else:
        parts = {k: float(np.max(v)) for k, v in per_inv.items() if v}
    viol, checked = 0, 0
    for (a0, a1, Ea), (b0, b1, Eb) in zip(energies[:-1], energies[1:]):
        checked += 1
        if Eb > Ea * (1 + cfg.energy_gain_tol) + 1e-6:
            viol += 1
    return {"S_inv": float(np.mean(list(parts.values()))), "parts": parts,
            "energy_gain_violations": viol, "n_contacts_checked": checked}


# ----------------------------------------------------------------------------- entry
def motion_extent(sh: np.ndarray, frames: np.ndarray) -> float:
    """Largest distance (m) of the median-filtered track from its first valid point."""
    from .common import running_median

    x = np.stack([running_median(sh[:, 0], 5), running_median(sh[:, 1], 5)], 1)
    idx = np.where(frames & np.all(np.isfinite(x), axis=1))[0]
    if len(idx) < 2:
        return 0.0
    return float(np.max(np.linalg.norm(x[idx] - x[idx[0]], axis=1)))


def plausibility(uv: np.ndarray, cam: Camera, plane: Plane, up: np.ndarray, g: float, contact_h: float,
                 kind: str, times: np.ndarray, eval_frames: np.ndarray, visible: np.ndarray,
                 n_components: np.ndarray | None, gt_extent_m: float, cfg: PlausibilityConfig,
                 methods: tuple[str, ...] = ("lsq", "pinn")) -> dict:
    """Plausibility of one generated track on the eval grid (frame 0 included as context).

    ``gt_extent_m`` is how far the simulator's ball gets from its frame-0 position; a
    generated ball that never gets 10% as far "never moves" (spatial extent, not path
    length, so tracker jitter cannot make a frozen ball look like it moves).
    """
    sh = lift_to_plane(uv, cam, plane, up)
    s, h = sh[:, 0].copy(), sh[:, 1].copy()
    ev = eval_frames.copy()
    found = np.isfinite(s) & np.isfinite(h)
    n_vis = int(np.sum(visible & ev))
    lost_frac = float(np.sum(visible & ev & ~found) / max(n_vis, 1))
    multi_frac = float(np.mean(n_components[ev] > 1)) if n_components is not None else 0.0
    extent = motion_extent(sh, ev)
    out = {"lost_frac": lost_frac, "multi_frac": multi_frac, "extent_m": extent}
    reason = None
    if lost_frac > cfg.discard_lost_frac:
        reason = "disappears"
    elif multi_frac > cfg.discard_multi_frac:
        reason = "duplicates"
    elif gt_extent_m > 0 and extent < cfg.discard_still_frac * gt_extent_m:
        reason = "never_moves"
    out["discard"] = reason
    if reason is not None:
        out.update({"S_dyn": 0.0, "S_dyn_lsq": 0.0, "S_dyn_pinn": 0.0, "S_inv": 0.0})
        return out
    # evaluate on generated frames (+ frame 0 as the start of the first segment)
    sel = np.where(ev | (np.arange(len(ev)) == 0))[0]
    s_e, h_e, t_e = s[sel], h[sel], times[sel]
    if kind == "rolling":
        segs, contacts = [Segment(0, len(sel), "roll")], []
        # a rolling ball that leaves the floor is split like a bounce
        if np.nanmax(h_e) - contact_h > 3 * cfg.contact_tol_m:
            segs, contacts = segment_ballistic(h_e, contact_h, cfg.contact_tol_m, cfg.min_segment_frames)
    else:
        segs, contacts = segment_ballistic(h_e, contact_h, cfg.contact_tol_m, cfg.min_segment_frames)
    out["n_segments"] = len(segs)
    out["n_contacts"] = len(contacts)
    for m in methods:
        out[f"S_dyn_{m}"] = dynamical_score(t_e, s_e, h_e, segs, g, m, cfg)["S_dyn"]
    out["S_dyn"] = out.get(f"S_dyn_{cfg.method}", out.get("S_dyn_lsq", 0.0))
    inv = invariance_score(t_e, s_e, h_e, segs, g, contact_h, int(ev.sum()), cfg)
    out["S_inv"] = inv["S_inv"]
    out["S_inv_parts"] = inv["parts"]
    out["energy_gain_violations"] = inv["energy_gain_violations"]
    out["n_contacts_checked"] = inv["n_contacts_checked"]
    return out
