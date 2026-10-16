"""Put a model's output on the benchmark's clock and pixel grid.

Video APIs differ in what they return: some return only the new frames, some return the
input followed by the continuation, some repeat the last input frame first, and many use
their own frame rate and resolution. ``align_output`` handles all of these and records
what it did, so every generated video ends up as

    frame 0 = the last conditioning frame, then the continuation,

with a timestamp (seconds after the last conditioning frame) for every frame and a scale
from native to reference pixels. Trajectories are later resampled to the 24 fps grid by
timestamp, never by frame index, so a 16 fps or 30 fps output is compared correctly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class AlignedVideo:
    frames: np.ndarray  # (T, h, w, 3): frame 0 = last conditioning frame, then generated frames
    times: np.ndarray  # (T,) seconds after the last conditioning frame
    scale: tuple[float, float]  # native pixel * scale = reference pixel
    info: dict = field(default_factory=dict)


def _roi_gray(frames: np.ndarray, roi: tuple[int, int, int, int] | None, size=(96, 96)) -> np.ndarray:
    out = []
    for f in frames:
        g = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
        if roi is not None:
            x0, y0, x1, y1 = roi
            g = g[max(0, y0):max(y0 + 1, y1), max(0, x0):max(x0 + 1, x1)]
        out.append(cv2.resize(g, size, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0)
    return np.stack(out)


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def _replay_cost(g_out: np.ndarray, g_in: np.ndarray, start: int, ratio: float, n: int) -> float:
    """Mean ROI difference if output frame k replays input frame start + k * ratio (k < n)."""
    ks = np.arange(min(n, len(g_out)))
    js = start + np.round(ks * ratio).astype(int)
    ok = js < len(g_in)
    if ok.sum() < 3:
        return float("inf")
    return float(np.mean([_mse(g_out[k], g_in[j]) for k, j in zip(ks[ok], js[ok])]))


def align_output(out_frames: np.ndarray, out_fps: float, input_frames: np.ndarray, ref_fps: float,
                 roi_ref: tuple[int, int, int, int] | None = None, includes_input: str | bool = "auto",
                 extra_offset: int = 0, expected_duration_s: float = 4.0,
                 repeats_last_input: bool = False) -> AlignedVideo:
    """Align a generated video to the end of its conditioning clip.

    out_frames: model output (T, h, w, 3) at its native size and ``out_fps``.
    input_frames: the conditioning frames (N, H, W, 3) at reference size and ``ref_fps``.
    roi_ref: (x0, y0, x1, y1) reference-pixel box around the ball's path in the input;
        matching is done inside it, where the moving ball makes frames distinguishable.
    includes_input: "auto" | True | False -- whether the output starts with the input.
        Replaying the input is unambiguous (the first output frames match the input's
        sequence), so it is detected automatically, including an API that keeps only the
        last part of the input. The replay fixes the clock: output frame k shows time
        k / out_fps after the input frame the replay starts with.
    repeats_last_input: set True for an API whose continuation starts with a copy of the
        last input frame. NOT auto-detected: a first frame equal to the last input frame
        is also what a model that lags (or freezes) produces, and silently dropping it
        would hide exactly the errors the benchmark measures. The match is still recorded
        (``first_frame_matches_last_input``) so it can be checked on the first outputs.
    extra_offset: manual shift for odd APIs: the continuation starts this many output frames
        later than detected.

    Frame 0 of the result is always the (resized) last input frame itself at t = 0, where the
    ground truth is exact; the tracker is initialised there. Replayed frames are dropped.
    """
    out_frames = np.asarray(out_frames)
    T, h, w = out_frames.shape[:3]
    N, H, W = input_frames.shape[:3]
    scale = (W / w, H / h)
    aspect_ok = abs((w / h) - (W / H)) < 0.01 * (W / H)
    inp = np.stack([cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) for f in input_frames]) if (w, h) != (W, H) else input_frames
    roi = None
    if roi_ref is not None:
        x0, y0, x1, y1 = roi_ref
        roi = (int(x0 / scale[0]), int(y0 / scale[1]), int(np.ceil(x1 / scale[0])), int(np.ceil(y1 / scale[1])))
    ratio = ref_fps / out_fps  # input frames per output frame
    n_replay_max = int(np.ceil(N / ratio)) + 8
    g_out = _roi_gray(out_frames[: min(T, n_replay_max + 8)], roi)
    g_in = _roi_gray(inp, roi)
    last_in = g_in[-1]
    one_step = max(_mse(g_in[-1], g_in[-2]), 1e-8)  # how much one input frame of motion changes the ROI

    info: dict = {"out_fps": float(out_fps), "out_frames": int(T), "out_size": [int(w), int(h)],
                  "aspect_ok": bool(aspect_ok)}
    # where would a replay start? match the first ~0.5 s of output against the input sequence
    n_match = max(6, int(round(0.5 * out_fps)))
    starts = range(0, max(1, N - 5))
    costs = np.array([_replay_cost(g_out, g_in, s, ratio, n_match) for s in starts])
    best = int(np.argmin(costs))
    # prefer a replay of the whole input unless a later start is clearly better
    replay_start = 0 if costs[0] <= 2.0 * costs[best] + 1e-9 else best
    d_last = _mse(g_out[0], last_in)
    detected = includes_input
    if includes_input == "auto":
        detected = bool(costs[replay_start] < 0.5 * d_last and len(g_out) > (N - replay_start) / ratio / 2)
        info["auto_detect"] = {"replay_cost": float(costs[replay_start]), "mse_first_vs_last_input": d_last}
    info["includes_input"] = bool(detected)

    if detected:
        # output frame k shows time k / out_fps after input frame replay_start
        t_out = (np.arange(T) - int(extra_offset)) / float(out_fps) + (replay_start - (N - 1)) / float(ref_fps)
        keep = t_out > 1e-6  # replayed input (t <= 0) is dropped; frame 0 is the input itself
        cont, t_cont = out_frames[keep], t_out[keep]
        k_last = int(np.argmin(np.abs(np.where(t_out <= 1e-6, t_out, -np.inf))))
        info["replay_starts_at_input_frame"] = int(replay_start)
        info["last_input_index_in_output"] = k_last
        # diagnostic: the replayed frame that best matches the last input frame should be
        # where the clock puts it; a mismatch means the API's frame rate is not what it says
        lo, hi = max(0, k_last - 6), min(len(g_out) - 1, k_last + 6)
        k_best = lo + int(np.argmin([_mse(g_out[k], last_in) for k in range(lo, hi + 1)]))
        info["replay_clock_mismatch_frames"] = int(k_best - k_last)
    else:
        # continuation only
        info["first_frame_matches_last_input"] = bool(d_last < 0.25 * one_step)
        info["repeats_last_input"] = bool(repeats_last_input)
        start = (1 if repeats_last_input else 0) + int(extra_offset)
        cont = out_frames[start:]
        t_cont = np.arange(1, len(cont) + 1) / float(out_fps)
    frames = np.concatenate([inp[-1][None], cont], axis=0)
    times = np.concatenate([[0.0], t_cont])
    duration = float(times[-1]) if len(times) > 1 else 0.0
    info["first_generated_time_s"] = round(float(t_cont[0]), 5) if len(t_cont) else None
    info["continuation_s"] = round(duration, 4)
    info["short_output"] = bool(duration < expected_duration_s - 0.5 / ref_fps)
    return AlignedVideo(frames=frames, times=times, scale=scale, info=info)


def camera_motion(frames: np.ndarray, exclude_uv: np.ndarray | None = None, exclude_radius: float = 0.0,
                  max_corners: int = 300, min_good: int = 12, fb_tol_px: float = 1.0, step: int = 2) -> dict:
    """How far the image content moves relative to frame 0 (pixels of these frames).

    Tracks background corners from frame 0 with pyramidal Lucas-Kanade (well-conditioned
    corners only, forward-backward checked, away from the ball: ``exclude_uv`` (T, 2),
    ``exclude_radius``), fits a similarity motion (shift, zoom, roll) with RANSAC, and
    reports the largest, over frames, of the median displacement of a grid of image points
    under that motion.

    The result is None ("unknown") when the background has too little 2D texture to tell:
    a plain wall and floor only constrain motion across their boundary (aperture problem),
    and whole-frame methods (phase correlation, ECC) then lock onto the ball or noise.
    """
    g0 = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
    H, W = g0.shape
    mask = np.full(g0.shape, 255, np.uint8)
    rad = int(exclude_radius * 2 + 4)
    if exclude_uv is not None and np.all(np.isfinite(exclude_uv[0])):
        cv2.circle(mask, (int(exclude_uv[0][0]), int(exclude_uv[0][1])), rad, 0, -1)
    pts = cv2.goodFeaturesToTrack(g0, maxCorners=max_corners, qualityLevel=0.01, minDistance=8, mask=mask)
    n_feat = 0 if pts is None else len(pts)
    gy, gx = np.mgrid[0:8, 0:8]
    grid = np.stack([(gx.ravel() + 0.5) * W / 8, (gy.ravel() + 0.5) * H / 8], 1)
    disp, n_used = [], []
    ts = list(range(step, len(frames), step))
    if len(frames) > 1 and (not ts or ts[-1] != len(frames) - 1):
        ts.append(len(frames) - 1)
    if n_feat >= min_good:
        p0 = pts.reshape(-1, 2)
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        kw = dict(winSize=(21, 21), maxLevel=3, criteria=crit)  # default eigenvalue threshold
        for t in ts:
            g = cv2.cvtColor(frames[t], cv2.COLOR_RGB2GRAY)
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(g0, g, pts, None, **kw)
            back, st2, _ = cv2.calcOpticalFlowPyrLK(g, g0, nxt, None, **kw)
            good = (st.ravel() == 1) & (st2.ravel() == 1)
            good &= np.linalg.norm((back - pts).reshape(-1, 2), axis=1) < fb_tol_px
            if exclude_uv is not None and t < len(exclude_uv) and np.all(np.isfinite(exclude_uv[t])):
                good &= np.linalg.norm(p0 - exclude_uv[t], axis=1) > rad  # the ball passed over it
            if good.sum() < min_good:
                continue
            # similarity (shift, zoom, roll): unlike a full affine it stays well-posed when the
            # usable corners lie along one edge, e.g. where the floor meets the wall
            A, inl = cv2.estimateAffinePartial2D(p0[good], nxt.reshape(-1, 2)[good], method=cv2.RANSAC,
                                                 ransacReprojThreshold=1.0)
            if A is None or inl.sum() < min_good or inl.mean() < 0.5:
                continue
            moved = grid @ A[:, :2].T + A[:, 2]
            disp.append(float(np.median(np.linalg.norm(moved - grid, axis=1))))
            n_used.append(int(inl.sum()))
    return {
        "n_features": int(n_feat),
        "frames_checked": len(disp),
        "frames_total": len(ts),
        "min_features_used": min(n_used) if n_used else 0,
        "max_median_disp_px": max(disp) if len(disp) >= max(1, len(ts) // 2) else None,
    }
