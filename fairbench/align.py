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
        Replaying the input is unambiguous (output frame 0 matches input frame 0), so it is
        detected automatically.
    repeats_last_input: set True for an API whose continuation starts with a copy of the
        last input frame. NOT auto-detected: a first frame equal to the last input frame
        is also what a model that lags (or freezes) produces, and silently dropping it
        would hide exactly the errors the benchmark measures. The match is still recorded
        (``first_frame_matches_last_input``) so it can be checked on the first outputs.
    extra_offset: manual shift (frames) applied after detection, for odd APIs.
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
    g_out = _roi_gray(out_frames[: min(T, N + 24)], roi)
    g_in = _roi_gray(inp, roi)
    last_in = g_in[-1]
    one_step = max(_mse(g_in[-1], g_in[-2]), 1e-8)  # how much one input frame of motion changes the ROI

    info: dict = {"out_fps": float(out_fps), "out_frames": int(T), "out_size": [int(w), int(h)],
                  "aspect_ok": bool(aspect_ok)}
    detected = includes_input
    if includes_input == "auto":
        # an output that replays the input matches input frame 0 better than the last frame
        d_first = _mse(g_out[0], g_in[0])
        d_last = _mse(g_out[0], last_in)
        detected = bool(d_first < 0.5 * d_last and len(g_out) > N // 2)
        info["auto_detect"] = {"mse_vs_first_input": d_first, "mse_vs_last_input": d_last}
    info["includes_input"] = bool(detected)

    if detected:
        # index k in the output that corresponds to the last conditioning frame
        n_in_out = int(round(N * out_fps / ref_fps))
        lo, hi = max(0, n_in_out - 6), min(len(g_out) - 1, n_in_out + 6)
        costs = []
        for k in range(lo, hi + 1):
            ctx = [_mse(g_out[k - m], g_in[max(0, N - 1 - int(round(m * ref_fps / out_fps)))])
                   for m in range(0, min(k, 4) + 1)]
            costs.append(np.mean(ctx))
        k0 = lo + int(np.argmin(costs)) + int(extra_offset)
        frame0 = out_frames[k0]
        cont = out_frames[k0 + 1:]
        info["last_input_index_in_output"] = int(k0)
        # the replayed input shares the input's clock: output frame k shows time k / out_fps
        # from the start of the input, so frame k0 may sit slightly off the last input frame
        t0_offset = k0 / float(out_fps) - (N - 1) / float(ref_fps)
        info["frame0_time_offset_s"] = round(t0_offset, 5)
    else:
        # continuation only
        info["first_frame_matches_last_input"] = bool(_mse(g_out[0], last_in) < 0.25 * one_step)
        info["repeats_last_input"] = bool(repeats_last_input)
        start = (1 if repeats_last_input else 0) + int(extra_offset)
        frame0 = inp[-1]
        cont = out_frames[start:]
        t0_offset = 0.0
    frames = np.concatenate([frame0[None], cont], axis=0)
    times = np.arange(len(frames)) / float(out_fps) + t0_offset
    times[0] = 0.0 if not detected else times[0]
    duration = (len(frames) - 1) / float(out_fps)
    info["continuation_s"] = round(duration, 4)
    info["short_output"] = bool(duration < expected_duration_s - 0.5 / ref_fps)
    return AlignedVideo(frames=frames, times=times, scale=scale, info=info)


def camera_motion(frames: np.ndarray, exclude_uv: np.ndarray | None = None, exclude_radius: float = 0.0,
                  max_corners: int = 300) -> dict:
    """How far the background moves relative to frame 0 (pixels of these frames).

    Tracks corner features from frame 0 with pyramidal Lucas-Kanade and reports the
    largest per-frame median displacement. Also returns a phase-correlation estimate, which
    still works on walls with few corners. ``exclude_uv`` (T, 2) masks out the ball.
    """
    g0 = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
    mask = np.full(g0.shape, 255, np.uint8)
    if exclude_uv is not None and np.all(np.isfinite(exclude_uv[0])):
        cv2.circle(mask, (int(exclude_uv[0][0]), int(exclude_uv[0][1])), int(exclude_radius * 2 + 4), 0, -1)
    pts = cv2.goodFeaturesToTrack(g0, maxCorners=max_corners, qualityLevel=0.01, minDistance=8, mask=mask)
    n_feat = 0 if pts is None else len(pts)
    lk_disp = []
    win = cv2.createHanningWindow(g0.shape[::-1], cv2.CV_32F)
    pc_disp = []
    f0 = g0.astype(np.float32)
    for f in frames[1:]:
        g = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
        if n_feat >= 15:
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(g0, g, pts, None, winSize=(21, 21), maxLevel=3)
            good = st.ravel() == 1
            if good.sum() >= 10:
                lk_disp.append(float(np.median(np.linalg.norm((nxt - pts)[good].reshape(-1, 2), axis=1))))
        (dx, dy), _ = cv2.phaseCorrelate(f0, g.astype(np.float32), win)
        pc_disp.append(float(np.hypot(dx, dy)))
    return {
        "n_features": int(n_feat),
        "lk_max_median_disp_px": max(lk_disp) if lk_disp else None,
        "phasecorr_max_disp_px": max(pc_disp) if pc_disp else None,
    }
