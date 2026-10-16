"""SAM 2.1 video tracker (the tracker named in the proposal).

Requires the official ``sam2`` package and a SAM 2.1 checkpoint::

    pip install "git+https://github.com/facebookresearch/sam2.git"
    # checkpoint, e.g. sam2.1_hiera_large.pt, from the SAM 2 release page

The ball is prompted on frame 0 (the last conditioning frame, where its position is known
from the simulator) with a box around the projected ball plus a positive click at its
center. Masks are propagated through the generated frames. Each frame's center is the
probability-weighted centroid of the mask (sub-pixel refinement); frames with no mask
are marked missing, and extra blobs are counted as possible duplicates.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np

from ..video import write_jpeg_folder
from .base import Tracker, TrackResult


class Sam2Tracker(Tracker):
    name = "sam2"

    def __init__(self, checkpoint: str | None = None, config: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
                 device: str | None = None, min_area_px: float = 4.0, dup_area_frac: float = 0.3,
                 jpeg_quality: int = 95):
        try:
            import torch
            from sam2.build_sam import build_sam2_video_predictor
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError(
                "SAM 2 is not installed. Install it with\n"
                "  pip install 'git+https://github.com/facebookresearch/sam2.git'\n"
                "or use --tracker color for synthetic data."
            ) from e
        if checkpoint is None or not Path(checkpoint).exists():
            raise FileNotFoundError(
                f"SAM 2.1 checkpoint not found: {checkpoint!r}. Set tracking.sam2.checkpoint in configs/metrics.yaml."
            )
        if device is None:
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.device = device
        self.torch = torch
        self.predictor = build_sam2_video_predictor(config, checkpoint, device=device)
        self.min_area_px = min_area_px
        self.dup_area_frac = dup_area_frac
        self.jpeg_quality = jpeg_quality

    def track(self, frames, init_uv, init_radius_px, ref_rgb=None) -> TrackResult:
        torch = self.torch
        frames = np.asarray(frames)
        T = len(frames)
        u, v = map(float, init_uv)
        r = float(init_radius_px)
        box = np.array([u - 1.25 * r, v - 1.25 * r, u + 1.25 * r, v + 1.25 * r], dtype=np.float32)
        centers = np.full((T, 2), np.nan)
        area = np.zeros(T)
        ncomp = np.zeros(T, dtype=int)
        with tempfile.TemporaryDirectory() as tmp:
            write_jpeg_folder(tmp, frames, quality=self.jpeg_quality)
            autocast = torch.autocast("cuda", dtype=torch.bfloat16) if self.device == "cuda" else _nullcontext()
            with torch.inference_mode(), autocast:
                state = self.predictor.init_state(video_path=tmp)
                self.predictor.add_new_points_or_box(
                    inference_state=state, frame_idx=0, obj_id=1, box=box,
                    points=np.array([[u, v]], dtype=np.float32), labels=np.array([1], dtype=np.int32),
                )
                expected = np.pi * r * r
                for fi, obj_ids, logits in self.predictor.propagate_in_video(state):
                    lg = logits[0, 0].float().cpu().numpy()
                    mask = lg > 0
                    if mask.sum() < self.min_area_px:
                        continue
                    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
                    areas = stats[1:, cv2.CC_STAT_AREA]
                    ncomp[fi] = int(np.sum(areas >= self.dup_area_frac * expected))
                    i = 1 + int(np.argmax(areas))  # largest blob is the ball
                    ys, xs = np.nonzero(lab == i)
                    w = 1.0 / (1.0 + np.exp(-lg[ys, xs]))  # mask probability as weight
                    centers[fi] = [np.sum(w * (xs + 0.5)) / w.sum(), np.sum(w * (ys + 0.5)) / w.sum()]
                    area[fi] = areas[i - 1]
                    expected = 0.7 * expected + 0.3 * area[fi]
                self.predictor.reset_state(state)
        return TrackResult(centers=centers, area=area, n_components=ncomp, info={"tracker": self.name})


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False
