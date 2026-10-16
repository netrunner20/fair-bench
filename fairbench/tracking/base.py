"""Tracker interface: one ball, one video, frame 0 = the last conditioning frame."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TrackResult:
    centers: np.ndarray  # (T, 2) mask centroids in native pixels, NaN where the ball is not found
    area: np.ndarray  # (T,) mask area in native pixels
    n_components: np.ndarray  # (T,) number of ball-like blobs (>1 suggests a duplicated ball)
    info: dict = field(default_factory=dict)

    @property
    def found(self) -> np.ndarray:
        return np.all(np.isfinite(self.centers), axis=1)

    def to_npz_dict(self) -> dict:
        return {"centers": self.centers, "area": self.area, "n_components": self.n_components}


class Tracker:
    """Track the ball through ``frames`` given where it is in frame 0.

    Parameters of :meth:`track`
    ---------------------------
    frames:  uint8 (T, H, W, 3); frame 0 is the last conditioning frame.
    init_uv: (2,) ball center in frame 0, native pixels.
    init_radius_px: ball radius in frame 0, native pixels.
    ref_rgb: optional RGB of the ball (sampled from frame 0 when missing).
    """

    name = "base"

    def track(self, frames: np.ndarray, init_uv: np.ndarray, init_radius_px: float,
              ref_rgb: tuple | None = None) -> TrackResult:  # pragma: no cover - interface
        raise NotImplementedError


def get_tracker(name: str, **kwargs) -> Tracker:
    name = name.lower()
    if name in ("color", "colour"):
        from .color import ColorTracker

        return ColorTracker(**kwargs)
    if name in ("sam2", "sam2.1", "sam"):
        from .sam2_tracker import Sam2Tracker

        return Sam2Tracker(**kwargs)
    raise ValueError(f"unknown tracker '{name}' (choose 'sam2' or 'color')")
