"""Mock 'models' for checking the pipeline before spending money.

Each mock builds its output from the simulator's own continuation, so the expected
scores are known in advance:

* ``oracle``   -- the reference continuation: every metric should be at its ceiling.
* ``freeze``   -- the last input frame repeated: perfectly invariant to every
                  intervention, yet physically wrong. Shows why robustness alone
                  cannot be the benchmark score.
* ``lag``      -- the reference delayed by N frames, optionally for one variant only
                  (``lag_by_variant``): VIMS should drop to about e^-1 for a one-frame lag.
* ``timewarp`` -- the reference played faster/slower per variant (``factor_by_variant``):
                  a model whose motion depends on the ball's color.
* ``noise``    -- the reference plus pixel noise: tests tracker robustness.

``emulate`` reproduces API quirks (``includes_input``, ``fps``, ``size``,
``repeat_last``) so the alignment code is exercised too.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import cv2
import numpy as np

from ..video import read_video, write_video
from .base import GenerationRequest, JobStatus, VideoModel


class MockModel(VideoModel):
    adapter = "mock"

    def __init__(self, name: str, cfg: dict):
        super().__init__(name, cfg)
        self._done: dict[str, JobStatus] = {}

    def _continuation(self, req: GenerationRequest) -> tuple[np.ndarray, np.ndarray, float]:
        ref_path = Path(req.params["reference_video"])
        frames, fps = read_video(ref_path)
        n_in = int(req.params.get("input_frames", 48))
        n_gen = len(frames) - n_in
        variant = req.params.get("variant")
        mode = self.cfg.get("mode", "oracle")
        last = n_in - 1
        j = np.arange(1, n_gen + 1)
        if mode == "oracle":
            idx = last + j
        elif mode == "freeze":
            idx = np.full(n_gen, last)
        elif mode == "lag":
            lag = int(self.cfg.get("lag_by_variant", {}).get(variant, self.cfg.get("lag", 1)))
            idx = np.clip(last + j - lag, last, len(frames) - 1)
        elif mode == "timewarp":
            fac = float(self.cfg.get("factor_by_variant", {}).get(variant, self.cfg.get("factor", 1.0)))
            idx = np.clip(np.round(last + j * fac).astype(int), last, len(frames) - 1)
        elif mode == "noise":
            idx = last + j
        else:
            raise ValueError(f"unknown mock mode {mode!r}")
        out = frames[idx].astype(np.float32)
        if mode == "noise":
            rng = np.random.default_rng(req.seed or 0)
            out = out + rng.normal(0, float(self.cfg.get("sigma", 6.0)), out.shape)
        if mode in ("oracle", "lag", "timewarp", "noise") and float(self.cfg.get("jitter_px", 0)) > 0:
            # whole-frame sub-pixel shake: emulates small tracking-irrelevant changes
            rng = np.random.default_rng((req.seed or 0) + 1)
            for k in range(len(out)):
                dx, dy = rng.normal(0, self.cfg["jitter_px"], 2)
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                out[k] = cv2.warpAffine(out[k], M, (out.shape[2], out.shape[1]), borderMode=cv2.BORDER_REPLICATE)
        return np.clip(out, 0, 255).astype(np.uint8), frames[:n_in], fps

    def submit(self, req: GenerationRequest) -> str:
        cont, inp, fps = self._continuation(req)
        em = self.cfg.get("emulate", {}) or {}
        frames = cont
        if em.get("repeat_last"):
            frames = np.concatenate([inp[-1:], frames])
        if em.get("includes_input"):
            frames = np.concatenate([inp, frames])
        out_fps = float(em.get("fps", fps))
        if out_fps != fps:
            # resample by time (nearest frame), as a model generating at another rate would;
            # frame k of the output shows time k / out_fps of the 24 fps sequence
            offset = 1 if not (em.get("includes_input") or em.get("repeat_last")) else 0
            t_src = (np.arange(len(frames)) + offset) / fps
            t_dst = (np.arange(int(np.floor(t_src[-1] * out_fps)) + 1 - offset) + offset) / out_fps
            idx = np.clip(np.round(t_dst * fps).astype(int) - offset, 0, len(frames) - 1)
            frames = frames[idx]
        if em.get("size"):
            w, h = em["size"]
            frames = np.stack([cv2.resize(f, (int(w), int(h)), interpolation=cv2.INTER_AREA) for f in frames])
        write_video(req.out_path, frames, fps=out_fps)
        job = f"mock-{uuid.uuid4().hex[:10]}"
        self._done[job] = JobStatus("done", result=str(req.out_path), meta={"mode": self.cfg.get("mode", "oracle")})
        return job

    def poll(self, job_id: str) -> JobStatus:
        return self._done.get(job_id, JobStatus("failed", error="unknown mock job (mocks are not resumable)"))

    def fetch(self, status: JobStatus, req: GenerationRequest) -> Path:
        return Path(status.result)
