"""Videos generated outside the harness (a web console, a lab GPU, a teammate).

Put the files in one folder named ``<clip_id>__s<sample>.mp4`` (e.g.
``thrown_color_r__s0.mp4``) and run ``fairbench import-outputs --model NAME --dir FOLDER``;
they are copied into the run layout with a provenance record, exactly like API outputs.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from .base import GenerationRequest, JobStatus, VideoModel


class ManualModel(VideoModel):
    adapter = "manual"

    def submit(self, req: GenerationRequest) -> str:
        src_dir = Path(self.cfg.get("import_dir", ""))
        src = src_dir / f"{req.clip_id}__s{req.sample}.mp4"
        if not src.exists():
            raise FileNotFoundError(f"manual model {self.name}: expected {src}")
        req.out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, req.out_path)
        return f"manual:{src.name}"

    def poll(self, job_id: str) -> JobStatus:
        return JobStatus("done", result=job_id)

    def fetch(self, status: JobStatus, req: GenerationRequest) -> Path:
        return req.out_path


def load_model(name: str, cfg: dict):
    """Instantiate the adapter named in a model's config entry."""
    adapter = cfg.get("adapter", "manual")
    if adapter == "mock":
        from .mock import MockModel

        return MockModel(name, cfg)
    if adapter == "fal":
        from .fal import FalModel

        return FalModel(name, cfg)
    if adapter == "replicate":
        from .replicate import ReplicateModel

        return ReplicateModel(name, cfg)
    if adapter == "http":
        from .http import HttpModel

        return HttpModel(name, cfg)
    if adapter == "manual":
        return ManualModel(name, cfg)
    raise ValueError(f"model {name}: unknown adapter {adapter!r}")
