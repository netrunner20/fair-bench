"""Replicate adapter.

    pip install replicate
    export REPLICATE_API_TOKEN=...

Config (copy the model id and input names from the model's API page on Replicate)::

    my-model:
      adapter: replicate
      model_ref: <owner>/<model>             # or <owner>/<model>:<version>
      input:
        video: "{video_file}"                # file inputs are uploaded by the client
        prompt: "{prompt}"
        seed: "{seed}"
      output_path: ""                         # '' if the output is the URL itself; '0' for a list

Extra placeholders: {video_file} {image_file} (open file handles).
"""

from __future__ import annotations

from pathlib import Path

from .base import GenerationRequest, JobStatus, VideoModel, download, fill_template, get_path


class ReplicateModel(VideoModel):
    adapter = "replicate"

    def __init__(self, name: str, cfg: dict):
        super().__init__(name, cfg)
        if not cfg.get("model_ref"):
            raise ValueError(f"model {name}: set 'model_ref' in configs/models.yaml ({cfg.get('endpoint_hint', '')})")
        try:
            import replicate  # noqa: F401
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError("pip install replicate  (and export REPLICATE_API_TOKEN)") from e

    def submit(self, req: GenerationRequest) -> str:
        import replicate

        values = req.placeholders(req.params)
        values["video_file"] = open(req.input_video, "rb")
        values["image_file"] = open(req.last_frame, "rb")
        try:
            inp = fill_template(self.cfg.get("input", {}), values)
            ref = self.cfg["model_ref"]
            if ":" in ref:
                pred = replicate.predictions.create(version=ref.split(":", 1)[1], input=inp)
            else:
                pred = replicate.predictions.create(model=ref, input=inp)
        finally:
            values["video_file"].close()
            values["image_file"].close()
        return pred.id

    def poll(self, job_id: str) -> JobStatus:
        import replicate

        p = replicate.predictions.get(job_id)
        if p.status == "succeeded":
            return JobStatus("done", result=p.output, meta={"prediction_id": job_id, "metrics": p.metrics})
        if p.status in ("starting",):
            return JobStatus("queued")
        if p.status == "processing":
            return JobStatus("running")
        return JobStatus("failed", error=f"{p.status}: {p.error}")

    def fetch(self, status: JobStatus, req: GenerationRequest) -> Path:
        out = get_path(status.result, self.cfg.get("output_path", ""))
        url = getattr(out, "url", out)
        return download(str(url), req.out_path)
