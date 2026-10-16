"""fal.ai adapter (many V2V candidates in the proposal are served there).

Install the client and set your key::

    pip install fal-client
    export FAL_KEY=...

Everything model-specific lives in configs/models.yaml -- the endpoint id and the
argument names are copied from the model's page on fal (its "API" tab), e.g.::

    my-model:
      adapter: fal
      endpoint: <owner>/<model>/<task>        # exactly as shown on fal
      arguments:                              # request body; {placeholders} are filled in
        video_url: "{video_url}"
        prompt: "{prompt}"
        seed: "{seed}"
      output_path: video.url                  # where the result's URL is in the response

Placeholders: {video_url} {image_url} {prompt} {negative_prompt} {seed} {duration_s}
{duration_int} {clip_id} {sample}, plus anything under ``params``.
"""

from __future__ import annotations

from pathlib import Path

from .base import GenerationRequest, JobStatus, VideoModel, download, fill_template, get_path


class FalModel(VideoModel):
    adapter = "fal"

    def __init__(self, name: str, cfg: dict):
        super().__init__(name, cfg)
        if not cfg.get("endpoint"):
            hint = cfg.get("endpoint_hint", "")
            raise ValueError(f"model {name}: set 'endpoint' in configs/models.yaml (hint from the proposal: {hint})")
        try:
            import fal_client  # noqa: F401
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError("pip install fal-client  (and export FAL_KEY)") from e
        self.endpoint = cfg["endpoint"]
        self._requests: dict[str, dict] = {}

    def submit(self, req: GenerationRequest) -> str:
        import fal_client

        values = req.placeholders(req.params)
        values["video_url"] = fal_client.upload_file(str(req.input_video))
        if "{image_url}" in str(self.cfg.get("arguments", {})):
            values["image_url"] = fal_client.upload_file(str(req.last_frame))
        args = fill_template(self.cfg.get("arguments", {}), values)
        handle = fal_client.submit(self.endpoint, arguments=args)
        self._requests[handle.request_id] = args
        return handle.request_id

    def poll(self, job_id: str) -> JobStatus:
        import fal_client

        st = fal_client.status(self.endpoint, job_id, with_logs=False)
        kind = type(st).__name__.lower()
        if "completed" in kind:
            res = fal_client.result(self.endpoint, job_id)
            return JobStatus("done", result=res, meta={"request_id": job_id})
        if "queued" in kind:
            return JobStatus("queued")
        if "progress" in kind:
            return JobStatus("running")
        return JobStatus("failed", error=str(st))

    def fetch(self, status: JobStatus, req: GenerationRequest) -> Path:
        url = get_path(status.result, self.cfg.get("output_path", "video.url"))
        return download(url, req.out_path)
