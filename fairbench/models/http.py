"""Generic REST adapter for providers without a Python client (xAI, BytePlus, Alibaba, ...).

Most video APIs follow the same pattern: POST a job (video as a URL or base64), get a job
id, GET its status until done, download the result URL. This adapter does exactly that,
with every provider-specific detail in configs/models.yaml::

    my-model:
      adapter: http
      api_key_env: PROVIDER_API_KEY
      submit:
        url: https://api.example.com/v1/videos/extend
        headers: {Authorization: "Bearer {api_key}"}
        json:                                  # request body
          model: <model id>
          video: "{video_b64_data_uri}"        # or "{video_url}" with an uploader below
          prompt: "{prompt}"
          seed: "{seed}"
        job_id_path: id                        # where the job id is in the response
      poll:
        url: https://api.example.com/v1/videos/{job_id}
        headers: {Authorization: "Bearer {api_key}"}
        status_path: status
        done_values: [succeeded, completed]
        failed_values: [failed, error, cancelled]
        result_url_path: output.url
      upload: none                             # none | fal  (to turn the clip into a public URL)

Extra placeholders: {api_key} {video_b64} {video_b64_data_uri} {video_url} {image_b64}
{job_id}. Read the provider's docs for the exact field names -- nothing here is guessed.
"""

from __future__ import annotations

import base64
from pathlib import Path

from .base import GenerationRequest, JobStatus, VideoModel, download, fill_template, get_path


class HttpModel(VideoModel):
    adapter = "http"

    def __init__(self, name: str, cfg: dict):
        super().__init__(name, cfg)
        for key in ("submit", "poll"):
            if not cfg.get(key) or not cfg[key].get("url"):
                raise ValueError(f"model {name}: http adapter needs '{key}.url' in configs/models.yaml")

    def _values(self, req: GenerationRequest | None, extra: dict | None = None) -> dict:
        v = req.placeholders(req.params) if req is not None else {}
        v["api_key"] = self.api_key()
        v.update(extra or {})
        return v

    def submit(self, req: GenerationRequest) -> str:
        import requests

        b64 = base64.b64encode(Path(req.input_video).read_bytes()).decode()
        extra = {
            "video_b64": b64,
            "video_b64_data_uri": f"data:video/mp4;base64,{b64}",
            "image_b64": base64.b64encode(Path(req.last_frame).read_bytes()).decode(),
        }
        if self.cfg.get("upload") == "fal":
            import fal_client

            extra["video_url"] = fal_client.upload_file(str(req.input_video))
        s = self.cfg["submit"]
        vals = self._values(req, extra)
        r = requests.post(fill_template(s["url"], vals), headers=fill_template(s.get("headers", {}), vals),
                          json=fill_template(s.get("json", {}), vals), timeout=float(s.get("timeout", 120)))
        r.raise_for_status()
        return str(get_path(r.json(), s.get("job_id_path", "id")))

    def poll(self, job_id: str) -> JobStatus:
        import requests

        p = self.cfg["poll"]
        vals = self._values(None, {"job_id": job_id})
        r = requests.get(fill_template(p["url"], vals), headers=fill_template(p.get("headers", {}), vals),
                         timeout=float(p.get("timeout", 60)))
        r.raise_for_status()
        body = r.json()
        state = str(get_path(body, p.get("status_path", "status"))).lower()
        if state in [s.lower() for s in p.get("done_values", ["succeeded", "completed", "done"])]:
            return JobStatus("done", result=body, meta={"job_id": job_id})
        if state in [s.lower() for s in p.get("failed_values", ["failed", "error", "cancelled", "canceled"])]:
            return JobStatus("failed", error=str(body)[:500])
        return JobStatus("running")

    def fetch(self, status: JobStatus, req: GenerationRequest) -> Path:
        url = get_path(status.result, self.cfg["poll"].get("result_url_path", "output.url"))
        return download(str(url), req.out_path)
