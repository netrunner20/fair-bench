"""Model adapter interface.

Every video model is reached through an adapter with three steps -- ``submit`` a job,
``poll`` it, ``fetch`` the finished video -- so the runner can write the provider's job id
to the run journal right after submitting. If the process dies while a job is running,
the next run re-polls that job instead of paying for a second generation.

Adapters that run synchronously (local models, mocks) do the work in ``submit`` and
return status "done" on the first ``poll``.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class GenerationRequest:
    model: str  # key in configs/models.yaml
    clip_id: str
    sample: int  # 0 .. samples_per_clip-1
    seed: int | None
    input_video: Path  # the 2 s conditioning clip (mp4, reference size, 24 fps)
    last_frame: Path  # PNG of the last conditioning frame
    prompt: str
    negative_prompt: str | None
    duration_s: float  # requested continuation length
    params: dict = field(default_factory=dict)  # model-specific extras from the config
    out_path: Path = Path("out.mp4")

    def placeholders(self, extra: dict | None = None) -> dict:
        d = {
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt or "",
            "seed": self.seed,
            "duration_s": self.duration_s,
            "duration_int": int(round(self.duration_s)),
            "clip_id": self.clip_id,
            "sample": self.sample,
        }
        d.update(extra or {})
        return d

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("input_video", "last_frame", "out_path"):
            d[k] = str(d[k])
        return d


@dataclass
class JobStatus:
    state: str  # "queued" | "running" | "done" | "failed"
    result: Any = None  # adapter-specific handle for fetch (url, local path, response dict)
    meta: dict = field(default_factory=dict)
    error: str | None = None


class VideoModel:
    """Base adapter. Subclasses implement submit / poll / fetch."""

    adapter = "base"

    def __init__(self, name: str, cfg: dict):
        self.name = name
        self.cfg = cfg

    # -- the three steps --------------------------------------------------------
    def submit(self, req: GenerationRequest) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    def poll(self, job_id: str) -> JobStatus:  # pragma: no cover - interface
        raise NotImplementedError

    def fetch(self, status: JobStatus, req: GenerationRequest) -> Path:  # pragma: no cover - interface
        raise NotImplementedError

    # -- helpers -----------------------------------------------------------------
    def api_key(self) -> str | None:
        env = self.cfg.get("api_key_env")
        if not env:
            return None
        key = os.environ.get(env)
        if not key:
            raise RuntimeError(f"model {self.name}: environment variable {env} is not set")
        return key

    def describe(self) -> dict:
        safe = {k: v for k, v in self.cfg.items() if "key" not in k.lower() and "secret" not in k.lower()}
        return {"name": self.name, "adapter": self.adapter, **safe}


_PLACEHOLDER = re.compile(r"^\{(\w+)\}$")


def fill_template(obj: Any, values: dict) -> Any:
    """Replace "{name}" placeholders in a nested config structure.

    A string that is exactly "{name}" is replaced by the value itself (keeping its type,
    e.g. an int seed); placeholders inside longer strings are formatted as text. Keys whose
    value resolves to None are dropped, so an unset seed is simply not sent.
    """
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            fv = fill_template(v, values)
            if fv is not None:
                out[k] = fv
        return out
    if isinstance(obj, list):
        return [fill_template(v, values) for v in obj]
    if isinstance(obj, str):
        m = _PLACEHOLDER.match(obj)
        if m:
            return values.get(m.group(1))
        try:
            return obj.format(**{k: ("" if v is None else v) for k, v in values.items()})
        except (KeyError, IndexError):
            return obj
    return obj


def get_path(obj: Any, path: str | None) -> Any:
    """Read a dotted path ('video.url', 'output.0') from a nested response."""
    if not path:
        return obj
    cur = obj
    for part in path.split("."):
        if isinstance(cur, (list, tuple)):
            cur = cur[int(part)]
        elif isinstance(cur, dict):
            cur = cur[part]
        else:
            cur = getattr(cur, part)
    return cur


def download(url: str, out_path: Path, timeout: float = 300.0) -> Path:
    import requests

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        tmp = out_path.with_suffix(out_path.suffix + ".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
        tmp.replace(out_path)
    return out_path
