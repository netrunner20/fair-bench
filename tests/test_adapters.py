"""Adapters fill the configured request and read the configured response (no network)."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from fairbench.models.base import GenerationRequest, fill_template, get_path


def _req(tmp_path: Path) -> GenerationRequest:
    vid = tmp_path / "in.mp4"
    png = tmp_path / "in.png"
    vid.write_bytes(b"x")
    png.write_bytes(b"x")
    return GenerationRequest(model="m", clip_id="thrown_color_r", sample=0, seed=1234, input_video=vid,
                             last_frame=png, prompt="Continue this video.", negative_prompt=None, duration_s=4.0,
                             out_path=tmp_path / "out" / "s0.mp4")


def test_fill_template_keeps_types_and_drops_none():
    out = fill_template({"seed": "{seed}", "prompt": "{prompt}", "neg": "{negative_prompt}", "n": "{missing}",
                         "text": "clip {clip_id}"}, {"seed": 7, "prompt": "go", "negative_prompt": "", "clip_id": "c"})
    assert out == {"seed": 7, "prompt": "go", "neg": "", "text": "clip c"}
    assert get_path({"video": {"url": "u"}}, "video.url") == "u"
    assert get_path({"output": ["a", "b"]}, "output.1") == "b"


def test_fal_adapter(tmp_path, monkeypatch):
    fal_client = pytest.importorskip("fal_client")
    import fairbench.models.fal as F

    sent = {}
    monkeypatch.setattr(fal_client, "upload_file", lambda p: f"https://files/{Path(p).name}")
    monkeypatch.setattr(fal_client, "submit", lambda app, arguments: sent.update(app=app, args=arguments)
                        or SimpleNamespace(request_id="r1"))
    monkeypatch.setattr(fal_client, "status", lambda app, rid, with_logs=False: fal_client.Completed(logs=None, metrics={}))
    monkeypatch.setattr(fal_client, "result", lambda app, rid: {"video": {"url": "https://cdn/out.mp4"}})
    monkeypatch.setattr(F, "download", lambda url, out: (out.parent.mkdir(parents=True, exist_ok=True),
                                                         out.write_text(url), out)[-1])
    cfg = {"adapter": "fal", "endpoint": "owner/model/extend",
           "arguments": {"video_url": "{video_url}", "prompt": "{prompt}", "seed": "{seed}"}, "output_path": "video.url"}
    m = F.FalModel("m", cfg)
    req = _req(tmp_path)
    jid = m.submit(req)
    assert jid == "r1" and sent["app"] == "owner/model/extend"
    assert sent["args"] == {"video_url": "https://files/in.mp4", "prompt": "Continue this video.", "seed": 1234}
    st = m.poll(jid)
    assert st.state == "done"
    assert m.fetch(st, req).read_text() == "https://cdn/out.mp4"


def test_fal_adapter_requires_endpoint():
    pytest.importorskip("fal_client")
    from fairbench.models.fal import FalModel

    with pytest.raises(ValueError, match="endpoint"):
        FalModel("m", {"adapter": "fal", "endpoint": None, "endpoint_hint": "fal ltx-2.3-22b/extend"})
