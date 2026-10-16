"""The runner must never pay twice for one generation, and must respect the budget."""

import json

import numpy as np
import pytest

import fairbench.generate as G
from fairbench.models.base import JobStatus, VideoModel
from fairbench.synthetic import build_synthetic_dataset
from fairbench.video import write_video


@pytest.fixture(scope="module")
def tiny_ds(tmp_path_factory):
    return build_synthetic_dataset(tmp_path_factory.mktemp("tiny"), width=96, height=54, ss=1,
                                   groups=["thrown/color"], verbose=False)


class FakeAPI(VideoModel):
    adapter = "fake"

    def __init__(self, name, cfg, poll_errors=0, provider_failures=0):
        super().__init__(name, cfg)
        self.submits, self.poll_errors, self.provider_failures = 0, poll_errors, provider_failures
        self.jobs = {}

    def submit(self, req):
        self.submits += 1
        jid = f"job{self.submits}"
        self.jobs[jid] = req
        return jid

    def poll(self, job_id):
        if self.poll_errors > 0:
            self.poll_errors -= 1
            raise ConnectionError("network down")
        if self.provider_failures > 0:
            self.provider_failures -= 1
            return JobStatus("failed", error="provider error")
        return JobStatus("done", result=job_id)

    def fetch(self, status, req):
        write_video(req.out_path, np.zeros((96, 54, 96, 3), np.uint8), fps=24)
        return req.out_path


def _run(ds, tmp_path, monkeypatch, fake, budget=10.0, cost=0.5, clips=None):
    monkeypatch.setattr(G, "load_model", lambda name, cfg: fake)
    models = {"fake": {"adapter": "fake", "cost_per_generation_usd": cost, "max_concurrency": 1}}
    gen = {"samples_per_clip": 1, "poll_interval_s": 0, "retry_backoff_s": 0.0, "max_retries": 2, "base_seed": 0}
    jobs = G.plan_jobs(ds, tmp_path, models, gen, ["fake"], clips=clips or ["thrown_color_r"])
    return G.run_jobs(ds, tmp_path, jobs, models, gen, budget_usd=budget, log=lambda *a: None), jobs


def test_poll_errors_do_not_resubmit(tiny_ds, tmp_path, monkeypatch):
    fake = FakeAPI("fake", {}, poll_errors=2)
    res, jobs = _run(tiny_ds, tmp_path, monkeypatch, fake)
    assert res["ok"] == 1 and fake.submits == 1 and jobs[0].out_path.exists()


def test_provider_failure_resubmits(tiny_ds, tmp_path, monkeypatch):
    fake = FakeAPI("fake", {}, provider_failures=1)
    res, _ = _run(tiny_ds, tmp_path, monkeypatch, fake)
    assert res["ok"] == 1 and fake.submits == 2


def test_budget_guard(tiny_ds, tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="exceeds the budget"):
        _run(tiny_ds, tmp_path, monkeypatch, FakeAPI("fake", {}), budget=1.0, cost=0.5,
             clips=["thrown_color_r", "thrown_color_g", "thrown_color_b"])


def test_resume_repolls_instead_of_paying_again(tiny_ds, tmp_path, monkeypatch):
    journal = tmp_path / "journal.jsonl"
    journal.write_text(json.dumps({"event": "submitted", "job": "fake/thrown_color_r/s0", "job_id": "old-job"}) + "\n")
    fake = FakeAPI("fake", {})
    res, _ = _run(tiny_ds, tmp_path, monkeypatch, fake)
    assert res["ok"] == 1 and fake.submits == 0
