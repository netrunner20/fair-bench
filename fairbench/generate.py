"""Generation runner: plan -> (dry run) -> generate, with a cost guard and full provenance.

Layout of a run (``runs/<run>/``)::

    inputs/<clip_id>.mp4            the 2 s conditioning clip given to every model
    inputs/<clip_id>_last.png       its last frame
    <model>/<clip_id>/s<k>.mp4      generated video, sample k
    <model>/<clip_id>/s<k>.json     provenance: request, settings, job id, timings, cost
    journal.jsonl                   every submit / finish / failure, in order (resume source)
    ledger.jsonl                    estimated spend per finished generation

A job whose video already exists is skipped. A job that was submitted but not finished
(process killed, network down) is re-polled by its job id on the next run rather than
submitted again, so an interruption never pays twice.
"""

from __future__ import annotations

import json
import subprocess
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

import imageio.v2 as imageio

from .manifest import MOTION_KIND, Dataset
from .models.base import GenerationRequest
from .models.manual import load_model
from .video import read_video, write_video


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_commit() -> str | None:
    try:
        # "-dirty" marks outputs made with uncommitted harness changes
        return subprocess.run(["git", "describe", "--always", "--dirty", "--abbrev=7"], capture_output=True, text=True,
                              cwd=Path(__file__).parent, check=True).stdout.strip()
    except Exception:
        return None


# --------------------------------------------------------------------------- inputs
def prepare_inputs(ds: Dataset, run_dir: Path) -> dict[str, dict]:
    """Cut the first ``input_frames`` of every clip into a 2 s mp4 (+ last-frame PNG)."""
    out = {}
    d = run_dir / "inputs"
    d.mkdir(parents=True, exist_ok=True)
    for c in ds.clips.values():
        vid, png = d / f"{c.id}.mp4", d / f"{c.id}_last.png"
        if not (vid.exists() and png.exists()):
            frames, fps = read_video(c.video)
            inp = frames[: ds.input_frames]
            write_video(vid, inp, fps=ds.fps, crf=10)
            imageio.imwrite(png, inp[-1])
        out[c.id] = {"input_video": vid, "last_frame": png}
    return out


# ----------------------------------------------------------------------------- jobs
@dataclass
class Job:
    model: str
    clip_id: str
    sample: int
    seed: int | None
    out_path: Path
    cost_usd: float

    @property
    def key(self) -> str:
        return f"{self.model}/{self.clip_id}/s{self.sample}"


def required_window_s(ds: Dataset, motion: str) -> float:
    """How much of the end of the input a model must see for the future to be determined.

    Ballistic scenes: everything from the last floor contact in the input onward (plus two
    frames of margin). Rolling scene: a quarter second to see the constant speed.
    """
    dur = ds.input_frames / ds.fps
    ev = ds.motions[motion].events or {}
    if MOTION_KIND.get(motion) == "ballistic" and ev.get("input_contacts_s"):
        return dur - max(ev["input_contacts_s"]) + 2.0 / ds.fps
    return 0.25


def check_model_fits(ds: Dataset, name: str, mcfg: dict) -> list[str]:
    """Reasons a model cannot be evaluated fairly on this dataset (empty = fine)."""
    problems = []
    if mcfg.get("available") is False:
        problems.append(f"{name} is marked unavailable ({mcfg.get('endpoint_hint', '')})")
    win = mcfg.get("conditioning_window_s")
    if win is not None:
        for motion in ds.motions:
            need = required_window_s(ds, motion)
            if float(win) < need:
                problems.append(f"{name} conditions on the last {win} s only; {motion} needs {need:.2f} s "
                                f"(the last bounce would be cut off, so the future is not determined)")
    mif = mcfg.get("min_input_frames")
    if mif is not None and int(mif) > ds.input_frames:
        problems.append(f"{name} needs at least {mif} input frames; the benchmark gives {ds.input_frames}")
    return problems


def plan_jobs(ds: Dataset, run_dir: Path, models: dict[str, dict], gen: dict, names: list[str],
              clips: list[str] | None = None, samples: int | None = None) -> list[Job]:
    K = int(samples if samples is not None else gen.get("samples_per_clip", 1))
    base = gen.get("base_seed")
    jobs = []
    for name in names:
        m = models[name]
        per_gen = float(m.get("cost_per_generation_usd") or 0.0)
        for cid in sorted(ds.clips):
            if clips and cid not in clips:
                continue
            for k in range(K):
                seed = None if base is None or m.get("supports_seed") is False else int(base) + k
                jobs.append(Job(name, cid, k, seed, run_dir / name / cid / f"s{k}.mp4", per_gen))
    return jobs


# ------------------------------------------------------------------------ bookkeeping
class Journal:
    def __init__(self, path: Path):
        self.path = path
        self.lock = Lock()
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, **rec) -> None:
        rec["time"] = _now()
        with self.lock, open(self.path, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    def pending_job_ids(self) -> dict[str, str]:
        """job key -> provider job id for jobs submitted but not finished."""
        if not self.path.exists():
            return {}
        open_ids: dict[str, str] = {}
        for line in self.path.read_text().splitlines():
            r = json.loads(line)
            if r.get("event") == "submitted":
                open_ids[r["job"]] = r["job_id"]
            elif r.get("event") in ("finished", "failed"):
                open_ids.pop(r["job"], None)
        return open_ids


def spent_usd(run_dir: Path) -> float:
    p = run_dir / "ledger.jsonl"
    if not p.exists():
        return 0.0
    return float(sum(json.loads(l).get("cost_usd", 0.0) for l in p.read_text().splitlines() if l.strip()))


# ----------------------------------------------------------------------------- run
def run_jobs(ds: Dataset, run_dir: Path, jobs: list[Job], models: dict[str, dict], gen: dict,
             budget_usd: float | None = None, dry_run: bool = False, log=print) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    todo = [j for j in jobs if not j.out_path.exists()]
    est = sum(j.cost_usd for j in todo)
    spent = spent_usd(run_dir)
    budget = budget_usd if budget_usd is not None else gen.get("budget_usd")
    log(f"{len(jobs)} jobs, {len(jobs) - len(todo)} already done, {len(todo)} to run; "
        f"estimated cost ${est:.2f} (spent so far ${spent:.2f}{'' if budget is None else f', budget ${budget:.2f}'})")
    if dry_run or not todo:
        return {"planned": len(todo), "estimated_usd": est, "ran": 0}
    if budget is not None and spent + est > float(budget) + 1e-9:
        raise RuntimeError(f"estimated total ${spent + est:.2f} exceeds the budget ${float(budget):.2f}; "
                           "raise generation.budget_usd or pass --max-usd")
    inputs = prepare_inputs(ds, run_dir)
    journal = Journal(run_dir / "journal.jsonl")
    pending = journal.pending_job_ids()
    ledger_lock = Lock()
    adapters = {name: load_model(name, models[name]) for name in {j.model for j in todo}}
    results = {"ok": 0, "failed": 0}

    def one(job: Job):
        m = models[job.model]
        clip = ds.clips[job.clip_id]
        params = dict(m.get("params", {}) or {})
        if m.get("adapter") == "mock":
            params.update({"reference_video": str(clip.video), "input_frames": ds.input_frames, "variant": clip.variant})
        req = GenerationRequest(
            model=job.model, clip_id=job.clip_id, sample=job.sample, seed=job.seed,
            input_video=inputs[job.clip_id]["input_video"], last_frame=inputs[job.clip_id]["last_frame"],
            prompt=m.get("prompt", gen.get("prompt", "")), negative_prompt=m.get("negative_prompt", gen.get("negative_prompt")),
            duration_s=float(gen.get("duration_s", ds.n_gen / ds.fps)), params=params, out_path=job.out_path,
        )
        ad = adapters[job.model]
        t0 = time.time()
        tries = int(gen.get("max_retries", 2)) + 1
        backoff = float(gen.get("retry_backoff_s", 2.0))
        job_id = pending.get(job.key)
        if job_id is not None:
            log(f"  resuming {job.key} (job {job_id})")
        last_err = None
        submits = 0
        poll_errors = 0
        while True:
            # 1. submit -- the only step that costs money; never repeated while a job id is alive
            if job_id is None:
                if submits >= tries:
                    break
                try:
                    submits += 1
                    job_id = ad.submit(req)
                    journal.write(event="submitted", job=job.key, job_id=job_id, attempt=submits)
                except Exception as e:  # noqa: BLE001 - nothing was created (or the API gave no id)
                    last_err = e
                    journal.write(event="submit_error", job=job.key, attempt=submits, error=repr(e),
                                  trace=traceback.format_exc(limit=3))
                    time.sleep(min(60.0, backoff * 2 ** submits))
                    continue
            # 2. poll + fetch -- errors here (network, rate limits) keep the job id and retry
            try:
                deadline = time.time() + float(gen.get("timeout_s", 1800))
                while True:
                    st = ad.poll(job_id)
                    if st.state in ("done", "failed"):
                        break
                    if time.time() > deadline:
                        # leave it pending in the journal: the next run re-polls instead of paying again
                        log(f"  TIMEOUT {job.key}: job {job_id} still running; re-run later to collect it")
                        return False
                    time.sleep(float(gen.get("poll_interval_s", 10)))
                if st.state == "failed":
                    # the provider says the job failed: a new submission is legitimate
                    last_err = RuntimeError(st.error or "provider reported failure")
                    journal.write(event="failed", job=job.key, job_id=job_id, error=str(st.error)[:500])
                    job_id = None
                    continue
                path = ad.fetch(st, req)
            except Exception as e:  # noqa: BLE001
                last_err = e
                poll_errors += 1
                journal.write(event="poll_error", job=job.key, job_id=job_id, error=repr(e),
                              trace=traceback.format_exc(limit=3))
                if poll_errors > tries:
                    log(f"  GAVE UP polling {job.key} (job {job_id} stays pending for the next run): {e!r}")
                    return False
                time.sleep(min(60.0, backoff * 2 ** poll_errors))
                continue
            sidecar = {
                "job": job.key, "job_id": job_id, "request": req.to_dict(), "model": ad.describe(),
                "provider_meta": st.meta, "harness_commit": git_commit(), "started": t0,
                "seconds": round(time.time() - t0, 1), "cost_usd_estimate": job.cost_usd, "finished": _now(),
                "submissions": submits,
            }
            if m.get("adapter") == "mock":
                sidecar["request"].pop("params", None)  # mocks read the reference; keep it out of the record
            path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2, default=str))
            journal.write(event="finished", job=job.key, job_id=job_id)
            with ledger_lock, open(run_dir / "ledger.jsonl", "a") as f:
                f.write(json.dumps({"job": job.key, "cost_usd": job.cost_usd * max(submits, 1), "time": _now()}) + "\n")
            return True
        log(f"  FAILED {job.key}: {last_err!r}")
        return False

    by_model: dict[str, list[Job]] = {}
    for j in todo:
        by_model.setdefault(j.model, []).append(j)
    for name, js in by_model.items():
        workers = int(models[name].get("max_concurrency", 2))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(one, j): j for j in js}
            for f in as_completed(futs):
                ok = f.result()
                results["ok" if ok else "failed"] += 1
        log(f"  {name}: {results['ok']} ok, {results['failed']} failed so far")
    results["spent_usd"] = spent_usd(run_dir)
    return results


def import_outputs(ds: Dataset, run_dir: Path, model: str, src_dir: Path, log=print) -> int:
    """Copy hand-made generations (<clip_id>__s<k>.mp4) into the run layout."""
    import shutil

    n = 0
    for p in sorted(Path(src_dir).glob("*__s*.mp4")):
        cid, s = p.stem.rsplit("__s", 1)
        if cid not in ds.clips:
            log(f"  skipping {p.name}: unknown clip id")
            continue
        dst = run_dir / model / cid / f"s{int(s)}.mp4"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
        dst.with_suffix(".json").write_text(json.dumps(
            {"job": f"{model}/{cid}/s{int(s)}", "imported_from": str(p), "imported": _now(),
             "harness_commit": git_commit()}, indent=2))
        n += 1
    log(f"imported {n} videos for {model}")
    return n
