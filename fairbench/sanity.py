"""End-to-end check on synthetic data: every stage runs and every score lands where it must.

    python -m fairbench sanity            # ~5-10 min on a laptop CPU (320x180)
    python -m fairbench sanity --quick    # fewer models, no PINN

Expected outcomes (asserted at the end, printed as PASS/FAIL):
  oracle (video)            R, P, F near 1; the PINN reaches the LSQ dynamical score
  oracle with API quirks    R as for the oracle (alignment handles fps, size, replayed
                            input); F a little lower, because the mock makes its 16 fps
                            frames by picking the nearest 24 fps frame (up to 1/48 s off)
  static cameras            no video flagged as "camera moved" (plain walls: "unknown")
  freeze                    R near 1 but P = 0 (discarded: never moves)
  lag on variant 2          angle group VIMS about e^-1; bg group drops; color groups unaffected
  timewarp on blue          color groups drop; bg and angle groups unaffected
  baselines                 physics_from_input close to oracle; constant_velocity implausible
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .baselines import baseline_tracks
from .config import load_configs, model_entries
from .generate import plan_jobs, run_jobs
from .metrics.calibration import load_or_calibrate
from .report import plot_tracks, write_report
from .score import collect_tracks, score_tracks, write_scores
from .synthetic import build_synthetic_dataset
from .tracking.color import ColorTracker
from .validate import reference_floor


def run_sanity(out: Path, res: str = "320x180", quick: bool = False, workers: int = 1) -> dict:
    t0 = time.time()
    out = Path(out)
    cfg = load_configs()
    mcfg = cfg["metrics"]
    # LSQ is the primary dynamical score here (fast, exact); the PINN is fitted on the oracle
    # only, with fewer epochs than a real run, as a check that it reaches the same answer
    mcfg["plausibility"]["method"] = "lsq"
    mcfg["plausibility"]["pinn_epochs"] = 3000
    w, h = map(int, res.split("x"))
    data_dir = out / "data"
    if (data_dir / "manifest.json").exists():
        from .manifest import Dataset

        ds = Dataset.load(data_dir / "manifest.json")
        print(f"[sanity] using existing synthetic dataset in {data_dir}")
    else:
        print(f"[sanity] rendering synthetic dataset ({res}) ...")
        ds = build_synthetic_dataset(data_dir, width=w, height=h, ss=2, verbose=False)
    cal = load_or_calibrate(ds, noise_px=float(mcfg["noise_px"]), n_mc=20, median_k=int(mcfg["median_k"]))
    models = model_entries(cfg["models"])
    names = ["mock-oracle", "mock-freeze", "mock-lag-variant2", "mock-timewarp-blue"]
    if not quick:
        names += ["mock-oracle-api-quirks", "mock-noise"]
    gen = dict(cfg["generation"])
    gen.update({"samples_per_clip": 1, "poll_interval_s": 0, "budget_usd": 0.0})
    run_dir = out / "run"
    samples = {n: (2 if n == "mock-noise" else 1) for n in names}
    for n in names:
        jobs = plan_jobs(ds, run_dir, models, gen, [n], samples=samples[n])
        run_jobs(ds, run_dir, jobs, models, gen, log=lambda *a: None)
    tracker = ColorTracker()
    floor = reference_floor(ds, tracker, mcfg, cal)
    print("[sanity] tracker floor (reference renders): " + ", ".join(f"{g} {v['S_rob']:.3f}" for g, v in floor.items()))
    summaries = {}
    gens = {}
    for n in names:
        tracks = collect_tracks(ds, run_dir, n, tracker, mcfg, models[n].get("align", {}), log=lambda *a: None)
        methods = ("lsq", "pinn") if (n == "mock-oracle" and not quick) else ("lsq",)
        g, grp, s = score_tracks(ds, tracks, cal, mcfg, methods=methods, workers=workers, log=lambda *a: None)
        write_scores(run_dir / n / "scores", g, grp, s, {"model": n})
        summaries[n], gens[n] = s, g
        if n in ("mock-oracle", "mock-timewarp-blue"):
            plot_tracks(ds, tracks, n, run_dir / "report" / "plots")
        print(f"[sanity] scored {n}")
    for b in ("oracle", "freeze", "constant_velocity", "physics_from_input"):
        tracks = baseline_tracks(ds, b, noise_px=0.3)
        g, grp, s = score_tracks(ds, tracks, cal, mcfg, methods=("lsq",))
        write_scores(run_dir / "baselines" / b / "scores", g, grp, s, {"model": f"baseline:{b}"})
        summaries[f"baseline:{b}"] = s
    rep = write_report(summaries, run_dir / "report", baselines=tuple(k for k in summaries if k.startswith("baseline:")),
                       note="Synthetic sanity run (fairbench.synthetic): mock models built from the simulator's own continuation.")

    def G(m, gid, key):
        return summaries[m]["groups"][gid][key]

    def mean_over(m, key, groups=None):
        gs = groups or list(summaries[m]["groups"])
        return float(np.mean([G(m, g, key) for g in gs]))

    color_groups = ["thrown/color", "bounce/color", "occlude/color"]
    gids = list(summaries["mock-oracle"]["groups"])
    checks = {
        "tracker floor: reference renders score >= 0.8 in every group": min(v["S_rob"] for v in floor.values()) >= 0.8,
        "oracle: robustness within 0.1 of the tracker floor in every group": all(
            abs(G("mock-oracle", g, "S_rob") - floor[g]["S_rob"]) <= 0.1 for g in gids),
        "oracle: reference similarity >= 0.8 on average": mean_over("mock-oracle", "S_ref") >= 0.8,
        "oracle: plausibility >= 0.85 on average": mean_over("mock-oracle", "S_plaus") >= 0.85,
        # a frozen ball is perfectly invariant; only where the tracker itself is invariant
        # (floor >= 0.99) can that show up as a score of ~1 (elsewhere its static bias leaks in)
        "freeze: robustness >= 0.95 wherever the tracker floor is >= 0.99 (trivially invariant)": all(
            G("mock-freeze", g, "S_rob") >= 0.95 for g in gids if floor[g]["S_rob"] >= 0.99),
        "freeze: plausibility = 0 (discarded as never moving)": mean_over("mock-freeze", "S_plaus") == 0.0,
        "lag on variant 2: VIMS between 0.2 and 0.6 (expected ~e^-1)": 0.2 <= G("mock-lag-variant2", "thrown/angle", "S_rob") <= 0.6,
        "lag on variant 2: color groups unaffected (>= 0.9)": min(G("mock-lag-variant2", g, "S_rob") for g in color_groups) >= 0.9,
        "timewarp blue: color groups drop below 0.5": max(G("mock-timewarp-blue", g, "S_rob") for g in color_groups) < 0.5,
        "timewarp blue: angle group unaffected (>= 0.9)": G("mock-timewarp-blue", "thrown/angle", "S_rob") >= 0.9,
        "baseline physics_from_input: reference similarity >= 0.6": mean_over("baseline:physics_from_input", "S_ref") >= 0.6,
        "baseline freeze: plausibility = 0": mean_over("baseline:freeze", "S_plaus") == 0.0,
        "static camera: no generation flagged as camera moved": all(
            summaries[n]["flags"]["camera_moved"] == 0 for n in names),
    }
    if not quick:
        og = gens["mock-oracle"]
        checks["oracle: PINN and LSQ dynamical scores agree (mean |difference| <= 0.03)"] = bool(
            (og["S_dyn_pinn"] - og["S_dyn_lsq"]).abs().mean() <= 0.03)
    if "mock-oracle-api-quirks" in summaries:
        checks["oracle with API quirks: robustness within 0.15 of the tracker floor"] = all(
            abs(G("mock-oracle-api-quirks", g, "S_rob") - floor[g]["S_rob"]) <= 0.15 for g in gids)
        checks["oracle with API quirks: reference similarity >= 0.7"] = mean_over("mock-oracle-api-quirks", "S_ref") >= 0.7
    print("\n[sanity] checks")
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print(f"\n[sanity] report: {rep}   ({time.time() - t0:.0f} s)")
    result = {"tracker_floor": floor, "checks": checks, "all_pass": all(checks.values()), "report": str(rep), "seconds": time.time() - t0}
    (run_dir / "sanity.json").write_text(json.dumps(result, indent=2))
    return result
