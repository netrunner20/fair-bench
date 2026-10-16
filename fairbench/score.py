"""From generated videos (or baseline trajectories) to the three FAIR-Bench scores.

    videos -> align -> track -> clean -> eval-grid tracks -> metrics -> tables

Outputs per scored model (``<run>/<model>/scores/``):
    per_generation.csv   one row per generated video: reference error, plausibility, flags
    per_group.csv        one row per intervention group: robustness (raw + 0-1), seed floor
    summary.json         per-group scores and every flag needed to read them
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .align import align_output, camera_motion
from .manifest import GROUP_VARIANTS, MOTION_KIND, Clip, Dataset
from .metrics.common import exp_score
from .metrics.plausibility import PlausibilityConfig, lift_to_plane, motion_extent, plausibility
from .metrics.reference import reference_error
from .metrics.trajectory import group_distance
from .metrics.vims import group_vims
from .tracking.postprocess import correct_sphere_centers, remove_single_frame_jumps, resample_track
from .video import read_video


@dataclass
class TrackRecord:
    clip_id: str
    sample: int
    uv: np.ndarray  # (n_gen+1, 2) ball-center pixels (reference size), eval grid, NaN = not found
    n_components: np.ndarray  # (n_gen+1,)
    info: dict = field(default_factory=dict)

    def save(self, path: Path) -> None:
        np.savez_compressed(path.with_suffix(".track.npz"), uv=self.uv, n_components=self.n_components)
        path.with_suffix(".track.json").write_text(json.dumps(self.info, indent=2, default=_jsonable))

    @classmethod
    def load(cls, path: Path, clip_id: str, sample: int) -> "TrackRecord":
        z = np.load(path.with_suffix(".track.npz"))
        info = json.loads(path.with_suffix(".track.json").read_text())
        return cls(clip_id, sample, z["uv"], z["n_components"], info)


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


# --------------------------------------------------------------------------- tracking
def _ball_roi(ds: Dataset, clip: Clip, margin_r: float = 3.0) -> tuple[int, int, int, int]:
    uv = ds.gt_pixels(clip, eval_grid=False)[: ds.input_frames]
    r = float(np.max(ds.gt_radius_px(clip, eval_grid=False)[: ds.input_frames]))
    m = margin_r * r
    return (int(np.floor(uv[:, 0].min() - m)), int(np.floor(uv[:, 1].min() - m)),
            int(np.ceil(uv[:, 0].max() + m)), int(np.ceil(uv[:, 1].max() + m)))


def track_video(ds: Dataset, clip: Clip, video: Path, tracker, mcfg: dict, align_cfg: dict | None = None) -> TrackRecord:
    align_cfg = align_cfg or {}
    out_frames, out_fps = read_video(video)
    in_frames, _ = read_video(clip.video)
    in_frames = in_frames[: ds.input_frames]
    av = align_output(out_frames, out_fps, in_frames, ds.fps, roi_ref=_ball_roi(ds, clip),
                      includes_input=align_cfg.get("includes_input", "auto"),
                      extra_offset=int(align_cfg.get("extra_offset", 0)),
                      expected_duration_s=ds.n_gen / ds.fps,
                      repeats_last_input=bool(align_cfg.get("repeats_last_input", False)))
    sx, sy = av.scale
    gt0 = ds.gt_pixels(clip)[0]
    r0 = float(ds.gt_radius_px(clip)[0])
    res = tracker.track(av.frames, gt0 / np.array([sx, sy]), r0 / (0.5 * (sx + sy)), ref_rgb=clip.ball_color)
    uv_native = res.centers * np.array([sx, sy])
    jump_px = float(mcfg.get("jump_thresh_radii", 3.0)) * r0
    uv_clean, n_jumps = remove_single_frame_jumps(uv_native, jump_px)
    t_eval = ds.eval_times()
    uv_eval = resample_track(av.times, uv_clean, t_eval, max_gap_s=1.5 / av.info["out_fps"],
                             end_tolerance_s=0.5 / av.info["out_fps"])
    ncomp = resample_track(av.times, res.n_components.astype(float), t_eval, max_gap_s=10.0, end_tolerance_s=1.0)[:, 0]
    uv_eval = correct_sphere_centers(uv_eval, clip.camera, ds.gt_depth(clip), ds.ball_radius)
    cm = camera_motion(av.frames, exclude_uv=(res.centers if np.isfinite(res.centers[0]).all() else None),
                       exclude_radius=r0 / (0.5 * (sx + sy)))
    thr = float(mcfg.get("camera_motion_px", 2.0))
    disp = cm.get("max_median_disp_px")
    # None = too little background texture to tell; reported as unknown, not as "still"
    cam_px = None if disp is None else float(disp) * 0.5 * (sx + sy)
    info = {"source": str(video), "tracker": res.info.get("tracker"), "align": av.info,
            "jumps_removed": n_jumps, "camera_motion": cm, "camera_motion_px": cam_px,
            "camera_moved": bool(cam_px is not None and cam_px > thr)}
    return TrackRecord(clip.id, -1, uv_eval, np.nan_to_num(ncomp, nan=0.0), info)


def collect_tracks(ds: Dataset, run_dir: Path, model: str, tracker, mcfg: dict, align_cfg: dict | None = None,
                   retrack: bool = False, log=print) -> dict[str, list[TrackRecord]]:
    out: dict[str, list[TrackRecord]] = {}
    mdir = run_dir / model
    for cid, clip in sorted(ds.clips.items()):
        vids = sorted((mdir / cid).glob("s*.mp4"))
        for v in vids:
            k = int(v.stem[1:])
            cached = v.with_suffix(".track.npz").exists() and not retrack
            if cached:
                rec = TrackRecord.load(v, cid, k)
                if rec.info.get("tracker") != getattr(tracker, "name", None):
                    cached = False  # tracked with another tracker: redo
            if not cached:
                rec = track_video(ds, clip, v, tracker, mcfg, align_cfg)
                rec.sample = k
                rec.save(v)
            rec.sample = k
            out.setdefault(cid, []).append(rec)
        if vids:
            log(f"  tracked {cid}: {len(vids)} video(s)")
    return out


# --------------------------------------------------------------------------- bootstrap
def _ci(x: np.ndarray) -> list[float]:
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def bootstrap_robustness(by_var: dict[str, list[np.ndarray]], variants: tuple[str, ...], angle: bool, cams: dict,
                         depth: np.ndarray, focal: float, vis: np.ndarray, ev: np.ndarray, gcal: dict,
                         penalty: float | None, k_med: int, B: int, rng: np.random.Generator) -> list[float] | None:
    """95% interval of a group's robustness score, resampling the samples of each variant."""
    from .manifest import iter_pairs
    from .metrics.trajectory import pair_distance
    from .metrics.vims import vims_E

    Ks = [len(by_var[v]) for v in variants]
    if min(Ks) < 2 or B <= 0:
        return None
    if angle:
        S = np.zeros(Ks)
        for i in range(Ks[0]):
            for j in range(Ks[1]):
                for k in range(Ks[2]):
                    uvs = [by_var[variants[0]][i], by_var[variants[1]][j], by_var[variants[2]][k]]
                    E = vims_E(uvs, [cams[v] for v in variants], vis, ev, penalty, k_med)["E_m"]
                    S[i, j, k] = exp_score(E, gcal["ceil_m"], gcal["ref_m"])
        boots = []
        for _ in range(B):
            idx = [rng.integers(0, K, K) for K in Ks]
            boots.append(S[np.ix_(*idx)].mean())
        return _ci(np.array(boots))
    mats = {}
    for a, b in iter_pairs(variants):
        mats[(a, b)] = np.array([[pair_distance(ta, tb, depth, focal, vis, ev, penalty, k_med)["D_m"]
                                  for tb in by_var[b]] for ta in by_var[a]])
    boots = []
    for _ in range(B):
        idx = {v: rng.integers(0, len(by_var[v]), len(by_var[v])) for v in variants}
        D = np.mean([m[np.ix_(idx[a], idx[b])].mean() for (a, b), m in mats.items()])
        boots.append(exp_score(float(D), gcal["ceil_m"], gcal["ref_m"]))
    return _ci(np.array(boots))


def bootstrap_mean(df: pd.DataFrame, col: str, B: int, rng: np.random.Generator) -> list[float] | None:
    """95% interval of a group mean, resampling generations within each clip."""
    by_clip = [g[col].to_numpy(dtype=float) for _, g in df.groupby("clip")]
    if not by_clip or min(len(x) for x in by_clip) < 2 or B <= 0:
        return None
    boots = [np.mean(np.concatenate([x[rng.integers(0, len(x), len(x))] for x in by_clip])) for _ in range(B)]
    return _ci(np.array(boots))


# ---------------------------------------------------------------------------- scoring
def _plaus_cfg(mcfg: dict) -> PlausibilityConfig:
    p = dict(mcfg.get("plausibility", {}) or {})
    known = set(PlausibilityConfig.__dataclass_fields__)
    return PlausibilityConfig(**{k: v for k, v in p.items() if k in known and k != "combine"})


def _combine_plaus(S_dyn: float, S_inv: float, how: str) -> float:
    if how == "dyn_only":
        return S_dyn
    if how == "product":
        return S_dyn * S_inv
    return 0.5 * (S_dyn + S_inv)


def _plausibility_job(args: tuple) -> dict:
    try:  # one thread per worker: the PINN's tensors are tiny, threads only add overhead
        import torch

        torch.set_num_threads(1)
    except ImportError:
        pass
    *pos, methods = args
    return plausibility(*pos, methods=methods)


def _run_plausibility(jobs: list[tuple], workers: int) -> list[dict]:
    """Plausibility of every generation; the PINN dominates, so spread it over processes."""
    if workers > 1 and len(jobs) > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(min(workers, len(jobs)), mp_context=mp.get_context("spawn")) as ex:
            return list(ex.map(_plausibility_job, jobs, chunksize=1))
    return [_plausibility_job(j) for j in jobs]


def score_tracks(ds: Dataset, tracks: dict[str, list[TrackRecord]], cal: dict, mcfg: dict,
                 methods: tuple[str, ...] = ("lsq", "pinn"), log=print,
                 workers: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    ev = np.arange(ds.n_gen + 1) >= 1
    full = float(mcfg.get("visible_full", 0.99))
    k_med = int(mcfg.get("median_k", 5))
    policy = mcfg.get("missing_policy", "penalty")
    pcfg = _plaus_cfg(mcfg)
    combine = (mcfg.get("plausibility") or {}).get("combine", "mean")
    workers = int(workers if workers is not None else mcfg.get("workers", 1))
    gen_rows, p_jobs = [], []
    for cid, recs in sorted(tracks.items()):
        c = ds.clips[cid]
        vis = ds.gt_visible(c, full)
        X = ds.gt_world(c)
        gt_uv = ds.gt_pixels(c)
        depth = ds.gt_depth(c)
        ccal = cal["clips"][cid]
        pen = ccal["penalty_m"] if policy == "penalty" else None
        plane = ds.plane(c)
        gt_extent = motion_extent(lift_to_plane(gt_uv, c.camera, plane, ds.up), ev)
        for rec in recs:
            row = {"clip": cid, "group": c.group, "variant": c.variant, "sample": rec.sample,
                   "camera_moved": rec.info.get("camera_moved", False),
                   "camera_motion_px": rec.info.get("camera_motion_px", np.nan),
                   "short_output": rec.info.get("align", {}).get("short_output", False),
                   "aspect_ok": rec.info.get("align", {}).get("aspect_ok", True)}
            for mode in ("depth_scaled", "ray"):
                r = reference_error(rec.uv, gt_uv, X, c.camera, depth, vis, ev, mode, pen, k_med)
                row[f"ref_err_{mode}_m"] = r["err_m"]
            r = reference_error(rec.uv, gt_uv, X, c.camera, depth, vis, ev, ccal["mode"], pen, k_med)
            row["ref_err_m"] = r["err_m"]
            row["ref_final_err_m"] = r["final_err_m"]
            row["lost_frac"] = r["lost_frac"]
            row["S_ref"] = exp_score(r["err_m"], ccal["ceil_m"], ccal["ref_m"])
            p_jobs.append((rec.uv, c.camera, plane, ds.up, ds.g, ds.floor_contact_height(),
                           MOTION_KIND.get(c.motion, "ballistic"), ds.eval_times(), ev, vis,
                           rec.n_components, gt_extent, pcfg, tuple(methods)))
            gen_rows.append(row)
    t0 = time.time()
    # only the PINN is slow enough to be worth extra processes
    for row, p in zip(gen_rows, _run_plausibility(p_jobs, workers if "pinn" in methods else 1)):
        row.update({k: v for k, v in p.items() if not isinstance(v, dict)})
        row["S_plaus"] = _combine_plaus(p["S_dyn"], p["S_inv"], combine)
    if "pinn" in methods and p_jobs:
        log(f"  plausibility of {len(p_jobs)} generation(s): {time.time() - t0:.0f} s ({workers} worker(s))")
    gen = pd.DataFrame(gen_rows)

    group_rows = []
    B = int(mcfg.get("bootstrap", 200))
    rng = np.random.default_rng(int(mcfg.get("bootstrap_seed", 0)))
    for gid, clips in ds.groups().items():
        variants = GROUP_VARIANTS[gid]
        by_var = {c.variant: [r.uv for r in tracks.get(c.id, [])] for c in clips}
        if not all(by_var.values()):
            continue
        vis = np.logical_and.reduce([ds.gt_visible(c, full) for c in clips])
        gcal = cal["groups"][gid]
        pen = gcal["penalty_m"] if policy == "penalty" else None
        if gid.endswith("/angle"):
            cams = {c.variant: c.camera for c in clips}
            r = group_vims(by_var, variants, cams, vis, ev, gcal["ceil_m"], gcal["ref_m"], pen, k_med)
            ci = bootstrap_robustness(by_var, variants, True, cams, None, 0.0, vis, ev, gcal, pen, k_med, B, rng)
            group_rows.append({"group": gid, "metric": "VIMS", "raw_m": r["E_m"], "S_rob": r["VIMS"], "S_rob_ci": ci,
                               "seed_raw_m": np.nan, "lost_frac": r["lost_frac"], "n_combos": r["n_triplets"]})
        else:
            c0 = clips[0]
            r = group_distance(by_var, variants, ds.gt_depth(c0), c0.camera.focal, vis, ev, pen, k_med)
            ci = bootstrap_robustness(by_var, variants, False, {}, ds.gt_depth(c0), c0.camera.focal, vis, ev, gcal,
                                      pen, k_med, B, rng)
            row = {"group": gid, "metric": "trajectory", "raw_m": r["D_m"],
                   "S_rob": exp_score(r["D_m"], gcal["ceil_m"], gcal["ref_m"]), "S_rob_ci": ci,
                   "seed_raw_m": r["seed_D_m"], "lost_frac": r["lost_frac"],
                   "n_combos": len(by_var[variants[0]]) * len(by_var[variants[1]])}
            row.update({f"pair_{k}_m": v for k, v in r["pairs"].items()})
            group_rows.append(row)
    grp = pd.DataFrame(group_rows)

    summary: dict = {"groups": {}}
    for gid in ds.groups():
        g_gen = gen[gen["group"] == gid] if len(gen) else gen
        g_grp = grp[grp["group"] == gid] if len(grp) else grp
        summary["groups"][gid] = {
            "S_rob": float(g_grp["S_rob"].iloc[0]) if len(g_grp) else float("nan"),
            "rob_raw_m": float(g_grp["raw_m"].iloc[0]) if len(g_grp) else float("nan"),
            "S_plaus": float(g_gen["S_plaus"].mean()) if len(g_gen) else float("nan"),
            "S_dyn": float(g_gen["S_dyn"].mean()) if len(g_gen) else float("nan"),
            "S_inv": float(g_gen["S_inv"].mean()) if len(g_gen) else float("nan"),
            "S_ref": float(g_gen["S_ref"].mean()) if len(g_gen) else float("nan"),
            "ref_err_m": float(g_gen["ref_err_m"].mean()) if len(g_gen) else float("nan"),
            "discard_rate": float(g_gen["discard"].notna().mean()) if len(g_gen) else float("nan"),
            "seed_raw_m": float(g_grp["seed_raw_m"].iloc[0]) if len(g_grp) else float("nan"),
            "S_rob_ci": g_grp["S_rob_ci"].iloc[0] if len(g_grp) else None,
            "S_plaus_ci": bootstrap_mean(g_gen, "S_plaus", B, rng) if len(g_gen) else None,
            "S_ref_ci": bootstrap_mean(g_gen, "S_ref", B, rng) if len(g_gen) else None,
        }
    if len(gen):
        summary["flags"] = {
            "n_generations": int(len(gen)),
            "camera_moved": int(gen["camera_moved"].sum()),
            "camera_motion_unknown": int(pd.to_numeric(gen["camera_motion_px"], errors="coerce").isna().sum()),
            "short_output": int(gen["short_output"].sum()),
            "aspect_mismatch": int((~gen["aspect_ok"].astype(bool)).sum()),
            "discarded": {str(k): int(v) for k, v in gen["discard"].value_counts().items()},
            "mean_lost_frac": float(gen["lost_frac"].mean()),
        }
    return gen, grp, summary


def write_scores(out_dir: Path, gen: pd.DataFrame, grp: pd.DataFrame, summary: dict, extra: dict | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    gen.to_csv(out_dir / "per_generation.csv", index=False)
    grp.to_csv(out_dir / "per_group.csv", index=False)
    summary.update(extra or {})
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=_jsonable))
