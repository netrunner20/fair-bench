"""Dataset checks before any model is run (``fairbench validate``).

Reports what the paper's dataset section needs and what the metrics depend on:
clip inventory and video format, the determined-future check (floor contacts inside the
input), how well the tracked reference ball matches the projected ground truth (and any
systematic offset, e.g. a half-pixel convention mismatch), the score constants
(E_ceil, E_ref, ...), VIMS of the reference renders, and the angle between the camera
views' lines of sight.
"""

from __future__ import annotations

import numpy as np

from .geometry import angle_between_deg
from .manifest import MOTION_KIND, Dataset, expected_clip_ids
from .metrics.calibration import calibrate
from .metrics.vims import vims_E
from .video import probe


def _contacts(ds: Dataset, motion: str) -> list[float]:
    """Floor contacts of the ground truth (frame times; the true contact can fall between frames)."""
    from .metrics.plausibility import detect_contacts

    h = ds.height_of(ds.motions[motion].positions)
    idx = detect_contacts(h, ds.floor_contact_height(), 0.03)
    return [round(float(i / ds.fps), 3) for i in idx]


def reference_floor(ds: Dataset, tracker, mcfg: dict, cal: dict, ref_tracks: dict | None = None) -> dict:
    """Intervention robustness of the simulator's own renders, as measured by the tracker.

    The three variants of a group show exactly the same motion, so any distance found here
    is measurement error -- and if it differs between groups (e.g. the ball is harder to
    track on one background), the tracker itself is not invariant to the intervention.
    Model scores in a group should be read against this floor.
    """
    from .metrics.common import exp_score
    from .metrics.trajectory import group_distance
    from .metrics.vims import group_vims
    from .manifest import GROUP_VARIANTS
    from .score import track_video

    if ref_tracks is None:
        ref_tracks = {c.id: track_video(ds, c, c.video, tracker, mcfg, {"includes_input": True}).uv
                      for c in ds.clips.values()}
    ev = np.arange(ds.n_gen + 1) >= 1
    k = int(mcfg.get("median_k", 5))
    full = float(mcfg.get("visible_full", 0.99))
    out = {}
    for gid, clips in ds.groups().items():
        var = GROUP_VARIANTS[gid]
        tr = {c.variant: [ref_tracks[c.id]] for c in clips}
        vis = np.logical_and.reduce([ds.gt_visible(c, full) for c in clips])
        g = cal["groups"][gid]
        if gid.endswith("/angle"):
            r = group_vims(tr, var, {c.variant: c.camera for c in clips}, vis, ev, g["ceil_m"], g["ref_m"], None, k)
            out[gid] = {"raw_mm": round(r["E_m"] * 1000, 3), "S_rob": round(r["VIMS"], 4)}
        else:
            c0 = clips[0]
            r = group_distance(tr, var, ds.gt_depth(c0), c0.camera.focal, vis, ev, None, k)
            out[gid] = {"raw_mm": round(r["D_m"] * 1000, 3), "S_rob": round(exp_score(r["D_m"], g["ceil_m"], g["ref_m"]), 4)}
    return out


def validate(ds: Dataset, mcfg: dict, tracker=None, track_reference: bool = True) -> dict:
    rep: dict = {"root": str(ds.root)}
    missing = [c for c in expected_clip_ids() if c not in ds.clips]
    rep["clips"] = {"n": len(ds.clips), "missing": missing}
    fmt = {}
    for c in ds.clips.values():
        try:
            info = probe(c.video)
            ok = info.n_frames == ds.num_frames and abs(info.fps - ds.fps) < 1e-3 and (info.width, info.height) == (ds.width, ds.height)
            fmt[c.id] = {**info.to_dict(), "ok": bool(ok)}
        except Exception as e:  # noqa: BLE001
            fmt[c.id] = {"error": repr(e), "ok": False}
    rep["video_format_ok"] = all(v.get("ok") for v in fmt.values())
    rep["video_format"] = {k: v for k, v in fmt.items() if not v.get("ok")} or "all clips match"
    det = {}
    for m in ds.motions:
        cts = _contacts(ds, m)
        in_input = [t for t in cts if t < ds.input_frames / ds.fps]
        det[m] = {"kind": MOTION_KIND.get(m), "floor_contacts_s": cts, "contacts_in_input": in_input,
                  "declared": ds.motions[m].events}
    rep["determined_future"] = det
    groups = ds.groups()
    if "thrown/angle" in groups:
        cams = [c.camera for c in groups["thrown/angle"]]
        X = ds.gt_world(groups["thrown/angle"][0])
        ang = []
        for i in range(3):
            for j in range(i + 1, 3):
                a = [angle_between_deg(X[t] - cams[i].C, X[t] - cams[j].C) for t in range(len(X))]
                ang.append((round(min(a), 1), round(max(a), 1)))
        rep["angle_views_line_of_sight_deg"] = ang
    cal = calibrate(ds, noise_px=float(mcfg.get("noise_px", 0.3)), n_mc=int(mcfg.get("calibration_mc", 50)),
                    median_k=int(mcfg.get("median_k", 5)), full_visible=float(mcfg.get("visible_full", 0.99)))
    rep["score_constants_mm"] = {g: {k: round(v * 1000, 3) for k, v in d.items() if k.endswith("_m")}
                                 for g, d in cal["groups"].items()}
    if track_reference:
        from .score import track_video

        if tracker is None:
            from .tracking.color import ColorTracker

            tracker = ColorTracker()
        rep["tracker"] = tracker.name
        errs, offs, ref_tracks = {}, {}, {}
        for c in ds.clips.values():
            # the reference render itself, aligned like a model output that replays its input
            rec = track_video(ds, c, c.video, tracker, mcfg, {"includes_input": True})
            gt = ds.gt_pixels(c)
            vis = ds.gt_visible(c)
            d = rec.uv[vis] - gt[vis]
            ok = np.all(np.isfinite(d), axis=1)
            e = np.linalg.norm(d[ok], axis=1)
            errs[c.id] = {"median_px": round(float(np.median(e)), 3) if len(e) else None,
                          "p95_px": round(float(np.percentile(e, 95)), 3) if len(e) else None,
                          "found_frac": round(float(ok.mean()), 3)}
            offs[c.id] = np.round(np.nanmean(d, axis=0), 3).tolist()
            ref_tracks[c.id] = rec.uv
        rep["reference_tracking_error"] = errs
        mean_off = np.nanmean(np.array(list(offs.values())), axis=0)
        rep["reference_tracking_mean_offset_px"] = np.round(mean_off, 3).tolist()
        if np.any(np.abs(mean_off) > 0.3):
            rep["warning"] = ("systematic offset between tracked and projected ball: check the pixel-center "
                              "convention of K (this package puts pixel centers at +0.5, docs/DATA_FORMAT.md)")
        if "thrown/angle" in groups:
            cl = groups["thrown/angle"]
            vis = np.logical_and.reduce([ds.gt_visible(c) for c in cl])
            ev = np.arange(ds.n_gen + 1) >= 1
            r = vims_E([ref_tracks[c.id] for c in cl], [c.camera for c in cl], vis, ev, None, int(mcfg.get("median_k", 5)))
            rep["reference_VIMS_E_mm"] = round(r["E_m"] * 1000, 3)
        rep["reference_robustness_floor"] = reference_floor(ds, None, mcfg, cal, ref_tracks)
        low = {g: v for g, v in rep["reference_robustness_floor"].items() if v["S_rob"] < 0.9}
        if low:
            rep["warning_tracker_not_invariant"] = (
                f"tracked reference renders score below 0.9 in {sorted(low)}: the tracker's error depends on the "
                "intervened factor there, so model differences in these groups may be measurement artifacts")
    return rep
