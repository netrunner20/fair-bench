import numpy as np

from fairbench.metrics.calibration import calibrate
from fairbench.metrics.common import exp_score, running_median
from fairbench.metrics.reference import reference_error
from fairbench.metrics.trajectory import group_distance
from fairbench.metrics.vims import deviating_view, group_vims, vims_E


def _lag(uv, k=1):
    out = uv.copy()
    out[k:] = uv[:-k]
    return out


def test_running_median_removes_two_frame_glitches_only():
    x = np.zeros(20)
    x[5:7] = 10.0  # 2-frame glitch: removed
    x[12:15] = 10.0  # 3-frame disagreement: kept
    m = running_median(x, 5)
    assert np.all(m[4:8] == 0) and np.all(m[12:15] == 10)


def test_exp_score_anchors():
    assert exp_score(0.001, 0.001, 0.010) == 1.0
    assert abs(exp_score(0.010, 0.001, 0.010) - np.exp(-1)) < 1e-12


def test_calibration_constants_are_ordered(mem_ds):
    cal = calibrate(mem_ds, n_mc=10)
    for gid, g in cal["groups"].items():
        assert 0 < g["ceil_m"] < g["ref_m"], gid
    ang = cal["groups"]["thrown/angle"]
    assert ang["E_exact_m"] < 1e-9  # exact GT tracks meet at one point


def test_vims_separates_same_and_different_motion(mem_ds):
    clips = mem_ds.groups()["thrown/angle"]
    cams = [c.camera for c in clips]
    gts = [mem_ds.gt_pixels(c) for c in clips]
    vis = np.logical_and.reduce([mem_ds.gt_visible(c) for c in clips])
    ev = np.arange(mem_ds.n_gen + 1) >= 1
    rng = np.random.default_rng(0)
    cal = calibrate(mem_ds, n_mc=10)["groups"]["thrown/angle"]
    same = [vims_E([g + rng.normal(0, 0.3, g.shape) for g in gts], cams, vis, ev)["E_m"] for _ in range(20)]
    # a view showing a different motion: speed +5% (time-scaled trajectory)
    X = mem_ds.gt_world(clips[0])
    t = np.arange(len(X))
    Xf = np.stack([np.interp(t * 1.05, t, X[:, k]) for k in range(3)], 1)
    diff = gts.copy()
    diff[1] = clips[1].camera.project(Xf)
    E_diff = vims_E(diff, cams, vis, ev)["E_m"]
    assert max(same) < E_diff
    assert deviating_view(diff, cams, vis, ev) == 1
    lag = [gts[0], _lag(gts[1]), gts[2]]
    E_lag = vims_E(lag, cams, vis, ev)["E_m"]
    assert abs(E_lag - cal["ref_m"]) / cal["ref_m"] < 0.5


def test_group_vims_and_distance_on_identical_tracks(mem_ds):
    ev = np.arange(mem_ds.n_gen + 1) >= 1
    cal = calibrate(mem_ds, n_mc=10)
    clips = mem_ds.groups()["thrown/angle"]
    tracks = {c.variant: [mem_ds.gt_pixels(c)] for c in clips}
    vis = np.logical_and.reduce([mem_ds.gt_visible(c) for c in clips])
    g = cal["groups"]["thrown/angle"]
    r = group_vims(tracks, ("1", "2", "3"), {c.variant: c.camera for c in clips}, vis, ev, g["ceil_m"], g["ref_m"], None)
    assert r["VIMS"] == 1.0
    cc = mem_ds.groups()["thrown/color"]
    tracks = {c.variant: [mem_ds.gt_pixels(c)] * 2 for c in cc}
    vis = mem_ds.gt_visible(cc[0])
    d = group_distance(tracks, ("r", "g", "b"), mem_ds.gt_depth(cc[0]), cc[0].camera.focal, vis, ev, None)
    assert d["D_m"] == 0 and d["seed_D_m"] == 0


def test_missing_frames_are_penalized(mem_ds):
    ev = np.arange(mem_ds.n_gen + 1) >= 1
    c = mem_ds.clips["thrown_color_r"]
    gt = mem_ds.gt_pixels(c)
    lost = gt.copy()
    lost[40:60] = np.nan  # ball lost for 20 frames
    vis = mem_ds.gt_visible(c)
    args = (gt, mem_ds.gt_world(c), c.camera, mem_ds.gt_depth(c), vis, ev, "depth_scaled")
    e_none = reference_error(lost, *args, penalty_m=None)["err_m"]
    e_pen = reference_error(lost, *args, penalty_m=1.0)["err_m"]
    assert e_none < 1e-9 and e_pen > 0.15
