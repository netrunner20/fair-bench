import numpy as np
import pytest

from fairbench.manifest import MOTION_KIND
from fairbench.metrics.plausibility import (PlausibilityConfig, lift_to_plane, morpheus_constancy, motion_extent,
                                           plausibility)


def _score(ds, clip, uv, cfg=None, methods=("lsq",)):
    cfg = cfg or PlausibilityConfig(method="lsq")
    ev = np.arange(ds.n_gen + 1) >= 1
    vis = ds.gt_visible(clip)
    plane = ds.plane(clip)
    gt_extent = motion_extent(lift_to_plane(ds.gt_pixels(clip), clip.camera, plane, ds.up), ev)
    return plausibility(uv, clip.camera, plane, ds.up, ds.g, ds.floor_contact_height(), MOTION_KIND[clip.motion],
                        ds.eval_times(), ev, vis, np.ones(len(uv)), gt_extent, cfg, methods=methods)


def test_morpheus_constancy_formula():
    assert morpheus_constancy(np.full(10, 9.81)) == 1.0
    x = np.array([9.0, 10.0, 11.0])
    assert abs(morpheus_constancy(x) - 1 / (1 + np.std(x) / 10.0)) < 1e-12


@pytest.mark.parametrize("cid", ["thrown_color_r", "bounce_color_r", "occlude_color_r", "thrown_angle_3"])
def test_ground_truth_is_plausible(mem_ds, cid):
    c = mem_ds.clips[cid]
    rng = np.random.default_rng(1)
    uv = mem_ds.gt_pixels(c) + rng.normal(0, 0.3, (mem_ds.n_gen + 1, 2))
    p = _score(mem_ds, c, uv)
    assert p["discard"] is None
    assert p["S_dyn_lsq"] > 0.98, p
    assert p["S_inv"] > 0.85, p


def test_wrong_gravity_and_freeze_are_not(mem_ds):
    c = mem_ds.clips["thrown_color_r"]
    X = mem_ds.gt_world(c)
    t = np.arange(len(X))
    # the same motion played 30% slower: effective gravity 0.49 g
    Xs = np.stack([np.interp(t * 0.7, t, X[:, k]) for k in range(3)], 1)
    good = _score(mem_ds, c, c.camera.project(X))
    slow = _score(mem_ds, c, c.camera.project(Xs))
    assert slow["S_dyn_lsq"] < good["S_dyn_lsq"] - 0.05
    frozen = _score(mem_ds, c, np.repeat(c.camera.project(X[:1]), len(X), axis=0))
    assert frozen["discard"] == "never_moves" and frozen["S_dyn"] == 0.0


def test_energy_gain_is_flagged(mem_ds):
    from fairbench.synthetic import simulate_ballistic

    c = mem_ds.clips["bounce_color_r"]
    # every bounce higher than the one before (restitution 1.1: energy from nowhere) --
    # each flight is a perfect parabola, so only the contact check can see it
    t = (np.arange(mem_ds.n_gen + 1)) / mem_ds.fps
    X = simulate_ballistic((-0.7, 0.0, 0.4), (0.3, 0.0, 0.0), t, 0.1, 1.1)
    p = _score(mem_ds, c, c.camera.project(X))
    assert p["S_dyn_lsq"] > 0.95  # per-flight physics is fine...
    assert p["energy_gain_violations"] >= 1  # ...but energy grows at the contacts


def test_pinn_runs_and_agrees_with_lsq_on_clean_data(mem_ds):
    pytest.importorskip("torch")
    c = mem_ds.clips["occlude_color_r"]
    uv = mem_ds.gt_pixels(c)
    cfg = PlausibilityConfig(method="pinn", pinn_epochs=1500, pinn_patience=300)
    p = _score(mem_ds, c, uv, cfg, methods=("lsq", "pinn"))
    assert p["S_dyn_lsq"] > 0.99
    assert p["S_dyn_pinn"] > 0.9
