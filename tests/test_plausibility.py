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


def _pinn_alone(t, s, h, kind, g, cfg):
    """Reference: one segment, one network, trained on its own (the textbook loop)."""
    import torch

    torch.manual_seed(cfg.seed)
    t0, T = float(t.min()), max(float(t.max() - t.min()), 1e-3)
    s0, h0 = float(np.mean(s)), float(np.mean(h))
    td = torch.tensor((t - t0)[:, None], dtype=torch.float32)
    yd = torch.tensor(np.stack([s - s0, h - h0], 1), dtype=torch.float32)
    tp = torch.linspace(0.0, T, cfg.pinn_phys_points)[:, None]
    H = cfg.pinn_hidden
    net = torch.nn.Sequential(torch.nn.Linear(1, H), torch.nn.Tanh(), torch.nn.Linear(H, H), torch.nn.Tanh(),
                              torch.nn.Linear(H, 2))
    a = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.Adam(list(net.parameters()) + ([a] if kind == "roll" else []), lr=cfg.pinn_lr)
    for _ in range(cfg.pinn_epochs):
        opt.zero_grad()
        tq = tp.clone().requires_grad_(True)
        y = net(tq)
        d1 = [torch.autograd.grad(y[:, i:i + 1], tq, torch.ones_like(tq), create_graph=True)[0] for i in (0, 1)]
        d2 = [torch.autograd.grad(d, tq, torch.ones_like(d), create_graph=True)[0] for d in d1]
        phys = torch.mean(d2[0] ** 2 + (d2[1] + g) ** 2) if kind == "flight" else torch.mean((d2[0] - a) ** 2 + d2[1] ** 2)
        (torch.mean((net(td) - yd) ** 2) + cfg.pinn_lambda * phys).backward()
        opt.step()
    with torch.no_grad():
        p = net(td).numpy()
    return p[:, 0] + s0, p[:, 1] + h0


def test_parallel_scoring_matches_serial(mem_ds):
    """--workers N must give the same numbers (and everything it ships must pickle)."""
    from fairbench.score import _run_plausibility

    jobs = []
    for cid in ("thrown_color_r", "bounce_color_r", "occlude_color_r"):
        c = mem_ds.clips[cid]
        ev = np.arange(mem_ds.n_gen + 1) >= 1
        plane = mem_ds.plane(c)
        uv = mem_ds.gt_pixels(c)
        ext = motion_extent(lift_to_plane(uv, c.camera, plane, mem_ds.up), ev)
        jobs.append((uv, c.camera, plane, mem_ds.up, mem_ds.g, mem_ds.floor_contact_height(), MOTION_KIND[c.motion],
                     mem_ds.eval_times(), ev, mem_ds.gt_visible(c), np.ones(len(uv)), ext,
                     PlausibilityConfig(method="lsq"), ("lsq",)))
    serial = _run_plausibility(jobs, 1)
    parallel = _run_plausibility(jobs, 2)
    for a, b in zip(serial, parallel):
        assert a["S_dyn"] == b["S_dyn"] and a["S_inv"] == b["S_inv"]


def test_batched_pinn_equals_separate_training():
    pytest.importorskip("torch")
    from fairbench.metrics.plausibility import _pinn_segments

    rng = np.random.default_rng(0)
    t1 = np.arange(0, 0.5, 1 / 24)
    t2 = np.arange(0.55, 1.4, 1 / 24)
    t3 = np.arange(1.45, 2.0, 1 / 24)
    segs = [(t1, 1.0 * t1 + rng.normal(0, 1e-3, len(t1)), 0.8 + 2 * t1 - 4.9 * t1**2, "flight"),
            (t2, 1.0 * t2, 0.1 + 3 * (t2 - 0.55) - 4.9 * (t2 - 0.55) ** 2, "flight"),
            (t3, 2.0 * t3 - 0.3 * t3**2, np.full(len(t3), 0.1), "roll")]
    cfg = PlausibilityConfig(pinn_epochs=150, pinn_patience=10**6)
    batched = _pinn_segments(segs, 9.81, cfg)
    for seg, (ps, ph) in zip(segs, batched):
        rs, rh = _pinn_alone(*seg, 9.81, cfg)
        assert np.allclose(ps, rs, atol=1e-4) and np.allclose(ph, rh, atol=1e-4)
