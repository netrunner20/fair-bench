"""Comparison report across scored models (markdown + CSV + figures)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .aggregate import GROUPS, candidates, rank_stability  # noqa: E402
from .manifest import GROUP_VARIANTS, Dataset  # noqa: E402

INK, QUIET, GRID = "#1f2328", "#6e7781", "#d0d7de"
SERIES = ["#2f6fdb", "#d1495b", "#2a9d8f", "#e9a03b", "#7a5195", "#57606a", "#8ab17d", "#bc5090"]


def _fmt(x, nd=3):
    return "" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def load_summaries(score_dirs: dict[str, Path]) -> dict[str, dict]:
    return {m: json.loads((d / "summary.json").read_text()) for m, d in score_dirs.items() if (d / "summary.json").exists()}


def group_table(summaries: dict[str, dict], key: str) -> pd.DataFrame:
    rows = []
    for m, s in summaries.items():
        row = {"model": m}
        for g in GROUPS:
            row[g] = s["groups"].get(g, {}).get(key, np.nan)
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def _md_table(df: pd.DataFrame, nd: int = 3) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join([df.index.name or ""] + cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for idx, r in df.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [_fmt(r[c], nd) if isinstance(r[c], (float, np.floating)) else str(r[c]) for c in cols]) + " |")
    return "\n".join(lines)


def plot_group_bars(tab: pd.DataFrame, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.6), dpi=150)
    n = len(tab)
    w = 0.8 / max(n, 1)
    x = np.arange(len(GROUPS))
    for i, (m, r) in enumerate(tab.iterrows()):
        ax.bar(x + (i - (n - 1) / 2) * w, [r[g] for g in GROUPS], w, label=m, color=SERIES[i % len(SERIES)])
    ax.set_xticks(x, [g.replace("/", "\n") for g in GROUPS], fontsize=8, color=INK)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("score (0-1)", color=INK)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=7, frameon=False, ncol=min(n, 4), loc="lower left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_seed_vs_intervention(summaries: dict[str, dict], path: Path) -> bool:
    """Cross-variant distance against the same-input (seed) distance, per model and group.

    Points above the diagonal: the intervention moves the generated ball more than
    re-sampling the same input does.
    """
    pts = [(m, g, v["seed_raw_m"] * 1000, v["rob_raw_m"] * 1000) for m, s in summaries.items()
           for g, v in s["groups"].items() if np.isfinite(v.get("seed_raw_m", np.nan)) and np.isfinite(v["rob_raw_m"])]
    if not pts:
        return False
    fig, ax = plt.subplots(figsize=(4.8, 4.2), dpi=150)
    models = sorted({p[0] for p in pts})
    for i, m in enumerate(models):
        xs = [p[2] for p in pts if p[0] == m]
        ys = [p[3] for p in pts if p[0] == m]
        ax.scatter(xs, ys, s=22, color=SERIES[i % len(SERIES)], label=m, zorder=3)
    hi = max(max(p[2] for p in pts), max(p[3] for p in pts)) * 1.1 + 1e-6
    ax.plot([0, hi], [0, hi], color=QUIET, lw=0.8, ls="--", zorder=1)
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_xlabel("same input, two samples (mm)", fontsize=8)
    ax.set_ylabel("across interventions (mm)", fontsize=8)
    ax.set_title("Intervention effect vs. sampling spread", loc="left", fontsize=9, color=INK)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def plot_robustness_vs_plausibility(summaries: dict[str, dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 4.2), dpi=150)
    for i, (m, s) in enumerate(summaries.items()):
        xs = [v["S_plaus"] for v in s["groups"].values()]
        ys = [v["S_rob"] for v in s["groups"].values()]
        ax.scatter(xs, ys, s=22, color=SERIES[i % len(SERIES)], label=m, zorder=3)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("physical plausibility P", fontsize=8)
    ax.set_ylabel("intervention robustness R", fontsize=8)
    ax.set_title("Robustness vs. plausibility (one point per group)", loc="left", fontsize=9, color=INK)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, frameon=False, loc="lower left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_vims_over_time(ds: Dataset, tracks: dict, model: str, out_dir: Path) -> Path | None:
    """VIMS per-frame inconsistency e_t of the camera-angle group (first sample of each view)."""
    from .metrics.vims import vims_frames

    groups = ds.groups()
    if "thrown/angle" not in groups:
        return None
    clips = groups["thrown/angle"]
    if not all(tracks.get(c.id) for c in clips):
        return None
    vis = np.logical_and.reduce([ds.gt_visible(c) for c in clips])
    e, _, _ = vims_frames([tracks[c.id][0].uv for c in clips], [c.camera for c in clips], vis, None)
    t = ds.eval_times()
    fig, ax = plt.subplots(figsize=(5.2, 2.6), dpi=150)
    ax.plot(t[1:], e[1:] * 1000, color=SERIES[0], lw=1.2)
    ax.set_xlabel("seconds after the input ends", fontsize=8)
    ax.set_ylabel("e_t (mm)", fontsize=8)
    ax.set_title(f"{model} · VIMS inconsistency over time", loc="left", fontsize=9, color=INK)
    ax.grid(color=GRID, lw=0.6)
    ax.tick_params(labelsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"vims_over_time_{model}.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def plot_tracks(ds: Dataset, tracks: dict, model: str, out_dir: Path) -> list[Path]:
    """One panel per group: the generated tracks of the 3 variants vs. the simulator's."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for gid, clips in ds.groups().items():
        fig, ax = plt.subplots(figsize=(5.2, 3.0), dpi=150)
        for i, c in enumerate(clips):
            gt = ds.gt_pixels(c)
            ax.plot(gt[:, 0], gt[:, 1], color=QUIET, lw=1.0, ls="--", label="simulator" if i == 0 else None)
            for k, rec in enumerate(tracks.get(c.id, [])):
                ax.plot(rec.uv[:, 0], rec.uv[:, 1], color=SERIES[i % len(SERIES)], lw=1.1, alpha=0.9,
                        label=f"variant {c.variant}" if k == 0 else None)
        ax.invert_yaxis()
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"{model} · {gid}", loc="left", fontsize=9, color=INK)
        ax.set_xlabel("u (px)", fontsize=8)
        ax.set_ylabel("v (px)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, frameon=False)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        p = out_dir / f"tracks_{model}_{gid.replace('/', '_')}.png"
        fig.savefig(p)
        plt.close(fig)
        paths.append(p)
    v = plot_vims_over_time(ds, tracks, model, out_dir)
    if v is not None:
        paths.append(v)
    return paths


def write_report(summaries: dict[str, dict], out_dir: Path, baselines: tuple[str, ...] = (), note: str = "") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    R = group_table(summaries, "S_rob")
    P = group_table(summaries, "S_plaus")
    F = group_table(summaries, "S_ref")
    raw = group_table(summaries, "rob_raw_m") * 1000.0
    err = group_table(summaries, "ref_err_m") * 1000.0
    disc = group_table(summaries, "discard_rate")
    for name, tab in {"robustness": R, "plausibility": P, "reference": F}.items():
        tab.to_csv(out_dir / f"{name}_by_group.csv")
        plot_group_bars(tab, {"robustness": "Intervention robustness by group",
                              "plausibility": "Physical plausibility by group",
                              "reference": "Reference similarity by group"}[name], out_dir / f"{name}_by_group.png")
    has_seed = plot_seed_vs_intervention(summaries, out_dir / "intervention_vs_seed.png")
    plot_robustness_vs_plausibility(summaries, out_dir / "robustness_vs_plausibility.png")
    ci_rows = []
    for m, s in summaries.items():
        row = {"model": m}
        for g in GROUPS:
            ci = s["groups"].get(g, {}).get("S_rob_ci")
            row[g] = "" if not ci else f"{ci[0]:.2f}-{ci[1]:.2f}"
        ci_rows.append(row)
    ci_tab = pd.DataFrame(ci_rows).set_index("model")
    cands = candidates(summaries)
    cands.to_csv(out_dir / "aggregation_candidates.csv", index=False)
    stab = rank_stability(cands, exclude=baselines)
    stab.to_csv(out_dir / "rank_stability.csv", index=False)
    mean_rule = cands[cands["group_rule"] == "mean"].set_index("model")[["R", "P", "F", "final_equal", "final_robustness_main", "final_geometric", "final_gated"]]
    lines = [
        "# FAIR-Bench results", "",
        note, "",
        "Scores are 0-1 (higher is better). R = intervention robustness, P = physical plausibility, "
        "F = reference similarity. Group-level raw values are in millimeters.", "",
        "## Overview (groups combined by mean)", "", _md_table(mean_rule), "",
        "## Intervention robustness by group", "", _md_table(R), "",
        "Raw distance (mm): trajectory distance for color/background, VIMS E for camera angle.", "",
        _md_table(raw, 1), "",
        "95% bootstrap interval of R (resampling generations; empty with one sample per clip):", "",
        _md_table(ci_tab), "",
        ("Figures: robustness_vs_plausibility.png, intervention_vs_seed.png (cross-variant vs. same-input distance)."
         if has_seed else "Figures: robustness_vs_plausibility.png (intervention_vs_seed.png needs >= 2 samples per clip)."), "",
        "## Physical plausibility by group", "", _md_table(P), "",
        "Discard rate (ball vanishes, duplicates or never moves):", "", _md_table(disc, 2), "",
        "## Reference similarity by group", "", _md_table(F), "",
        "Reference error (mm):", "", _md_table(err, 1), "",
        "## How much does the aggregation choice matter?", "",
    ]
    if len(stab):
        lines += [f"Kendall's tau between model rankings over {len(stab)} pairs of candidate rules "
                  f"(baselines excluded): min {stab['tau'].min():.2f}, median {stab['tau'].median():.2f}.",
                  "Full table: rank_stability.csv; every candidate score: aggregation_candidates.csv.", ""]
    else:
        lines += ["Needs at least 3 non-baseline models.", ""]
    flags = {m: s.get("flags", {}) for m, s in summaries.items() if s.get("flags")}
    if flags:
        lines += ["## Flags", "", "| model | generations | camera moved (unknown) | short output | aspect mismatch | mean lost frac |",
                  "|---|---|---|---|---|---|"]
        for m, f in flags.items():
            lines.append(f"| {m} | {f.get('n_generations')} | {f.get('camera_moved')} ({f.get('camera_motion_unknown', 0)}) | "
                         f"{f.get('short_output')} | {f.get('aspect_mismatch')} | {_fmt(f.get('mean_lost_frac'), 3)} |")
        lines.append("")
    p = out_dir / "report.md"
    p.write_text("\n".join(lines))
    return p
