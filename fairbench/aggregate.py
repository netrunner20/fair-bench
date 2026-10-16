"""Candidate ways to combine scores -- computed side by side, chosen later.

The proposal leaves two choices open until real scores exist: how to combine the six
intervention-group scores, and the weights of the three scores (intervention robustness,
physical plausibility, reference similarity). This module computes every candidate below
for every model and measures how much the model ranking depends on the choice
(Kendall's tau between rankings). If the ranking barely moves, the choice matters little
and the simplest rule can be taken; if it moves, the paper should say so.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

GROUPS = ("thrown/color", "thrown/bg", "thrown/angle", "bounce/color", "bounce/bg", "occlude/color")
EPS = 1e-3


def over_groups(values: dict[str, float], how: str) -> float:
    v = np.array([values[g] for g in GROUPS if g in values and np.isfinite(values[g])], dtype=float)
    if len(v) == 0:
        return float("nan")
    if how == "mean":
        return float(v.mean())
    if how == "median":
        return float(np.median(v))
    if how == "worst":
        return float(v.min())
    if how == "geomean":
        return float(np.exp(np.mean(np.log(np.maximum(v, EPS)))))
    if how == "macro_intervention":
        by = {}
        for g in GROUPS:
            if g in values and np.isfinite(values[g]):
                by.setdefault(g.split("/")[1], []).append(values[g])
        return float(np.mean([np.mean(x) for x in by.values()]))
    raise ValueError(how)


GROUP_RULES = ("mean", "macro_intervention", "geomean", "worst", "median")


def final_scores(R: float, P: float, F: float) -> dict[str, float]:
    return {
        "equal": (R + P + F) / 3,
        "robustness_main": 0.5 * R + 0.25 * P + 0.25 * F,
        "geometric": float(np.cbrt(max(R, EPS) * max(P, EPS) * max(F, EPS))),
        "gated": R * P,  # robustness only counts as far as the motion is physically possible
    }


def candidates(summaries: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for model, s in summaries.items():
        g = s["groups"]
        for rule in GROUP_RULES:
            R = over_groups({k: v["S_rob"] for k, v in g.items()}, rule)
            P = over_groups({k: v["S_plaus"] for k, v in g.items()}, rule)
            F = over_groups({k: v["S_ref"] for k, v in g.items()}, rule)
            row = {"model": model, "group_rule": rule, "R": R, "P": P, "F": F}
            row.update({f"final_{k}": v for k, v in final_scores(R, P, F).items()})
            rows.append(row)
    return pd.DataFrame(rows)


def rank_stability(cands: pd.DataFrame, exclude: tuple[str, ...] = ()) -> pd.DataFrame:
    """Kendall tau between the model rankings of every pair of candidate rules."""
    from scipy.stats import kendalltau

    df = cands[~cands["model"].isin(exclude)]
    cols = [c for c in df.columns if c.startswith("final_")] + ["R"]
    keys, ranks = [], []
    for rule in GROUP_RULES:
        sub = df[df["group_rule"] == rule].set_index("model")
        for c in cols:
            keys.append(f"{rule}:{c}")
            ranks.append(sub[c])
    out = []
    for (ka, ra), (kb, rb) in combinations(zip(keys, ranks), 2):
        common = ra.index.intersection(rb.index)
        if len(common) < 3:
            continue
        tau = kendalltau(ra[common], rb[common]).statistic
        out.append({"a": ka, "b": kb, "tau": float(tau)})
    return pd.DataFrame(out)
