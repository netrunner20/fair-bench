# FAIR-Bench

Ongoing research on the FAIR-Bench paper: **does a video generation model's physics stay
the same when only physically irrelevant things change?** FAIR = *Framework for Assessing
Intervention Robustness*. See `Proposal.pdf` for the full proposal.

Each benchmark clip shows a ball (thrown, bouncing, or rolling behind a box) for 2 s; the
model generates the next 4 s. Every event the future depends on is visible in the input,
so exactly one continuation is correct. Three variants of each clip differ only in ball
color, back wall, or camera angle, and a model that understands physics must produce the
same motion for all three.

| Score | Question | How |
| --- | --- | --- |
| Intervention robustness (main) | Same motion across variants? | tracked-trajectory distance (color, background); VIMS triangulation (camera angle) |
| Physical plausibility | Does the motion obey physics? | Morpheus-style physics-informed scores |
| Reference error | How close to the simulator? | per-frame distance in meters at the ball's depth |

## This repository

```
fairbench/            evaluation harness (python -m fairbench ...)
  manifest.py         dataset format: 3 motions x 6 groups x 3 variants
  geometry.py         cameras, lines of sight, triangulation, sphere-center correction
  models/             adapters: fal, Replicate, generic REST, manual import, offline mocks
  generate.py         plan -> dry run -> generate, budget guard, resumable, provenance logs
  align.py            put any output (fps, size, replayed input) on the benchmark clock
  tracking/           SAM 2.1 tracker (paper), color tracker (tests), clean-up
  metrics/            trajectory distance, VIMS, Morpheus-style plausibility, reference error
  score.py            videos -> tracks -> scores (+ bootstrap intervals, seed floor)
  baselines.py        oracle, freeze, constant velocity, physics-from-input
  aggregate.py        every candidate way to combine scores + ranking stability
  report.py           comparison tables and figures
  synthetic.py        ray-traced stand-in dataset for testing the pipeline
  sanity.py           end-to-end check with known expected outcomes
configs/              dataset, generation, models (all proposal candidates), metrics
docs/                 EXPERIMENTS.md (runbook), METRICS.md (definitions), DATA_FORMAT.md, DECISIONS.md
tests/                unit tests (pytest)
```

## Quick start

```bash
pip install -e ".[pinn,api,dev]"
pytest                               # unit tests
python -m fairbench sanity           # whole pipeline on synthetic data, with PASS/FAIL checks
```

Then follow `docs/EXPERIMENTS.md`: import the Kubric renders, `validate`, add one model's
API details to `configs/models.yaml`, `plan`, a one-clip `generate`, the full run,
`score`, `report`.

Choices the proposal leaves open (model list, samples per clip, prompt, how scores are
combined, ...) have defaults so the pipeline runs; they are listed in `docs/DECISIONS.md`
and none of them is a decision.
