# Running the experiments

## 0. Install

```bash
pip install -e ".[pinn,api,dev]"                      # numpy, opencv, torch (PINN), fal/replicate clients, pytest
pip install "git+https://github.com/facebookresearch/sam2.git"
# SAM 2.1 checkpoint (e.g. sam2.1_hiera_large.pt) -> checkpoints/  (path set in configs/metrics.yaml)
pytest                                                # 20 tests, ~10 s
python -m fairbench sanity                            # end-to-end on synthetic data, ~5-15 min on CPU
```

`sanity` renders a small synthetic copy of the 18-clip design, runs the mock models
(oracle, freeze, one-frame lag on variant 2, faster blue ball, API quirks, noise) and the
baselines through generation -> tracking -> scoring -> report, then prints PASS/FAIL for
the outcomes each one must produce. Run it after any change to the metric code.

## 1. Point the harness at the Kubric dataset

```bash
python -m fairbench import-kubric --src <folder with one subfolder per clip> --out data/fairbench_v1
#   or write data/fairbench_v1/manifest.json by hand (docs/DATA_FORMAT.md)
python -m fairbench validate --tracker sam2
```

Read the `validate` output before generating anything:

* `score_constants_mm.thrown/angle`: `ceil_m` and `ref_m` should be close to the
  proposal's E_ceil = 1.3 mm and E_ref = 14.7 mm. A large difference means the noise
  model or the lag definition differs from the one used for the proposal.
* `reference_VIMS_E_mm` should be close to the proposal's 0.25 mm.
* `reference_tracking_error` (px) per clip, and `reference_tracking_mean_offset_px`
  (a systematic ~0.5 px offset means a pixel-center convention mismatch in `K`).
* `reference_robustness_floor`: robustness of the simulator's own renders per group.
  Model scores in a group are read against this floor.
* `determined_future`: floor contacts found in the input vs. the declared ones.

## 2. Add a model

In `configs/models.yaml`, every candidate from the proposal is listed with its price and
known constraints. For the one you want:

1. Choose the provider route; open the model's API page.
2. fal: set `endpoint` (exact id from the page), check the argument names in `arguments`
   and where the video URL is in the response (`output_path`). Replicate: `model_ref`,
   `input`. Anything else: the `http` adapter's `submit` / `poll` blocks
   (`fairbench/models/http.py` explains each field).
3. Export the key named in `api_key_env` (`FAL_KEY` for fal, `REPLICATE_API_TOKEN` for Replicate).
4. If the provider returns input + continuation, `align.includes_input: auto` handles it.
   If its continuation starts with a copy of the last input frame, set
   `align.repeats_last_input: true` (this is never guessed; see `align.py`).

## 3. Generate (cheap first)

```bash
python -m fairbench plan --models ltx-2.3                     # jobs, $ estimate, constraint check
python -m fairbench generate --models ltx-2.3 --dry-run
python -m fairbench generate --models ltx-2.3 --clips thrown_color_r --samples 1
#   look at runs/main/ltx-2.3/thrown_color_r/s0.mp4 and its .json record
python -m fairbench generate --models ltx-2.3                 # all 18 clips x samples_per_clip
```

* `--max-usd` / `generation.budget_usd`: a run refuses to start jobs whose estimate
  would exceed the budget (spend is tracked in `runs/<run>/ledger.jsonl`).
* Interrupted? Re-run the same command: finished videos are skipped and submitted jobs
  are re-polled by their job id (`journal.jsonl`), not paid for again.
* Generated elsewhere (console, lab GPU)? Name files `<clip_id>__s<k>.mp4` and run
  `python -m fairbench import-outputs --model NAME --dir FOLDER`.

## 4. Score and compare

```bash
python -m fairbench score --model ltx-2.3 --plots
for b in oracle freeze constant_velocity physics_from_input; do
  python -m fairbench score --model baseline:$b
done
python -m fairbench report --models ltx-2.3 <other models> baseline:oracle baseline:freeze \
    baseline:constant_velocity baseline:physics_from_input
```

Outputs:

* `runs/<run>/<model>/scores/per_generation.csv`: one row per video (reference error,
  plausibility parts, discard reason, lost-ball share, camera-moved flag, ...).
* `.../per_group.csv`: robustness per group (raw mm, 0-1, 95% interval, seed floor).
* `.../summary.json`, and `plots/` with tracked paths per group and VIMS error over time.
* `runs/<run>/report/report.md` + CSVs + figures: all models side by side, every
  aggregation candidate (`aggregation_candidates.csv`) and how much the ranking changes
  between them (`rank_stability.csv`).

Tracks are cached next to each video (`*.track.npz`); `--retrack` recomputes them.

## Configuration overrides

Any config value can be overridden on the command line, e.g.
`--set metrics.missing_policy=exclude --set metrics.plausibility.method=lsq`.
