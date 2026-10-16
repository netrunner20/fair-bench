#!/usr/bin/env bash
# One model, end to end: plan -> one-clip trial -> full generation -> score -> report.
# Usage: scripts/run_model.sh <model-name> [run-name]
# Stops after the trial so you can look at the video before paying for the full run.
set -euo pipefail
MODEL="$1"
RUN="${2:-main}"

python -m fairbench plan --models "$MODEL" --run "$RUN"
if [ ! -f "runs/$RUN/$MODEL/thrown_color_r/s0.mp4" ]; then
  python -m fairbench generate --models "$MODEL" --run "$RUN" --clips thrown_color_r --samples 1
  echo "Trial video: runs/$RUN/$MODEL/thrown_color_r/s0.mp4 -- check it, then run this script again."
  exit 0
fi
python -m fairbench generate --models "$MODEL" --run "$RUN"
python -m fairbench score --model "$MODEL" --run "$RUN" --plots
for b in oracle freeze constant_velocity physics_from_input; do
  [ -f "runs/$RUN/baselines/$b/scores/summary.json" ] || python -m fairbench score --model "baseline:$b" --run "$RUN"
done
python -m fairbench report --run "$RUN" --models "$MODEL" baseline:oracle baseline:freeze \
  baseline:constant_velocity baseline:physics_from_input
