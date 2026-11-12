#!/usr/bin/env bash
# Drive one phase of the comparison. Arms are selected so the cheap ones can run on a
# laptop while the fusion arm runs wherever a GPU is free.
#
# The first argument is the size tag 01_subsample.py wrote (t<targets>n<pairs>), so a
# grid at a different size lands in its own run folders instead of overwriting this one.
#
#   ./run_phase.sh t300n30000 "cold_drug scaffold cold_target" "siamese flaml"
#   ./run_phase.sh t300n100000 "cold_drug scaffold cold_target cold_both" "fusion" "0 1 2"
set -euo pipefail

TAG=${1:-t300n30000}
SPLITS=${2:-"cold_drug scaffold cold_target"}
ARMS=${3:-"siamese fusion flaml"}
SEEDS=${4:-"0"}
FLAML_BUDGET=${FLAML_BUDGET:-1800}
# --no-sync runs against the environment as it stands, rather than re-resolving the lock;
# override UV if you want different behaviour.
UV=${UV:-"uv run --no-sync"}

cd "$(dirname "$0")/../../.."   # track_ml-baselines
EXP=experiments/exp2-comparing_models1
mkdir -p "$EXP/logs"

for seed in $SEEDS; do
  for split in $SPLITS; do
    for arm in $ARMS; do
      case "$arm" in
        siamese|fusion) names="$arm" ;;
        flaml)          names="flaml-ecfp flaml-chemberta" ;;
        *) echo "unknown arm '$arm'" >&2; exit 1 ;;
      esac
      for name in $names; do
        run="${TAG}_${split}_${name}_s${seed}"
        if [ -f "$EXP/runs/$run/metrics.json" ]; then
          echo "skip $run (already done)"
          continue
        fi
        echo "=== $run ==="
        if [ "$arm" = flaml ]; then
          # unbuffered: a segfault would otherwise swallow the log
          $UV python -u "$EXP/scripts/04_run_flaml.py" \
            --drug-feature "${name#flaml-}" --split "$split" --tag "$TAG" --seed "$seed" \
            --time-budget "$FLAML_BUDGET" 2>&1 | tee "$EXP/logs/$run.log" | { grep -vE "^\[flaml" || true; }
        else
          $UV python -u "$EXP/scripts/03_run_neural.py" \
            --model "$arm" --split "$split" --tag "$TAG" --seed "$seed" \
            2>&1 | tee "$EXP/logs/$run.log" | { grep -E "^run|^split|parameters|^epoch|early stop" || true; }
        fi
      done
    done
  done
done
echo "done: tag=$TAG splits=[$SPLITS] arms=[$ARMS] seeds=[$SEEDS]"
