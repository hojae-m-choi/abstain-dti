#!/usr/bin/env bash
# Launch the whole Phase 2 grid on a two-GPU box, detached.
#
# Fusion is the long pole, so it gets a GPU to itself; siamese takes the other;
# FLAML is CPU-only and runs alongside both. Detached, so the grid outlives the shell.
#
# CPU budget matters more than it looks: the first attempt left FLAML's boosters
# free to take all 16 cores, which starved the two GPU jobs' data loading and held
# both at 0% GPU for minutes. Each arm now gets a slice, and the loaders get workers.
#
#   ./launch_phase2.sh [start] | status | stop
set -euo pipefail

cd "$(dirname "$0")/.."           # experiments/exp2-comparing_models1
EXP_ROOT=$(pwd)
N=100000
SPLITS="cold_drug scaffold cold_target cold_both"
SEEDS="0 1 2"
TOTAL_RUNS=48
PIDFILE=logs/phase2.pids
mkdir -p logs runs

running_pids() {  # launcher pids from the pidfile that are still alive
  [ -f "$PIDFILE" ] || return 0
  while read -r pid _; do
    kill -0 "$pid" 2>/dev/null && echo "$pid"
  done < "$PIDFILE"
}

case "${1:-start}" in
status)
  echo "=== launchers ==="
  for pid in $(running_pids); do echo "  pid $pid  $(grep "^$pid " "$PIDFILE" | cut -d' ' -f2-)"; done
  [ -n "$(running_pids)" ] || echo "  (none)"
  echo "=== gpu ==="
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
  echo "=== load (cores: $(nproc)) ==="; uptime
  echo "=== finished: $(ls runs/*/metrics.json 2>/dev/null | wc -l) / $TOTAL_RUNS ==="
  for f in logs/n${N}_*.log; do
    [ -f "$f" ] || continue
    printf '  %-45s %s\n' "$(basename "$f" .log)" "$(grep -cE '^epoch' "$f" || true) epochs"
  done
  exit 0
  ;;
stop)
  for pid in $(running_pids); do kill "$pid" 2>/dev/null || true; done
  sleep 2
  # the launchers' children (uv -> python) do not die with the parent
  pkill -u "$USER" -f 'scripts/0[34]_run_' 2>/dev/null || true
  sleep 3
  echo "stopped; still alive: $(pgrep -u "$USER" -cf 'scripts/0[34]_run_' || true)"
  exit 0
  ;;
esac

if [ -n "$(running_pids)" ]; then
  echo "already running; './launch_phase2.sh stop' first" >&2
  exit 1
fi

: > "$PIDFILE"
start() {  # name arm env...
  local name=$1 arm=$2
  shift 2
  env "$@" nohup "$EXP_ROOT/scripts/run_phase.sh" "$N" "$SPLITS" "$arm" "$SEEDS" \
    > "$EXP_ROOT/logs/phase2_$name.log" 2>&1 &
  echo "$! $name ($arm)" >> "$PIDFILE"
  echo "$name -> pid $! ($arm)"
}

# 16 cores split four ways, leaving headroom for the OS and the uv wrappers
start fusion  fusion  CUDA_VISIBLE_DEVICES=0 DTI_NUM_WORKERS=4
start siamese siamese CUDA_VISIBLE_DEVICES=1 DTI_NUM_WORKERS=4
start flaml   flaml   CUDA_VISIBLE_DEVICES= FLAML_BUDGET=1800 FLAML_N_JOBS=4 \
                      OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
disown -a
sleep 30
echo
"$0" status
