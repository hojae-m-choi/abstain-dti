#!/usr/bin/env bash
# Run Boltz-2 structure + affinity prediction on every YAML in an input dir.
#
#   ./run_boltz2.sh [input_dir] [out_dir]
#
# ACCEL=gpu  -> CUDA on Linux, MPS on Apple Silicon (Lightning picks it)
# ACCEL=cpu  -> portable fallback (slow: minutes per complex)
# FAST=1     -> fewer diffusion/recycling steps, for pipeline walkthroughs only
#
# MSAs come from the public ColabFold MMseqs2 server (--use_msa_server): only the
# protein sequence leaves the machine. Point MSA_SERVER_URL at a private server if needed.
set -euo pipefail
cd "$(dirname "$0")"
IN=${1:-inputs}
OUT=${2:-out}
ACCEL=${ACCEL:-gpu}
EXTRA=()
if [[ "${FAST:-0}" == "1" ]]; then
  EXTRA+=(--recycling_steps 1 --sampling_steps 50 --sampling_steps_affinity 50 --diffusion_samples_affinity 2)
fi
if [[ -n "${MSA_SERVER_URL:-}" ]]; then
  EXTRA+=(--msa_server_url "$MSA_SERVER_URL")
fi
export PYTORCH_ENABLE_MPS_FALLBACK=1
uv run boltz predict "$IN" \
  --out_dir "$OUT" \
  --model boltz2 \
  --accelerator "$ACCEL" \
  --use_msa_server \
  --output_format pdb \
  --num_workers 0 \
  --seed 0 \
  ${EXTRA[@]+"${EXTRA[@]}"}
uv run python collect_affinity.py "$OUT" labels.csv
