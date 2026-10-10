#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 FINAL_150M_DAGGER_CHECKPOINT [CONFIG] [HYDRA_OVERRIDES ...]" >&2
  exit 2
fi

release_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${release_root}"

export LANAV_DAGGER_CHECKPOINT="$1"
shift
config="${1:-config/experiments/wsla_ppo.yaml}"
if [[ $# -gt 0 ]]; then
  shift
fi

python_bin="${PYTHON:-python}"
"${python_bin}" tools/validate_checkpoint.py \
  "${LANAV_DAGGER_CHECKPOINT}" \
  --stage dagger \
  --minimum-steps "${MIN_DAGGER_STEPS:-150000000}"

num_gpus="${NUM_GPUS:-8}"
if [[ "${num_gpus}" == "1" ]]; then
  "${python_bin}" -m ovon.run \
    --run-type train \
    --exp-config "${config}" \
    "$@"
else
  "${python_bin}" -m torch.distributed.run \
    --standalone \
    --nproc_per_node="${num_gpus}" \
    -m ovon.run \
    --run-type train \
    --exp-config "${config}" \
    "$@"
fi
