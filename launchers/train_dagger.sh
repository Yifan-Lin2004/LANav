#!/usr/bin/env bash
set -euo pipefail

release_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${release_root}"

num_gpus="${NUM_GPUS:-8}"
python_bin="${PYTHON:-python}"
config="${1:-config/experiments/wsla_dagger.yaml}"
shift || true

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
