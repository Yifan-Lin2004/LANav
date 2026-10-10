#!/usr/bin/env bash
set -euo pipefail

release_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
external_root="${release_root}/external"
python_bin="${PYTHON:-python}"
mkdir -p "${external_root}"

"${python_bin}" -c 'import habitat_sim; assert habitat_sim.__version__ == "0.3.3", f"LANav requires habitat-sim 0.3.3, got {habitat_sim.__version__}"'

habitat_root="${external_root}/habitat-lab"
frontier_root="${external_root}/frontier_exploration"

if [[ ! -d "${habitat_root}/.git" ]]; then
  git clone https://github.com/facebookresearch/habitat-lab.git "${habitat_root}"
fi
git -C "${habitat_root}" checkout 2c7519b1baead8d5bc9557dd53d2a01e4aa9c6e5
if git -C "${habitat_root}" apply --check "${release_root}/patches/habitat_lab_training.patch"; then
  git -C "${habitat_root}" apply "${release_root}/patches/habitat_lab_training.patch"
elif ! git -C "${habitat_root}" apply --reverse --check "${release_root}/patches/habitat_lab_training.patch"; then
  echo "Habitat-Lab checkout is neither clean nor already patched." >&2
  exit 1
fi

if [[ ! -d "${frontier_root}/.git" ]]; then
  git clone https://github.com/naokiyokoyama/frontier_exploration.git "${frontier_root}"
fi
git -C "${frontier_root}" checkout a8890d68cfa0d10254238abe9266a76856cb1f17
if git -C "${frontier_root}" apply --check "${release_root}/patches/frontier_exploration_compat.patch"; then
  git -C "${frontier_root}" apply "${release_root}/patches/frontier_exploration_compat.patch"
elif ! git -C "${frontier_root}" apply --reverse --check "${release_root}/patches/frontier_exploration_compat.patch"; then
  echo "frontier_exploration checkout is neither clean nor already patched." >&2
  exit 1
fi

"${python_bin}" -m pip install -e "${habitat_root}/habitat-lab"
"${python_bin}" -m pip install -e "${habitat_root}/habitat-baselines"
"${python_bin}" -m pip install -e "${frontier_root}"
"${python_bin}" -m pip install -e "${release_root}"

echo "External training dependencies installed from pinned commits plus release patches."
