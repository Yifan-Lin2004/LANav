# Third-party notices

## NVIDIA GatedDeltaNet kernel

`third_party/GatedDeltaNet/lit_gpt/gated_delta_rule_ops/` is derived from the
operator sources distributed by NVlabs/GatedDeltaNet, with release-local import
compatibility changes. It is covered by the NVIDIA Source Code License-NC copied to
`third_party/GatedDeltaNet/LICENSE`. Its use and derivatives are restricted to
non-commercial research or evaluation, and redistribution must preserve the
license and notices.

## Habitat-Lab

`patches/habitat_lab_training.patch` targets Habitat-Lab commit
`2c7519b1baead8d5bc9557dd53d2a01e4aa9c6e5`. Habitat-Lab is MIT licensed; a
copy is included as `third_party/Habitat-Lab-LICENSE`. The retained DAgger/VER
training plumbing in `ovon/algos/dagger.py`,
`ovon/utils/dagger_environment_worker.py`, and `ovon/trainers/` is adapted from
Habitat-Lab and the original OVON training tree; Habitat-derived portions keep
the same MIT attribution.

## frontier_exploration

`patches/frontier_exploration_compat.patch` targets frontier_exploration commit
`a8890d68cfa0d10254238abe9266a76856cb1f17`. The current upstream project is
MIT licensed; a copy is included as `third_party/Frontier-Exploration-LICENSE`.

## Data and pretrained encoders

HM3D scenes, HM3D-OVON episodes, SigLIP weights, and generated checkpoints are
not part of this source folder. HM3D must be obtained directly from Matterport
under the Matterport Academic Use EULA. HM3D-OVON episodes and the SigLIP goal
cache are obtained from their official dataset release, and the frozen encoder
weights are obtained from the corresponding `timm` model repositories. See the
README for pinned sources and checksums.

## Project-owned files

Project-owned portions are provided as part of this research code release.
No additional license for project-owned portions is specified in this
repository. Third-party components remain subject to their included licenses
and notices.
