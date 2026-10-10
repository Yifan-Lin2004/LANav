# LANav

Research code for **Beyond Transformers: Linear Attention Policy for
Open-Vocabulary Object Goal Navigation**. This repository provides the LANav
model implementation and the two-stage training configuration for the Weighted
State-Expansion Linear Attention (WSLA) backbone on HM3D-OVON.

[Paper (arXiv)](https://arxiv.org/abs/2607.18794) ·
[Project page](https://yifan-lin2004.github.io/LANav/) ·
[Robot video](https://yifan-lin2004.github.io/LANav/video.html)

The repository root contains the research code. The project website and its
media resources are maintained under [`docs/`](docs/).

LANav receives an egocentric RGB observation, a language-specified object goal,
and the previous discrete action. Frozen SigLIP encoders produce the visual and
goal features, and a causal WSLA policy integrates the observation history to
predict the next navigation action. The supplied launchers cover DAgger
pre-training followed by PPO fine-tuning.

## Reference configuration

The default configuration corresponds to the LANav (WSLA) setting described in
the manuscript.

| Component | Setting |
|---|---|
| Task | HM3D Open-Vocabulary Object Goal Navigation |
| Policy inputs | RGB, language goal, previous action |
| RGB feature | Frozen SigLIP, 768 dimensions |
| Goal feature | Frozen SigLIP, 768 dimensions |
| Previous-action embedding | 32 dimensions |
| Policy token | 1,568 dimensions projected to a hidden size of 512 |
| Temporal backbone | 4 causal WSLA layers |
| WSLA states | 4 attention heads, 4 expanded sub-states per head |
| Context length | 500 steps |
| Action space | move-forward, turn-left, turn-right, look-up, look-down, stop |
| Stage 1 | 150 million environment steps of DAgger |
| Stage 2 | 100 million environment steps of PPO initialized from Stage 1 |
| Distributed setting | 8 GPUs, 32 environments per GPU |
| Rollout and optimization | 500-step rollouts, 16 mini-batches per update |
| Optimizer | Adam, learning rate `2.5e-4`, epsilon `1e-5`, gradient clipping `0.2` |

The hardware reported in the manuscript is 8 NVIDIA A100 80 GB GPUs. This is a
reference training configuration rather than a minimum hardware requirement.
Each full launcher invocation trains one seed; the HM3D-OVON results in Table 1
of the manuscript are reported as mean and standard deviation over three seeds.

## Repository structure

```text
config/experiments/   DAgger, PPO, and integration-smoke configurations
config/tasks/         HM3D ObjectNav task configuration
launchers/            Single-GPU and single-node distributed entry points
ovon/models/          Frozen SigLIP encoder and LANav/WSLA policy
ovon/algos/           DAgger objective and policy wrapper
ovon/trainers/        DAgger/VER and PPO/VER trainers
ovon/dataset/         HM3D-OVON episode loader
ovon/task/            Simulator, goal sensor, and step sensor
patches/              Compatibility patches for pinned external dependencies
third_party/          CUDA operators and third-party licenses
tools/                Setup, data preparation, configuration, and checkpoint tools
tests/                Model, state-history, DAgger, and configuration tests
docs/                 Project website, paper, and robot demonstration resources
```

## Installation

The reference software environment uses Python 3.9, PyTorch 2.0.1 with CUDA
11.8, Habitat-Sim 0.3.3, and the pinned Habitat-Lab and
`frontier_exploration` revisions installed by the setup script.

From the root of this repository:

```bash
conda env create -f environment.yml
conda activate lanav

conda install habitat-sim=0.3.3 headless \
  -c conda-forge -c aihabitat

bash tools/setup_external_dependencies.sh
```

The setup script clones the pinned external repositories into `external/`,
applies the included compatibility patches, and installs LANav in editable
mode. Internet access is required during installation and for the first SigLIP
weight download.

Verify the installation before starting a training job:

```bash
python -c "import habitat, habitat_sim, habitat_baselines, frontier_exploration, ovon; print('imports OK')"
python tools/check_paper_config.py
python -m pytest
```

## Data preparation

Datasets and pretrained encoder weights are not included in the research code release.
HM3D scenes are license-gated and substantially larger than the source package;
HM3D-OVON episodes and the goal-feature cache are downloaded separately from
their official release.

The default configuration expects this layout:

```text
data/
├── datasets/ovon/hm3d/
│   ├── train/train.json.gz
│   ├── train/content/*.json.gz
│   ├── val_seen/...
│   ├── val_seen_synonyms/...
│   └── val_unseen/...
├── scene_datasets/hm3d/
└── text_embeddings/siglip.pkl
```

### 1. HM3D v0.2 scenes

Request HM3D access from the official
[Matterport HM3D page](https://matterport.com/habitat-matterport-3d-research-dataset)
and create an API token after accepting the academic-use terms. Download the
training scenes with Habitat-Sim's official data utility:

```bash
export MATTERPORT_TOKEN_ID=YOUR_TOKEN_ID
export MATTERPORT_TOKEN_SECRET=YOUR_TOKEN_SECRET

python -m habitat_sim.utils.datasets_download \
  --username "$MATTERPORT_TOKEN_ID" \
  --password "$MATTERPORT_TOKEN_SECRET" \
  --uids hm3d_train_v0.2 \
  --data-path data
```

The validation scenes are also required when preparing a complete evaluation
environment:

```bash
python -m habitat_sim.utils.datasets_download \
  --username "$MATTERPORT_TOKEN_ID" \
  --password "$MATTERPORT_TOKEN_SECRET" \
  --uids hm3d_val_v0.2 \
  --data-path data
```

Do not place Matterport credentials in configuration files, logs, or submitted
artifacts.

### 2. HM3D-OVON episodes

Download the episode archive from the official
[HM3D-OVON dataset](https://huggingface.co/datasets/nyokoyama/hm3d_ovon)
at the fixed revision used by this artifact:

```bash
mkdir -p data/datasets/ovon

curl -L --fail --retry 3 \
  https://huggingface.co/datasets/nyokoyama/hm3d_ovon/resolve/f04c75812660b9464bc62ce5d50912cdd8c5ffcd/hm3d.tar.gz \
  -o data/datasets/ovon/hm3d.tar.gz

echo "ae68e49a26a0dec27e42d77f15407753095c956c0595a05ef4e08aea119c2f71  data/datasets/ovon/hm3d.tar.gz" \
  | sha256sum --check -

tar --exclude='._*' --exclude='*/._*' \
  -xzf data/datasets/ovon/hm3d.tar.gz \
  -C data/datasets/ovon
```

The exclusions remove AppleDouble metadata entries present in the upstream
archive; those entries have `.json.gz` suffixes but are not gzip-compressed
episode files.

### 3. SigLIP goal embeddings

The fixed HM3D-OVON release provides the 768-dimensional goal-feature cache
used by the policy:

```bash
mkdir -p data/text_embeddings

curl -L --fail --retry 3 \
  https://huggingface.co/datasets/nyokoyama/hm3d_ovon/resolve/f04c75812660b9464bc62ce5d50912cdd8c5ffcd/text_embeddings/siglip.pkl \
  -o data/text_embeddings/siglip.pkl

echo "9e135b787842dd7f5af3579148964287c39a436e7dea48b68bb2b10ffe74c539  data/text_embeddings/siglip.pkl" \
  | sha256sum --check -
```

The cache is a Python pickle and should be obtained from the pinned official
source above. It can instead be regenerated from the downloaded episodes:

```bash
python tools/cache_siglip_text_embeddings.py \
  --dataset-root data/datasets/ovon/hm3d \
  --output data/text_embeddings/siglip.pkl
```

The regeneration command downloads the frozen SigLIP text encoder on first use.
The cache builder pins the text encoder to
`hf-hub:timm/ViT-B-16-SigLIP-256`. The policy pins the frozen RGB encoder to
`vit_base_patch16_siglip_256.v2_webli`, which `timm` downloads automatically
when the policy is first constructed.

## Integration smoke test

The smoke configurations execute eight environment steps per stage and validate
the DAgger-to-PPO checkpoint handoff. They test the training path and are not a
substitute for paper-scale training.

```bash
NUM_GPUS=1 bash launchers/train_dagger.sh \
  config/experiments/wsla_dagger_smoke.yaml

python tools/validate_checkpoint.py \
  outputs/smoke/dagger/ckpt.final.pth \
  --stage dagger \
  --minimum-steps 8

NUM_GPUS=1 MIN_DAGGER_STEPS=8 bash launchers/train_ppo.sh \
  outputs/smoke/dagger/ckpt.final.pth \
  config/experiments/wsla_ppo_smoke.yaml

python tools/validate_checkpoint.py \
  outputs/smoke/ppo/ckpt.final.pth \
  --stage ppo \
  --minimum-steps 8
```

Use an empty output directory for a fresh smoke test. Habitat VER maintains a
resume-state file in an existing checkpoint directory and may resume an
interrupted run automatically.

## Paper-scale training

The launchers use one process per GPU on a single node. With the default
configuration, each process owns 32 Habitat environments.

### Stage 1: DAgger pre-training

```bash
NUM_GPUS=8 bash launchers/train_dagger.sh
```

After normal completion, validate the stable final checkpoint:

```bash
python tools/validate_checkpoint.py \
  outputs/wsla/dagger/ckpt.final.pth \
  --stage dagger \
  --minimum-steps 150000000
```

### Stage 2: PPO fine-tuning

The PPO launcher validates the DAgger architecture, training stage, and recorded
step count before loading the checkpoint:

```bash
NUM_GPUS=8 bash launchers/train_ppo.sh \
  outputs/wsla/dagger/ckpt.final.pth
```

Validate the completed PPO checkpoint with:

```bash
python tools/validate_checkpoint.py \
  outputs/wsla/ppo/ckpt.final.pth \
  --stage ppo \
  --minimum-steps 100000000
```

Periodic checkpoints are written as `ckpt.N.pth`; a normally completed stage
also writes `ckpt.final.pth`. Logs and TensorBoard events are stored under
`outputs/`.

To reproduce the training portion of the manuscript's three-seed protocol,
repeat both stages independently with distinct Habitat seeds and separate
checkpoint, TensorBoard, and log paths. Do not reuse a DAgger checkpoint across
nominally independent seeds. Producing the reported aggregate also requires the
a separate evaluation and aggregation procedure; see **Artifact scope and
implementation notes** for the release limitations.

## Custom data locations

Hydra overrides may be appended after the configuration argument. The PPO
launcher takes the DAgger checkpoint as its first positional argument and the
configuration file as its second:

```bash
NUM_GPUS=8 bash launchers/train_dagger.sh \
  config/experiments/wsla_dagger.yaml \
  habitat.dataset.data_path='PATH_TO_OVON/{split}/{split}.json.gz' \
  habitat.dataset.scenes_dir='PATH_TO_SCENE_DATASETS' \
  habitat.task.lab_sensors.clip_objectgoal_sensor.cache='PATH_TO_SIGLIP_CACHE'

NUM_GPUS=8 bash launchers/train_ppo.sh \
  outputs/wsla/dagger/ckpt.final.pth \
  config/experiments/wsla_ppo.yaml \
  habitat.dataset.data_path='PATH_TO_OVON/{split}/{split}.json.gz' \
  habitat.dataset.scenes_dir='PATH_TO_SCENE_DATASETS' \
  habitat.task.lab_sensors.clip_objectgoal_sensor.cache='PATH_TO_SIGLIP_CACHE'
```

## Artifact scope and implementation notes

This release provides the LANav/WSLA architecture, frozen SigLIP feature
interfaces, HM3D-OVON data loading, and the DAgger-to-PPO training pipeline.
The supplied `ovon/run.py` entry point supports training.

The package does not include trained policies or checkpoints, datasets,
evaluation and three-seed aggregation scripts, an HSSD benchmark runner, or the
Unitree Go2 deployment runtime. It therefore does not by itself reproduce every
experiment reported in the paper. The robot video is a demonstration resource.
Obtain datasets and encoder weights separately as described above; train and
validate the required checkpoints before evaluation.

## Citation

```bibtex
@misc{zhang2026lanav,
  title = {Beyond Transformers: Linear Attention Policy for
           Open-Vocabulary Object Goal Navigation},
  author = {Zhang, Jiahong and Lin, Yifan and Zhang, Yandong and
            Shen, Sijun and Wang, Kexin and Pan, Yuqi and
            Pei, Hongjuan and Wang, Wei and Li, Guoqi},
  year = {2026},
  eprint = {2607.18794},
  archivePrefix = {arXiv},
  primaryClass = {cs.RO},
  doi = {10.48550/arXiv.2607.18794},
  url = {https://arxiv.org/abs/2607.18794}
}
```

## Third-party assets and licenses

HM3D is not redistributed here. Each user must obtain it from Matterport and
comply with the
[Matterport Academic Use EULA](https://matterport.com/legal/matterport-end-user-license-agreement-academic-use-model-data).
The CUDA operators bundled under `third_party/GatedDeltaNet/` are covered by
the NVIDIA Source Code License-NC. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and the licenses under
`third_party/` before use or redistribution.
