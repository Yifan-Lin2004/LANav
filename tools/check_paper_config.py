#!/usr/bin/env python3
"""Static assertions for the architecture and training values stated in the paper."""

from pathlib import Path

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]


def load(name):
    return OmegaConf.load(ROOT / "config" / "experiments" / name)


def check_common(cfg):
    policy = cfg.habitat_baselines.rl.policy
    model = policy.wsla_config
    ppo = cfg.habitat_baselines.rl.ppo
    assert policy.name == "LANavWSLAPolicy"
    assert policy.rgb_only and policy.use_prev_action and policy.freeze_backbone
    assert (model.n_layers, model.n_hidden) == (4, 512)
    assert (model.n_heads, model.expanded_states) == (4, 4)
    assert (model.num_block, model.overlap) == (1, 0)
    assert model.max_context_length == 500
    assert ppo.num_mini_batch == 16
    assert ppo.num_steps == 500
    assert ppo.lr == 2.5e-4
    assert ppo.eps == 1e-5
    assert ppo.max_grad_norm == 0.2
    assert cfg.habitat_baselines.num_environments == 32
    assert cfg.habitat_baselines.rl.ddppo.train_encoder is False
    assert (
        cfg.habitat.dataset.data_path
        == "data/datasets/ovon/hm3d/{split}/{split}.json.gz"
    )
    assert (
        cfg.habitat.task.lab_sensors.clip_objectgoal_sensor.cache
        == "data/text_embeddings/siglip.pkl"
    )


def main():
    dagger = load("wsla_dagger.yaml")
    ppo = load("wsla_ppo.yaml")
    task = OmegaConf.load(ROOT / "config" / "tasks" / "objectnav_stretch_hm3d.yaml")
    check_common(dagger)
    check_common(ppo)
    assert task.habitat.environment.max_episode_steps == 500
    assert dagger.habitat_baselines.total_num_steps == 150_000_000
    assert ppo.habitat_baselines.total_num_steps == 100_000_000
    assert dagger.habitat_baselines.trainer_name == "ver_dagger"
    assert ppo.habitat_baselines.trainer_name == "wsla_ver"
    print("OK: paper architecture and DAgger/PPO schedule are locked in both configs")


if __name__ == "__main__":
    main()
