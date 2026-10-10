import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.validate_checkpoint import validate_checkpoint


def strict_state():
    """Build an allocation-free, independent fixture for all 278 state keys."""

    shapes = {
        "net.visual_encoder.model.pos_embed": (1, 256, 768),
        "net.visual_encoder.model.patch_embed.proj.weight": (768, 3, 16, 16),
        "net.visual_encoder.model.patch_embed.proj.bias": (768,),
    }
    for block in range(12):
        prefix = f"net.visual_encoder.model.blocks.{block}"
        shapes.update(
            {
                f"{prefix}.norm1.weight": (768,),
                f"{prefix}.norm1.bias": (768,),
                f"{prefix}.attn.qkv.weight": (2304, 768),
                f"{prefix}.attn.qkv.bias": (2304,),
                f"{prefix}.attn.proj.weight": (768, 768),
                f"{prefix}.attn.proj.bias": (768,),
                f"{prefix}.norm2.weight": (768,),
                f"{prefix}.norm2.bias": (768,),
                f"{prefix}.mlp.fc1.weight": (3072, 768),
                f"{prefix}.mlp.fc1.bias": (3072,),
                f"{prefix}.mlp.fc2.weight": (768, 3072),
                f"{prefix}.mlp.fc2.bias": (768,),
            }
        )
    pool = "net.visual_encoder.model.attn_pool"
    shapes.update(
        {
            "net.visual_encoder.model.norm.weight": (768,),
            "net.visual_encoder.model.norm.bias": (768,),
            f"{pool}.latent": (1, 1, 768),
            f"{pool}.q.weight": (768, 768),
            f"{pool}.q.bias": (768,),
            f"{pool}.kv.weight": (1536, 768),
            f"{pool}.kv.bias": (1536,),
            f"{pool}.proj.weight": (768, 768),
            f"{pool}.proj.bias": (768,),
            f"{pool}.norm.weight": (768,),
            f"{pool}.norm.bias": (768,),
            f"{pool}.mlp.fc1.weight": (3072, 768),
            f"{pool}.mlp.fc1.bias": (3072,),
            f"{pool}.mlp.fc2.weight": (768, 3072),
            f"{pool}.mlp.fc2.bias": (768,),
            "net.prev_action_embedding.weight": (7, 32),
            "net.state_encoder.input_projection.weight": (512, 1568),
            "net.state_encoder.input_projection.bias": (512,),
        }
    )
    for layer in range(4):
        block = f"net.state_encoder.layers.{layer}"
        attention = f"{block}.attention"
        shapes.update(
            {
                f"{block}.input_norm.weight": (512,),
                f"{attention}.A_log": (16,),
                f"{attention}.dt_bias": (16,),
                f"{attention}.expert_logits": (4,),
                f"{attention}.q_proj.weight": (384, 512),
                f"{attention}.k_proj.weight": (384, 512),
                f"{attention}.v_proj.weight": (768, 512),
                f"{attention}.beta_proj.weight": (16, 512),
                f"{attention}.decay_proj.weight": (16, 512),
                f"{attention}.q_conv.weight": (384, 1, 4),
                f"{attention}.k_conv.weight": (384, 1, 4),
                f"{attention}.v_conv.weight": (768, 1, 4),
                f"{attention}.output_gate_proj.weight": (768, 512),
                f"{attention}.output_norm.weight": (768,),
                f"{attention}.output_proj.weight": (512, 768),
                f"{block}.post_attention_norm.weight": (512,),
                f"{block}.mlp.gate_proj.weight": (2048, 512),
                f"{block}.mlp.up_proj.weight": (2048, 512),
                f"{block}.mlp.down_proj.weight": (512, 2048),
            }
        )
        for head in range(4):
            shapes[f"{attention}.q_proj_expand.{head}.weight"] = (384, 96)
            shapes[f"{attention}.k_proj_expand.{head}.weight"] = (384, 96)
    shapes.update(
        {
            "net.state_encoder.final_norm.weight": (512,),
            "action_distribution.linear.weight": (6, 512),
            "action_distribution.linear.bias": (6,),
            "critic.fc.weight": (1, 512),
            "critic.fc.bias": (1,),
        }
    )
    assert len(shapes) == 278
    return {
        f"actor_critic.{key}": torch.empty(shape, device="meta")
        for key, shape in shapes.items()
    }


def strict_config(*, stage="dagger", n_heads=4):
    if stage == "dagger":
        trainer_name = "ver_dagger"
        policy_name = "DAggerPolicy"
        stage_fields = {
            "original_name": "LANavWSLAPolicy",
            "teacher_forcing": True,
        }
    else:
        trainer_name = "wsla_ver"
        policy_name = "LANavWSLAPolicy"
        stage_fields = {}
    return OmegaConf.create(
        {
            "habitat": {"environment": {"max_episode_steps": 500}},
            "habitat_baselines": {
                "trainer_name": trainer_name,
                "rl": {
                    "policy": {
                        "name": policy_name,
                        "backbone": "siglip",
                        "rgb_only": True,
                        "use_prev_action": True,
                        "freeze_backbone": True,
                        "wsla_config": {
                            "n_layers": 4,
                            "n_hidden": 512,
                            "n_heads": n_heads,
                            "expanded_states": 4,
                            "head_dim": 96,
                            "max_context_length": 500,
                            "intermediate_size": 2048,
                            "norm_eps": 1e-5,
                            "use_mlp": True,
                            "expand_v": 2.0,
                            "num_block": 1,
                            "overlap": 0,
                            "conv_size": 4,
                        },
                        **stage_fields,
                    },
                    "ppo": {"hidden_size": 512},
                    "ddppo": {"train_encoder": False},
                },
            }
        }
    )


def make_checkpoint(*, stage="dagger", n_heads=4, state=None):
    return {
        "state_dict": strict_state() if state is None else state,
        "config": strict_config(stage=stage, n_heads=n_heads),
        "extra_state": {"step": 150_000_000},
    }


def rejection_message(checkpoint, stage="dagger"):
    try:
        validate_checkpoint(checkpoint, stage, minimum_steps=150_000_000)
    except SystemExit as error:
        return str(error)
    raise AssertionError("validator unexpectedly accepted an incompatible checkpoint")


def test_strict_dagger_checkpoint_is_accepted():
    step = validate_checkpoint(
        make_checkpoint(), "dagger", minimum_steps=150_000_000
    )
    assert step == 150_000_000


def test_wrong_head_count_is_rejected():
    message = rejection_message(make_checkpoint(n_heads=8))
    assert "n_heads" in message


def test_episode_limit_longer_than_context_is_rejected():
    checkpoint = make_checkpoint()
    checkpoint["config"].habitat.environment.max_episode_steps = 1000
    message = rejection_message(checkpoint)
    assert "max_episode_steps" in message


def test_ppo_checkpoint_is_rejected_as_dagger():
    message = rejection_message(make_checkpoint(stage="ppo"), stage="dagger")
    assert "trainer_name" in message


def test_missing_frozen_siglip_tensor_is_rejected():
    state = strict_state()
    missing = "actor_critic.net.visual_encoder.model.pos_embed"
    state.pop(missing)
    message = rejection_message(make_checkpoint(state=state))
    assert "missing keys" in message
    assert "visual_encoder.model.pos_embed" in message


def test_extra_actor_critic_tensor_is_rejected():
    state = strict_state()
    state["actor_critic.net.unexpected.weight"] = torch.empty(1, device="meta")
    message = rejection_message(make_checkpoint(state=state))
    assert "extra keys" in message
    assert "net.unexpected.weight" in message


def test_wrong_wsla_tensor_shape_is_rejected():
    state = strict_state()
    key = "actor_critic.net.state_encoder.layers.2.attention.decay_proj.weight"
    state[key] = torch.empty(8, 512, device="meta")
    message = rejection_message(make_checkpoint(state=state))
    assert "shape mismatches" in message
    assert "decay_proj.weight=(8, 512), expected (16, 512)" in message
