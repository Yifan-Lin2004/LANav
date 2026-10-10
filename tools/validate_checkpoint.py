#!/usr/bin/env python3
"""Validate LANav checkpoint architecture, stage provenance, and step count."""

import argparse
from functools import lru_cache
from pathlib import Path
import sys
from typing import Mapping

import torch
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def find_step(checkpoint):
    candidates = [checkpoint.get("extra_state", {}), checkpoint.get("requeue_stats", {})]
    for candidate in candidates:
        if isinstance(candidate, dict):
            for key in ("step", "num_steps_done", "num_steps"):
                if key in candidate:
                    return int(candidate[key])
    return None


EXPECTED_CONFIG = {
    "habitat.environment.max_episode_steps": 500,
    "habitat_baselines.rl.policy.backbone": "siglip",
    "habitat_baselines.rl.policy.rgb_only": True,
    "habitat_baselines.rl.policy.use_prev_action": True,
    "habitat_baselines.rl.policy.freeze_backbone": True,
    "habitat_baselines.rl.policy.wsla_config.n_layers": 4,
    "habitat_baselines.rl.policy.wsla_config.n_hidden": 512,
    "habitat_baselines.rl.policy.wsla_config.n_heads": 4,
    "habitat_baselines.rl.policy.wsla_config.expanded_states": 4,
    "habitat_baselines.rl.policy.wsla_config.head_dim": 96,
    "habitat_baselines.rl.policy.wsla_config.max_context_length": 500,
    "habitat_baselines.rl.policy.wsla_config.intermediate_size": 2048,
    "habitat_baselines.rl.policy.wsla_config.norm_eps": 1e-5,
    "habitat_baselines.rl.policy.wsla_config.use_mlp": True,
    "habitat_baselines.rl.policy.wsla_config.expand_v": 2.0,
    "habitat_baselines.rl.policy.wsla_config.num_block": 1,
    "habitat_baselines.rl.policy.wsla_config.overlap": 0,
    "habitat_baselines.rl.policy.wsla_config.conv_size": 4,
    "habitat_baselines.rl.ppo.hidden_size": 512,
    "habitat_baselines.rl.ddppo.train_encoder": False,
}


def require_config_value(config, path, expected):
    actual = OmegaConf.select(config, path)
    if actual != expected:
        raise SystemExit(
            f"INCOMPATIBLE: config {path} is {actual!r}, expected {expected!r}"
        )


@lru_cache(maxsize=1)
def expected_actor_critic_shapes():
    """Return the complete paper-release actor-critic shape contract.

    Constructing SigLIP on the meta device gives us timm's authoritative key
    names and shapes without allocating its roughly 93 million parameters.
    The release pins both torch and timm, so this also detects checkpoints made
    with a different SigLIP implementation or timm model definition.
    """

    try:
        import timm
    except (ImportError, ModuleNotFoundError) as error:
        raise SystemExit(
            "INCOMPATIBLE: timm is required to validate the frozen SigLIP state"
        ) from error

    with torch.device("meta"):
        siglip = timm.create_model(
            "vit_base_patch16_siglip_256.v2_webli",
            pretrained=False,
            num_classes=0,
        )
    expected = {
        f"net.visual_encoder.model.{key}": tuple(tensor.shape)
        for key, tensor in siglip.state_dict().items()
    }

    expected.update(
        {
            "net.prev_action_embedding.weight": (7, 32),
            "net.state_encoder.input_projection.weight": (512, 1568),
            "net.state_encoder.input_projection.bias": (512,),
        }
    )
    for layer in range(4):
        block = f"net.state_encoder.layers.{layer}"
        attention = f"{block}.attention"
        expected.update(
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
            expected[f"{attention}.q_proj_expand.{head}.weight"] = (384, 96)
            expected[f"{attention}.k_proj_expand.{head}.weight"] = (384, 96)

    expected.update(
        {
            "net.state_encoder.final_norm.weight": (512,),
            "action_distribution.linear.weight": (6, 512),
            "action_distribution.linear.bias": (6,),
            "critic.fc.weight": (1, 512),
            "critic.fc.bias": (1,),
        }
    )
    return expected


def _summarize(items, limit=4):
    items = list(items)
    shown = ", ".join(str(item) for item in items[:limit])
    if len(items) > limit:
        shown += f", ... (+{len(items) - limit} more)"
    return shown


def validate_actor_critic_state(state):
    """Require the exact actor-critic key set and every expected tensor shape."""

    if not isinstance(state, Mapping):
        raise SystemExit("INCOMPATIBLE: state_dict is not a mapping")

    actual = {}
    invalid = []
    prefix = "actor_critic."
    for key, tensor in state.items():
        if not isinstance(key, str) or not key.startswith(prefix):
            invalid.append(key)
            continue
        name = key[len(prefix) :]
        if not hasattr(tensor, "shape"):
            invalid.append(key)
            continue
        actual[name] = tuple(tensor.shape)

    expected = expected_actor_critic_shapes()
    missing = sorted(set(expected).difference(actual))
    extra = sorted(set(actual).difference(expected))
    mismatched = sorted(
        (name, actual[name], expected[name])
        for name in set(actual).intersection(expected)
        if actual[name] != expected[name]
    )
    if invalid or missing or extra or mismatched:
        details = []
        if invalid:
            details.append(f"invalid/extra keys: {_summarize(invalid)}")
        if missing:
            details.append(f"missing keys: {_summarize(missing)}")
        if extra:
            details.append(f"extra keys: {_summarize(extra)}")
        if mismatched:
            details.append(
                "shape mismatches: "
                + _summarize(
                    f"{name}={actual_shape}, expected {expected_shape}"
                    for name, actual_shape, expected_shape in mismatched
                )
            )
        raise SystemExit(
            "INCOMPATIBLE: actor_critic state_dict contract mismatch; "
            + "; ".join(details)
        )


def validate_stage(config, stage):
    if stage == "dagger":
        require_config_value(config, "habitat_baselines.trainer_name", "ver_dagger")
        require_config_value(
            config, "habitat_baselines.rl.policy.name", "DAggerPolicy"
        )
        require_config_value(
            config,
            "habitat_baselines.rl.policy.original_name",
            "LANavWSLAPolicy",
        )
        require_config_value(
            config, "habitat_baselines.rl.policy.teacher_forcing", True
        )
    else:
        require_config_value(config, "habitat_baselines.trainer_name", "wsla_ver")
        require_config_value(
            config, "habitat_baselines.rl.policy.name", "LANavWSLAPolicy"
        )


def validate_checkpoint(checkpoint, stage, minimum_steps=0):
    """Validate an already-loaded checkpoint and return its recorded step."""

    if not isinstance(checkpoint, Mapping):
        raise SystemExit("INCOMPATIBLE: checkpoint is not a mapping")
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, Mapping):
        raise SystemExit("INCOMPATIBLE: state_dict is not a mapping")
    keys = tuple(state)
    bad = [
        key
        for key in keys
        if isinstance(key, str)
        and ("o_expert_proj" in key or ".router" in key)
    ]
    if bad:
        raise SystemExit(
            "INCOMPATIBLE: checkpoint contains historical linear/router WSLA keys: "
            + ", ".join(bad[:4])
        )
    logits = sorted(
        key
        for key in keys
        if isinstance(key, str) and key.endswith(".expert_logits")
    )
    if len(logits) != 4:
        raise SystemExit(
            f"INCOMPATIBLE: expected one expert_logits vector in each of 4 layers, got {len(logits)}"
        )
    for key in logits:
        if not hasattr(state[key], "shape") or tuple(state[key].shape) != (4,):
            actual = tuple(state[key].shape) if hasattr(state[key], "shape") else None
            raise SystemExit(
                f"INCOMPATIBLE: {key} has shape {actual}, expected (4,)"
            )

    config = checkpoint.get("config")
    if config is None:
        raise SystemExit("INCOMPATIBLE: checkpoint has no embedded training config")
    for path, expected in EXPECTED_CONFIG.items():
        require_config_value(config, path, expected)
    validate_stage(config, stage)
    validate_actor_critic_state(state)

    step = find_step(checkpoint)
    if minimum_steps and (step is None or step < minimum_steps):
        raise SystemExit(
            f"INCOMPATIBLE: checkpoint step is {step}; required >= {minimum_steps}"
        )
    return step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--stage", choices=("dagger", "ppo"), required=True)
    parser.add_argument("--minimum-steps", type=int, default=0)
    args = parser.parse_args()

    # Checkpoint configs can be pickled structured dataclasses. Import the
    # release definitions immediately before torch.load resolves their symbols.
    import ovon.config  # noqa: F401

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    step = validate_checkpoint(checkpoint, args.stage, args.minimum_steps)
    print(
        "OK: strict LANav WSLA checkpoint "
        f"(stage={args.stage}, 1568->512, 4 layers, H=4, E=4), step={step}"
    )


if __name__ == "__main__":
    main()
