"""LANav policy with frozen SigLIP features and a WSLA state encoder.

Only the three modalities described in the paper are consumed: egocentric
RGB, the language goal embedding and the previous discrete action.  No pose,
GPS, compass, depth, detector or dynamic-routing features are present.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .wsla_encoder import WSLAEncoder


try:
    from habitat_baselines.common.baseline_registry import baseline_registry
    from habitat_baselines.rl.ddppo.policy import PointNavResNetNet
    from habitat_baselines.rl.ppo import Net, NetPolicy

    PRECOMPUTED_VISUAL_FEATURES_KEY = PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY
except (ImportError, ModuleNotFoundError):  # Keep model tests lightweight.
    baseline_registry = None
    PRECOMPUTED_VISUAL_FEATURES_KEY = "visual_features"

    class Net(nn.Module):  # type: ignore[no-redef]
        pass

    class NetPolicy(nn.Module):  # type: ignore[no-redef]
        def __init__(self, net: nn.Module, action_space: Any, **_: Any) -> None:
            super().__init__()
            self.net = net
            self.action_space = action_space


def _register_policy(cls: type) -> type:
    if baseline_registry is not None:
        return baseline_registry.register_policy(cls)
    return cls


def _nested_attr(config: Any, path: str, default: Any = None) -> Any:
    current = config
    for part in path.split("."):
        if isinstance(current, Mapping):
            if part not in current:
                return default
            current = current[part]
        elif hasattr(current, part):
            current = getattr(current, part)
        else:
            return default
    return current


class FrozenSigLIPEncoder(nn.Module):
    """Frozen SigLIP2 B/16-256 RGB encoder (768-D output)."""

    output_size = 768
    output_shape = (768,)
    model_name = "vit_base_patch16_siglip_256.v2_webli"

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        try:
            import timm
        except (ImportError, ModuleNotFoundError) as error:
            raise ImportError(
                "FrozenSigLIPEncoder requires timm; install the release environment"
            ) from error
        # Pin the pretrained tag explicitly. In timm 1.0.22 the untagged
        # architecture resolves to v2_webli, but that default may change.
        self.model = timm.create_model(
            self.model_name, pretrained=pretrained, num_classes=0
        )
        self.model.eval()
        self.model.requires_grad_(False)

    def train(self, mode: bool = True) -> "FrozenSigLIPEncoder":
        # The paper keeps SigLIP frozen during both DAgger and PPO.
        super().train(False)
        self.model.eval()
        return self

    def forward(self, observations: Mapping[str, Tensor], *_: Any, **__: Any) -> Tensor:
        rgb = observations["rgb"]
        if rgb.ndim != 4:
            raise ValueError("rgb must be NHWC or NCHW")
        if rgb.shape[-1] == 3:
            rgb = rgb.permute(0, 3, 1, 2)
        elif rgb.shape[1] != 3:
            raise ValueError("rgb must have three channels")
        if rgb.dtype == torch.uint8:
            rgb = rgb.float().div_(255.0)
        else:
            rgb = rgb.float()
        rgb = F.interpolate(rgb, size=(256, 256), mode="bicubic", align_corners=False)
        rgb = (rgb - 0.5) / 0.5
        with torch.inference_mode():
            features = self.model(rgb)
        if features.shape[-1] != self.output_size:
            raise RuntimeError(f"SigLIP returned {features.shape[-1]} instead of 768 features")
        return features


class WSLANet(Net):
    """Habitat ``Net`` implementing the LANav observation-to-state path."""

    RGB_EMBED_DIM = 768
    GOAL_EMBED_DIM = 768
    ACTION_EMBED_DIM = 32
    TOKEN_DIM = RGB_EMBED_DIM + GOAL_EMBED_DIM + ACTION_EMBED_DIM
    GOAL_SENSOR_UUID = "clip_objectgoal"

    is_wsla = True
    is_transformer = False

    def __init__(
        self,
        observation_space: Any,
        action_space: Any,
        wsla_config: Any,
        *,
        hidden_size: int = 512,
        visual_encoder: Optional[nn.Module] = None,
        goal_sensor_uuid: str = GOAL_SENSOR_UUID,
    ) -> None:
        super().__init__()
        del observation_space  # The runtime tensors are validated in forward().
        if not hasattr(action_space, "n"):
            raise TypeError("LANav requires a discrete action space")
        if int(action_space.n) != 6:
            raise ValueError(
                "the paper policy uses six actions: move-forward, turn-left, "
                "turn-right, look-up, look-down, stop"
            )

        configured_hidden = int(_nested_attr(wsla_config, "n_hidden", hidden_size))
        if configured_hidden != hidden_size:
            raise ValueError(
                f"policy hidden_size ({hidden_size}) and WSLA n_hidden "
                f"({configured_hidden}) must match"
            )
        self._hidden_size = hidden_size
        self.goal_sensor_uuid = goal_sensor_uuid
        self.visual_encoder = visual_encoder or FrozenSigLIPEncoder()
        self.visual_encoder.requires_grad_(False)
        self.prev_action_embedding = nn.Embedding(action_space.n + 1, self.ACTION_EMBED_DIM)
        self.state_encoder = WSLAEncoder(input_size=self.TOKEN_DIM, config=wsla_config)

    @property
    def output_size(self) -> int:
        return self._hidden_size

    @property
    def is_blind(self) -> bool:
        return False

    @property
    def num_recurrent_layers(self) -> int:
        return 1

    @property
    def recurrent_hidden_state_size(self) -> int:
        return self.state_encoder.recurrent_hidden_state_size

    @property
    def perception_embedding_size(self) -> int:
        return self.RGB_EMBED_DIM

    def _visual_features(self, observations: Mapping[str, Tensor]) -> Tensor:
        if PRECOMPUTED_VISUAL_FEATURES_KEY in observations:
            features = observations[PRECOMPUTED_VISUAL_FEATURES_KEY]
        else:
            features = self.visual_encoder(observations)
        if features.ndim != 2 or features.shape[-1] != self.RGB_EMBED_DIM:
            raise ValueError("RGB SigLIP features must be [batch, 768]")
        return features

    def forward(
        self,
        observations: Dict[str, Tensor],
        rnn_hidden_states: Optional[Tensor],
        prev_actions: Tensor,
        masks: Tensor,
        rnn_build_seq_info: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Optional[Tensor], Dict[str, Tensor]]:
        visual_features = self._visual_features(observations)
        goal_features = observations[self.goal_sensor_uuid]
        if goal_features.ndim != 2 or goal_features.shape[-1] != self.GOAL_EMBED_DIM:
            raise ValueError("goal SigLIP features must be [batch, 768]")
        if goal_features.shape[0] != visual_features.shape[0]:
            raise ValueError("RGB and goal batch dimensions must match")

        batch_size = visual_features.shape[0]
        previous = prev_actions.reshape(batch_size, -1)[:, -1].long()
        active = masks.reshape(batch_size, -1)[:, -1].bool()
        # Index zero is a dedicated start token; action ids are shifted by one.
        action_indices = torch.where(active, previous + 1, torch.zeros_like(previous))
        action_features = self.prev_action_embedding(action_indices)

        token = torch.cat([visual_features, goal_features, action_features], dim=-1)
        if token.shape[-1] != self.TOKEN_DIM:
            raise RuntimeError("LANav token construction must produce 1,568 features")
        output, rnn_hidden_states = self.state_encoder(
            token,
            rnn_hidden_states,
            masks,
            rnn_build_seq_info=rnn_build_seq_info,
        )
        auxiliary = {
            "rnn_output": output,
            "perception_embed": visual_features,
        }
        return output, rnn_hidden_states, auxiliary


@_register_policy
class LANavWSLAPolicy(NetPolicy):
    """Policy registered with Habitat Baselines for DAgger and PPO."""

    is_wsla = True
    is_transformer = False

    def __init__(
        self,
        observation_space: Any,
        action_space: Any,
        wsla_config: Any,
        *,
        hidden_size: int = 512,
        policy_config: Any = None,
        aux_loss_config: Any = None,
        **_: Any,
    ) -> None:
        if _nested_attr(policy_config, "action_distribution_type", "categorical") != "categorical":
            raise ValueError("LANavWSLAPolicy supports only categorical actions")
        super().__init__(
            WSLANet(
                observation_space=observation_space,
                action_space=action_space,
                wsla_config=wsla_config,
                hidden_size=hidden_size,
            ),
            action_space=action_space,
            policy_config=policy_config,
            aux_loss_config=aux_loss_config,
        )

    @classmethod
    def from_config(
        cls, config: Any, observation_space: Any, action_space: Any, **kwargs: Any
    ) -> "LANavWSLAPolicy":
        policy_config = _nested_attr(config, "habitat_baselines.rl.policy")
        wsla_config = _nested_attr(policy_config, "wsla_config")
        if wsla_config is None:
            raise KeyError("habitat_baselines.rl.policy.wsla_config is required")
        hidden_size = int(
            _nested_attr(config, "habitat_baselines.rl.ppo.hidden_size", 512)
        )
        return cls(
            observation_space=observation_space,
            action_space=action_space,
            wsla_config=wsla_config,
            hidden_size=hidden_size,
            policy_config=policy_config,
            aux_loss_config=_nested_attr(
                config, "habitat_baselines.rl.auxiliary_losses"
            ),
            **kwargs,
        )

    @property
    def recurrent_hidden_size(self) -> int:
        return self.net.recurrent_hidden_state_size

    @property
    def num_recurrent_layers(self) -> int:
        return 1

    def freeze_visual_encoders(self) -> None:
        self.net.visual_encoder.requires_grad_(False)

    def unfreeze_visual_encoders(self) -> None:
        raise RuntimeError("SigLIP is frozen in the paper and cannot be unfrozen")


__all__ = ["FrozenSigLIPEncoder", "LANavWSLAPolicy", "WSLANet"]
