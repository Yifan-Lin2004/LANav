"""Resize RGB observations before they enter shared-memory rollouts."""

import copy
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from gym import spaces
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.obs_transformers import ObservationTransformer
from habitat_baselines.config.default_structured_configs import ObsTransformConfig
from habitat_baselines.utils.common import overwrite_gym_box_shape
from hydra.core.config_store import ConfigStore


@baseline_registry.register_obs_transformer()
class ResizeRGB(ObservationTransformer):
    def __init__(self, size: Tuple[int, int] = (256, 256)):
        super().__init__()
        self.size = tuple(size)

    def transform_observation_space(self, observation_space: spaces.Dict):
        observation_space = copy.deepcopy(observation_space)
        if "rgb" in observation_space.spaces:
            observation_space.spaces["rgb"] = overwrite_gym_box_shape(
                observation_space.spaces["rgb"], self.size
            )
        return observation_space

    def forward(self, observations: Dict[str, torch.Tensor]):
        if "rgb" not in observations:
            return observations
        rgb = torch.as_tensor(observations["rgb"])
        squeeze = rgb.ndim == 3
        if squeeze:
            rgb = rgb.unsqueeze(0)
        rgb = rgb.permute(0, 3, 1, 2)
        rgb = torch.nn.functional.interpolate(
            rgb.float(), size=self.size, mode="bilinear", align_corners=False
        ).to(observations["rgb"].dtype)
        observations["rgb"] = rgb.permute(0, 2, 3, 1)
        if squeeze:
            observations["rgb"] = observations["rgb"].squeeze(0)
        return observations

    @classmethod
    def from_config(cls, config):
        return cls(tuple(config.size))


@dataclass
class ResizeRGBConfig(ObsTransformConfig):
    type: str = ResizeRGB.__name__
    size: Tuple[int, int] = (256, 256)


ConfigStore.instance().store(
    package="habitat_baselines.rl.policy.obs_transforms.resize",
    group="habitat_baselines/rl/policy/obs_transforms",
    name="resize",
    node=ResizeRGBConfig,
)
