"""Only the language-goal and step sensors used by the paper policy."""

from typing import Any

import numpy as np
from gym import spaces
from habitat.core.registry import registry
from habitat.core.simulator import Sensor, SensorTypes

from ovon.utils.utils import load_pickle


@registry.register_sensor
class ClipObjectGoalSensor(Sensor):
    cls_uuid = "clip_objectgoal"

    def __init__(self, *args: Any, config, **kwargs: Any):
        self.cache = load_pickle(config.cache)
        if not self.cache:
            raise ValueError(f"Empty SigLIP text embedding cache: {config.cache}")
        dims = {np.asarray(value).shape for value in self.cache.values()}
        if len(dims) != 1 or len(next(iter(dims))) != 1:
            raise ValueError("Every cached text embedding must be a 1-D vector")
        self._embed_dim = next(iter(dims))[0]
        if self._embed_dim != 768:
            raise ValueError(
                f"The paper policy expects 768-D SigLIP text features, got {self._embed_dim}"
            )
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.SEMANTIC

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._embed_dim,),
            dtype=np.float32,
        )

    def get_observation(self, observations, *args: Any, episode, **kwargs: Any):
        category = episode.object_category
        if category not in self.cache:
            raise KeyError(f"No cached SigLIP text embedding for {category!r}")
        return np.asarray(self.cache[category], dtype=np.float32)


@registry.register_sensor
class StepIDSensor(Sensor):
    cls_uuid = "step_id"

    def __init__(self, *args: Any, config, **kwargs: Any):
        self._episode_key = None
        self._step = 0
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(0, np.iinfo(np.int64).max, shape=(1,), dtype=np.int64)

    def get_observation(self, *args: Any, episode, **kwargs: Any):
        episode_key = (episode.scene_id, episode.episode_id)
        if episode_key != self._episode_key:
            self._episode_key = episode_key
            self._step = 0
        else:
            self._step += 1
        return np.asarray([self._step], dtype=np.int64)
