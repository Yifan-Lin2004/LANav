"""Distance-to-goal measure used by OVON training."""

from typing import Any, List, Optional, Tuple

import numpy as np
from habitat.core.embodied_task import Measure
from habitat.core.registry import registry
from habitat.core.simulator import Simulator
from habitat.tasks.nav.nav import NavigationEpisode, NavigationTask


@registry.register_measure
class OVONDistanceToGoal(Measure):
    cls_uuid = "distance_to_goal"

    def __init__(self, sim: Simulator, config, *args: Any, **kwargs: Any):
        self._sim = sim
        self._config = config
        self._previous_position: Optional[Tuple[float, float, float]] = None
        self._episode_view_points: Optional[List[Tuple[float, float, float]]] = None
        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        self._previous_position = None
        goals = task._dataset.goals_by_category[episode.goals_key]
        self._episode_view_points = [
            view_point.agent_state.position
            for goal in goals
            for view_point in goal.view_points
        ]
        for category in episode.children_object_categories or []:
            scene_id = episode.scene_id.split("/")[-1]
            key = f"{scene_id}_{category}"
            for goal in task._dataset.goals_by_category.get(key, []):
                self._episode_view_points.extend(
                    view_point.agent_state.position
                    for view_point in goal.view_points
                )
        self.update_metric(episode=episode, task=task, *args, **kwargs)

    def update_metric(
        self,
        episode: NavigationEpisode,
        task: NavigationTask,
        *args: Any,
        **kwargs: Any,
    ):
        current_position = self._sim.get_agent_state().position
        if self._previous_position is not None and np.allclose(
            self._previous_position, current_position, atol=1e-4
        ):
            return
        if self._config.distance_to == "POINT":
            goals = task._dataset.goals_by_category[episode.goals_key]
            targets = [goal.position for goal in goals]
        elif self._config.distance_to == "VIEW_POINTS":
            targets = self._episode_view_points
        else:
            raise ValueError(f"Unknown distance_to={self._config.distance_to!r}")
        self._metric = self._sim.geodesic_distance(
            current_position, targets, episode
        )
        self._previous_position = tuple(current_position)
