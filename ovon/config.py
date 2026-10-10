"""Structured Hydra configuration for LANav training."""

from dataclasses import dataclass, field
from pathlib import Path

from habitat.config.default_structured_configs import (
    HabitatConfig,
    LabSensorConfig,
    MeasurementConfig,
    SimulatorConfig,
)
from habitat_baselines.config.default_structured_configs import (
    HabitatBaselinesBaseConfig,
    HabitatBaselinesRLConfig,
    PolicyConfig,
    RLConfig,
)
from hydra.core.config_search_path import ConfigSearchPath
from hydra.core.config_store import ConfigStore
from hydra.plugins.search_path_plugin import SearchPathPlugin


@dataclass
class ClipObjectGoalSensorConfig(LabSensorConfig):
    type: str = "ClipObjectGoalSensor"
    cache: str = "data/text_embeddings/siglip.pkl"


@dataclass
class StepIDSensorConfig(LabSensorConfig):
    type: str = "StepIDSensor"


@dataclass
class OVONDistanceToGoalConfig(MeasurementConfig):
    type: str = "OVONDistanceToGoal"
    distance_to: str = "VIEW_POINTS"


@dataclass
class NavmeshSettings:
    agent_max_climb: float = 0.20
    cell_height: float = 0.20


@dataclass
class OVONSimulatorConfig(SimulatorConfig):
    type: str = "OVONSim-v0"
    navmesh_settings: NavmeshSettings = field(default_factory=NavmeshSettings)


@dataclass
class OVONHabitatConfig(HabitatConfig):
    simulator: SimulatorConfig = field(default_factory=OVONSimulatorConfig)


@dataclass
class WSLAConfig(HabitatBaselinesBaseConfig):
    """Paper architecture plus code-level WSLA dimensions."""

    n_layers: int = 4
    n_hidden: int = 512
    n_heads: int = 4
    expanded_states: int = 4
    head_dim: int = 96
    max_context_length: int = 500
    intermediate_size: int = 2048
    norm_eps: float = 1e-5
    use_mlp: bool = True
    expand_v: float = 2.0
    # One full state per (head, expanded state), exactly as in the paper.
    num_block: int = 1
    overlap: int = 0
    conv_size: int = 4


@dataclass
class LANavPolicyConfig(PolicyConfig):
    name: str = "LANavWSLAPolicy"
    backbone: str = "siglip"
    rgb_only: bool = True
    use_prev_action: bool = True
    freeze_backbone: bool = True
    wsla_config: WSLAConfig = field(default_factory=WSLAConfig)


@dataclass
class LANavRLConfig(RLConfig):
    policy: LANavPolicyConfig = field(default_factory=LANavPolicyConfig)


@dataclass
class LANavBaselinesRLConfig(HabitatBaselinesRLConfig):
    rl: LANavRLConfig = field(default_factory=LANavRLConfig)


cs = ConfigStore.instance()
cs.store(
    group="habitat_baselines",
    name="habitat_baselines_rl_config_base",
    node=LANavBaselinesRLConfig(),
)
cs.store(group="habitat", name="habitat_config_base", node=OVONHabitatConfig)
cs.store(
    package="habitat.task.lab_sensors.clip_objectgoal_sensor",
    group="habitat/task/lab_sensors",
    name="clip_objectgoal_sensor",
    node=ClipObjectGoalSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.step_id_sensor",
    group="habitat/task/lab_sensors",
    name="step_id_sensor",
    node=StepIDSensorConfig,
)
cs.store(
    package="habitat.task.measurements.distance_to_goal",
    group="habitat/task/measurements",
    name="ovon_distance_to_goal",
    node=OVONDistanceToGoalConfig,
)


class LANavConfigPlugin(SearchPathPlugin):
    """Add this release's config tree and Habitat package configs to Hydra."""

    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        config_dir = Path(__file__).resolve().parents[1] / "config"
        search_path.append(provider="lanav", path=f"file://{config_dir}")
        search_path.append(provider="habitat", path="pkg://habitat.config")
        search_path.append(
            provider="habitat_baselines",
            path="pkg://habitat_baselines.config",
        )
