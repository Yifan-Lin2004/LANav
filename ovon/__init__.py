"""Registration entry point for LANav."""

from ovon import config  # noqa: F401
from ovon.dataset import ovon_dataset  # noqa: F401
from ovon.measurements import nav  # noqa: F401
from ovon.models import wsla_policy  # noqa: F401
from ovon.obs_transformers import relabel_teacher_actions, resize  # noqa: F401
from ovon.task import sensors, simulator  # noqa: F401
from ovon.trainers import dagger_trainer, wsla_trainer  # noqa: F401

# The DAgger oracle is an external, pinned dependency installed by
# tools/setup_external_dependencies.sh. Importing it registers its sensors.
import frontier_exploration  # noqa: F401,E402
