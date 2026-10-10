# Adapted from Habitat-Lab/OVON imitation-learning training code.
# Habitat-derived portions are covered by third_party/Habitat-Lab-LICENSE;
# project-owned modifications remain subject to the pending project license.

import torch
from habitat import logger
from habitat.config import read_write
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.rl.ddppo.policy import PointNavResNetNet  # noqa: F401.
from habitat_baselines.rl.ver.worker_common import WorkerQueues
from omegaconf import DictConfig, open_dict

from ovon.algos.dagger import DAgger, DAggerPolicy, DDPDAgger
from ovon.trainers.wsla_trainer import WSLAVERTrainer
from ovon.utils.dagger_environment_worker import construct_il_environment_workers

try:
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
except AttributeError:
    pass


def _uses_oracle_actions(trainer_name: str) -> bool:
    """Whether collection returns the oracle action to VER as the executed action."""

    return trainer_name == "ver_dagger"


@baseline_registry.register_trainer(name="ver_dagger")
class VERDAggerTrainer(WSLAVERTrainer):
    def __init__(self, config: DictConfig):
        with open_dict(config.habitat_baselines.rl.policy):  # allow new keys
            with read_write(config.habitat_baselines.rl.policy):  # allow write
                if hasattr(config.habitat_baselines.rl.policy, "original_name"):
                    original_name = config.habitat_baselines.rl.policy.original_name
                else:
                    original_name = config.habitat_baselines.rl.policy.name
                if original_name != "LANavWSLAPolicy":
                    raise ValueError(
                        "the LANav DAgger trainer requires LANavWSLAPolicy"
                    )
                config.habitat_baselines.rl.policy["original_name"] = (
                    # add new key "original_name"
                    original_name
                )
                # The retained environment worker executes the oracle action.
                # Return that same action from the policy so VER records it as
                # the next ``prev_actions`` value. Otherwise WSLA would receive
                # a sampled student action that was never executed.
                config.habitat_baselines.rl.policy["teacher_forcing"] = (
                    _uses_oracle_actions(config.habitat_baselines.trainer_name)
                )
                config.habitat_baselines.rl.policy.name = DAggerPolicy.__name__
        super().__init__(config)

    def _setup_actor_critic_agent(self, ppo_cfg: "DictConfig") -> None:
        r"""Same as PPOTrainer._setup_actor_critic_agent but mixes the policy class with
        DAgger mixin so that evaluate_actions induces the correct gradients, and allows
        the usage of DAgger agent instead of DDPPO or PPO. Also, the critic is gone, so
        we don't need to reset it.

        Args:
            ppo_cfg: config node with relevant params

        Returns:
            None
        """
        logger.add_filehandler(self.config.habitat_baselines.log_file)

        policy = baseline_registry.get_policy(
            self.config.habitat_baselines.rl.policy.name
        )
        observation_space = self.obs_space
        self.obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            observation_space, self.obs_transforms
        )

        self.actor_critic = policy.from_config(
            self.config,
            observation_space,
            self.policy_action_space,
            orig_action_space=self.orig_policy_action_space,
        )
        self.obs_space = observation_space
        self.actor_critic.to(self.device)

        if (
            self.config.habitat_baselines.rl.ddppo.pretrained_encoder
            or self.config.habitat_baselines.rl.ddppo.pretrained
        ):
            pretrained_state = torch.load(
                self.config.habitat_baselines.rl.ddppo.pretrained_weights,
                map_location="cpu",
            )

        if self.config.habitat_baselines.rl.ddppo.pretrained:
            self.actor_critic.load_state_dict(
                {  # type: ignore
                    k[len("actor_critic.") :]: v
                    for k, v in pretrained_state["state_dict"].items()
                }
            )
        elif self.config.habitat_baselines.rl.ddppo.pretrained_encoder:
            prefix = "actor_critic.net.visual_encoder."
            self.actor_critic.net.visual_encoder.load_state_dict(
                {
                    k[len(prefix) :]: v
                    for k, v in pretrained_state["state_dict"].items()
                    if k.startswith(prefix)
                }
            )

        if not self.config.habitat_baselines.rl.ddppo.train_encoder and hasattr(
            self.actor_critic, "net"
        ):
            self._static_encoder = True
            for param in self.actor_critic.net.visual_encoder.parameters():
                param.requires_grad_(False)

        self.agent = (DDPDAgger if self._is_distributed else DAgger).from_config(
            self.actor_critic, ppo_cfg
        )

    def _construct_environment_workers(self):
        """Override to use IL environment workers that import ovon in subprocess."""
        return construct_il_environment_workers(
            self.config, self.mp_ctx, self.queues
        )
