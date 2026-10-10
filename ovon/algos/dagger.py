#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license copied at
# third_party/Habitat-Lab-LICENSE in this release.

import collections
import inspect
from dataclasses import dataclass
from typing import Dict, Optional, Union

import torch
import torch.optim as optim
from gym import spaces
from habitat.utils import profiling_wrapper
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.config.default_structured_configs import PolicyConfig
from habitat_baselines.rl.ddppo.algo.ddppo import DecentralizedDistributedMixin
from habitat_baselines.rl.ddppo.policy import PointNavResNetNet  # noqa: F401.
from habitat_baselines.rl.ppo import PPO, Policy
from habitat_baselines.rl.ppo.policy import NetPolicy
from habitat_baselines.rl.ver.ver_rollout_storage import VERRolloutStorage
from habitat_baselines.utils.common import (
    CategoricalNet,
    GaussianNet,
    LagrangeInequalityCoefficient,
    inference_mode,
)
from omegaconf import DictConfig
from torch import nn

from ovon.obs_transformers.relabel_teacher_actions import RelabelTeacherActions

EPS_PPO = 1e-5


def _compute_dagger_loss(
    log_probs: torch.Tensor,
    inflections: Optional[torch.Tensor] = None,
    inflection_weight: float = 2.11,
    is_coeffs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return the sample-weighted oracle negative log likelihood.

    Inflection weighting is normalized so that, without VER importance
    coefficients, this is exactly the historical weighted-mean objective. VER
    coefficients are applied before the final reduction, as required to correct
    its non-uniform per-environment sampling.
    """

    per_sample_loss = -log_probs
    if inflections is not None:
        if inflections.shape != log_probs.shape:
            raise ValueError("inflections and log_probs must have identical shapes")
        inflection_weights = torch.where(
            inflections == 1,
            torch.full_like(log_probs, inflection_weight),
            torch.ones_like(log_probs),
        )
        normalization = inflection_weights.mean(dim=0, keepdim=True)
        per_sample_loss = per_sample_loss * (inflection_weights / normalization)

    if is_coeffs is not None:
        if is_coeffs.shape != log_probs.shape:
            raise ValueError("is_coeffs and log_probs must have identical shapes")
        per_sample_loss = per_sample_loss * is_coeffs.clamp(max=1.0).to(
            dtype=log_probs.dtype, device=log_probs.device
        )

    return per_sample_loss.mean()


class DAgger(PPO):
    def __init__(
        self,
        actor_critic: NetPolicy,
        clip_param: float,
        ppo_epoch: int,
        num_mini_batch: int,
        value_loss_coef: float,
        entropy_coef: float,
        lr: Optional[float] = None,
        eps: Optional[float] = None,
        max_grad_norm: Optional[float] = None,
        use_clipped_value_loss: bool = False,
        use_normalized_advantage: bool = True,
        entropy_target_factor: float = 0.0,
        use_adaptive_entropy_pen: bool = False,
    ) -> None:
        nn.Module.__init__(self)

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.inflection_weight = 2.11

        if hasattr(self.actor_critic, "net"):
            self.device = next(actor_critic.parameters()).device

        if (
            use_adaptive_entropy_pen
            and hasattr(self.actor_critic, "num_actions")
            and getattr(self.actor_critic, "action_distribution_type", None)
            == "gaussian"
        ):
            num_actions = self.actor_critic.num_actions

            self.entropy_coef = LagrangeInequalityCoefficient(
                -float(entropy_target_factor) * num_actions,
                init_alpha=entropy_coef,
                alpha_max=1.0,
                alpha_min=1e-4,
                greater_than=True,
            ).to(device=self.device)

        self.use_normalized_advantage = use_normalized_advantage

        params = list(filter(lambda p: p.requires_grad, self.parameters()))

        if len(params) > 0:
            optim_cls = optim.Adam
            optim_kwargs = dict(
                params=params,
                lr=lr,
                eps=eps,
            )
            signature = inspect.signature(optim_cls.__init__)
            if "foreach" in signature.parameters:
                optim_kwargs["foreach"] = True
            else:
                try:
                    import torch.optim._multi_tensor
                except ImportError:
                    pass
                else:
                    optim_cls = torch.optim._multi_tensor.Adam

            self.optimizer = optim_cls(**optim_kwargs)
        else:
            self.optimizer = None

        self.non_ac_params = [
            p
            for name, p in self.named_parameters()
            if not name.startswith("actor_critic.")
        ]

    def update(self, rollouts: RolloutStorage) -> Dict[str, float]:
        learner_metrics = collections.defaultdict(list)

        def record_min_mean_max(t: torch.Tensor, prefix: str):
            for name, op in (
                ("min", torch.min),
                ("mean", torch.mean),
                ("max", torch.max),
            ):
                learner_metrics[f"{prefix}_{name}"].append(op(t))

        for epoch in range(self.ppo_epoch):
            profiling_wrapper.range_push("DAgger.update epoch")
            data_generator = rollouts.recurrent_generator(None, self.num_mini_batch)

            for _bid, batch in enumerate(data_generator):
                self._set_grads_to_none()
                # TODO: see if casting from torch.uint8 to long is necessary
                teacher_actions = batch["observations"][
                    RelabelTeacherActions.TEACHER_LABEL
                ].type(torch.long)
                log_probs = self._evaluate_actions(
                    batch["observations"],
                    batch["recurrent_hidden_states"],
                    batch["prev_actions"],
                    batch["masks"],
                    teacher_actions,
                    batch["rnn_build_seq_info"],
                )
                inflections = (
                    batch["observations"]["inflection"]
                    if "inflection" in batch["observations"]
                    else None
                )
                loss = _compute_dagger_loss(
                    log_probs,
                    inflections=inflections,
                    inflection_weight=self.inflection_weight,
                )

                if "is_coeffs" in batch:
                    assert isinstance(batch["is_coeffs"], torch.Tensor)
                    total_loss = _compute_dagger_loss(
                        log_probs,
                        inflections=inflections,
                        inflection_weight=self.inflection_weight,
                        is_coeffs=batch["is_coeffs"],
                    )
                else:
                    total_loss = loss

                total_loss = self.before_backward(total_loss)
                total_loss.backward()
                self.after_backward(total_loss)

                grad_norm = self.before_step()
                self.optimizer.step()
                self.after_step()

                with inference_mode():
                    if "is_coeffs" in batch:
                        record_min_mean_max(batch["is_coeffs"], "ver_is_coeffs")
                    learner_metrics["loss"].append(loss)
                    learner_metrics["grad_norm"].append(grad_norm)

                    # Probability assigned to the oracle action. Values near
                    # one indicate stronger student/oracle agreement.
                    action_agreement = torch.exp(log_probs).mean()
                    learner_metrics["action_agreement"].append(action_agreement)

                    # Negative log probability is retained as a training metric.
                    learner_metrics["neg_log_prob"].append(-log_probs.mean())

                    if "is_stale" in batch:
                        assert isinstance(batch["is_stale"], torch.Tensor)
                        learner_metrics["fraction_stale"].append(
                            batch["is_stale"].float().mean()
                        )

                    if isinstance(rollouts, VERRolloutStorage):
                        assert isinstance(batch["policy_version"], torch.Tensor)
                        record_min_mean_max(
                            (
                                rollouts.current_policy_version
                                - batch["policy_version"]
                            ).float(),
                            "policy_version_difference",
                        )

            profiling_wrapper.range_pop()  # PPO.update epoch

        self._set_grads_to_none()

        with inference_mode():
            return {
                k: float(
                    torch.stack(
                        [torch.as_tensor(v, dtype=torch.float32) for v in vs]
                    ).mean()
                )
                for k, vs in learner_metrics.items()
            }

    def _evaluate_actions(self, *args, **kwargs):
        r"""Internal method that calls Policy.evaluate_actions.  This is used instead of
        calling that directly so that that call can be overrided with inheritance
        """
        return self.actor_critic.evaluate_actions(*args, **kwargs)


class DDPDAgger(DecentralizedDistributedMixin, DAgger):
    pass


class DAggerPolicyMixin:
    """Avoids computing value or action_log_probs, which are RL-only, and
    .evaluate_actions() will be overridden to produce the correct gradients."""

    action_distribution: Union[CategoricalNet, GaussianNet]
    critic: nn.Module
    net: nn.Module
    action_distribution_type: str

    def __init__(self, teacher_forcing: bool, *args, **kwargs):
        self.teacher_forcing = teacher_forcing
        super().__init__(*args, **kwargs)

    @property
    def policy_components(self):
        """Same except critic weights are not included."""
        return self.net, self.action_distribution

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        if not hasattr(self, "action_distribution"):
            # For the dummy policy class that has no neural net
            return super().act(  # type: ignore
                observations, rnn_hidden_states, prev_actions, masks
            )

        if self.teacher_forcing and "teacher_label" in observations:
            # The oracle still chooses the executed action, but WSLA must see
            # every collected observation so its stored 500-step history is
            # valid when an episode crosses a rollout or mini-batch boundary.
            with torch.no_grad():
                _, rnn_hidden_states, _ = self.net(
                    observations, rnn_hidden_states, prev_actions, masks
                )
            return self.cheat(observations, rnn_hidden_states)

        """Skips computing values and action_log_probs, which are RL-only."""
        features, rnn_hidden_states, _ = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)

        with torch.no_grad():
            if deterministic:
                if self.action_distribution_type == "categorical":
                    action = distribution.mode()
                elif self.action_distribution_type == "gaussian":
                    action = distribution.mean
                else:
                    raise NotImplementedError(
                        "Distribution type {} is not supported".format(
                            self.action_distribution_type
                        )
                    )
            else:
                action = distribution.sample()
        n = action.shape[0]
        value = torch.zeros(n, 1, device=action.device)
        action_log_probs = torch.zeros(n, 1, device=action.device)
        return value, action, action_log_probs, rnn_hidden_states

    @staticmethod
    def cheat(observations, rnn_hidden_states):
        action = observations["teacher_label"].long()

        num_envs = observations["teacher_label"].shape[0]
        device = observations["teacher_label"].device

        action_log_probs = torch.zeros(num_envs, 1).to(device)
        value = torch.zeros(num_envs, 1).to(device)

        return value, action, action_log_probs, rnn_hidden_states

    def evaluate_actions(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        action,
        rnn_build_seq_info,
    ):
        """Given a state and action, computes the policy's action distribution for that
        state and then returns the log probability of the given action under this
        distribution."""
        features, _, _ = self.net(
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
            rnn_build_seq_info,
        )
        distribution = self.action_distribution(features)
        log_probs = distribution.log_probs(action)
        return log_probs


@baseline_registry.register_policy
class DAggerPolicy(Policy):
    @classmethod
    def from_config(
        cls,
        config: "DictConfig",
        observation_space: spaces.Dict,
        action_space,
        **kwargs,
    ):
        if (
            config.habitat_baselines.rl.policy.original_name
            != "LANavWSLAPolicy"
        ):
            raise ValueError("DAggerPolicy requires LANavWSLAPolicy")
        original_cls = baseline_registry.get_policy(
            config.habitat_baselines.rl.policy.original_name
        )
        # fmt: off
        class MixedPolicy(DAggerPolicyMixin, original_cls): pass  # noqa
        # fmt: on
        return MixedPolicy.from_config(
            config,
            observation_space,
            action_space,
            teacher_forcing=config.habitat_baselines.rl.policy.teacher_forcing,
            **kwargs,
        )


@dataclass
class DAggerPolicyConfig(PolicyConfig):
    original_name: str = ""
    teacher_forcing: bool = False
