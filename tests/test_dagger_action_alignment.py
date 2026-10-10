import torch

from ovon.algos.dagger import DAggerPolicyMixin, _compute_dagger_loss
from ovon.trainers.dagger_trainer import _uses_oracle_actions


def test_oracle_forced_dagger_records_the_executed_action():
    assert _uses_oracle_actions("ver_dagger")
    observations = {"teacher_label": torch.tensor([[4], [2]])}
    hidden = torch.randn(2, 1, 3)
    _, actions, _, returned_hidden = DAggerPolicyMixin.cheat(observations, hidden)
    assert torch.equal(actions, observations["teacher_label"])
    assert returned_hidden is hidden


def test_oracle_forced_collection_still_updates_wsla_history():
    class Harness:
        teacher_forcing = True
        action_distribution = object()

        @staticmethod
        def net(observations, hidden, prev_actions, masks):
            del observations, prev_actions, masks
            output = torch.zeros(hidden.shape[0], 1)
            return output, hidden + 1, {}

        cheat = staticmethod(DAggerPolicyMixin.cheat)
        act = DAggerPolicyMixin.act

    observations = {"teacher_label": torch.tensor([[4], [2]])}
    hidden = torch.zeros(2, 1, 3)
    _, actions, _, returned_hidden = Harness().act(
        observations,
        hidden,
        prev_actions=torch.zeros(2, 1, dtype=torch.long),
        masks=torch.ones(2, 1, dtype=torch.bool),
    )
    assert torch.equal(actions, observations["teacher_label"])
    torch.testing.assert_close(returned_hidden, torch.ones_like(hidden))


def test_ppo_is_not_oracle_forced():
    assert not _uses_oracle_actions("wsla_ver")
    assert not _uses_oracle_actions("ver_il")


def test_ver_importance_coefficients_weight_each_dagger_sample():
    log_probs = torch.tensor([[-1.0], [-3.0]])
    inflections = torch.tensor([[0], [1]])
    is_coeffs = torch.tensor([[1.0], [0.25]])

    actual = _compute_dagger_loss(
        log_probs,
        inflections=inflections,
        inflection_weight=2.0,
        is_coeffs=is_coeffs,
    )

    # Inflection weights [1, 2] are normalized by their mean, then each sample
    # receives its own VER coefficient before reduction.
    expected = torch.tensor(
        ((1.0 * (1.0 / 1.5)) + (0.25 * 3.0 * (2.0 / 1.5))) / 2.0
    )
    wrong_scalar_reweighting = torch.tensor(
        ((1.0 + 6.0) / 3.0) * ((1.0 + 0.25) / 2.0)
    )
    torch.testing.assert_close(actual, expected)
    assert not torch.isclose(actual, wrong_scalar_reweighting)
