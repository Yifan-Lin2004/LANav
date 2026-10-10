from types import SimpleNamespace

import torch
from torch import nn

from ovon.models.wsla import WSLALayer
from ovon.models.wsla_encoder import WSLAEncoder, WSLAEncoderConfig
from ovon.models.wsla_policy import (
    PRECOMPUTED_VISUAL_FEATURES_KEY,
    FrozenSigLIPEncoder,
    WSLANet,
)


SMALL_CONFIG = {
    "n_layers": 2,
    "n_hidden": 16,
    "n_heads": 2,
    "head_dim": 4,
    "max_context_length": 3,
    "intermediate_size": 32,
    "expand_v": 1.0,
    "expanded_states": 4,
    "num_block": 1,
    "overlap": 0,
    "conv_size": 2,
    "use_short_conv": False,
    "kernel_backend": "reference",
}


def test_static_softmax_readout_is_shared_across_time_and_heads() -> None:
    layer = WSLALayer(
        hidden_size=16,
        num_heads=2,
        head_dim=4,
        num_subheads=4,
        expand_v=1.0,
        num_blocks=1,
        overlap=0,
        use_short_conv=False,
        kernel_backend="reference",
    )
    with torch.no_grad():
        layer.expert_logits.copy_(torch.tensor([-1.0, 0.0, 1.0, 2.0]))

    expert_outputs = torch.randn(2, 5, 2, 4, 4)
    combined = layer.combine_experts(expert_outputs)
    weights = torch.softmax(layer.expert_logits, dim=0)
    expected = torch.einsum("bthde,e->bthd", expert_outputs, weights)

    torch.testing.assert_close(combined, expected)
    torch.testing.assert_close(layer.expert_weights.sum(), torch.tensor(1.0))
    assert layer.expert_logits.shape == (4,)
    assert layer.num_states == 2 * 4
    # Paper Eq. (18): concatenate all head outputs, then normalize the full u_t.
    assert layer.output_norm.weight.shape == (layer.value_dim,)
    assert layer.output_norm.weight.shape != (layer.head_value_dim,)
    # A single E-vector, rather than [time, head, E], makes sharing explicit.
    assert layer.expert_weights.ndim == 1


def test_wsla_shape_gradient_and_no_dynamic_router() -> None:
    layer = WSLALayer(
        hidden_size=16,
        num_heads=2,
        head_dim=4,
        num_subheads=4,
        expand_v=1.0,
        num_blocks=1,
        overlap=0,
        use_short_conv=False,
        kernel_backend="reference",
    )
    inputs = torch.randn(2, 5, 16, requires_grad=True)
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.bool)
    output = layer(inputs, attention_mask=mask)

    assert output.shape == inputs.shape
    output.square().mean().backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    assert layer.expert_logits.grad is not None
    assert all("router" not in name.lower() for name, _ in layer.named_parameters())
    assert not hasattr(layer, "router")


def test_wsla_rejects_extra_implementation_blocks() -> None:
    try:
        WSLALayer(
            hidden_size=16,
            num_heads=2,
            head_dim=4,
            num_subheads=4,
            expand_v=1.0,
            num_blocks=2,
            overlap=0,
            use_short_conv=False,
        )
    except ValueError as error:
        assert "H*E" in str(error)
    else:  # pragma: no cover
        raise AssertionError("non-paper K*H*E states must be rejected")


def test_paper_defaults_are_four_layers_four_heads_four_subheads() -> None:
    config = WSLAEncoderConfig()
    assert config.n_layers == 4
    assert config.n_hidden == 512
    assert config.n_heads == 4
    assert config.expanded_states == 4
    assert config.n_heads * config.expanded_states == 16
    assert config.max_context_length == 500

    encoder = WSLAEncoder(input_size=1568, config=config)
    assert encoder.input_projection.in_features == 1568
    assert encoder.input_projection.out_features == 512
    assert len(encoder.layers) == 4
    logits = [block.attention.expert_logits for block in encoder.layers]
    assert all(parameter.shape == (4,) for parameter in logits)
    assert all(block.attention.num_states == 16 for block in encoder.layers)
    assert all(
        block.attention.output_norm.weight.shape == (768,)
        for block in encoder.layers
    )
    assert len({parameter.data_ptr() for parameter in logits}) == 4


def test_rgb_checkpoint_tag_is_explicitly_pinned() -> None:
    assert FrozenSigLIPEncoder.model_name == "vit_base_patch16_siglip_256.v2_webli"


def test_encoder_sequence_shape() -> None:
    encoder = WSLAEncoder(input_size=12, config={**SMALL_CONFIG, "max_context_length": 5})
    inputs = torch.randn(2, 5, 12)
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.bool)
    output = encoder.forward_sequence(inputs, attention_mask=mask)
    assert output.shape == (2, 5, 16)
    torch.testing.assert_close(output[1, 3:], torch.zeros_like(output[1, 3:]))


def _single_sequence_info(history: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "select_inds": torch.tensor([0]),
        "cpu_num_seqs_at_step": torch.tensor([1]),
        "cpu_sequence_lengths": torch.tensor([1]),
        "sequence_starts": torch.tensor([0]),
        "rnn_state_batch_inds": torch.tensor([0]),
        "sequence_start_hidden_states": history,
    }


def test_packed_update_prepends_valid_rollout_history() -> None:
    encoder = WSLAEncoder(input_size=12, config=SMALL_CONFIG)
    current = torch.randn(1, 12)
    history = torch.zeros(1, 1, encoder.recurrent_hidden_state_size)
    reshaped = history.reshape(1, 3, 17)
    reshaped[0, 0, :16] = torch.randn(16)
    reshaped[0, 0, -1] = 1

    actual, _ = encoder(
        current,
        history,
        masks=torch.ones(1, 1, dtype=torch.bool),
        rnn_build_seq_info=_single_sequence_info(history),
    )
    projected = torch.cat(
        [
            reshaped[:, :1, :16],
            encoder.input_projection(current).unsqueeze(1),
        ],
        dim=1,
    )
    expected = encoder._encode_projected_sequence(projected)[:, -1]
    torch.testing.assert_close(actual, expected)


def test_packed_update_discards_history_at_episode_reset() -> None:
    encoder = WSLAEncoder(input_size=12, config=SMALL_CONFIG)
    current = torch.randn(1, 12)
    populated = torch.zeros(1, 1, encoder.recurrent_hidden_state_size)
    populated_view = populated.reshape(1, 3, 17)
    populated_view[0, 0, :16] = torch.randn(16)
    populated_view[0, 0, -1] = 1
    empty = torch.zeros_like(populated)
    reset_mask = torch.zeros(1, 1, dtype=torch.bool)

    with_history, _ = encoder(
        current,
        populated,
        reset_mask,
        rnn_build_seq_info=_single_sequence_info(populated),
    )
    without_history, _ = encoder(
        current,
        empty,
        reset_mask,
        rnn_build_seq_info=_single_sequence_info(empty),
    )
    torch.testing.assert_close(with_history, without_history)


def test_packed_update_uses_each_sequence_start_history() -> None:
    encoder = WSLAEncoder(input_size=12, config=SMALL_CONFIG)
    current = torch.randn(2, 12)
    sequence_histories = torch.zeros(
        2, 1, encoder.recurrent_hidden_state_size
    )
    history_view = sequence_histories.reshape(2, 3, 17)
    history_view[:, 0, :16] = torch.randn(2, 16)
    history_view[:, 0, -1] = 1
    sequence_info = {
        "select_inds": torch.tensor([0, 1]),
        "cpu_num_seqs_at_step": torch.tensor([2]),
        "cpu_sequence_lengths": torch.tensor([1, 1]),
        "sequence_starts": torch.tensor([0, 1]),
        # Both sequences can belong to one environment; VER's conventional
        # unique-env recurrent state is therefore insufficient by itself.
        "rnn_state_batch_inds": torch.tensor([0, 0]),
        "sequence_start_hidden_states": sequence_histories,
    }
    unique_env_history = sequence_histories[:1]
    actual, _ = encoder(
        current,
        unique_env_history,
        masks=torch.ones(2, 1, dtype=torch.bool),
        rnn_build_seq_info=sequence_info,
    )
    for index in range(2):
        projected = torch.cat(
            [
                history_view[index : index + 1, :1, :16],
                encoder.input_projection(current[index : index + 1]).unsqueeze(1),
            ],
            dim=1,
        )
        expected = encoder._encode_projected_sequence(projected)[:, -1]
        torch.testing.assert_close(actual[index : index + 1], expected)


def test_packed_update_repacking_preserves_original_sample_order() -> None:
    encoder = WSLAEncoder(input_size=12, config=SMALL_CONFIG)
    current = torch.randn(3, 12)
    sequence_histories = torch.zeros(
        2, 1, encoder.recurrent_hidden_state_size
    )
    history_view = sequence_histories.reshape(2, 3, 17)
    history_view[0, 0, :16] = torch.randn(16)
    history_view[0, 0, -1] = 1
    history_view[1, :2, :16] = torch.randn(2, 16)
    history_view[1, 0, -1] = 2
    sequence_info = {
        # Packed rows are: sequence 0 step 0, sequence 1 step 0,
        # sequence 0 step 1. Original learner rows are deliberately shuffled.
        "select_inds": torch.tensor([2, 0, 1]),
        "cpu_num_seqs_at_step": torch.tensor([2, 1]),
        "cpu_sequence_lengths": torch.tensor([2, 1]),
        "sequence_starts": torch.tensor([2, 0]),
        "sequence_start_hidden_states": sequence_histories,
    }
    actual, _ = encoder(
        current,
        sequence_histories,
        masks=torch.ones(3, 1, dtype=torch.bool),
        rnn_build_seq_info=sequence_info,
    )

    sequence_zero = torch.cat(
        [
            history_view[0:1, :1, :16],
            encoder.input_projection(current[[2, 1]]).unsqueeze(0),
        ],
        dim=1,
    )
    sequence_one = torch.cat(
        [
            history_view[1:2, :2, :16],
            encoder.input_projection(current[0:1]).unsqueeze(0),
        ],
        dim=1,
    )
    expected_zero = encoder._encode_projected_sequence(sequence_zero)
    expected_one = encoder._encode_projected_sequence(sequence_one)
    torch.testing.assert_close(actual[2], expected_zero[0, 1])
    torch.testing.assert_close(actual[1], expected_zero[0, 2])
    torch.testing.assert_close(actual[0], expected_one[0, 2])


def test_packed_update_rejects_episode_longer_than_context() -> None:
    encoder = WSLAEncoder(input_size=12, config=SMALL_CONFIG)
    current = torch.randn(2, 12)
    history = torch.zeros(1, 1, encoder.recurrent_hidden_state_size)
    history_view = history.reshape(1, 3, 17)
    history_view[0, :2, :16] = torch.randn(2, 16)
    history_view[0, 0, -1] = 2
    sequence_info = {
        "select_inds": torch.tensor([0, 1]),
        "cpu_num_seqs_at_step": torch.tensor([1, 1]),
        "cpu_sequence_lengths": torch.tensor([2]),
        "sequence_starts": torch.tensor([0]),
        "sequence_start_hidden_states": history,
    }
    try:
        encoder(
            current,
            history,
            masks=torch.ones(2, 1, dtype=torch.bool),
            rnn_build_seq_info=sequence_info,
        )
    except ValueError as error:
        assert "exceeds the 3-step WSLA context" in str(error)
    else:  # pragma: no cover
        raise AssertionError("history plus current sequence must respect context")


class _UnusedVisualEncoder(nn.Module):
    def forward(self, observations):  # pragma: no cover - precomputed path is required.
        raise AssertionError("precomputed visual features should bypass RGB encoding")


def test_policy_token_is_768_plus_768_plus_32_and_habitat_shape() -> None:
    action_space = SimpleNamespace(n=6)
    net = WSLANet(
        observation_space=None,
        action_space=action_space,
        wsla_config=SMALL_CONFIG,
        hidden_size=16,
        visual_encoder=_UnusedVisualEncoder(),
    )
    assert net.TOKEN_DIM == 1568
    assert net.state_encoder.input_projection.in_features == 1568
    assert not hasattr(net, "gps_embedding")
    assert not hasattr(net, "compass_embedding")
    assert not hasattr(net, "depth_encoder")

    observations = {
        PRECOMPUTED_VISUAL_FEATURES_KEY: torch.randn(2, 768),
        "clip_objectgoal": torch.randn(2, 768),
        # Extra simulator observations must not enter the LANav token.
        "gps": torch.randn(2, 2),
        "depth": torch.randn(2, 8, 8, 1),
    }
    hidden = torch.zeros(2, 1, net.recurrent_hidden_state_size)
    output, next_hidden, auxiliary = net(
        observations,
        hidden,
        prev_actions=torch.tensor([[1], [2]]),
        masks=torch.tensor([[False], [True]]),
    )
    assert output.shape == (2, 16)
    assert next_hidden is not None
    assert next_hidden.shape == (2, 1, 3 * (16 + 1))
    assert auxiliary["perception_embed"].shape == (2, 768)
