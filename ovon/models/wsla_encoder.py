"""Four-layer WSLA state encoder used by LANav.

The encoder accepts the paper's 1,568-dimensional token (768-D RGB SigLIP,
768-D text SigLIP and a 32-D previous-action embedding), projects it to 512
dimensions and applies four causal WSLA decoder blocks.  During environment
interaction the Habitat recurrent-state slot stores the most recent projected
tokens, capped at the paper context length of 500.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .wsla import RMSNorm, WSLALayer


def _config_value(config: Any, name: str, default: Any) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


@dataclass(frozen=True)
class WSLAEncoderConfig:
    """Paper configuration plus implementation-level WSLA parameters."""

    n_layers: int = 4
    n_hidden: int = 512
    n_heads: int = 4
    head_dim: int = 96
    max_context_length: int = 500
    intermediate_size: int = 2048
    norm_eps: float = 1e-5
    use_mlp: bool = True
    expand_v: float = 2.0
    expanded_states: int = 4  # Number E of WSLA sub-states.
    num_block: int = 1
    overlap: int = 0
    conv_size: int = 4
    use_short_conv: bool = True
    kernel_backend: str = "auto"


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class WSLABlock(nn.Module):
    """Pre-normalized WSLA and SwiGLU residual block."""

    def __init__(self, config: WSLAEncoderConfig, layer_idx: int) -> None:
        super().__init__()
        self.input_norm = RMSNorm(config.n_hidden, eps=config.norm_eps)
        self.attention = WSLALayer(
            hidden_size=config.n_hidden,
            num_heads=config.n_heads,
            head_dim=config.head_dim,
            num_subheads=config.expanded_states,
            expand_v=config.expand_v,
            num_blocks=config.num_block,
            overlap=config.overlap,
            conv_size=config.conv_size,
            use_short_conv=config.use_short_conv,
            norm_eps=config.norm_eps,
            layer_idx=layer_idx,
            kernel_backend=config.kernel_backend,
        )
        self.use_mlp = config.use_mlp
        if self.use_mlp:
            self.post_attention_norm = RMSNorm(config.n_hidden, eps=config.norm_eps)
            self.mlp = SwiGLU(config.n_hidden, config.intermediate_size)

    def forward(self, x: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        x = x + self.attention(self.input_norm(x), attention_mask=attention_mask)
        if self.use_mlp:
            x = x + self.mlp(self.post_attention_norm(x))
        return x


class WSLAEncoder(nn.Module):
    """Habitat-compatible causal WSLA state encoder."""

    def __init__(self, input_size: int = 1568, config: Any = None) -> None:
        super().__init__()
        defaults = WSLAEncoderConfig()
        values = {
            field_name: _config_value(config, field_name, getattr(defaults, field_name))
            for field_name in WSLAEncoderConfig.__dataclass_fields__
        }
        self.config = WSLAEncoderConfig(**values)
        self.input_size = input_size
        self.n_layers = self.config.n_layers
        self.n_hidden = self.config.n_hidden
        self.n_heads = self.config.n_heads
        self.max_context_length = self.config.max_context_length
        if self.n_layers < 1 or self.max_context_length < 1:
            raise ValueError("n_layers and max_context_length must be positive")

        self.input_projection = nn.Linear(input_size, self.n_hidden)
        self.layers = nn.ModuleList(
            WSLABlock(self.config, layer_idx=index) for index in range(self.n_layers)
        )
        self.final_norm = RMSNorm(self.n_hidden, eps=self.config.norm_eps)

        # One extra scalar in each history slot stores the sequence length.
        self._history_slot_size = self.n_hidden + 1

    @property
    def recurrent_hidden_state_size(self) -> int:
        return self.max_context_length * self._history_slot_size

    @property
    def num_recurrent_layers(self) -> int:
        # Habitat treats the flattened history as one recurrent-state layer.
        return 1

    def forward_sequence(
        self, features: Tensor, attention_mask: Optional[Tensor] = None
    ) -> Tensor:
        """Encode a padded batch of causal sequences."""

        if features.ndim != 3 or features.shape[-1] != self.input_size:
            raise ValueError(
                f"expected [batch, time, {self.input_size}], got {tuple(features.shape)}"
            )
        if features.shape[1] > self.max_context_length:
            raise ValueError(
                f"sequence length {features.shape[1]} exceeds context "
                f"length {self.max_context_length}"
            )
        return self._encode_projected_sequence(
            self.input_projection(features), attention_mask=attention_mask
        )

    def _encode_projected_sequence(
        self, projected: Tensor, attention_mask: Optional[Tensor] = None
    ) -> Tensor:
        """Encode already-projected tokens, including detached rollout history."""

        if projected.ndim != 3 or projected.shape[-1] != self.n_hidden:
            raise ValueError(
                f"expected [batch, time, {self.n_hidden}], got "
                f"{tuple(projected.shape)}"
            )
        if projected.shape[1] > self.max_context_length:
            raise ValueError(
                f"sequence length {projected.shape[1]} exceeds context "
                f"length {self.max_context_length}"
            )
        x = projected
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)
        x = self.final_norm(x)
        if attention_mask is not None:
            x = x * attention_mask.to(x.dtype).unsqueeze(-1)
        return x

    def _empty_history(self, batch_size: int, device: torch.device) -> Tensor:
        return torch.zeros(
            batch_size,
            self.max_context_length,
            self._history_slot_size,
            device=device,
            dtype=torch.float32,
        )

    def _reshape_history(
        self,
        hidden_states: Optional[Tensor],
        batch_size: int,
        device: torch.device,
    ) -> Tensor:
        if hidden_states is None or hidden_states.numel() == 0:
            return self._empty_history(batch_size, device)
        expected = self.recurrent_hidden_state_size
        if (
            hidden_states.shape[0] != batch_size
            or hidden_states.numel() != batch_size * expected
        ):
            raise ValueError("recurrent state does not match the WSLA history size")
        return hidden_states.reshape(
            batch_size, self.max_context_length, self._history_slot_size
        ).to(device=device, dtype=torch.float32)

    def _forward_step(
        self, features: Tensor, hidden_states: Optional[Tensor], masks: Tensor
    ) -> Tuple[Tensor, Tensor]:
        if features.ndim == 3:
            if features.shape[1] != 1:
                raise ValueError("environment inference expects one token per batch item")
            features = features[:, 0]
        batch_size = features.shape[0]
        history = self._reshape_history(hidden_states, batch_size, features.device)
        history_features = history[..., : self.n_hidden].clone()
        lengths = (
            history[:, 0, -1].long().clamp(0, self.max_context_length).clone()
        )

        reset = ~masks.reshape(batch_size, -1)[:, -1].bool()
        history_features[reset] = 0
        lengths[reset] = 0

        projected = self.input_projection(features).to(history_features.dtype)
        for batch_index in range(batch_size):
            length = int(lengths[batch_index].item())
            if length < self.max_context_length:
                history_features[batch_index, length] = projected[batch_index]
                lengths[batch_index] = length + 1
            else:
                history_features[batch_index, :-1] = history_features[
                    batch_index, 1:
                ].clone()
                history_features[batch_index, -1] = projected[batch_index]

        max_length = max(int(lengths.max().item()), 1)
        sequence = history_features[:, :max_length].to(
            next(self.parameters()).dtype
        )
        attention_mask = (
            torch.arange(max_length, device=features.device)[None, :] < lengths[:, None]
        )
        x = self._encode_projected_sequence(
            sequence, attention_mask=attention_mask
        )
        last_indices = (lengths - 1).clamp_min(0)
        output = x[torch.arange(batch_size, device=features.device), last_indices]

        new_history = self._empty_history(batch_size, features.device)
        new_history[..., : self.n_hidden] = history_features
        new_history[:, 0, -1] = lengths.float()
        return output, new_history.reshape(batch_size, 1, -1)

    def _forward_packed(
        self,
        features: Tensor,
        masks: Tensor,
        sequence_info: Dict[str, Tensor],
    ) -> Tensor:
        """Encode Habitat VER sequences with their stored pre-sequence history.

        VER supplies one recurrent state for the first retained step of every
        packed sequence before compressing recurrent state to one entry per
        environment.  A false mask at a sequence start resets that history.
        Since the released task and WSLA context both have a 500-step limit,
        valid history plus the retained sequence never exceeds the paper
        context length.
        """

        required = {
            "select_inds",
            "cpu_num_seqs_at_step",
            "cpu_sequence_lengths",
            "sequence_starts",
            "sequence_start_hidden_states",
        }
        missing = required.difference(sequence_info)
        if missing:
            raise KeyError(f"missing packed sequence metadata: {sorted(missing)}")

        select_inds = sequence_info["select_inds"].to(features.device)
        counts = sequence_info["cpu_num_seqs_at_step"]
        lengths = sequence_info["cpu_sequence_lengths"].long()
        num_packed = int(select_inds.numel())
        max_length = int(counts.numel())
        if max_length > self.max_context_length:
            raise ValueError(
                f"packed sequence length {max_length} exceeds context "
                f"length {self.max_context_length}"
            )
        num_sequences = int(counts[0].item()) if max_length else 0
        if num_sequences == 0:
            return features.new_zeros((features.shape[0], self.n_hidden))

        projected = self.input_projection(features)
        packed_projected = projected.index_select(0, select_inds)
        padded_current = torch.zeros(
            num_sequences,
            max_length,
            self.n_hidden,
            dtype=projected.dtype,
            device=features.device,
        )
        offset = 0
        for step, count_tensor in enumerate(counts):
            count = int(count_tensor.item())
            padded_current[:count, step] = packed_projected[
                offset : offset + count
            ]
            offset += count

        sequence_start_states = sequence_info["sequence_start_hidden_states"]
        expected = self.recurrent_hidden_state_size
        if sequence_start_states.numel() != num_sequences * expected:
            raise ValueError(
                "sequence_start_hidden_states must contain one WSLA history "
                "for every packed sequence"
            )
        history = sequence_start_states.reshape(
            num_sequences, self.max_context_length, self._history_slot_size
        ).to(device=features.device, dtype=torch.float32)

        history_features = history[..., : self.n_hidden].to(projected.dtype)
        history_lengths = history[:, 0, -1].long().clamp(
            0, self.max_context_length
        )
        sequence_starts = sequence_info["sequence_starts"].to(
            device=features.device, dtype=torch.long
        )
        if sequence_starts.numel() != num_sequences:
            raise ValueError("sequence_starts must contain one entry per sequence")
        start_masks = (
            masks.reshape(features.shape[0], -1)[:, -1]
            .bool()
            .index_select(0, sequence_starts)
        )
        history_lengths = torch.where(
            start_masks, history_lengths, torch.zeros_like(history_lengths)
        )

        current_lengths = lengths.to(features.device)
        total_lengths = history_lengths + current_lengths
        if torch.any(total_lengths > self.max_context_length):
            raise ValueError(
                "stored history plus packed sequence exceeds the "
                f"{self.max_context_length}-step WSLA context; keep "
                "habitat.environment.max_episode_steps at or below "
                "max_context_length"
            )
        combined_length = max(int(total_lengths.max().item()), 1)
        combined = torch.zeros(
            num_sequences,
            combined_length,
            self.n_hidden,
            dtype=projected.dtype,
            device=features.device,
        )
        for sequence_index in range(num_sequences):
            history_length = int(history_lengths[sequence_index].item())
            current_length = int(current_lengths[sequence_index].item())
            if history_length:
                combined[sequence_index, :history_length] = history_features[
                    sequence_index, :history_length
                ]
            combined[
                sequence_index,
                history_length : history_length + current_length,
            ] = padded_current[sequence_index, :current_length]

        attention_mask = (
            torch.arange(combined_length, device=features.device)[None, :]
            < total_lengths[:, None]
        )
        encoded = self._encode_projected_sequence(
            combined, attention_mask=attention_mask
        )

        repacked = torch.empty(
            num_packed, self.n_hidden, dtype=encoded.dtype, device=encoded.device
        )
        offset = 0
        for step, count_tensor in enumerate(counts):
            count = int(count_tensor.item())
            positions = history_lengths[:count] + step
            repacked[offset : offset + count] = encoded[
                torch.arange(count, device=features.device), positions
            ]
            offset += count
        output = torch.zeros(
            features.shape[0],
            self.n_hidden,
            dtype=encoded.dtype,
            device=encoded.device,
        )
        output.index_copy_(0, select_inds, repacked)
        return output

    def forward(
        self,
        features: Tensor,
        rnn_hidden_states: Optional[Tensor],
        masks: Tensor,
        rnn_build_seq_info: Optional[Dict[str, Tensor]] = None,
        **_: Any,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        is_packed_update = (
            rnn_build_seq_info is not None
            and "select_inds" in rnn_build_seq_info
            and "cpu_num_seqs_at_step" in rnn_build_seq_info
        )
        if is_packed_update:
            return (
                self._forward_packed(features, masks, rnn_build_seq_info),
                rnn_hidden_states,
            )
        return self._forward_step(features, rnn_hidden_states, masks)


__all__ = ["WSLABlock", "WSLAEncoder", "WSLAEncoderConfig"]
