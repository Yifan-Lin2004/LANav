"""Weighted State-Expansion Linear Attention (WSLA).

This module is a direct implementation of the WSLA equations in the LANav
paper.  Every original attention head is expanded into ``num_subheads``
independent gated-delta states.  Their outputs are combined with one learned,
static softmax vector per layer::

    w = softmax(expert_logits)
    o[b, t, h] = sum_e w[e] * o_subhead[b, t, h, e]

The same vector is therefore shared across the batch, sequence positions and
original heads.  There is deliberately no input-dependent expert router in
this implementation.

The optimized GatedDeltaNet Triton operator is used when it is importable and
the inputs are CUDA tensors.  A differentiable PyTorch recurrence is kept as
the reference backend for tests and CPU smoke runs.
"""

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _load_triton_kernel() -> Tuple[Optional[Callable], Optional[Exception]]:
    """Load only GatedDeltaNet's operator package, not its training package."""

    release_root = Path(__file__).resolve().parents[2]
    gdn_root = release_root / "third_party" / "GatedDeltaNet"
    fla_root = release_root / "third_party" / "fla_compat"
    try:
        if gdn_root.is_dir():
            if fla_root.is_dir() and str(fla_root) not in sys.path:
                sys.path.insert(0, str(fla_root))
        module = importlib.import_module(
            "third_party.GatedDeltaNet.lit_gpt.gated_delta_rule_ops"
        )
        return module.chunk_gated_delta_rule, None
    except Exception as error:  # Triton is optional for CPU-only tests.
        return None, error


_triton_gated_delta_rule, _triton_import_error = _load_triton_kernel()


class RMSNorm(nn.Module):
    """Root-mean-square normalization used by the LANav decoder."""

    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: Tensor) -> Tensor:
        normalized = x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return normalized.to(x.dtype) * self.weight.to(x.dtype)


class CausalDepthwiseConv1d(nn.Module):
    """A small causal convolution applied independently to every channel."""

    def __init__(self, channels: int, kernel_size: int = 4, bias: bool = False) -> None:
        super().__init__()
        if kernel_size < 1:
            raise ValueError("kernel_size must be positive")
        self.channels = channels
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.empty(channels, 1, kernel_size))
        self.bias = nn.Parameter(torch.zeros(channels)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[-1] != self.channels:
            raise ValueError(
                f"expected [batch, time, {self.channels}], got {tuple(x.shape)}"
            )
        y = F.conv1d(
            x.transpose(1, 2),
            self.weight,
            self.bias,
            padding=self.kernel_size - 1,
            groups=self.channels,
        )
        return y[..., : x.shape[1]].transpose(1, 2)


def reference_gated_delta_rule(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    beta: Tensor,
    g: Tensor,
    initial_state: Optional[Tensor] = None,
    output_final_state: bool = False,
    **_: object,
) -> Tuple[Tensor, Optional[Tensor]]:
    """Differentiable reference implementation of the gated delta recurrence.

    Args:
        q, k: ``[batch, heads, time, key_dim]``.
        v: ``[batch, heads, time, value_dim]``.
        beta, g: ``[batch, heads, time]``. ``g`` is log decay and is non-positive.
        initial_state: optional ``[batch, heads, key_dim, value_dim]`` state.

    The recurrence follows the paper's delta update.  Query scaling matches the
    optimized GatedDeltaNet operator.
    """

    if not (q.ndim == k.ndim == v.ndim == 4 and beta.ndim == g.ndim == 3):
        raise ValueError("invalid gated-delta tensor ranks")
    if q.shape != k.shape or q.shape[:3] != v.shape[:3]:
        raise ValueError("q, k and v batch/head/time dimensions must match")
    if beta.shape != q.shape[:3] or g.shape != q.shape[:3]:
        raise ValueError("beta and g must match q's batch/head/time dimensions")

    batch, heads, time, key_dim = q.shape
    value_dim = v.shape[-1]
    compute_dtype = torch.float32
    if initial_state is None:
        state = torch.zeros(
            batch,
            heads,
            key_dim,
            value_dim,
            dtype=compute_dtype,
            device=q.device,
        )
    else:
        state = initial_state.to(compute_dtype)

    q_float = q.to(compute_dtype) * (key_dim**-0.5)
    k_float = k.to(compute_dtype)
    v_float = v.to(compute_dtype)
    beta_float = beta.to(compute_dtype)
    g_float = g.to(compute_dtype)

    outputs = []
    for step in range(time):
        q_t = q_float[:, :, step]
        k_t = k_float[:, :, step]
        v_t = v_float[:, :, step]
        beta_t = beta_float[:, :, step]

        state = state * g_float[:, :, step].exp()[..., None, None]
        state_value = torch.einsum("bhkm,bhk->bhm", state, k_t)
        delta = (v_t - state_value) * beta_t[..., None]
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        outputs.append(torch.einsum("bhk,bhkm->bhm", q_t, state))

    output = torch.stack(outputs, dim=2).to(v.dtype)
    final_state = state.to(v.dtype) if output_final_state else None
    return output, final_state


class WSLALayer(nn.Module):
    """One paper-consistent WSLA layer.

    The paper settings are the defaults: four original heads and four sub-states
    per original head.  There are exactly ``num_heads * num_subheads`` complete
    key/value states, as specified by the paper.
    """

    supports_recurrent_inference = False

    def __init__(
        self,
        hidden_size: int = 512,
        *,
        num_heads: int = 4,
        head_dim: int = 96,
        num_subheads: int = 4,
        expand_v: float = 2.0,
        num_blocks: int = 1,
        overlap: int = 0,
        conv_size: int = 4,
        use_short_conv: bool = True,
        norm_eps: float = 1e-5,
        layer_idx: Optional[int] = None,
        kernel_backend: str = "auto",
        kernel: Optional[Callable[..., Tuple[Tensor, Optional[Tensor]]]] = None,
    ) -> None:
        super().__init__()
        if num_heads < 1 or num_subheads < 1:
            raise ValueError("num_heads and num_subheads must be positive")
        if num_blocks != 1 or overlap != 0:
            raise ValueError(
                "paper-consistent WSLA has exactly H*E states and therefore "
                "requires num_blocks=1 and overlap=0"
            )
        if kernel_backend not in {"auto", "triton", "reference"}:
            raise ValueError("kernel_backend must be auto, triton or reference")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_subheads = num_subheads
        self.num_states = num_heads * num_subheads
        self.expand_v = expand_v
        self.num_blocks = num_blocks
        self.overlap = overlap
        self.layer_idx = layer_idx
        self.kernel_backend = kernel_backend
        self._kernel_override = kernel

        self.key_dim = num_heads * head_dim
        self.head_value_dim = int(head_dim * expand_v)
        self.value_dim = num_heads * self.head_value_dim

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.q_proj_expand = nn.ModuleList(
            nn.Linear(head_dim, head_dim * num_subheads, bias=False)
            for _ in range(num_heads)
        )
        self.k_proj_expand = nn.ModuleList(
            nn.Linear(head_dim, head_dim * num_subheads, bias=False)
            for _ in range(num_heads)
        )

        self.beta_proj = nn.Linear(hidden_size, num_heads * num_subheads, bias=False)
        self.decay_proj = nn.Linear(hidden_size, num_heads * num_subheads, bias=False)

        decay_rate = torch.empty(num_heads * num_subheads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(decay_rate.clamp_min(1e-4)))
        self.A_log._no_weight_decay = True  # type: ignore[attr-defined]

        dt = torch.exp(
            torch.empty(num_heads * num_subheads).uniform_(
                math.log(0.001), math.log(0.1)
            )
        ).clamp_min(1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.dt_bias._no_weight_decay = True  # type: ignore[attr-defined]

        self.use_short_conv = use_short_conv
        if use_short_conv:
            # Sub-states are folded into the batch, so convolution weights are
            # shared across them exactly as in the training implementation.
            self.q_conv = CausalDepthwiseConv1d(self.key_dim, conv_size)
            self.k_conv = CausalDepthwiseConv1d(self.key_dim, conv_size)
            self.v_conv = CausalDepthwiseConv1d(self.value_dim, conv_size)

        self.output_gate_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        # Equation (18) normalizes u_t after concatenating all H head outputs.
        self.output_norm = RMSNorm(self.value_dim, eps=norm_eps)
        self.output_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        # This is lambda in the paper.  There is one vector per WSLA layer.
        self.expert_logits = nn.Parameter(torch.zeros(num_subheads))
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=2**-2.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @property
    def expert_weights(self) -> Tensor:
        """Static normalized sub-state weights, shaped ``[num_subheads]``."""

        return F.softmax(self.expert_logits, dim=0)

    def combine_experts(self, expert_outputs: Tensor) -> Tensor:
        """Apply the shared static WSLA readout.

        ``expert_outputs`` must end in ``[head_value_dim, num_subheads]``; all
        preceding dimensions (including batch, time and head) share the same
        softmax vector.
        """

        if expert_outputs.shape[-1] != self.num_subheads:
            raise ValueError(
                f"expected {self.num_subheads} subheads, got {expert_outputs.shape[-1]}"
            )
        return torch.sum(expert_outputs * self.expert_weights, dim=-1)

    def _select_kernel(self, x: Tensor) -> Callable[..., Tuple[Tensor, Optional[Tensor]]]:
        if self._kernel_override is not None:
            return self._kernel_override
        if self.kernel_backend == "reference":
            return reference_gated_delta_rule
        if self.kernel_backend == "triton":
            if _triton_gated_delta_rule is None:
                raise RuntimeError(
                    "kernel_backend='triton' requires the bundled GatedDeltaNet "
                    f"operator; import failed with: {_triton_import_error!r}"
                ) from _triton_import_error
            if not x.is_cuda:
                raise RuntimeError("the Triton WSLA backend requires CUDA tensors")
            return _triton_gated_delta_rule
        if x.is_cuda:
            if _triton_gated_delta_rule is None:
                raise RuntimeError(
                    "WSLA received CUDA tensors but the bundled Triton kernel "
                    f"could not be imported: {_triton_import_error!r}. Use "
                    "kernel_backend='reference' only for explicit debugging."
                ) from _triton_import_error
            return _triton_gated_delta_rule
        return reference_gated_delta_rule

    def forward(self, hidden_states: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"expected [batch, time, {self.hidden_size}], got {tuple(hidden_states.shape)}"
            )
        batch, time, _ = hidden_states.shape
        if attention_mask is not None:
            if attention_mask.shape != (batch, time):
                raise ValueError("attention_mask must be [batch, time]")
            mask = attention_mask.to(dtype=hidden_states.dtype)
        else:
            mask = None

        q = self.q_proj(hidden_states).reshape(batch, time, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).reshape(batch, time, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).reshape(
            batch, time, self.num_heads, self.head_value_dim
        )

        q = torch.stack(
            [projection(q[:, :, head]) for head, projection in enumerate(self.q_proj_expand)],
            dim=2,
        )
        k = torch.stack(
            [projection(k[:, :, head]) for head, projection in enumerate(self.k_proj_expand)],
            dim=2,
        )
        q = q.reshape(batch, time, self.num_heads, self.num_subheads, self.head_dim)
        k = k.reshape(batch, time, self.num_heads, self.num_subheads, self.head_dim)
        q = q.permute(0, 1, 3, 2, 4)  # [B, T, E, H, Dk]
        k = k.permute(0, 1, 3, 2, 4)
        v = v[:, :, None].expand(-1, -1, self.num_subheads, -1, -1)

        if self.use_short_conv:
            def convolve_subheads(values: Tensor, conv: nn.Module) -> Tensor:
                width = values.shape[-2] * values.shape[-1]
                folded = values.reshape(batch, time, self.num_subheads, width)
                folded = folded.permute(2, 0, 1, 3).reshape(
                    self.num_subheads * batch, time, width
                )
                folded = F.silu(conv(folded))
                return folded.reshape(self.num_subheads, batch, time, self.num_heads, -1).permute(
                    1, 2, 0, 3, 4
                )

            q = convolve_subheads(q, self.q_conv)
            k = convolve_subheads(k, self.k_conv)
            v = convolve_subheads(v, self.v_conv)

        if mask is not None:
            expanded_mask = mask[:, :, None, None, None]
            q = q * expanded_mask
            k = k * expanded_mask
            v = v * expanded_mask

        beta = torch.sigmoid(self.beta_proj(hidden_states)).reshape(
            batch, time, self.num_subheads, self.num_heads
        )
        decay = -self.A_log.float().exp() * F.softplus(
            self.decay_proj(hidden_states).float() + self.dt_bias
        )
        decay = decay.reshape(batch, time, self.num_subheads, self.num_heads)
        if mask is not None:
            beta = beta * mask[:, :, None, None]
            decay = decay * mask[:, :, None, None]
        decay = decay.clamp(min=-20.0, max=0.0)

        q = F.normalize(q, p=2.0, dim=-1)
        k = F.normalize(k, p=2.0, dim=-1)

        total_heads = self.num_states
        q = q.reshape(batch, time, total_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        k = k.reshape(batch, time, total_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        v = v.reshape(batch, time, total_heads, self.head_value_dim).permute(0, 2, 1, 3).contiguous()
        beta = beta.reshape(batch, time, total_heads).permute(0, 2, 1).contiguous()
        decay = decay.reshape(batch, time, total_heads).permute(0, 2, 1).contiguous()

        output, _ = self._select_kernel(hidden_states)(
            q=q,
            k=k,
            v=v,
            beta=beta,
            g=decay,
            initial_state=None,
            output_final_state=False,
        )
        output = output.permute(0, 2, 1, 3).reshape(
            batch,
            time,
            self.num_subheads,
            self.num_heads,
            self.head_value_dim,
        )
        output = output.permute(0, 1, 3, 4, 2)  # [B,T,H,Dv,E]
        output = self.combine_experts(output)
        output = output.reshape(batch, time, self.value_dim)
        gate = self.output_gate_proj(hidden_states)
        output = self.output_norm(output) * F.silu(gate)
        return self.output_proj(output)


__all__ = [
    "CausalDepthwiseConv1d",
    "RMSNorm",
    "WSLALayer",
    "reference_gated_delta_rule",
]
