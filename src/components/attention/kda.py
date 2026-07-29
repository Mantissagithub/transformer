from __future__ import annotations

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.registry import ATTENTION

from .base import AttentionBase
from .kv_cache import KDACache


class _CausalShortConv(nn.Module):
    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        if kernel_size <= 0:
            raise ValueError(f"conv_size ({kernel_size}) must be positive")
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            groups=channels,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        history: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, _, c = x.shape
        if history is None:
            # left padding makes the depthwise convolution causal at prefill.
            history = x.new_zeros(b, self.kernel_size - 1, c)
        full = torch.cat([history, x], dim=1)
        out = self.conv(full.transpose(1, 2)).transpose(1, 2)
        if self.kernel_size == 1:
            next_history = full[:, :0]
        else:
            # decode only needs the inputs covered by the next convolution.
            next_history = full[:, -(self.kernel_size - 1) :]
        return F.silu(out), next_history


def _recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    b, seq_len, n_heads, head_dim = q.shape
    if initial_state is None:
        state = torch.zeros(
            b,
            n_heads,
            head_dim,
            head_dim,
            device=q.device,
            dtype=torch.float32,
        )
    else:
        state = initial_state.float()

    # the fp32 state is the fixed-size memory that replaces a growing kv cache.
    q = q.float() * (head_dim**-0.5)
    k = k.float()
    v = v.float()
    log_decay = log_decay.float()
    beta = beta.float()

    outputs = []
    for t in range(seq_len):
        q_t = q[:, t]
        k_t = k[:, t]
        v_t = v[:, t]
        beta_t = beta[:, t]

        # decay each key channel before correcting the stored k -> v map.
        state = state * log_decay[:, t].exp().unsqueeze(-1)
        prediction = torch.einsum("bhk,bhkv->bhv", k_t, state)
        error = v_t - prediction
        state = state + torch.einsum(
            "bhk,bhv->bhkv",
            beta_t.unsqueeze(-1) * k_t,
            error,
        )

        # the current token reads after its own delta-rule update.
        outputs.append(torch.einsum("bhk,bhkv->bhv", q_t, state))

    return torch.stack(outputs, dim=1), state


def _chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    gate_lower_bound: float,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    from fla.ops.kda import chunk_kda

    # tensor cores handle qkv while decay and the recurrent state stay fp32.
    kernel_dtype = v.dtype
    return chunk_kda(
        q=q.to(kernel_dtype),
        k=k.to(kernel_dtype),
        v=v,
        g=log_decay,
        beta=beta.to(kernel_dtype),
        scale=q.shape[-1] ** -0.5,
        initial_state=initial_state,
        output_final_state=output_final_state,
        safe_gate=gate_lower_bound >= -5.0,
        lower_bound=gate_lower_bound,
    )


@ATTENTION.register("kda")
class KimiDeltaAttention(AttentionBase):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        head_dim: int | None = None,
        conv_size: int = 4,
        gate_lower_bound: float = -5.0,
        norm_eps: float = 1e-5,
        dropout: float = 0.0,
    ) -> None:
        if n_heads <= 0:
            raise ValueError(f"n_heads ({n_heads}) must be positive")
        if head_dim is None:
            if d_model % n_heads != 0:
                raise ValueError(
                    f"d_model ({d_model}) not divisible by n_heads ({n_heads}); "
                    "set head_dim explicitly for expanded kda projections"
                )
            head_dim = d_model // n_heads
        if head_dim <= 0:
            raise ValueError(f"head_dim ({head_dim}) must be positive")
        if gate_lower_bound >= 0:
            raise ValueError(
                f"gate_lower_bound ({gate_lower_bound}) must be negative"
            )

        # kda can expand beyond d_model, so the base sees the projected width.
        super().__init__(n_heads * head_dim, n_heads, dropout)
        self.d_model = d_model
        self.head_dim = head_dim
        self.projection_dim = n_heads * head_dim
        self.conv_size = conv_size
        self.gate_lower_bound = gate_lower_bound

        self.q_proj = nn.Linear(d_model, self.projection_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.projection_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.projection_dim, bias=False)

        self.q_conv = _CausalShortConv(self.projection_dim, conv_size)
        self.k_conv = _CausalShortConv(self.projection_dim, conv_size)
        self.v_conv = _CausalShortConv(self.projection_dim, conv_size)

        # the decay bottleneck has the paper's one-head-width rank.
        self.alpha_down = nn.Linear(d_model, head_dim, bias=False)
        self.alpha_up = nn.Linear(head_dim, self.projection_dim, bias=False)
        self.decay_log_scale = nn.Parameter(torch.zeros(n_heads))

        # this initialization starts with long retention while letting channels
        # learn different decay rates from the first update.
        dt = torch.exp(
            torch.rand(self.projection_dim)
            * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        ).clamp(min=1e-4)
        self.alpha_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))

        self.beta_proj = nn.Linear(d_model, n_heads, bias=False)
        self.output_gate = nn.Linear(d_model, self.projection_dim, bias=False)
        self.output_norm = nn.RMSNorm(head_dim, eps=norm_eps)
        self.output_proj = nn.Linear(self.projection_dim, d_model, bias=False)

    def init_cache(self) -> KDACache:
        return KDACache()

    def _gates(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, seq_len, _ = x.shape
        z = self.alpha_up(self.alpha_down(x)).view(
            b,
            seq_len,
            self.n_heads,
            self.head_dim,
        )
        z = z.float() + self.alpha_bias.view(1, 1, self.n_heads, self.head_dim)
        scale = self.decay_log_scale.exp().view(1, 1, self.n_heads, 1)
        log_decay = self.gate_lower_bound * torch.sigmoid(scale * z)
        beta = torch.sigmoid(self.beta_proj(x).float())
        return log_decay, beta

    def _project_qkv(
        self,
        x: torch.Tensor,
        q_history: torch.Tensor | None,
        k_history: torch.Tensor | None,
        v_history: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        b, seq_len, _ = x.shape
        query, next_q_history = self.q_conv(self.q_proj(x), q_history)
        key, next_k_history = self.k_conv(self.k_proj(x), k_history)
        value, next_v_history = self.v_conv(self.v_proj(x), v_history)

        query = query.view(b, seq_len, self.n_heads, self.head_dim)
        key = key.view(b, seq_len, self.n_heads, self.head_dim)
        value = value.view(b, seq_len, self.n_heads, self.head_dim)

        # l2-normalized q and k keep the delta transition stable.
        query = F.normalize(query.float(), dim=-1)
        key = F.normalize(key.float(), dim=-1)
        histories = next_q_history, next_k_history, next_v_history
        return query, key, value, histories

    def forward(self, q, k, v, mask=None, past_kv=None, return_kv=False):
        if k is not q or v is not q:
            warnings.warn(
                "kda ignores k/v; treating q as the hidden state stream",
                stacklevel=2,
            )
        if past_kv is not None and not isinstance(past_kv, KDACache):
            raise TypeError(f"kda expected KDACache, got {type(past_kv).__name__}")

        x = q
        b, seq_len, _ = x.shape
        q_history = None if past_kv is None else past_kv.q_history
        k_history = None if past_kv is None else past_kv.k_history
        v_history = None if past_kv is None else past_kv.v_history

        query, key, value, histories = self._project_qkv(
            x,
            q_history,
            k_history,
            v_history,
        )
        next_q_history, next_k_history, next_v_history = histories

        log_decay, beta = self._gates(x)
        initial_state = None if past_kv is None else past_kv.recurrent_state
        needs_state = past_kv is not None or return_kv
        if query.is_cuda:
            recurrent_out, final_state = _chunk_kda(
                query,
                key,
                value,
                log_decay,
                beta,
                initial_state,
                needs_state,
                self.gate_lower_bound,
            )
        else:
            recurrent_out, final_state = _recurrent_kda(
                query,
                key,
                value,
                log_decay,
                beta,
                initial_state,
            )

        gate = torch.sigmoid(self.output_gate(x).float()).view(
            b,
            seq_len,
            self.n_heads,
            self.head_dim,
        )
        out = self.output_norm(recurrent_out) * gate
        out = out.reshape(b, seq_len, self.projection_dim).to(x.dtype)
        out = self.dropout(self.output_proj(out))

        if past_kv is not None or return_kv:
            if past_kv is None:
                past_kv = self.init_cache()
            assert final_state is not None
            past_kv.recurrent_state = final_state
            past_kv.q_history = next_q_history
            past_kv.k_history = next_k_history
            past_kv.v_history = next_v_history
            past_kv.total_seen += seq_len
        if return_kv:
            return out, past_kv
        return out
