# msa (minimax sparse attention) is a gqa-based block-sparse self-attention.
# the idea: don't let every query attend to every key. instead a cheap index
# branch scores the keys, those scores get max-pooled into per-block scores,
# and a top-k selector keeps only a few kv blocks per gqa group. the main
# sparse branch then runs ordinary gqa attention, but only over the keys that
# live in the selected blocks.
#
# the flow has two branches:
#   1) index branch — one lightweight index query per gqa group (q_idx, h heads)
#      attends to a single shared index key head (k_idx). that gives a score
#      matrix per group; block-max-pool collapses each key block to its best
#      score, then top-k picks the strongest blocks. one set of block indices
#      per gqa group (i_1, i_2, ... in the diagram).
#   2) sparse branch — the real q (H heads), k, v (h kv heads) partitioned into
#      blocks. each query gathers only the k/v inside its group's selected
#      blocks and attends over those, with the usual causal mask. heads inside
#      a gqa group share the group's block selection.
#
# this is intrinsically causal self-attention: there is no cross-attn path and
# selection/attention are masked to keys <= the query position, so msa runs
# through the causal-lm path only (see ATTENTION_CAPABILITIES in builder.py).
#
# no rope here — the diagram shows none; position enters purely through the
# causal mask. the gather is genuine: attention only ever sees top_k*block_size
# keys per query, not the full sequence, so the flop saving is real.

"""
shapes:
  s          = sequence length
  d          = hidden size (d_model)
  H          = number of query heads (n_heads)
  h          = number of kv / index-query heads (n_kv_heads)
  group_size = H // h           query heads per gqa group
  d_head     = d // H           main head dim
  d_idx      = index-branch head dim
  block_size = kv block length for selection
  top_k      = number of blocks kept per (group, query)

implemented:
  index branch (block-max-pool + top-k), gather-based sparse gqa attention,
  vectorized-over-t prefill, per-t serial reference, kv-cache for incremental
  decode (full k/v retained plus the index keys, since block selection needs
  every past block available to gather from).

not implemented (future):
  batched-over-bi vectorization (forward still loops batch elements, like the
  csa/hca siblings).
"""

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.registry import ATTENTION

try:
    from .base import AttentionBase
    from .kv_cache import MSACache
except ImportError:  # allows `python src/components/attention/msa.py` from repo root
    from src.components.attention.base import AttentionBase
    from src.components.attention.kv_cache import MSACache


def block_max_pool(scores: Tensor, block_size: int) -> Tensor:
    """max-pool the last (key) axis into blocks of block_size. keys that are
    masked out upstream should already be -inf so an all-masked block pools to
    -inf and never wins top-k. returns [..., n_blocks]."""
    *lead, s_k = scores.shape
    n_blocks = (s_k + block_size - 1) // block_size
    pad = n_blocks * block_size - s_k
    if pad:
        scores = F.pad(scores, (0, pad), value=float("-inf"))
    return scores.view(*lead, n_blocks, block_size).amax(dim=-1)


def select_topk_blocks(block_scores: Tensor, top_k: int) -> Tensor:
    """top-k block indices along the last axis. returns [..., k_eff] where
    k_eff = min(top_k, n_blocks)."""
    k_eff = min(top_k, block_scores.shape[-1])
    return block_scores.topk(k_eff, dim=-1).indices


def block_score_bias(block_scores: Tensor, sel: Tensor, block_size: int, group_size: int) -> Tensor:
    """turn selected block scores into an additive attention-logit bias.

    The forward sparse pattern stays hard top-k, but the selected blocks carry
    a differentiable prior from the index branch so w_iq / w_ik receive
    training signal from the LM loss.
    """
    sel_scores = block_scores.gather(dim=-1, index=sel)                      # [..., k_eff]
    sel_logp = torch.log_softmax(sel_scores, dim=-1)                         # [..., k_eff]
    bias = sel_logp[..., None].expand(*sel_logp.shape, block_size)
    bias = bias.reshape(*sel_logp.shape[:-1], sel_logp.shape[-1] * block_size)
    return bias.repeat_interleave(group_size, dim=0)


@ATTENTION.register("msa")
class MiniMaxSparseAttention(AttentionBase):
    def __init__(
        self,
        d_model:    int,
        n_heads:    int,
        n_kv_heads: int,
        block_size: int,
        top_k:      int,
        d_idx:      int,
        dropout:    float = 0.0,
        bias:       bool = True,
    ) -> None:
        super().__init__(d_model, n_heads, dropout)
        if n_heads % n_kv_heads != 0:
            raise ValueError(f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})")
        self.n_kv_heads = n_kv_heads
        self.group_size = n_heads // n_kv_heads
        self.block_size = block_size
        self.top_k = top_k
        self.d_idx = d_idx
        self.kv_dim = self.d_head * n_kv_heads

        # main gqa projections (mirror gqa.py)
        self.wq = nn.Linear(d_model, d_model, bias=bias)
        self.wk = nn.Linear(d_model, self.kv_dim, bias=bias)
        self.wv = nn.Linear(d_model, self.kv_dim, bias=bias)
        self.wo = nn.Linear(d_model, d_model, bias=bias)

        # index branch: one index query per gqa group, single shared index key.
        self.w_iq = nn.Linear(d_model, n_kv_heads * d_idx, bias=bias)
        self.w_ik = nn.Linear(d_model, d_idx, bias=bias)

    def init_cache(self) -> MSACache:
        return MSACache()

    def forward(self, q, k, v, mask=None, past_kv=None, return_kv=False):
        """msa is intrinsically causal self-attention. k/v/mask are ignored —
        causality lives in the index-branch and sparse-branch key masks
        (kj <= qi)."""
        if k is not q or v is not q:
            warnings.warn(
                "msa ignores k/v; treating q as the hidden state stream",
                stacklevel=2,
            )

        if past_kv is not None and past_kv.total_seen > 0 and q.shape[1] == 1:
            out = self._decode_step(q, past_kv)
            return (out, past_kv) if return_kv else out

        b, _, _ = q.shape
        outs = [self._forward_seq(q[bi]) for bi in range(b)]
        out = self.dropout(torch.stack(outs, dim=0))
        if return_kv:
            if past_kv is None:
                past_kv = self.init_cache()
            self._populate_cache_from_prefill(q, past_kv)
            return out, past_kv
        return out

    def _forward_seq(self, H: Tensor) -> Tensor:
        """vectorized-over-t causal forward for one sequence. [s, d] -> [s, d]."""
        s, _ = H.shape
        n_h, h, d_head = self.n_heads, self.n_kv_heads, self.d_head
        d_idx, block_size, group_size = self.d_idx, self.block_size, self.group_size

        q  = self.wq(H).view(s, n_h, d_head)
        k  = self.wk(H).view(s, h, d_head)
        v  = self.wv(H).view(s, h, d_head)
        iq = self.w_iq(H).view(s, h, d_idx)
        ik = self.w_ik(H)                                         # [s, d_idx]

        ts = torch.arange(s, device=H.device)
        key_ok = ts[None, :] <= ts[:, None]                      # [s_q, s_k] : kj <= qi

        # index branch: [h, s_q, s_k] scores, future keys masked before pooling.
        idx_scores = torch.einsum("qgi,ki->gqk", iq, ik) / math.sqrt(d_idx)
        idx_scores = idx_scores.masked_fill(~key_ok[None], float("-inf"))
        block_scores = block_max_pool(idx_scores, block_size)    # [h, s_q, n_blocks]
        sel = select_topk_blocks(block_scores, self.top_k)       # [h, s_q, k_eff]
        k_eff = sel.shape[-1]
        score_bias = block_score_bias(block_scores, sel, block_size, group_size)

        # expand selected blocks to key positions [h, s_q, k_eff*block_size]
        within = torch.arange(block_size, device=H.device)
        kpos = (sel[..., None] * block_size + within).reshape(h, s, k_eff * block_size)
        n_keys = kpos.shape[-1]
        valid = (kpos < s) & (kpos <= ts[None, :, None])         # in-range and causal
        kpos_c = kpos.clamp(max=s - 1)

        # gather the selected group kv, then expand groups to heads
        k_g = k.permute(1, 0, 2)                                 # [h, s, d_head]
        v_g = v.permute(1, 0, 2)
        gidx = torch.arange(h, device=H.device)[:, None, None].expand(h, s, n_keys)
        k_sel = k_g[gidx, kpos_c]                                # [h, s_q, n_keys, d_head]
        v_sel = v_g[gidx, kpos_c]
        k_sel = k_sel.repeat_interleave(group_size, dim=0)       # [H, s_q, n_keys, d_head]
        v_sel = v_sel.repeat_interleave(group_size, dim=0)
        valid_h = valid.repeat_interleave(group_size, dim=0)     # [H, s_q, n_keys]

        # sparse attention over the gathered keys only
        q_h = q.permute(1, 0, 2)                                 # [H, s_q, d_head]
        scores = torch.einsum("hqd,hqnd->hqn", q_h, k_sel) / math.sqrt(d_head)
        scores = scores + score_bias
        scores = scores.masked_fill(~valid_h, float("-inf"))
        any_valid = valid_h.any(dim=-1)                          # [H, s_q]
        # a query with no visible key would softmax an all-(-inf) row -> NaN;
        # zero those logits then zero the head output below.
        scores = scores.masked_fill(~any_valid[..., None], 0.0)
        weights = torch.softmax(scores, dim=-1)
        out = torch.einsum("hqn,hqnd->hqd", weights, v_sel)      # [H, s_q, d_head]
        out = out * any_valid[..., None]
        out = out.permute(1, 0, 2).reshape(s, n_h * d_head)
        return self.wo(out)

    def _populate_cache_from_prefill(self, H: Tensor, cache: MSACache) -> None:
        """build the decode-side state from the just-processed prefill batch.
        H is [b, s, d]; cache keeps the full head-split k/v plus index keys."""
        b, s, _ = H.shape
        k  = self.wk(H).view(b, s, self.n_kv_heads, self.d_head).permute(0, 2, 1, 3)
        v  = self.wv(H).view(b, s, self.n_kv_heads, self.d_head).permute(0, 2, 1, 3)
        ik = self.w_ik(H)                                        # [b, s, d_idx]
        cache.k, cache.v, cache.ik = k.contiguous(), v.contiguous(), ik.contiguous()
        cache.total_seen = s

    def _decode_step(self, q: Tensor, cache: MSACache) -> Tensor:
        """incremental decode for a single new token. q is [b, 1, d]; returns
        [b, 1, d]. mutates `cache` in place."""
        b, _, _ = q.shape
        n_h, h, d_head = self.n_heads, self.n_kv_heads, self.d_head
        d_idx, block_size, group_size = self.d_idx, self.block_size, self.group_size
        h_new = q[:, 0, :]                                       # [b, d]

        # project and append the new token's kv + index key.
        k_new  = self.wk(h_new).view(b, h, 1, d_head)
        v_new  = self.wv(h_new).view(b, h, 1, d_head)
        ik_new = self.w_ik(h_new)[:, None, :]                    # [b, 1, d_idx]
        cache.update(k_new, v_new, ik_new)
        t_abs = cache.total_seen - 1                             # position of the new query

        outs = []
        for bi in range(b):
            iq = self.w_iq(h_new[bi]).view(h, d_idx)             # [h, d_idx]
            ik = cache.ik[bi]                                    # [s_k, d_idx]
            # every cached key is causal for the new query, so no key mask here.
            idx_scores = (iq @ ik.T) / math.sqrt(d_idx)          # [h, s_k]
            block_scores = block_max_pool(idx_scores, block_size)
            sel = select_topk_blocks(block_scores, self.top_k)   # [h, k_eff]
            k_eff = sel.shape[-1]
            score_bias = block_score_bias(block_scores, sel, block_size, group_size)

            within = torch.arange(block_size, device=q.device)
            kpos = (sel[..., None] * block_size + within).reshape(h, k_eff * block_size)
            valid = kpos <= t_abs                                # [h, n_keys]
            kpos_c = kpos.clamp(max=t_abs)

            k_bi = cache.k[bi]                                   # [h, s_k, d_head]
            v_bi = cache.v[bi]
            gidx = torch.arange(h, device=q.device)[:, None].expand_as(kpos_c)
            k_sel = k_bi[gidx, kpos_c]                           # [h, n_keys, d_head]
            v_sel = v_bi[gidx, kpos_c]
            k_sel = k_sel.repeat_interleave(group_size, dim=0)   # [H, n_keys, d_head]
            v_sel = v_sel.repeat_interleave(group_size, dim=0)
            valid_h = valid.repeat_interleave(group_size, dim=0)

            q_bi = self.wq(h_new[bi]).view(n_h, d_head)          # [H, d_head]
            scores = torch.einsum("hd,hnd->hn", q_bi, k_sel) / math.sqrt(d_head)
            scores = scores + score_bias
            scores = scores.masked_fill(~valid_h, float("-inf"))
            any_valid = valid_h.any(dim=-1)
            scores = scores.masked_fill(~any_valid[:, None], 0.0)
            weights = torch.softmax(scores, dim=-1)
            o = torch.einsum("hn,hnd->hd", weights, v_sel)       # [H, d_head]
            o = o * any_valid[:, None]
            outs.append(self.wo(o.reshape(n_h * d_head)))

        return self.dropout(torch.stack(outs, dim=0)[:, None, :])  # [b, 1, d]

    def _forward_seq_serial(self, H: Tensor) -> Tensor:
        """per-t reference implementation, used to verify the vectorized path."""
        s, _ = H.shape
        n_h, h, d_head = self.n_heads, self.n_kv_heads, self.d_head
        d_idx, block_size, group_size = self.d_idx, self.block_size, self.group_size
        out = H.new_zeros(s, self.d_model)

        q  = self.wq(H).view(s, n_h, d_head)
        k  = self.wk(H).view(s, h, d_head)
        v  = self.wv(H).view(s, h, d_head)
        iq = self.w_iq(H).view(s, h, d_idx)
        ik = self.w_ik(H)

        for t in range(s):
            idx_scores = (iq[t] @ ik[: t + 1].T) / math.sqrt(d_idx)   # [h, t+1]
            block_scores = block_max_pool(idx_scores, block_size)     # [h, n_blocks]
            sel = select_topk_blocks(block_scores, self.top_k)        # [h, k_eff]
            k_eff = sel.shape[-1]
            score_bias = block_score_bias(block_scores, sel, block_size, group_size)

            within = torch.arange(block_size, device=H.device)
            kpos = (sel[..., None] * block_size + within).reshape(h, k_eff * block_size)
            valid = kpos <= t                                         # [h, n_keys]
            kpos_c = kpos.clamp(max=t)

            gidx = torch.arange(h, device=H.device)[:, None].expand_as(kpos_c)
            k_sel = k[kpos_c, gidx]                                   # [h, n_keys, d_head]
            v_sel = v[kpos_c, gidx]
            k_sel = k_sel.repeat_interleave(group_size, dim=0)       # [H, n_keys, d_head]
            v_sel = v_sel.repeat_interleave(group_size, dim=0)
            valid_h = valid.repeat_interleave(group_size, dim=0)

            scores = torch.einsum("hd,hnd->hn", q[t], k_sel) / math.sqrt(d_head)
            scores = scores + score_bias
            scores = scores.masked_fill(~valid_h, float("-inf"))
            any_valid = valid_h.any(dim=-1)
            scores = scores.masked_fill(~any_valid[:, None], 0.0)
            weights = torch.softmax(scores, dim=-1)
            o = torch.einsum("hn,hnd->hd", weights, v_sel)           # [H, d_head]
            o = o * any_valid[:, None]
            out[t] = self.wo(o.reshape(n_h * d_head))
        return out


if __name__ == "__main__":
    # smoke test for the module — not a unit suite.
    torch.manual_seed(0)
    mod = MiniMaxSparseAttention(
        d_model=128, n_heads=8, n_kv_heads=2,
        block_size=4, top_k=3, d_idx=16,
    )
    x = torch.randn(2, 64, 128)

    out = mod(x, x, x)
    assert out.shape == (2, 64, 128), out.shape
    assert not torch.isnan(out).any(), "msa produced nans"

    # equivalence check: vectorized path vs per-t serial reference.
    with torch.no_grad():
        ref = torch.stack([mod._forward_seq_serial(x[bi]) for bi in range(x.shape[0])], dim=0)
    diff = (out - ref).abs().max().item()
    assert diff < 1e-4, f"vectorized != serial, max abs diff {diff}"

    # prefill-vs-decode parity: run the cache one token at a time and compare.
    with torch.no_grad():
        full, cache = mod(x, x, x, past_kv=mod.init_cache(), return_kv=True)
        step_cache = mod.init_cache()
        steps = []
        for t in range(x.shape[1]):
            o_t, step_cache = mod(
                x[:, t : t + 1], x[:, t : t + 1], x[:, t : t + 1],
                past_kv=step_cache, return_kv=True,
            )
            steps.append(o_t)
        decoded = torch.cat(steps, dim=1)
    pd = (full - decoded).abs().max().item()
    assert pd < 1e-4, f"prefill != decode, max abs diff {pd}"

    # selected block scores also bias the sparse logits, so the index branch
    # receives gradient even though the gather pattern is still hard top-k.
    loss = out.sum()
    loss.backward()
    for name in ("wq", "wk", "wv", "wo", "w_iq", "w_ik"):
        p = getattr(mod, name).weight
        assert p.grad is not None, f"{name} did not receive a gradient"
    print(f"msa smoke test ok (vec vs serial {diff:.2e}, prefill vs decode {pd:.2e})")
