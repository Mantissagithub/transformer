import math

import torch

from src.components.attention.kda import KimiDeltaAttention, _recurrent_kda


def _module() -> KimiDeltaAttention:
    return KimiDeltaAttention(
        d_model=32,
        n_heads=4,
        head_dim=8,
        conv_size=4,
        gate_lower_bound=-5.0,
        dropout=0.0,
    )


def test_recurrent_kda_matches_two_step_equation():
    q = torch.tensor([[[[1.0, 0.0]], [[1.0, 0.0]]]])
    k = q.clone()
    v = torch.tensor([[[[2.0, 4.0]], [[1.0, 1.0]]]])
    decay = torch.tensor([[[[0.8, 0.9]], [[0.5, 0.25]]]])
    beta = torch.tensor([[[0.5], [0.5]]])

    out, state = _recurrent_kda(q, k, v, decay.log(), beta)

    scale = 1.0 / math.sqrt(2.0)
    expected_out = torch.tensor(
        [[[[1.0 * scale, 2.0 * scale]], [[0.75 * scale, 1.0 * scale]]]]
    )
    expected_state = torch.tensor([[[[0.75, 1.0], [0.0, 0.0]]]])
    assert torch.allclose(out, expected_out)
    assert torch.allclose(state, expected_state)


def test_kda_gate_stays_inside_k3_bounds():
    torch.manual_seed(0)
    mod = _module()
    x = torch.randn(2, 7, 32)

    log_decay, beta = mod._gates(x)

    assert bool((log_decay > -5.0).all())
    assert bool((log_decay < 0.0).all())
    assert bool((beta > 0.0).all())
    assert bool((beta < 1.0).all())


def test_kda_normalizes_query_and_key_per_head():
    torch.manual_seed(0)
    mod = _module()
    x = torch.randn(2, 7, 32)

    query, key, _, _ = mod._project_qkv(x, None, None, None)

    assert torch.allclose(torch.linalg.vector_norm(query, dim=-1), torch.ones(2, 7, 4))
    assert torch.allclose(torch.linalg.vector_norm(key, dim=-1), torch.ones(2, 7, 4))


def test_kda_shape_backward_and_finite_gradients():
    torch.manual_seed(0)
    mod = _module()
    x = torch.randn(2, 6, 32)

    out = mod(x, x, x)
    out.square().mean().backward()

    assert out.shape == (2, 6, 32)
    assert torch.isfinite(out).all()
    trained = [
        mod.q_proj.weight,
        mod.k_proj.weight,
        mod.v_proj.weight,
        mod.q_conv.conv.weight,
        mod.k_conv.conv.weight,
        mod.v_conv.conv.weight,
        mod.alpha_down.weight,
        mod.alpha_up.weight,
        mod.decay_log_scale,
        mod.alpha_bias,
        mod.beta_proj.weight,
        mod.output_gate.weight,
        mod.output_norm.weight,
        mod.output_proj.weight,
    ]
    assert all(param.grad is not None for param in trained)
    assert all(torch.isfinite(param.grad).all() for param in trained)


def test_kda_is_causal():
    torch.manual_seed(0)
    mod = _module().eval()
    x_a = torch.randn(1, 8, 32)
    x_b = x_a.clone()
    x_b[:, 5:] = torch.randn_like(x_b[:, 5:])

    with torch.no_grad():
        out_a = mod(x_a, x_a, x_a)
        out_b = mod(x_b, x_b, x_b)

    assert torch.allclose(out_a[:, :5], out_b[:, :5], atol=1e-6)
    assert not torch.allclose(out_a[:, 5:], out_b[:, 5:])


def test_kda_prefill_decode_matches_full_recompute():
    torch.manual_seed(0)
    mod = _module().eval()
    x = torch.randn(1, 8, 32)

    with torch.no_grad():
        full = mod(x, x, x)
        prefill_x = x[:, :5]
        prefill, cache = mod(prefill_x, prefill_x, prefill_x, return_kv=True)
        pieces = [prefill]
        for t in range(5, x.shape[1]):
            step = x[:, t : t + 1]
            out, cache = mod(
                step,
                step,
                step,
                past_kv=cache,
                return_kv=True,
            )
            pieces.append(out)

    cached = torch.cat(pieces, dim=1)
    assert torch.allclose(cached, full, atol=1e-5)
    assert cache.position() == x.shape[1]
    assert cache.recurrent_state.shape == (1, 4, 8, 8)
    assert cache.q_history.shape == (1, 3, 32)
    assert cache.k_history.shape == (1, 3, 32)
    assert cache.v_history.shape == (1, 3, 32)


def test_kda_cache_size_does_not_grow_with_sequence_length():
    torch.manual_seed(0)
    mod = _module().eval()
    first = torch.randn(1, 4, 32)

    with torch.no_grad():
        _, cache = mod(first, first, first, return_kv=True)
        state_size = cache.recurrent_state.numel()
        history_sizes = (
            cache.q_history.numel(),
            cache.k_history.numel(),
            cache.v_history.numel(),
        )
        for _ in range(5):
            step = torch.randn(1, 1, 32)
            _, cache = mod(step, step, step, past_kv=cache, return_kv=True)

    assert cache.recurrent_state.numel() == state_size
    assert (
        cache.q_history.numel(),
        cache.k_history.numel(),
        cache.v_history.numel(),
    ) == history_sizes
    assert cache.position() == 9
