import torch

from src.components.attention.msa import MiniMaxSparseAttention


def test_msa_index_branch_receives_gradients():
    torch.manual_seed(0)
    mod = MiniMaxSparseAttention(
        d_model=32,
        n_heads=4,
        n_kv_heads=2,
        block_size=2,
        top_k=2,
        d_idx=4,
        dropout=0.0,
    )
    x = torch.randn(2, 8, 32)

    out = mod(x, x, x)
    out.sum().backward()

    assert mod.w_iq.weight.grad is not None
    assert mod.w_ik.weight.grad is not None
    assert torch.count_nonzero(mod.w_iq.weight.grad).item() > 0
    assert torch.count_nonzero(mod.w_ik.weight.grad).item() > 0
