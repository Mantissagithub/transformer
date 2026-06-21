import torch

from src.components.attention.msa import MiniMaxSparseAttention, select_topk_blocks


def test_select_topk_blocks_always_includes_local_block():
    scores = torch.tensor(
        [
            [10.0, 9.0, -1.0],
            [8.0, -1.0, 7.0],
        ]
    )
    local = torch.tensor([2, 1])

    selected = select_topk_blocks(scores, top_k=2, local_blocks=local)

    assert selected.tolist() == [[2, 0], [1, 0]]


def test_msa_lm_loss_does_not_train_index_branch():
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

    assert mod.wq.weight.grad is not None
    assert mod.wk.weight.grad is not None
    assert mod.wv.weight.grad is not None
    assert mod.wo.weight.grad is not None
    assert mod.w_iq.weight.grad is None
    assert mod.w_ik.weight.grad is None


def test_msa_kl_alignment_trains_only_index_branch():
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

    loss = mod.kl_alignment_loss(x)
    loss.backward()

    assert torch.isfinite(loss)
    assert mod.w_iq.weight.grad is not None
    assert mod.w_ik.weight.grad is not None
    assert torch.count_nonzero(mod.w_iq.weight.grad).item() > 0
    assert torch.count_nonzero(mod.w_ik.weight.grad).item() > 0
    assert mod.wq.weight.grad is None
    assert mod.wk.weight.grad is None
    assert mod.wv.weight.grad is None
