from lean_sft.io import assign_splits


def test_assign_splits_is_deterministic():
    rows = [{"id": idx} for idx in range(100)]
    first = assign_splits(rows, seed=7, validation_fraction=0.1, test_fraction=0.1)
    second = assign_splits([{"id": idx} for idx in range(100)], seed=7, validation_fraction=0.1, test_fraction=0.1)
    assert first == second
    assert sum(row["split"] == "test" for row in first) == 10
    assert sum(row["split"] == "validation" for row in first) == 10
