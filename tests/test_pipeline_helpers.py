from app.pipeline import _build_dynamic_batches, _chunk_indices, _sort_indices_by_token_count


def test_chunk_indices_splits_list_into_batches() -> None:
    indices = [0, 1, 2, 3, 4]

    result = _chunk_indices(indices, 2)

    assert result == [[0, 1], [2, 3], [4]]


def test_chunk_indices_returns_empty_list_for_empty_input() -> None:
    result = _chunk_indices([], 3)
    assert result == []
    
def test_sort_indices_by_token_count_orders_indices_by_length() -> None:
    result = _sort_indices_by_token_count([4, 1, 3, 0], [5, 2, 8, 3, 1])
    assert result == [4, 1, 0, 3]


def test_build_dynamic_batches_respects_token_budget_and_batch_size() -> None:
    result = _build_dynamic_batches(
        [0, 1, 2, 3],
        [2, 3, 5, 6],
        max_batch_size=10,
        batch_token_budget=10,
    )

    assert result == [[0, 1, 2], [3]]
