from app.pipeline import _chunk_indices


def test_chunk_indices_splits_list_into_batches() -> None:
    indices = [0, 1, 2, 3, 4]

    result = _chunk_indices(indices, 2)

    assert result == [[0, 1], [2, 3], [4]]


def test_chunk_indices_returns_empty_list_for_empty_input() -> None:
    result = _chunk_indices([], 3)
    assert result == []