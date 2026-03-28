from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from app.translator import (
    ChunkedTranslationResult,
    LengthCheckResult,
    OpusTranslator,
)


def test_resolve_device_returns_cpu_when_cpu_requested() -> None:
    translator = OpusTranslator.__new__(OpusTranslator)
    resolved = translator._resolve_device("cpu")
    assert resolved == "cpu"


@patch("torch.cuda.is_available", return_value=False)
def test_resolve_device_falls_back_to_cpu_when_cuda_unavailable(
    mocked_cuda_available: MagicMock,
) -> None:
    translator = OpusTranslator.__new__(OpusTranslator)
    resolved = translator._resolve_device("cuda")

    assert resolved == "cpu"
    mocked_cuda_available.assert_called_once()


@patch("torch.cuda.is_available", return_value=True)
def test_resolve_device_returns_cuda_when_available(
    mocked_cuda_available: MagicMock,
) -> None:
    translator = OpusTranslator.__new__(OpusTranslator)
    resolved = translator._resolve_device("cuda")

    assert resolved == "cuda"
    mocked_cuda_available.assert_called_once()


def test_check_input_length_marks_text_as_too_long_when_token_count_exceeds_limit() -> None:
    translator = OpusTranslator.__new__(OpusTranslator)
    translator.max_input_length = 5
    translator.get_token_count = MagicMock(return_value=8)

    result = translator.check_input_length("some text")

    assert isinstance(result, LengthCheckResult)
    assert result.is_too_long is True
    assert result.token_count == 8
    assert result.max_input_length == 5


def test_check_input_length_marks_text_as_valid_when_token_count_within_limit() -> None:
    translator = OpusTranslator.__new__(OpusTranslator)
    translator.max_input_length = 10
    translator.get_token_count = MagicMock(return_value=7)

    result = translator.check_input_length("some text")

    assert result.is_too_long is False
    assert result.token_count == 7
    assert result.max_input_length == 10


def test_check_input_lengths_returns_result_for_each_text() -> None:
    translator = OpusTranslator.__new__(OpusTranslator)
    translator.check_input_length = MagicMock(
        side_effect=[
            LengthCheckResult(is_too_long=False, token_count=3, max_input_length=10),
            LengthCheckResult(is_too_long=True, token_count=12, max_input_length=10),
        ]
    )

    results = translator.check_input_lengths(["short", "very long text"])

    assert len(results) == 2
    assert results[0].is_too_long is False
    assert results[1].is_too_long is True


def test_translate_batch_returns_empty_list_for_empty_input() -> None:
    translator = OpusTranslator.__new__(OpusTranslator)

    result = translator.translate_batch([])

    assert result == []


def test_translate_in_chunks_raises_for_non_positive_batch_size() -> None:
    translator = OpusTranslator.__new__(OpusTranslator)

    try:
        translator.translate_in_chunks(["a", "b"], batch_size=0)
    except ValueError as exc:
        assert str(exc) == "batch_size must be greater than 0"
    else:
        raise AssertionError("Expected ValueError for batch_size=0")


def test_translate_in_chunks_keeps_result_order() -> None:
    translator = OpusTranslator.__new__(OpusTranslator)

    translator.translate_batch_with_metadata = MagicMock(
        side_effect=[
            SimpleNamespace(translations=["t1", "t2"], elapsed_seconds=0.1, batch_size=2),
            SimpleNamespace(translations=["t3"], elapsed_seconds=0.1, batch_size=1),
        ]
    )

    result = translator.translate_in_chunks(["a", "b", "c"], batch_size=2)

    assert result == ["t1", "t2", "t3"]


def test_hard_split_by_tokens_raises_for_non_positive_max_tokens() -> None:
    translator = OpusTranslator.__new__(OpusTranslator)

    try:
        translator._hard_split_by_tokens("some text", max_tokens=0)
    except ValueError as exc:
        assert str(exc) == "max_tokens must be greater than 0"
    else:
        raise AssertionError("Expected ValueError for max_tokens=0")


def test_split_text_into_token_aware_chunks_raises_for_non_positive_max_tokens() -> None:
    translator = OpusTranslator.__new__(OpusTranslator)

    try:
        translator.split_text_into_token_aware_chunks("some text", max_tokens=0)
    except ValueError as exc:
        assert str(exc) == "max_tokens must be greater than 0"
    else:
        raise AssertionError("Expected ValueError for max_tokens=0")


def test_split_text_into_token_aware_chunks_returns_single_chunk_when_text_fits() -> None:
    translator = OpusTranslator.__new__(OpusTranslator)
    translator.get_token_count = MagicMock(return_value=3)

    result = translator.split_text_into_token_aware_chunks("hello world", max_tokens=10)

    assert result == ["hello world"]


def test_split_text_into_token_aware_chunks_splits_by_spaces_using_token_limit() -> None:
    translator = OpusTranslator.__new__(OpusTranslator)

    token_counts = {
        "alpha beta gamma delta epsilon": 12,
        "alpha": 2,
        "beta": 2,
        "gamma": 2,
        "delta": 2,
        "epsilon": 2,
        "alpha beta": 4,
        "alpha beta gamma": 6,
        "alpha beta gamma delta": 8,
        "alpha beta gamma delta epsilon": 10,
        "delta epsilon": 4,
    }

    translator.get_token_count = MagicMock(side_effect=lambda text: token_counts[text])

    result = translator.split_text_into_token_aware_chunks(
        "alpha beta gamma delta epsilon",
        max_tokens=6,
    )

    assert result == ["alpha beta gamma", "delta epsilon"]


def test_split_text_into_token_aware_chunks_uses_hard_split_for_single_oversized_segment() -> None:
    translator = OpusTranslator.__new__(OpusTranslator)

    def fake_get_token_count(text: str) -> int:
        if text == "hugeblob":
            return 20
        if text == "tiny":
            return 1
        if text == "hugeblob tiny":
            return 21
        raise AssertionError(f"Unexpected text: {text}")

    translator.get_token_count = MagicMock(side_effect=fake_get_token_count)
    translator._hard_split_by_tokens = MagicMock(return_value=["huge", "blob"])

    result = translator.split_text_into_token_aware_chunks(
        "hugeblob tiny",
        max_tokens=5,
    )

    assert result == ["huge", "blob", "tiny"]
    translator._hard_split_by_tokens.assert_called_once_with("hugeblob", 5)


def test_translate_long_text_returns_empty_result_for_blank_input() -> None:
    translator = OpusTranslator.__new__(OpusTranslator)
    translator.max_input_length = 512

    result = translator.translate_long_text("   ")

    assert isinstance(result, ChunkedTranslationResult)
    assert result.translated_text == ""
    assert result.chunk_count == 0
    assert result.elapsed_seconds >= 0.0


def test_translate_long_text_translates_each_chunk_and_joins_results() -> None:
    translator = OpusTranslator.__new__(OpusTranslator)
    translator.max_input_length = 512
    translator.split_text_into_token_aware_chunks = MagicMock(
        return_value=["part one", "part two", "part three"]
    )
    translator.translate_batch = MagicMock(
        side_effect=[
            ["eins"],
            ["zwei"],
            ["drei"],
        ]
    )

    result = translator.translate_long_text(
        "very long input",
        chunk_token_limit=128,
    )

    assert result.translated_text == "eins zwei drei"
    assert result.chunk_count == 3
    assert result.elapsed_seconds >= 0.0

    translator.split_text_into_token_aware_chunks.assert_called_once_with(
        "very long input",
        max_tokens=128,
    )


def test_translate_long_text_uses_default_chunk_token_limit_when_not_provided() -> None:
    translator = OpusTranslator.__new__(OpusTranslator)
    translator.max_input_length = 200
    translator.split_text_into_token_aware_chunks = MagicMock(return_value=["chunk"])
    translator.translate_batch = MagicMock(return_value=["translated"])

    result = translator.translate_long_text("long text")

    assert result.translated_text == "translated"
    assert result.chunk_count == 1

    translator.split_text_into_token_aware_chunks.assert_called_once_with(
        "long text",
        max_tokens=160,
    )