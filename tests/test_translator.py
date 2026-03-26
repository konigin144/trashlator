from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from app.translator import LengthCheckResult, OpusTranslator


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