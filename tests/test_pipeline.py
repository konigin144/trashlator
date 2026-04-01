from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd

from app.config import AppConfig
from app.pipeline import run_pipeline


def _make_config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        input_path=tmp_path / "input.csv",
        output_path=tmp_path / "output.csv",
        model_name="fake-model",
        batch_size=2,
        device="cpu",
        log_level="INFO",
        log_file=None,
        limit=None,
        source_lang="en",
        target_lang="de",
        report_path=None,
        max_input_length=10,
        chunk_token_limit=160,
        max_new_tokens=512,
        chunk_overlap_tokens=0,
        num_beams=1,
        skip_url_like=True,
        text_column="message",
        label_column="label",
        input_encoding="utf-8-sig",
        output_encoding="utf-8-sig",
    )


def _validate_side_effect(*, source_text, translated_text, precomputed_status=None, precomputed_error_message=None):
    if precomputed_status == "translation_error":
        return SimpleNamespace(
            status="translation_error",
            placeholder_ok=True,
            error_message=precomputed_error_message,
        )

    if translated_text == "missing placeholder":
        return SimpleNamespace(
            status="placeholder_mismatch",
            placeholder_ok=False,
            error_message="Source and target placeholders do not match.",
        )

    if translated_text == "fixed <PERSON>":
        return SimpleNamespace(
            status="ok",
            placeholder_ok=True,
            error_message=None,
        )

    if translated_text == "still missing":
        return SimpleNamespace(
            status="placeholder_mismatch",
            placeholder_ok=False,
            error_message="Source and target placeholders do not match.",
        )

    return SimpleNamespace(
        status=precomputed_status or "ok",
        placeholder_ok=True,
        error_message=precomputed_error_message,
    )


@patch("app.pipeline.summarize_validation", return_value={"ok": 1})
@patch("app.pipeline.validate_translation")
@patch("app.pipeline.unmask_emojis", side_effect=lambda text, replacements: text)
@patch("app.pipeline.mask_emojis")
@patch("app.pipeline.is_url_like_text")
@patch("app.pipeline.OpusTranslator")
def test_run_pipeline_translates_long_record_with_chunking(
    mocked_translator_cls: MagicMock,
    mocked_is_url_like_text: MagicMock,
    mocked_mask_emojis: MagicMock,
    mocked_unmask_emojis: MagicMock,
    mocked_validate_translation: MagicMock,
    mocked_summarize_validation: MagicMock,
    tmp_path: Path,
) -> None:
    config = _make_config(tmp_path)

    df = pd.DataFrame(
        {
            "message": ["this is a very long phishing-like text"],
            "label": [1],
        }
    )
    df.to_csv(config.input_path, index=False, encoding=config.input_encoding)

    mocked_mask_emojis.return_value = SimpleNamespace(
        masked_text="this is a very long phishing-like text",
        replacements=[],
        contains_emoji=False,
    )
    mocked_is_url_like_text.return_value = False

    translator = MagicMock()
    translator.check_input_lengths.return_value = [
        SimpleNamespace(is_too_long=True, token_count=25),
    ]
    translator.max_input_length = 10
    translator.translate_long_text.return_value = SimpleNamespace(
        translated_text="dies ist ein sehr langer text",
        chunk_count=3,
        elapsed_seconds=0.25,
    )
    mocked_translator_cls.return_value = translator

    mocked_validate_translation.return_value = SimpleNamespace(
        status="ok_chunked",
        placeholder_ok=True,
        error_message=None,
    )

    run_pipeline(config)

    translator.translate_long_text.assert_called_once_with(
        "this is a very long phishing-like text",
        chunk_token_limit=160,
        chunk_overlap_tokens=0,
    )

    output_df = pd.read_csv(config.output_path, encoding=config.output_encoding)

    assert output_df.loc[0, "translated_text"] == "dies ist ein sehr langer text"
    assert output_df.loc[0, "status"] == "ok_chunked"
    assert bool(output_df.loc[0, "was_chunked"]) is True
    assert int(output_df.loc[0, "chunk_count"]) == 3
    assert bool(output_df.loc[0, "too_long_for_model"]) is True


@patch("app.pipeline.summarize_validation", return_value={"translation_error": 1})
@patch("app.pipeline.validate_translation")
@patch("app.pipeline.unmask_emojis", side_effect=lambda text, replacements: text)
@patch("app.pipeline.mask_emojis")
@patch("app.pipeline.is_url_like_text")
@patch("app.pipeline.OpusTranslator")
def test_run_pipeline_marks_translation_error_when_chunked_translation_fails(
    mocked_translator_cls: MagicMock,
    mocked_is_url_like_text: MagicMock,
    mocked_mask_emojis: MagicMock,
    mocked_unmask_emojis: MagicMock,
    mocked_validate_translation: MagicMock,
    mocked_summarize_validation: MagicMock,
    tmp_path: Path,
) -> None:
    config = _make_config(tmp_path)

    df = pd.DataFrame(
        {
            "message": ["this is a very long phishing-like text"],
            "label": [1],
        }
    )
    df.to_csv(config.input_path, index=False, encoding=config.input_encoding)

    mocked_mask_emojis.return_value = SimpleNamespace(
        masked_text="this is a very long phishing-like text",
        replacements=[],
        contains_emoji=False,
    )
    mocked_is_url_like_text.return_value = False

    translator = MagicMock()
    translator.check_input_lengths.return_value = [
        SimpleNamespace(is_too_long=True, token_count=25),
    ]
    translator.max_input_length = 10
    translator.translate_long_text.side_effect = RuntimeError("boom")
    mocked_translator_cls.return_value = translator

    mocked_validate_translation.return_value = SimpleNamespace(
        status="translation_error",
        placeholder_ok=True,
        error_message="Chunked translation failed: boom",
    )

    run_pipeline(config)

    output_df = pd.read_csv(config.output_path, encoding=config.output_encoding)

    assert pd.isna(output_df.loc[0, "translated_text"])
    assert output_df.loc[0, "status"] == "translation_error"
    assert bool(output_df.loc[0, "was_chunked"]) is False
    assert int(output_df.loc[0, "chunk_count"]) == 0


@patch("app.pipeline.summarize_validation", return_value={"ok": 1})
@patch("app.pipeline.validate_translation")
@patch("app.pipeline.unmask_emojis", side_effect=lambda text, replacements: text)
@patch("app.pipeline.mask_emojis")
@patch("app.pipeline.is_url_like_text")
@patch("app.pipeline.OpusTranslator")
def test_run_pipeline_retries_placeholder_mismatch_with_half_max_input_length(
    mocked_translator_cls: MagicMock,
    mocked_is_url_like_text: MagicMock,
    mocked_mask_emojis: MagicMock,
    mocked_unmask_emojis: MagicMock,
    mocked_validate_translation: MagicMock,
    mocked_summarize_validation: MagicMock,
    tmp_path: Path,
) -> None:
    config = _make_config(tmp_path)

    df = pd.DataFrame(
        {
            "message": ["hello <PERSON>"],
            "label": [0],
        }
    )
    df.to_csv(config.input_path, index=False, encoding=config.input_encoding)

    mocked_mask_emojis.return_value = SimpleNamespace(
        masked_text="hello <PERSON>",
        replacements=[],
        contains_emoji=False,
    )
    mocked_is_url_like_text.return_value = False
    mocked_validate_translation.side_effect = _validate_side_effect

    translator = MagicMock()
    translator.max_input_length = 10
    translator.check_input_lengths.return_value = [
        SimpleNamespace(is_too_long=False, token_count=3),
    ]
    translator.check_input_length.return_value = SimpleNamespace(
        is_too_long=False,
        token_count=3,
    )
    translator.translate_batch_with_metadata.return_value = SimpleNamespace(
        translations=["missing placeholder"],
        elapsed_seconds=0.1,
        batch_size=1,
    )

    def translate_retry(texts: list[str]) -> list[str]:
        assert translator.max_input_length == 5
        assert texts == ["hello <PERSON>"]
        return ["fixed <PERSON>"]

    translator.translate_batch.side_effect = translate_retry
    mocked_translator_cls.return_value = translator

    run_pipeline(config)

    assert translator.check_input_length.call_count == 1
    assert translator.check_input_length.call_args.args == ("hello <PERSON>",)
    assert translator.translate_batch.call_count == 1
    assert translator.translate_batch.call_args.args == (["hello <PERSON>"],)
    assert translator.max_input_length == 10

    output_df = pd.read_csv(config.output_path, encoding=config.output_encoding)

    assert output_df.loc[0, "translated_text"] == "fixed <PERSON>"
    assert output_df.loc[0, "status"] == "ok"


@patch("app.pipeline.summarize_validation", return_value={"placeholder_mismatch": 1})
@patch("app.pipeline.validate_translation")
@patch("app.pipeline.unmask_emojis", side_effect=lambda text, replacements: text)
@patch("app.pipeline.mask_emojis")
@patch("app.pipeline.is_url_like_text")
@patch("app.pipeline.OpusTranslator")
def test_run_pipeline_returns_second_retry_result_when_placeholder_mismatch_persists(
    mocked_translator_cls: MagicMock,
    mocked_is_url_like_text: MagicMock,
    mocked_mask_emojis: MagicMock,
    mocked_unmask_emojis: MagicMock,
    mocked_validate_translation: MagicMock,
    mocked_summarize_validation: MagicMock,
    tmp_path: Path,
) -> None:
    config = _make_config(tmp_path)

    df = pd.DataFrame(
        {
            "message": ["hello <PERSON>"],
            "label": [0],
        }
    )
    df.to_csv(config.input_path, index=False, encoding=config.input_encoding)

    mocked_mask_emojis.return_value = SimpleNamespace(
        masked_text="hello <PERSON>",
        replacements=[],
        contains_emoji=False,
    )
    mocked_is_url_like_text.return_value = False
    mocked_validate_translation.side_effect = _validate_side_effect

    translator = MagicMock()
    translator.max_input_length = 10
    translator.check_input_lengths.return_value = [
        SimpleNamespace(is_too_long=False, token_count=3),
    ]
    translator.check_input_length.return_value = SimpleNamespace(
        is_too_long=False,
        token_count=3,
    )
    translator.translate_batch_with_metadata.return_value = SimpleNamespace(
        translations=["missing placeholder"],
        elapsed_seconds=0.1,
        batch_size=1,
    )

    def translate_retry(texts: list[str]) -> list[str]:
        assert translator.max_input_length == 5
        assert texts == ["hello <PERSON>"]
        return ["still missing"]

    translator.translate_batch.side_effect = translate_retry
    mocked_translator_cls.return_value = translator

    run_pipeline(config)

    output_df = pd.read_csv(config.output_path, encoding=config.output_encoding)

    assert output_df.loc[0, "translated_text"] == "still missing"
    assert output_df.loc[0, "status"] == "placeholder_mismatch"


@patch("app.pipeline.summarize_validation", return_value={"ok": 1})
@patch("app.pipeline.validate_translation")
@patch("app.pipeline.unmask_emojis", side_effect=lambda text, replacements: text)
@patch("app.pipeline.mask_emojis")
@patch("app.pipeline.is_url_like_text")
@patch("app.pipeline.OpusTranslator")
def test_run_pipeline_uses_normal_batch_translation_for_short_record(
    mocked_translator_cls: MagicMock,
    mocked_is_url_like_text: MagicMock,
    mocked_mask_emojis: MagicMock,
    mocked_unmask_emojis: MagicMock,
    mocked_validate_translation: MagicMock,
    mocked_summarize_validation: MagicMock,
    tmp_path: Path,
) -> None:
    config = _make_config(tmp_path)

    df = pd.DataFrame(
        {
            "message": ["short text"],
            "label": [0],
        }
    )
    df.to_csv(config.input_path, index=False, encoding=config.input_encoding)

    mocked_mask_emojis.return_value = SimpleNamespace(
        masked_text="short text",
        replacements=[],
        contains_emoji=False,
    )
    mocked_is_url_like_text.return_value = False

    translator = MagicMock()
    translator.check_input_lengths.return_value = [
        SimpleNamespace(is_too_long=False, token_count=3),
    ]
    translator.max_input_length = 10
    translator.translate_batch_with_metadata.return_value = SimpleNamespace(
        translations=["kurzer text"],
        elapsed_seconds=0.1,
        batch_size=1,
    )
    mocked_translator_cls.return_value = translator

    mocked_validate_translation.return_value = SimpleNamespace(
        status="ok",
        placeholder_ok=True,
        error_message=None,
    )

    run_pipeline(config)

    translator.translate_batch_with_metadata.assert_called_once_with(["short text"])
    translator.translate_long_text.assert_not_called()

    output_df = pd.read_csv(config.output_path, encoding=config.output_encoding)

    assert output_df.loc[0, "translated_text"] == "kurzer text"
    assert output_df.loc[0, "status"] == "ok"
    assert bool(output_df.loc[0, "was_chunked"]) is False
    assert int(output_df.loc[0, "chunk_count"]) == 0
