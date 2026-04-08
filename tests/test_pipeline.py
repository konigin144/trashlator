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
        progress_path=tmp_path / "progress.json",
        max_input_length=10,
        chunk_token_limit=160,
        max_new_tokens=512,
        chunk_overlap_tokens=0,
        num_beams=1,
        skip_url_like=True,
        sort_batches_by_length=False,
        retry_placeholder_mismatch=True,
        checkpoint_interval=50,
        max_translate_tokens=1024,
        batch_token_budget=4096,
        input_chunk_size=5000,
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


@patch("app.pipeline.validate_translation")
@patch("app.pipeline.unmask_emojis", side_effect=lambda text, replacements: text)
@patch("app.pipeline.mask_emojis")
@patch("app.pipeline.is_url_like_text")
@patch("app.pipeline.OpusTranslator")
def test_run_pipeline_sorts_normal_batches_by_token_count_when_enabled(
    mocked_translator_cls: MagicMock,
    mocked_is_url_like_text: MagicMock,
    mocked_mask_emojis: MagicMock,
    mocked_unmask_emojis: MagicMock,
    mocked_validate_translation: MagicMock,
    tmp_path: Path,
) -> None:
    config = _make_config(tmp_path)
    config.batch_size = 2
    config.sort_batches_by_length = True

    df = pd.DataFrame(
        {
            "message": ["first", "second", "third"],
            "label": [0, 0, 0],
        }
    )
    df.to_csv(config.input_path, index=False, encoding=config.input_encoding)

    mocked_mask_emojis.side_effect = [
        SimpleNamespace(masked_text="first", replacements=[], contains_emoji=False),
        SimpleNamespace(masked_text="second", replacements=[], contains_emoji=False),
        SimpleNamespace(masked_text="third", replacements=[], contains_emoji=False),
    ]
    mocked_is_url_like_text.return_value = False
    mocked_validate_translation.return_value = SimpleNamespace(
        status="ok",
        placeholder_ok=True,
        error_message=None,
    )

    translator = MagicMock()
    translator.check_input_lengths.return_value = [
        SimpleNamespace(is_too_long=False, token_count=5),
        SimpleNamespace(is_too_long=False, token_count=2),
        SimpleNamespace(is_too_long=False, token_count=4),
    ]
    translator.translate_batch_with_metadata.side_effect = [
        SimpleNamespace(
            translations=["second-de", "third-de"],
            elapsed_seconds=0.1,
            batch_size=2,
        ),
        SimpleNamespace(
            translations=["first-de"],
            elapsed_seconds=0.1,
            batch_size=1,
        ),
    ]
    mocked_translator_cls.return_value = translator

    run_pipeline(config)

    first_batch = translator.translate_batch_with_metadata.call_args_list[0].args[0]
    second_batch = translator.translate_batch_with_metadata.call_args_list[1].args[0]

    assert first_batch == ["second", "third"]
    assert second_batch == ["first"]

    output_df = pd.read_csv(config.output_path, encoding=config.output_encoding)
    assert output_df["translated_text"].tolist() == ["first-de", "second-de", "third-de"]


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


@patch("app.pipeline.validate_translation")
@patch("app.pipeline.unmask_emojis", side_effect=lambda text, replacements: text)
@patch("app.pipeline.mask_emojis")
@patch("app.pipeline.is_url_like_text")
@patch("app.pipeline.OpusTranslator")
def test_run_pipeline_does_not_retry_placeholder_mismatch_when_disabled(
    mocked_translator_cls: MagicMock,
    mocked_is_url_like_text: MagicMock,
    mocked_mask_emojis: MagicMock,
    mocked_unmask_emojis: MagicMock,
    mocked_validate_translation: MagicMock,
    tmp_path: Path,
) -> None:
    config = _make_config(tmp_path)
    config.retry_placeholder_mismatch = False

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
    translator.translate_batch_with_metadata.return_value = SimpleNamespace(
        translations=["missing placeholder"],
        elapsed_seconds=0.1,
        batch_size=1,
    )
    mocked_translator_cls.return_value = translator

    run_pipeline(config)

    translator.check_input_length.assert_not_called()
    translator.translate_batch.assert_not_called()

    output_df = pd.read_csv(config.output_path, encoding=config.output_encoding)

    assert output_df.loc[0, "translated_text"] == "missing placeholder"
    assert output_df.loc[0, "status"] == "placeholder_mismatch"


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


@patch("app.pipeline.build_qe_columns")
@patch("app.pipeline.validate_translation")
@patch("app.pipeline.unmask_emojis", side_effect=lambda text, replacements: text)
@patch("app.pipeline.mask_emojis")
@patch("app.pipeline.is_url_like_text")
@patch("app.pipeline.OpusTranslator")
def test_run_pipeline_persists_checkpoint_before_unhandled_failure(
    mocked_translator_cls: MagicMock,
    mocked_is_url_like_text: MagicMock,
    mocked_mask_emojis: MagicMock,
    mocked_unmask_emojis: MagicMock,
    mocked_validate_translation: MagicMock,
    mocked_build_qe_columns: MagicMock,
    tmp_path: Path,
) -> None:
    config = _make_config(tmp_path)
    config.batch_size = 1
    config.checkpoint_interval = 1
    config.input_chunk_size = 1

    df = pd.DataFrame(
        {
            "message": ["first", "second"],
            "label": [0, 1],
        }
    )
    df.to_csv(config.input_path, index=False, encoding=config.input_encoding)

    mocked_mask_emojis.side_effect = [
        SimpleNamespace(masked_text="first", replacements=[], contains_emoji=False),
        SimpleNamespace(masked_text="second", replacements=[], contains_emoji=False),
    ]
    mocked_is_url_like_text.return_value = False
    mocked_validate_translation.return_value = SimpleNamespace(
        status="ok",
        placeholder_ok=True,
        error_message=None,
    )

    translator = MagicMock()
    translator.check_input_lengths.return_value = [
        SimpleNamespace(is_too_long=False, token_count=1),
        SimpleNamespace(is_too_long=False, token_count=1),
    ]
    translator.max_input_length = 10
    translator.translate_batch_with_metadata.side_effect = [
        SimpleNamespace(
            translations=["erste"],
            elapsed_seconds=0.1,
            batch_size=1,
        ),
        SimpleNamespace(
            translations=["zweite"],
            elapsed_seconds=0.1,
            batch_size=1,
        ),
    ]
    mocked_translator_cls.return_value = translator

    mocked_build_qe_columns.side_effect = [
        {
            "qe_score": None,
            "qe_label": None,
            "qe_error": None,
            "qe_backend": None,
            "qe_model_name": None,
        },
        RuntimeError("unexpected qe failure"),
    ]

    try:
        run_pipeline(config)
    except RuntimeError as exc:
        assert str(exc) == "unexpected qe failure"
    else:
        raise AssertionError("Expected run_pipeline to raise RuntimeError")

    output_df = pd.read_csv(config.output_path, encoding=config.output_encoding)

    assert output_df.loc[0, "translated_text"] == "erste"
    assert output_df.loc[0, "status"] == "ok"
    assert len(output_df) == 1
    assert config.progress_path.exists()


@patch("app.pipeline.validate_translation")
@patch("app.pipeline.unmask_emojis", side_effect=lambda text, replacements: text)
@patch("app.pipeline.mask_emojis")
@patch("app.pipeline.is_url_like_text")
@patch("app.pipeline.OpusTranslator")
def test_run_pipeline_skips_rows_exceeding_max_translate_tokens(
    mocked_translator_cls: MagicMock,
    mocked_is_url_like_text: MagicMock,
    mocked_mask_emojis: MagicMock,
    mocked_unmask_emojis: MagicMock,
    mocked_validate_translation: MagicMock,
    tmp_path: Path,
) -> None:
    config = _make_config(tmp_path)
    config.max_translate_tokens = 100

    df = pd.DataFrame(
        {
            "message": ["very long text"],
            "label": [1],
        }
    )
    df.to_csv(config.input_path, index=False, encoding=config.input_encoding)

    mocked_mask_emojis.return_value = SimpleNamespace(
        masked_text="very long text",
        replacements=[],
        contains_emoji=False,
    )
    mocked_is_url_like_text.return_value = False
    mocked_validate_translation.return_value = SimpleNamespace(
        status="skipped_max_translate_tokens",
        placeholder_ok=True,
        error_message="Record exceeds max_translate_tokens limit (150 > 100).",
    )

    translator = MagicMock()
    translator.check_input_lengths.return_value = [
        SimpleNamespace(is_too_long=False, token_count=150),
    ]
    mocked_translator_cls.return_value = translator

    run_pipeline(config)

    translator.translate_batch_with_metadata.assert_not_called()
    translator.translate_long_text.assert_not_called()

    output_df = pd.read_csv(config.output_path, encoding=config.output_encoding)
    assert output_df.loc[0, "status"] == "skipped_max_translate_tokens"
    assert bool(output_df.loc[0, "exceeds_max_translate_tokens"]) is True


@patch("app.pipeline.validate_translation")
@patch("app.pipeline.unmask_emojis", side_effect=lambda text, replacements: text)
@patch("app.pipeline.mask_emojis")
@patch("app.pipeline.is_url_like_text")
@patch("app.pipeline.OpusTranslator")
def test_run_pipeline_uses_dynamic_token_budget_for_normal_batches(
    mocked_translator_cls: MagicMock,
    mocked_is_url_like_text: MagicMock,
    mocked_mask_emojis: MagicMock,
    mocked_unmask_emojis: MagicMock,
    mocked_validate_translation: MagicMock,
    tmp_path: Path,
) -> None:
    config = _make_config(tmp_path)
    config.batch_size = 10
    config.batch_token_budget = 10

    df = pd.DataFrame(
        {
            "message": ["a", "b", "c", "d"],
            "label": [0, 0, 0, 0],
        }
    )
    df.to_csv(config.input_path, index=False, encoding=config.input_encoding)

    mocked_mask_emojis.side_effect = [
        SimpleNamespace(masked_text="a", replacements=[], contains_emoji=False),
        SimpleNamespace(masked_text="b", replacements=[], contains_emoji=False),
        SimpleNamespace(masked_text="c", replacements=[], contains_emoji=False),
        SimpleNamespace(masked_text="d", replacements=[], contains_emoji=False),
    ]
    mocked_is_url_like_text.return_value = False
    mocked_validate_translation.return_value = SimpleNamespace(
        status="ok",
        placeholder_ok=True,
        error_message=None,
    )

    translator = MagicMock()
    translator.check_input_lengths.return_value = [
        SimpleNamespace(is_too_long=False, token_count=2),
        SimpleNamespace(is_too_long=False, token_count=3),
        SimpleNamespace(is_too_long=False, token_count=5),
        SimpleNamespace(is_too_long=False, token_count=6),
    ]
    translator.translate_batch_with_metadata.side_effect = [
        SimpleNamespace(
            translations=["a-de", "b-de", "c-de"],
            elapsed_seconds=0.1,
            batch_size=3,
        ),
        SimpleNamespace(
            translations=["d-de"],
            elapsed_seconds=0.1,
            batch_size=1,
        ),
    ]
    mocked_translator_cls.return_value = translator

    run_pipeline(config)

    first_batch = translator.translate_batch_with_metadata.call_args_list[0].args[0]
    second_batch = translator.translate_batch_with_metadata.call_args_list[1].args[0]

    assert first_batch == ["a", "b", "c"]
    assert second_batch == ["d"]


@patch("app.pipeline.validate_translation")
@patch("app.pipeline.unmask_emojis", side_effect=lambda text, replacements: text)
@patch("app.pipeline.mask_emojis")
@patch("app.pipeline.is_url_like_text")
@patch("app.pipeline.OpusTranslator")
def test_run_pipeline_streams_input_in_chunks_and_appends_output(
    mocked_translator_cls: MagicMock,
    mocked_is_url_like_text: MagicMock,
    mocked_mask_emojis: MagicMock,
    mocked_unmask_emojis: MagicMock,
    mocked_validate_translation: MagicMock,
    tmp_path: Path,
) -> None:
    config = _make_config(tmp_path)
    config.input_chunk_size = 2
    config.batch_size = 2
    config.batch_token_budget = 100

    df = pd.DataFrame(
        {
            "message": ["one", "two", "three", "four"],
            "label": [0, 0, 1, 1],
        }
    )
    df.to_csv(config.input_path, index=False, encoding=config.input_encoding)

    mocked_mask_emojis.side_effect = [
        SimpleNamespace(masked_text="one", replacements=[], contains_emoji=False),
        SimpleNamespace(masked_text="two", replacements=[], contains_emoji=False),
        SimpleNamespace(masked_text="three", replacements=[], contains_emoji=False),
        SimpleNamespace(masked_text="four", replacements=[], contains_emoji=False),
    ]
    mocked_is_url_like_text.return_value = False
    mocked_validate_translation.return_value = SimpleNamespace(
        status="ok",
        placeholder_ok=True,
        error_message=None,
    )

    translator = MagicMock()
    translator.check_input_lengths.side_effect = [
        [
            SimpleNamespace(is_too_long=False, token_count=1),
            SimpleNamespace(is_too_long=False, token_count=1),
        ],
        [
            SimpleNamespace(is_too_long=False, token_count=1),
            SimpleNamespace(is_too_long=False, token_count=1),
        ],
    ]
    translator.translate_batch_with_metadata.side_effect = [
        SimpleNamespace(
            translations=["eins", "zwei"],
            elapsed_seconds=0.1,
            batch_size=2,
        ),
        SimpleNamespace(
            translations=["drei", "vier"],
            elapsed_seconds=0.1,
            batch_size=2,
        ),
    ]
    mocked_translator_cls.return_value = translator

    run_pipeline(config)

    output_df = pd.read_csv(config.output_path, encoding=config.output_encoding)

    assert output_df["translated_text"].tolist() == ["eins", "zwei", "drei", "vier"]
    assert output_df["input_row_number"].tolist() == [1, 2, 3, 4]
    assert not config.progress_path.exists()
