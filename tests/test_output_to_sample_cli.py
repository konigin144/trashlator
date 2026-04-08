import csv
from pathlib import Path

import pytest

from utils.output_to_sample_cli import (
    default_output_path,
    extract_sample_columns,
    lowercase_preserving_placeholders,
    remove_emoji_placeholders,
    remove_punctuation_preserving_placeholders,
)



def write_csv(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)



def read_csv(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.reader(handle))



def test_extract_sample_columns_maps_translated_text_to_message(tmp_path: Path) -> None:
    input_path = tmp_path / "translated.csv"
    output_path = tmp_path / "sample.csv"
    write_csv(
        input_path,
        [
            ["message", "label", "translated_text", "status"],
            ["hello", "1", "hallo", "ok"],
            ["world", "0", "welt", "ok"],
        ],
    )

    row_count = extract_sample_columns(input_path, output_path)

    assert row_count == 2
    assert read_csv(output_path) == [
        ["message", "label"],
        ["hallo", "1"],
        ["welt", "0"],
    ]



def test_remove_emoji_placeholders_removes_broken_variants() -> None:
    text = "earn money 👇__EMOJI_1_ _EMOJI_2__ EMOJI_3_ now"

    result = remove_emoji_placeholders(text)

    assert result == "earn money 👇 now"



def test_remove_emoji_placeholders_handles_lowercase_and_spaced_variants() -> None:
    text = "🔥 ___emoji_11_ __➖➖ emoji _➖ ____EMOJI_34______ end"

    result = remove_emoji_placeholders(text)

    assert "emoji" not in result.lower()
    assert result == "🔥 __➖➖ ➖ end"



def test_lowercase_preserving_placeholders_keeps_placeholder_tokens() -> None:
    text = "HELLO <PERSON> EMAIL <EMAIL_ADDRESS> CALL <PHONE_NUMBER> __EMOJI_1__ TODAY"

    result = lowercase_preserving_placeholders(text)

    assert result == "hello <PERSON> email <EMAIL_ADDRESS> call <PHONE_NUMBER> __EMOJI_1__ today"



def test_remove_punctuation_preserving_placeholders_keeps_angle_placeholders() -> None:
    text = "Hello, <PERSON>! Email: <EMAIL_ADDRESS>. Call? <PHONE_NUMBER>"

    result = remove_punctuation_preserving_placeholders(text)

    assert result == "Hello <PERSON> Email <EMAIL_ADDRESS> Call <PHONE_NUMBER>"



def test_extract_sample_columns_can_remove_emoji_placeholders_before_lowercasing(tmp_path: Path) -> None:
    input_path = tmp_path / "translated.csv"
    output_path = tmp_path / "sample.csv"
    write_csv(
        input_path,
        [
            ["message", "label", "translated_text", "status"],
            ["hello", "1", "GREET __EMOJI_1_ <PERSON> AT <EMAIL_ADDRESS>", "ok"],
        ],
    )

    row_count = extract_sample_columns(
        input_path,
        output_path,
        remove_emoji_placeholders_flag=True,
        lowercase_messages=True,
    )

    assert row_count == 1
    assert read_csv(output_path) == [
        ["message", "label"],
        ["greet <PERSON> at <EMAIL_ADDRESS>", "1"],
    ]



def test_extract_sample_columns_can_remove_punctuation(tmp_path: Path) -> None:
    input_path = tmp_path / "translated.csv"
    output_path = tmp_path / "sample.csv"
    write_csv(
        input_path,
        [
            ["message", "label", "translated_text", "status"],
            ["hello", "1", "GREET, <PERSON>! AT: <EMAIL_ADDRESS>.", "ok"],
        ],
    )

    row_count = extract_sample_columns(
        input_path,
        output_path,
        lowercase_messages=True,
        remove_punctuation=True,
    )

    assert row_count == 1
    assert read_csv(output_path) == [
        ["message", "label"],
        ["greet <PERSON> at <EMAIL_ADDRESS>", "1"],
    ]



def test_extract_sample_columns_autopopulates_skipped_url_like_rows(tmp_path: Path) -> None:
    input_path = tmp_path / "translated.csv"
    output_path = tmp_path / "sample.csv"
    write_csv(
        input_path,
        [
            ["message", "label", "translated_text", "status", "url_like_skipped"],
            ["httpsmyexamplecomloginverifytoken123", "0", "", "skipped_url_like", "True"],
            ["hello", "1", "hallo", "ok", "False"],
        ],
    )

    row_count = extract_sample_columns(input_path, output_path, autopopulate_url_like=True)

    assert row_count == 2
    assert read_csv(output_path) == [
        ["message", "label"],
        ["httpsmyexamplecomloginverifytoken123", "0"],
        ["hallo", "1"],
    ]



def test_extract_sample_columns_drops_skipped_max_translate_tokens_rows(tmp_path: Path) -> None:
    input_path = tmp_path / "translated.csv"
    output_path = tmp_path / "sample.csv"
    write_csv(
        input_path,
        [
            ["message", "label", "translated_text", "status"],
            ["original-1", "1", "translated-1", "ok"],
            ["original-2", "0", "", "skipped_max_translate_tokens"],
            ["original-3", "1", "translated-3", "ok"],
        ],
    )

    row_count = extract_sample_columns(
        input_path,
        output_path,
        drop_skipped_max_translate_tokens=True,
    )

    assert row_count == 2
    assert read_csv(output_path) == [
        ["message", "label"],
        ["translated-1", "1"],
        ["translated-3", "1"],
    ]



def test_default_output_path_adds_sample_suffix() -> None:
    input_path = Path("data/output/sample_100_de_out.csv")

    result = default_output_path(input_path)

    assert result == Path("data/output/sample_100_de_out_sample.csv")



def test_extract_sample_columns_requires_translated_text_and_label(tmp_path: Path) -> None:
    input_path = tmp_path / "translated.csv"
    write_csv(
        input_path,
        [
            ["message", "status"],
            ["hello", "ok"],
        ],
    )

    with pytest.raises(ValueError, match="label, translated_text|translated_text, label"):
        extract_sample_columns(input_path, tmp_path / "sample.csv")



def test_extract_sample_columns_requires_message_when_autopopulating(tmp_path: Path) -> None:
    input_path = tmp_path / "translated.csv"
    write_csv(
        input_path,
        [
            ["label", "translated_text", "status"],
            ["0", "", "skipped_url_like"],
        ],
    )

    with pytest.raises(ValueError, match="message"):
        extract_sample_columns(
            input_path,
            tmp_path / "sample.csv",
            autopopulate_url_like=True,
        )



def test_extract_sample_columns_requires_status_when_dropping_skipped_rows(tmp_path: Path) -> None:
    input_path = tmp_path / "translated.csv"
    write_csv(
        input_path,
        [
            ["label", "translated_text"],
            ["0", "hello"],
        ],
    )

    with pytest.raises(ValueError, match="status"):
        extract_sample_columns(
            input_path,
            tmp_path / "sample.csv",
            drop_skipped_max_translate_tokens=True,
        )