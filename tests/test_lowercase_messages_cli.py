import csv
from pathlib import Path

import pytest

from utils.lowercase_messages_cli import (
    default_output_path,
    lowercase_messages,
    lowercase_preserving_placeholders,
)



def write_csv(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)



def read_csv(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.reader(handle))



def test_lowercase_preserving_placeholders_keeps_placeholder_tokens() -> None:
    text = "HELLO <PERSON> EMAIL <EMAIL_ADDRESS> CALL <PHONE_NUMBER> TODAY"

    result = lowercase_preserving_placeholders(text)

    assert result == "hello <PERSON> email <EMAIL_ADDRESS> call <PHONE_NUMBER> today"



def test_lowercase_messages_updates_only_message_column(tmp_path: Path) -> None:
    input_path = tmp_path / "sample.csv"
    output_path = tmp_path / "sample_lowercase.csv"
    write_csv(
        input_path,
        [
            ["message", "label"],
            ["HELLO <PERSON>", "1"],
            ["MAIL <EMAIL_ADDRESS> NOW", "0"],
        ],
    )

    row_count = lowercase_messages(input_path, output_path)

    assert row_count == 2
    assert read_csv(output_path) == [
        ["message", "label"],
        ["hello <PERSON>", "1"],
        ["mail <EMAIL_ADDRESS> now", "0"],
    ]



def test_default_output_path_adds_lowercase_suffix() -> None:
    input_path = Path("data/samples/sample_100.csv")

    result = default_output_path(input_path)

    assert result == Path("data/samples/sample_100_lowercase.csv")



def test_lowercase_messages_requires_message_column(tmp_path: Path) -> None:
    input_path = tmp_path / "sample.csv"
    write_csv(
        input_path,
        [
            ["text", "label"],
            ["HELLO", "1"],
        ],
    )

    with pytest.raises(ValueError, match="message"):
        lowercase_messages(input_path, tmp_path / "out.csv")