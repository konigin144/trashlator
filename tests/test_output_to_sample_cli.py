import csv
from pathlib import Path

import pytest

from utils.output_to_sample_cli import default_output_path, extract_sample_columns


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