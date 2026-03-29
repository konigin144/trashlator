import csv
from pathlib import Path

import pytest

from utils.sample_cli import create_sample, default_output_path


def write_csv(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def read_csv(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.reader(handle))


def test_create_sample_keeps_header_and_first_n_rows(tmp_path: Path) -> None:
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "sample.csv"
    write_csv(
        input_path,
        [
            ["id", "text"],
            ["1", "alpha"],
            ["2", "beta"],
            ["3", "gamma"],
        ],
    )

    copied_rows = create_sample(input_path, output_path, rows=2)

    assert copied_rows == 2
    assert read_csv(output_path) == [
        ["id", "text"],
        ["1", "alpha"],
        ["2", "beta"],
    ]


def test_default_output_path_uses_row_count_suffix() -> None:
    input_path = Path("data/input/anonymized_dataset.csv")

    result = default_output_path(input_path, 25)

    assert result == Path("data/input/anonymized_dataset_first_25.csv")


def test_create_sample_requires_positive_row_count(tmp_path: Path) -> None:
    input_path = tmp_path / "input.csv"
    write_csv(input_path, [["id"], ["1"]])

    with pytest.raises(ValueError, match="positive integer"):
        create_sample(input_path, tmp_path / "sample.csv", rows=0)
