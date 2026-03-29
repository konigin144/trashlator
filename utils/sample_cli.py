from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_INPUT_PATH = Path("data/input/anonymized_dataset.csv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a CSV sample containing the header and the first N data rows."
    )
    parser.add_argument(
        "rows",
        type=int,
        help="Number of data rows to copy to the sample file.",
    )
    parser.add_argument(
        "--input",
        dest="input_path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to the source CSV file.",
    )
    parser.add_argument(
        "--output",
        dest="output_path",
        type=Path,
        help="Path to the sample CSV file. Defaults next to the input file.",
    )
    return parser


def default_output_path(input_path: Path, rows: int) -> Path:
    return input_path.with_name(f"{input_path.stem}_first_{rows}{input_path.suffix}")


def create_sample(input_path: Path, output_path: Path, rows: int) -> int:
    if rows <= 0:
        raise ValueError("rows must be a positive integer")
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    copied_rows = 0
    with input_path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.reader(source)

        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"Input CSV is empty: {input_path}") from exc

        with output_path.open("w", encoding="utf-8", newline="") as destination:
            writer = csv.writer(destination)
            writer.writerow(header)

            for copied_rows, row in enumerate(reader, start=1):
                if copied_rows > rows:
                    copied_rows -= 1
                    break
                writer.writerow(row)
            else:
                copied_rows = min(copied_rows, rows)

    return copied_rows


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    output_path = args.output_path or default_output_path(args.input_path, args.rows)
    copied_rows = create_sample(args.input_path, output_path, args.rows)
    print(f"Wrote {copied_rows} rows to {output_path}")


if __name__ == "__main__":
    main()
