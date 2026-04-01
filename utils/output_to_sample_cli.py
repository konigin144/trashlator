from __future__ import annotations

import argparse
import csv
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a translation output CSV into sample CSV format: message,label."
    )
    parser.add_argument(
        "--input",
        dest="input_path",
        type=Path,
        required=True,
        help="Path to the translation output CSV file.",
    )
    parser.add_argument(
        "--output",
        dest="output_path",
        type=Path,
        help="Path to the sample-format CSV file. Defaults next to the input file.",
    )
    return parser


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_sample{input_path.suffix}")


def extract_sample_columns(input_path: Path, output_path: Path) -> int:
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        fieldnames = reader.fieldnames or []
        required_columns = {"translated_text", "label"}
        missing_columns = required_columns.difference(fieldnames)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Input CSV is missing required columns: {missing}")

        with output_path.open("w", encoding="utf-8", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=["message", "label"])
            writer.writeheader()

            row_count = 0
            for row in reader:
                writer.writerow(
                    {
                        "message": row["translated_text"],
                        "label": row["label"],
                    }
                )
                row_count += 1

    return row_count


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    output_path = args.output_path or default_output_path(args.input_path)
    row_count = extract_sample_columns(args.input_path, output_path)
    print(f"Wrote {row_count} rows to {output_path}")


if __name__ == "__main__":
    main()