from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


PLACEHOLDER_PATTERN = re.compile(r"<[A-Z_]+>")



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Lowercase CSV messages while preserving placeholder tokens."
    )
    parser.add_argument(
        "--input",
        dest="input_path",
        type=Path,
        required=True,
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "--output",
        dest="output_path",
        type=Path,
        help="Path to the lowercased CSV file. Defaults next to the input file.",
    )
    return parser



def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_lowercase{input_path.suffix}")



def lowercase_preserving_placeholders(text: str) -> str:
    if not text:
        return text

    parts: list[str] = []
    last_index = 0
    for match in PLACEHOLDER_PATTERN.finditer(text):
        start, end = match.span()
        parts.append(text[last_index:start].lower())
        parts.append(match.group(0))
        last_index = end
    parts.append(text[last_index:].lower())
    return "".join(parts)



def lowercase_messages(input_path: Path, output_path: Path) -> int:
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        fieldnames = reader.fieldnames or []
        if "message" not in fieldnames:
            raise ValueError("Input CSV is missing required columns: message")

        with output_path.open("w", encoding="utf-8", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=fieldnames)
            writer.writeheader()

            row_count = 0
            for row in reader:
                row["message"] = lowercase_preserving_placeholders(row["message"])
                writer.writerow(row)
                row_count += 1

    return row_count



def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    output_path = args.output_path or default_output_path(args.input_path)
    row_count = lowercase_messages(args.input_path, output_path)
    print(f"Wrote {row_count} rows to {output_path}")


if __name__ == "__main__":
    main()