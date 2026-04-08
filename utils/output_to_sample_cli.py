from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from pathlib import Path

from app.preprocess import is_url_like_text


TRUE_VALUES = {"1", "true", "yes"}
SKIPPED_MAX_TRANSLATE_TOKENS = "skipped_max_translate_tokens"
ANGLE_PLACEHOLDER_PATTERN = r"<[A-Z_]+>"
EMOJI_PLACEHOLDER_PATTERN = r"__EMOJI_\d+__"
PROTECTED_TOKEN_PATTERN = re.compile(f"{ANGLE_PLACEHOLDER_PATTERN}|{EMOJI_PLACEHOLDER_PATTERN}")
BROKEN_EMOJI_PLACEHOLDER_PATTERN = re.compile(
    r"(?ix)"
    r"(?:[_\s]*)"
    r"e[_\s]*m[_\s]*o[_\s]*j[_\s]*i"
    r"(?:[_\s]*\d+)?"
    r"(?:[_\s]*)"
)



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
    parser.add_argument(
        "--autopopulate-url-like",
        dest="autopopulate_url_like",
        action="store_true",
        help="Reuse the original message for URL-like rows that were skipped during translation.",
    )
    parser.add_argument(
        "--drop-skipped-max-translate-tokens",
        dest="drop_skipped_max_translate_tokens",
        action="store_true",
        help="Exclude rows skipped because they exceeded max_translate_tokens.",
    )
    parser.add_argument(
        "--remove-emoji-placeholders",
        "--clean-broken-emoji-placeholders",
        dest="remove_emoji_placeholders",
        action="store_true",
        help="Remove emoji placeholders entirely, including broken variants like __EMOJI_1_.",
    )
    parser.add_argument(
        "--lowercase-messages",
        dest="lowercase_messages",
        action="store_true",
        help="Lowercase output messages while keeping placeholder tokens unchanged.",
    )
    parser.add_argument(
        "--remove-punctuation",
        dest="remove_punctuation",
        action="store_true",
        help="Remove punctuation from output messages while keeping placeholder tokens unchanged.",
    )
    return parser



def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_sample{input_path.suffix}")



def remove_emoji_placeholders(text: str) -> str:
    if not text:
        return text

    cleaned = BROKEN_EMOJI_PLACEHOLDER_PATTERN.sub(" ", text)
    cleaned = re.sub(r"(?<=\s)_{2,}(?=\s|$)", " ", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()



def lowercase_preserving_placeholders(text: str) -> str:
    if not text:
        return text

    parts: list[str] = []
    last_index = 0
    for match in PROTECTED_TOKEN_PATTERN.finditer(text):
        start, end = match.span()
        parts.append(text[last_index:start].lower())
        parts.append(match.group(0))
        last_index = end
    parts.append(text[last_index:].lower())
    return "".join(parts)



def remove_punctuation_preserving_placeholders(text: str) -> str:
    if not text:
        return text

    parts: list[str] = []
    last_index = 0
    for match in PROTECTED_TOKEN_PATTERN.finditer(text):
        start, end = match.span()
        parts.append(_strip_punctuation(text[last_index:start]))
        parts.append(match.group(0))
        last_index = end
    parts.append(_strip_punctuation(text[last_index:]))

    cleaned = "".join(parts)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()



def _strip_punctuation(text: str) -> str:
    chars: list[str] = []
    for char in text:
        if unicodedata.category(char).startswith("P"):
            chars.append(" ")
        else:
            chars.append(char)
    return "".join(chars)



def should_autopopulate_url_like(row: dict[str, str]) -> bool:
    status = (row.get("status") or "").strip()
    skipped_flag = (row.get("url_like_skipped") or "").strip().lower() in TRUE_VALUES
    source_message = row.get("message") or ""
    return status == "skipped_url_like" or skipped_flag or is_url_like_text(source_message)



def should_drop_row(row: dict[str, str], drop_skipped_max_translate_tokens: bool) -> bool:
    if not drop_skipped_max_translate_tokens:
        return False
    return (row.get("status") or "").strip() == SKIPPED_MAX_TRANSLATE_TOKENS



def extract_sample_columns(
    input_path: Path,
    output_path: Path,
    autopopulate_url_like: bool = False,
    drop_skipped_max_translate_tokens: bool = False,
    remove_emoji_placeholders_flag: bool = False,
    lowercase_messages: bool = False,
    remove_punctuation: bool = False,
) -> int:
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        fieldnames = reader.fieldnames or []
        required_columns = {"translated_text", "label"}
        if autopopulate_url_like:
            required_columns.add("message")
        if drop_skipped_max_translate_tokens:
            required_columns.add("status")
        missing_columns = required_columns.difference(fieldnames)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Input CSV is missing required columns: {missing}")

        with output_path.open("w", encoding="utf-8", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=["message", "label"])
            writer.writeheader()

            row_count = 0
            for row in reader:
                if should_drop_row(row, drop_skipped_max_translate_tokens):
                    continue

                message = row["translated_text"]
                if (
                    autopopulate_url_like
                    and not message
                    and should_autopopulate_url_like(row)
                ):
                    message = row["message"]
                if remove_emoji_placeholders_flag:
                    message = remove_emoji_placeholders(message)
                if lowercase_messages:
                    message = lowercase_preserving_placeholders(message)
                if remove_punctuation:
                    message = remove_punctuation_preserving_placeholders(message)

                writer.writerow(
                    {
                        "message": message,
                        "label": row["label"],
                    }
                )
                row_count += 1

    return row_count



def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    output_path = args.output_path or default_output_path(args.input_path)
    row_count = extract_sample_columns(
        args.input_path,
        output_path,
        autopopulate_url_like=args.autopopulate_url_like,
        drop_skipped_max_translate_tokens=args.drop_skipped_max_translate_tokens,
        remove_emoji_placeholders_flag=args.remove_emoji_placeholders,
        lowercase_messages=args.lowercase_messages,
        remove_punctuation=args.remove_punctuation,
    )
    print(f"Wrote {row_count} rows to {output_path}")


if __name__ == "__main__":
    main()