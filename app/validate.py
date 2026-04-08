from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass

logger = logging.getLogger(__name__)

PLACEHOLDER_PATTERN = re.compile(r"<[A-Z_]+>")


@dataclass(slots=True)
class ValidationResult:
    status: str
    placeholder_ok: bool
    source_placeholders: list[str]
    target_placeholders: list[str]
    error_message: str | None = None


def extract_placeholders(text: str) -> list[str]:
    if not text:
        return []
    return PLACEHOLDER_PATTERN.findall(text)


def placeholders_match(source_text: str, target_text: str) -> tuple[bool, list[str], list[str]]:
    source_placeholders = extract_placeholders(source_text)
    target_placeholders = extract_placeholders(target_text)

    source_counter = Counter(source_placeholders)
    target_counter = Counter(target_placeholders)

    return source_counter == target_counter, source_placeholders, target_placeholders


def validate_translation(
    source_text: str,
    translated_text: str | None,
    precomputed_status: str | None = None,
    precomputed_error_message: str | None = None,
) -> ValidationResult:
    if source_text is None:
        source_text = ""

    source_placeholders = extract_placeholders(source_text)

    if precomputed_status == "too_long_for_model":
        return ValidationResult(
            status="too_long_for_model",
            placeholder_ok=True,
            source_placeholders=source_placeholders,
            target_placeholders=[],
            error_message=precomputed_error_message or "Input exceeds model token limit.",
        )

    if precomputed_status == "translation_error":
        return ValidationResult(
            status="translation_error",
            placeholder_ok=True,
            source_placeholders=source_placeholders,
            target_placeholders=[],
            error_message=precomputed_error_message or "Translation failed.",
        )

    if precomputed_status == "skipped_url_like":
        return ValidationResult(
            status="skipped_url_like",
            placeholder_ok=True,
            source_placeholders=source_placeholders,
            target_placeholders=[],
            error_message=precomputed_error_message or "Record looks like a URL and was skipped.",
        )

    if precomputed_status == "skipped_max_translate_tokens":
        return ValidationResult(
            status="skipped_max_translate_tokens",
            placeholder_ok=True,
            source_placeholders=source_placeholders,
            target_placeholders=[],
            error_message=precomputed_error_message or "Record exceeds max_translate_tokens limit.",
        )

    if translated_text is None:
        translated_text = ""

    translated_stripped = translated_text.strip()
    if not translated_stripped:
        logger.warning("Empty translation detected")
        return ValidationResult(
            status="empty_translation",
            placeholder_ok=(len(source_placeholders) == 0),
            source_placeholders=source_placeholders,
            target_placeholders=[],
            error_message="Translated text is empty.",
        )

    placeholder_ok, source_placeholders, target_placeholders = placeholders_match(
        source_text, translated_text
    )

    if not placeholder_ok:
        logger.warning(
            "Placeholder mismatch detected: source=%s target=%s",
            source_placeholders,
            target_placeholders,
        )
        return ValidationResult(
            status="placeholder_mismatch",
            placeholder_ok=False,
            source_placeholders=source_placeholders,
            target_placeholders=target_placeholders,
            error_message="Source and target placeholders do not match.",
        )

    final_status = "ok_chunked" if precomputed_status == "ok_chunked" else "ok"

    return ValidationResult(
        status=final_status,
        placeholder_ok=True,
        source_placeholders=source_placeholders,
        target_placeholders=target_placeholders,
        error_message=None,
    )


def summarize_validation(results: list[ValidationResult]) -> dict[str, int]:
    summary: dict[str, int] = {
        "ok": 0,
        "ok_chunked": 0,
        "empty_translation": 0,
        "placeholder_mismatch": 0,
        "too_long_for_model": 0,
        "translation_error": 0,
        "skipped_url_like": 0,
        "skipped_max_translate_tokens": 0,
    }

    for result in results:
        summary[result.status] = summary.get(result.status, 0) + 1

    return summary