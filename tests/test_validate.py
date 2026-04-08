from app.validate import (
    extract_placeholders,
    placeholders_match,
    summarize_validation,
    validate_translation,
)


def test_extract_placeholders_returns_all_placeholders_in_order() -> None:
    text = "hello <PERSON> your email is <EMAIL_ADDRESS> and phone <PHONE_NUMBER>"
    result = extract_placeholders(text)

    assert result == ["<PERSON>", "<EMAIL_ADDRESS>", "<PHONE_NUMBER>"]


def test_extract_placeholders_returns_empty_list_for_plain_text() -> None:
    text = "hello this is a normal sentence"
    result = extract_placeholders(text)

    assert result == []


def test_placeholders_match_returns_true_for_identical_placeholder_sets() -> None:
    source = "hello <PERSON> contact <EMAIL_ADDRESS>"
    target = "cześć <PERSON> kontakt <EMAIL_ADDRESS>"

    is_match, source_placeholders, target_placeholders = placeholders_match(source, target)

    assert is_match is True
    assert source_placeholders == ["<PERSON>", "<EMAIL_ADDRESS>"]
    assert target_placeholders == ["<PERSON>", "<EMAIL_ADDRESS>"]


def test_placeholders_match_returns_true_when_order_differs_but_counts_match() -> None:
    source = "hello <PERSON> and <EMAIL_ADDRESS>"
    target = "kontakt <EMAIL_ADDRESS> do <PERSON>"

    is_match, _, _ = placeholders_match(source, target)

    assert is_match is True


def test_placeholders_match_returns_false_when_placeholder_missing() -> None:
    source = "hello <PERSON> contact <EMAIL_ADDRESS>"
    target = "cześć <PERSON> kontakt"

    is_match, source_placeholders, target_placeholders = placeholders_match(source, target)

    assert is_match is False
    assert source_placeholders == ["<PERSON>", "<EMAIL_ADDRESS>"]
    assert target_placeholders == ["<PERSON>"]


def test_validate_translation_returns_ok_for_valid_translation() -> None:
    source = "hello <PERSON> verify account <EMAIL_ADDRESS>"
    target = "cześć <PERSON> zweryfikuj konto <EMAIL_ADDRESS>"

    result = validate_translation(source, target)

    assert result.status == "ok"
    assert result.placeholder_ok is True
    assert result.error_message is None


def test_validate_translation_returns_empty_translation_status() -> None:
    source = "hello <PERSON>"
    target = "   "

    result = validate_translation(source, target)

    assert result.status == "empty_translation"
    assert result.placeholder_ok is False
    assert result.source_placeholders == ["<PERSON>"]
    assert result.target_placeholders == []
    assert result.error_message == "Translated text is empty."


def test_validate_translation_returns_placeholder_mismatch() -> None:
    source = "hello <PERSON> verify <EMAIL_ADDRESS>"
    target = "cześć <PERSON> zweryfikuj konto"

    result = validate_translation(source, target)

    assert result.status == "placeholder_mismatch"
    assert result.placeholder_ok is False
    assert result.source_placeholders == ["<PERSON>", "<EMAIL_ADDRESS>"]
    assert result.target_placeholders == ["<PERSON>"]


def test_validate_translation_respects_precomputed_too_long_status() -> None:
    source = "very long text <PERSON>"

    result = validate_translation(
        source_text=source,
        translated_text=None,
        precomputed_status="too_long_for_model",
        precomputed_error_message="Input has 999 tokens, exceeding limit 512.",
    )

    assert result.status == "too_long_for_model"
    assert result.placeholder_ok is True
    assert result.source_placeholders == ["<PERSON>"]
    assert result.target_placeholders == []
    assert result.error_message == "Input has 999 tokens, exceeding limit 512."


def test_validate_translation_respects_precomputed_translation_error_status() -> None:
    source = "hello <PERSON>"

    result = validate_translation(
        source_text=source,
        translated_text=None,
        precomputed_status="translation_error",
        precomputed_error_message="Batch translation failed.",
    )

    assert result.status == "translation_error"
    assert result.placeholder_ok is True
    assert result.source_placeholders == ["<PERSON>"]
    assert result.target_placeholders == []
    assert result.error_message == "Batch translation failed."


def test_validate_translation_respects_precomputed_skipped_url_like_status() -> None:
    source = "httpswwwexamplecomloginverifyhtml"

    result = validate_translation(
        source_text=source,
        translated_text=None,
        precomputed_status="skipped_url_like",
        precomputed_error_message="Record looks like a URL-like string and was skipped.",
    )

    assert result.status == "skipped_url_like"
    assert result.placeholder_ok is True
    assert result.source_placeholders == []
    assert result.target_placeholders == []
    assert result.error_message == "Record looks like a URL-like string and was skipped."


def test_validate_translation_respects_precomputed_skipped_max_translate_tokens_status() -> None:
    source = "very long text"

    result = validate_translation(
        source_text=source,
        translated_text=None,
        precomputed_status="skipped_max_translate_tokens",
        precomputed_error_message="Record exceeds max_translate_tokens limit.",
    )

    assert result.status == "skipped_max_translate_tokens"
    assert result.placeholder_ok is True
    assert result.source_placeholders == []
    assert result.target_placeholders == []
    assert result.error_message == "Record exceeds max_translate_tokens limit."


def test_validate_translation_respects_precomputed_ok_chunked_status() -> None:
    source = "hello <PERSON> verify account"
    target = "hallo <PERSON> konto prüfen"

    result = validate_translation(
        source_text=source,
        translated_text=target,
        precomputed_status="ok_chunked",
        precomputed_error_message=None,
    )

    assert result.status == "ok_chunked"
    assert result.placeholder_ok is True
    assert result.source_placeholders == ["<PERSON>"]
    assert result.target_placeholders == ["<PERSON>"]
    assert result.error_message is None


def test_summarize_validation_counts_statuses_correctly() -> None:
    results = [
        validate_translation("hello <PERSON>", "cześć <PERSON>"),
        validate_translation("hello <PERSON>", ""),
        validate_translation(
            source_text="long text",
            translated_text=None,
            precomputed_status="too_long_for_model",
        ),
        validate_translation(
            source_text="httpswwwexamplecom",
            translated_text=None,
            precomputed_status="skipped_url_like",
        ),
        validate_translation(
            source_text="hello",
            translated_text=None,
            precomputed_status="translation_error",
        ),
    ]

    summary = summarize_validation(results)

    assert summary["ok"] == 1
    assert summary["empty_translation"] == 1
    assert summary["too_long_for_model"] == 1
    assert summary["skipped_url_like"] == 1
    assert summary["translation_error"] == 1
    assert summary["placeholder_mismatch"] == 0
    assert summary["skipped_max_translate_tokens"] == 0


def test_summarize_validation_counts_ok_chunked_status() -> None:
    results = [
        validate_translation(
            source_text="hello <PERSON>",
            translated_text="hallo <PERSON>",
            precomputed_status="ok_chunked",
        ),
    ]

    summary = summarize_validation(results)

    assert summary["ok_chunked"] == 1
    assert summary["ok"] == 0