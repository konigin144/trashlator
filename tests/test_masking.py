from app.masking import mask_emojis, unmask_emojis


def test_mask_emojis_returns_same_text_when_no_emoji() -> None:
    text = "hello world"
    result = mask_emojis(text)

    assert result.masked_text == "hello world"
    assert result.replacements == []
    assert result.contains_emoji is False


def test_mask_emojis_replaces_single_emoji() -> None:
    text = "hello 🔥 world"
    result = mask_emojis(text)

    assert result.masked_text == "hello __EMOJI_0__ world"
    assert len(result.replacements) == 1
    assert result.replacements[0].token == "__EMOJI_0__"
    assert result.replacements[0].emoji_char == "🔥"
    assert result.contains_emoji is True


def test_mask_emojis_replaces_multiple_emojis_in_order() -> None:
    text = "click now ✅🔥"
    result = mask_emojis(text)

    assert result.masked_text == "click now __EMOJI_0__ __EMOJI_1__"
    assert len(result.replacements) == 2
    assert result.replacements[0].emoji_char == "✅"
    assert result.replacements[1].emoji_char == "🔥"
    assert result.contains_emoji is True


def test_unmask_emojis_restores_original_text() -> None:
    original = "verify now ✅🔥"
    masked = mask_emojis(original)

    restored = unmask_emojis(masked.masked_text, masked.replacements)

    assert restored == original


def test_unmask_emojis_restores_glued_emoji_inside_text() -> None:
    original = "promo💠mod info"
    masked = mask_emojis(original)

    assert masked.masked_text == "promo __EMOJI_0__ mod info"

    restored = unmask_emojis(masked.masked_text, masked.replacements)

    assert restored == original


def test_unmask_emojis_returns_input_when_no_replacements() -> None:
    text = "plain text"
    restored = unmask_emojis(text, [])

    assert restored == text