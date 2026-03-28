from __future__ import annotations

from dataclasses import dataclass

import emoji


@dataclass(slots=True)
class EmojiReplacement:
    token: str
    emoji_char: str
    glue_left: bool
    glue_right: bool


@dataclass(slots=True)
class EmojiMaskResult:
    masked_text: str
    replacements: list[EmojiReplacement]
    contains_emoji: bool


def _is_whitespace(char: str) -> bool:
    return char.isspace()


def mask_emojis(text: str) -> EmojiMaskResult:
    """
    Replace emoji characters with stable standalone placeholder tokens.

    Important:
    - placeholders are surrounded with spaces to reduce the risk of being merged
      with neighbouring text by the translation model
    - original left/right adjacency is stored so it can be restored later

    Example:
        "hello🔥world"
        -> "hello __EMOJI_0__ world"
        with glue_left=True, glue_right=True
    """
    if not text:
        return EmojiMaskResult(
            masked_text=text,
            replacements=[],
            contains_emoji=False,
        )

    emoji_matches = emoji.emoji_list(text)
    if not emoji_matches:
        return EmojiMaskResult(
            masked_text=text,
            replacements=[],
            contains_emoji=False,
        )

    parts: list[str] = []
    replacements: list[EmojiReplacement] = []
    last_index = 0

    for idx, match in enumerate(emoji_matches):
        start = match["match_start"]
        end = match["match_end"]
        emoji_char = match["emoji"]

        token = f"__EMOJI_{idx}__"

        left_char = text[start - 1] if start > 0 else ""
        right_char = text[end] if end < len(text) else ""

        glue_left = bool(left_char) and not _is_whitespace(left_char)
        glue_right = bool(right_char) and not _is_whitespace(right_char)

        replacements.append(
            EmojiReplacement(
                token=token,
                emoji_char=emoji_char,
                glue_left=glue_left,
                glue_right=glue_right,
            )
        )

        parts.append(text[last_index:start])

        # Force placeholder to become a standalone token for the model.
        # We intentionally add spaces around it.
        parts.append(f" {token} ")

        last_index = end

    parts.append(text[last_index:])

    masked_text = "".join(parts)

    # Light cleanup of excessive whitespace introduced by adjacent emojis/spaces.
    masked_text = " ".join(masked_text.split())

    return EmojiMaskResult(
        masked_text=masked_text,
        replacements=replacements,
        contains_emoji=True,
    )


def unmask_emojis(text: str, replacements: list[EmojiReplacement]) -> str:
    """
    Restore emoji placeholders and reconstruct original adjacency.

    If an emoji was originally glued to neighbouring text, remove the temporary
    spaces that were introduced around the placeholder.
    """
    if not text or not replacements:
        return text

    restored = text

    for replacement in replacements:
        token = replacement.token
        emoji_char = replacement.emoji_char

        spaced_token = f" {token} "

        if spaced_token in restored:
            left = "" if replacement.glue_left else " "
            right = "" if replacement.glue_right else " "
            restored = restored.replace(spaced_token, f"{left}{emoji_char}{right}")
            continue

        # Fallbacks in case model slightly altered surrounding whitespace
        if f"{token} " in restored:
            left = "" if replacement.glue_left else " "
            right = "" if replacement.glue_right else " "
            restored = restored.replace(f"{token} ", f"{left}{emoji_char}{right}")

        if f" {token}" in restored:
            left = "" if replacement.glue_left else " "
            right = "" if replacement.glue_right else " "
            restored = restored.replace(f" {token}", f"{left}{emoji_char}{right}")

        if token in restored:
            restored = restored.replace(token, emoji_char)

    return restored