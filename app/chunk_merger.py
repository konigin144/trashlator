from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re


@dataclass(frozen=True)
class OverlapMatch:
    size: int
    full_ratio: float
    inner_ratio: float
    score: float


@dataclass(frozen=True)
class PrefixAnchorMatch:
    start_idx_in_tail: int
    match_size: int
    score: float


class ChunkMerger:
    def _similarity(self, a: list[str], b: list[str]) -> float:
        if not a or not b:
            return 0.0
        return SequenceMatcher(None, a, b).ratio()

    def _trim_both_sides(self, words: list[str], trim: int) -> list[str]:
        if trim <= 0:
            return words
        if len(words) <= trim * 2:
            return []
        return words[trim:-trim]

    def _best_inner_similarity(
        self,
        left: list[str],
        right: list[str],
        max_edge_trim: int,
    ) -> float:
        best = 0.0

        # Try trimming 0..N tokens from both sides to ignore unstable boundaries
        for trim in range(max_edge_trim + 1):
            left_inner = self._trim_both_sides(left, trim)
            right_inner = self._trim_both_sides(right, trim)

            if not left_inner or not right_inner:
                continue

            ratio = self._similarity(left_inner, right_inner)
            if ratio > best:
                best = ratio

        return best

    def _find_best_overlap(
        self,
        merged_words: list[str],
        next_words: list[str],
        *,
        max_overlap: int = 30,
        min_overlap: int = 5,
        min_full_ratio: float = 0.72,
        min_inner_ratio: float = 0.85,
        max_edge_trim: int = 2,
        prefer_larger_overlap_bonus: float = 0.01,
    ) -> OverlapMatch | None:
        """
        Try standard fuzzy overlap detection:
        compare suffix of merged with prefix of next.

        If a good overlap is found, remove overlap from next.
        """
        max_size = min(len(merged_words), len(next_words), max_overlap)
        best_match: OverlapMatch | None = None

        for size in range(min_overlap, max_size + 1):
            left = merged_words[-size:]
            right = next_words[:size]

            full_ratio = self._similarity(left, right)
            inner_ratio = self._best_inner_similarity(
                left,
                right,
                max_edge_trim=max_edge_trim,
            )

            is_acceptable = (
                full_ratio >= min_full_ratio
                or inner_ratio >= min_inner_ratio
            )
            if not is_acceptable:
                continue

            # Prefer candidates with:
            # - better inner similarity
            # - better full similarity
            # - slightly larger overlap
            score = (
                (full_ratio * 0.45)
                + (inner_ratio * 0.55)
                + (size * prefer_larger_overlap_bonus)
            )

            candidate = OverlapMatch(
                size=size,
                full_ratio=full_ratio,
                inner_ratio=inner_ratio,
                score=score,
            )

            if best_match is None or candidate.score > best_match.score:
                best_match = candidate

        return best_match

    def _normalize_token(self, token: str) -> str:
        token = token.lower().strip()

        # Preserve placeholders as-is
        if token.startswith("<") and token.endswith(">"):
            return token

        # Remove punctuation around tokens
        token = re.sub(r"^[^\w<]+|[^\w>]+$", "", token, flags=re.UNICODE)
        return token

    def _find_prefix_seed_anchor(
        self,
        merged_words: list[str],
        next_words: list[str],
        *,
        tail_window: int = 80,
        min_seed_tokens: int = 2,
        max_seed_tokens: int = 3,
        prefer_later_match_bonus: float = 0.02,
    ) -> PrefixAnchorMatch | None:
        """
        Find a short exact seed from the beginning of next_words
        somewhere inside the tail of merged_words.

        This is intentionally simple and robust for cases where:
        - the beginning of next is still recognizable
        - the rest of the overlap was rewritten by MT
        """
        merged_tail = (
            merged_words[-tail_window:]
            if len(merged_words) > tail_window
            else merged_words
        )

        if not merged_tail or not next_words:
            return None

        normalized_tail = [self._normalize_token(t) for t in merged_tail]
        normalized_next = [self._normalize_token(t) for t in next_words]

        best: PrefixAnchorMatch | None = None
        max_seed = min(len(next_words), max_seed_tokens)

        # Try longer seeds first, then shorter ones
        for seed_size in range(max_seed, min_seed_tokens - 1, -1):
            seed = normalized_next[:seed_size]

            for start_idx in range(len(normalized_tail) - seed_size + 1):
                candidate = normalized_tail[start_idx:start_idx + seed_size]

                if candidate != seed:
                    continue

                # Prefer longer seeds and matches closer to the end
                score = seed_size + (start_idx * prefer_later_match_bonus)

                match = PrefixAnchorMatch(
                    start_idx_in_tail=start_idx,
                    match_size=seed_size,
                    score=score,
                )

                if best is None or match.score > best.score:
                    best = match

            if best is not None:
                return best

        return None

    def _replace_tail_from_anchor(
        self,
        merged_words: list[str],
        next_words: list[str],
        anchor: PrefixAnchorMatch,
        *,
        tail_window: int = 80,
    ) -> str:
        """
        Replace the tail of merged starting from the anchor position
        with the entire next chunk.

        merged = merged_prefix_before_anchor + next
        """
        tail_len = min(len(merged_words), tail_window)
        tail_start_abs = len(merged_words) - tail_len
        anchor_abs_idx = tail_start_abs + anchor.start_idx_in_tail

        new_words = merged_words[:anchor_abs_idx] + next_words
        return " ".join(new_words).strip()

    def merge_translated_chunks(
        self,
        translated_chunks: list[str],
        *,
        max_overlap: int = 30,
        min_overlap: int = 5,
        min_full_ratio: float = 0.72,
        min_inner_ratio: float = 0.85,
        max_edge_trim: int = 2,
        tail_window: int = 80,
        min_seed_tokens: int = 2,
        max_seed_tokens: int = 3,
    ) -> str:
        if not translated_chunks:
            return ""

        merged = translated_chunks[0].strip()

        for next_chunk in translated_chunks[1:]:
            next_chunk = next_chunk.strip()
            if not next_chunk:
                continue

            merged_words = merged.split()
            next_words = next_chunk.split()

            # 1. Try standard fuzzy boundary overlap first
            overlap_match = self._find_best_overlap(
                merged_words,
                next_words,
                max_overlap=max_overlap,
                min_overlap=min_overlap,
                min_full_ratio=min_full_ratio,
                min_inner_ratio=min_inner_ratio,
                max_edge_trim=max_edge_trim,
            )

            if overlap_match is not None:
                merged = " ".join(
                    merged_words + next_words[overlap_match.size:]
                ).strip()
                continue

            # 2. Fallback: find a short prefix seed from next in merged tail
            #    and replace merged tail from that point with the full next chunk
            prefix_anchor = self._find_prefix_seed_anchor(
                merged_words,
                next_words,
                tail_window=tail_window,
                min_seed_tokens=min_seed_tokens,
                max_seed_tokens=max_seed_tokens,
            )

            if prefix_anchor is not None:
                merged = self._replace_tail_from_anchor(
                    merged_words,
                    next_words,
                    anchor=prefix_anchor,
                    tail_window=tail_window,
                )
            else:
                # 3. If nothing matches, concatenate as a fallback
                merged = f"{merged} {next_chunk}".strip()

        return merged.strip()