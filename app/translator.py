from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from difflib import SequenceMatcher

import torch
from transformers import MarianMTModel, MarianTokenizer

from app.chunk_merger import ChunkMerger

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TranslationBatchResult:
    translations: list[str]
    elapsed_seconds: float
    batch_size: int


@dataclass(slots=True)
class LengthCheckResult:
    is_too_long: bool
    token_count: int
    max_input_length: int


@dataclass(slots=True)
class ChunkedTranslationResult:
    translated_text: str
    chunk_count: int
    elapsed_seconds: float


@dataclass(slots=True)
class TextChunk:
    text: str
    start_token: int
    end_token: int


class OpusTranslator:
    """
    Translator wrapper for OPUS-MT / MarianMT models from Hugging Face.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        max_input_length: int = 512,
        max_new_tokens: int = 512,
        num_beams: int = 1,
    ) -> None:
        self.model_name = model_name
        self.requested_device = device
        self.max_input_length = max_input_length
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams

        self.device = self._resolve_device(device)
        logger.info(
            (
                "Initializing translator with model=%s, requested_device=%s, "
                "resolved_device=%s, max_input_length=%d, max_new_tokens=%d, num_beams=%d"
            ),
            self.model_name,
            self.requested_device,
            self.device,
            self.max_input_length,
            self.max_new_tokens,
            self.num_beams,
        )

        self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)
        self.model = MarianMTModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        self.chunk_merger = ChunkMerger()

        logger.info("Translator initialized successfully")

    def _resolve_device(self, device: str) -> str:
        if device == "cuda":
            if torch.cuda.is_available():
                return "cuda"

            logger.warning("CUDA requested but not available. Falling back to CPU.")
            return "cpu"

        return "cpu"

    def get_token_count(self, text: str) -> int:
        """
        Count tokens for a single text without truncation.
        """
        if not text:
            return 0

        encoded = self.tokenizer(
            text,
            return_attention_mask=False,
            return_tensors=None,
            truncation=False,
        )
        return len(encoded["input_ids"])

    def check_input_length(self, text: str) -> LengthCheckResult:
        """
        Check whether a text exceeds model input length limit.
        """
        token_count = self.get_token_count(text)
        return LengthCheckResult(
            is_too_long=(token_count > self.max_input_length),
            token_count=token_count,
            max_input_length=self.max_input_length,
        )

    def check_input_lengths(self, texts: list[str]) -> list[LengthCheckResult]:
        """
        Check multiple texts and return per-text length results.
        """
        return [self.check_input_length(text) for text in texts]

    def translate_batch(self, texts: list[str]) -> list[str]:
        if not texts:
            logger.debug("Received empty batch for translation")
            return []

        result = self.translate_batch_with_metadata(texts)
        return result.translations

    def translate_batch_with_metadata(self, texts: list[str]) -> TranslationBatchResult:
        if not texts:
            logger.debug("Received empty batch for translation")
            return TranslationBatchResult(
                translations=[],
                elapsed_seconds=0.0,
                batch_size=0,
            )

        logger.debug("Translating batch of %d records", len(texts))

        start_time = time.perf_counter()

        try:
            encoded = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_input_length,
            )

            encoded = {key: value.to(self.device) for key, value in encoded.items()}

            with torch.no_grad():
                generated_tokens = self.model.generate(
                    **encoded,
                    max_new_tokens=self.max_new_tokens,
                    num_beams=self.num_beams,
                    early_stopping=True,
                )

            translations = self.tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=True,
            )

        except Exception:
            logger.exception("Batch translation failed")
            raise

        elapsed = time.perf_counter() - start_time

        logger.debug(
            "Batch translated successfully: batch_size=%d elapsed=%.2fs",
            len(texts),
            elapsed,
        )

        return TranslationBatchResult(
            translations=translations,
            elapsed_seconds=elapsed,
            batch_size=len(texts),
        )

    def translate_in_chunks(
        self,
        texts: list[str],
        batch_size: int,
    ) -> list[str]:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")

        all_translations: list[str] = []
        total = len(texts)

        logger.info(
            "Starting chunked translation: total_records=%d batch_size=%d",
            total,
            batch_size,
        )

        for start_idx in range(0, total, batch_size):
            end_idx = min(start_idx + batch_size, total)
            batch = texts[start_idx:end_idx]
            batch_number = (start_idx // batch_size) + 1
            total_batches = (total + batch_size - 1) // batch_size

            logger.info(
                "Processing batch %d/%d (%d records)",
                batch_number,
                total_batches,
                len(batch),
            )

            result = self.translate_batch_with_metadata(batch)

            logger.info(
                "Completed batch %d/%d in %.2fs",
                batch_number,
                total_batches,
                result.elapsed_seconds,
            )

            all_translations.extend(result.translations)

        logger.info("Chunked translation finished successfully")
        return all_translations

    def _hard_split_by_tokens(self, text: str, max_tokens: int) -> list[str]:
        """
        Fallback splitter used when a single segment is too large to fit even on its own.
        Splits strictly by token count.
        """
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than 0")

        encoded = self.tokenizer(
            text,
            return_attention_mask=False,
            return_tensors=None,
            truncation=False,
            add_special_tokens=False,
        )

        input_ids = encoded["input_ids"]
        chunks: list[str] = []

        for start in range(0, len(input_ids), max_tokens):
            chunk_ids = input_ids[start:start + max_tokens]
            chunk_text = self.tokenizer.decode(
                chunk_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()

            if chunk_text:
                chunks.append(chunk_text)

        return chunks

    def split_text_into_token_aware_chunks(
        self,
        text: str,
        max_tokens: int,
    ) -> list[str]:
        """
        Split long text into chunks that fit within max_tokens.

        Strategy:
        1. Build chunks from whitespace-separated segments.
        2. Keep adding segments while token count stays within the limit.
        3. If a single segment is too long, fall back to hard token split.

        This is intentionally conservative and suitable for noisy one-line texts.
        """
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than 0")

        stripped_text = text.strip()
        if not stripped_text:
            return [""]

        if self.get_token_count(stripped_text) <= max_tokens:
            return [stripped_text]

        segments = stripped_text.split()
        chunks: list[str] = []
        current_parts: list[str] = []

        for segment in segments:
            if not current_parts:
                if self.get_token_count(segment) <= max_tokens:
                    current_parts.append(segment)
                else:
                    chunks.extend(self._hard_split_by_tokens(segment, max_tokens))
                continue

            candidate_parts = current_parts + [segment]
            candidate_text = " ".join(candidate_parts)

            if self.get_token_count(candidate_text) <= max_tokens:
                current_parts.append(segment)
                continue

            chunks.append(" ".join(current_parts))

            if self.get_token_count(segment) <= max_tokens:
                current_parts = [segment]
            else:
                chunks.extend(self._hard_split_by_tokens(segment, max_tokens))
                current_parts = []

        if current_parts:
            chunks.append(" ".join(current_parts))

        return chunks
    
    def split_text_into_overlapping_token_chunks(
        self,
        text: str,
        chunk_token_limit: int,
        chunk_overlap_tokens: int,
    ) -> list[TextChunk]:
        if chunk_token_limit <= 0:
            raise ValueError("chunk_token_limit must be greater than 0")

        if chunk_overlap_tokens < 0:
            raise ValueError("chunk_overlap_tokens must be non-negative")

        if chunk_overlap_tokens >= chunk_token_limit:
            raise ValueError("chunk_overlap_tokens must be smaller than chunk_token_limit")

        stripped_text = text.strip()
        if not stripped_text:
            return [TextChunk(text="", start_token=0, end_token=0)]

        encoded = self.tokenizer(
            stripped_text,
            return_attention_mask=False,
            return_tensors=None,
            truncation=False,
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"]

        if len(input_ids) <= chunk_token_limit:
            return [
                TextChunk(
                    text=stripped_text,
                    start_token=0,
                    end_token=len(input_ids),
                )
            ]

        chunks: list[TextChunk] = []
        step = chunk_token_limit - chunk_overlap_tokens

        for start in range(0, len(input_ids), step):
            end = min(start + chunk_token_limit, len(input_ids))
            chunk_ids = input_ids[start:end]

            chunk_text = self.tokenizer.decode(
                chunk_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()

            if chunk_text:
                chunks.append(
                    TextChunk(
                        text=chunk_text,
                        start_token=start,
                        end_token=end,
                    )
                )

            if end >= len(input_ids):
                break

        return chunks
    
    def _merge_translated_chunks(self, translated_chunks: list[str]) -> str:
        if not translated_chunks:
            return ""

        merged = translated_chunks[0].strip()

        for next_chunk in translated_chunks[1:]:
            next_chunk = next_chunk.strip()

            if not next_chunk:
                continue

            merged_words = merged.split()
            next_words = next_chunk.split()

            max_overlap = min(len(merged_words), len(next_words), 30)
            overlap_size = 0

            for size in range(max_overlap, 0, -1):
                if merged_words[-size:] == next_words[:size]:
                    overlap_size = size
                    break

            if overlap_size > 0:
                merged = " ".join(merged_words + next_words[overlap_size:])
            else:
                merged = f"{merged} {next_chunk}".strip()

        return merged.strip()

    def translate_long_text(
        self,
        text: str,
        chunk_token_limit: int | None = None,
        chunk_overlap_tokens: int = 0,
    ) -> ChunkedTranslationResult:
        stripped_text = text.strip()
        if not stripped_text:
            return ChunkedTranslationResult(
                translated_text="",
                chunk_count=0,
                elapsed_seconds=0.0,
            )

        limit = chunk_token_limit or 160

        if chunk_overlap_tokens > 0:
            chunks = self.split_text_into_overlapping_token_chunks(
                stripped_text,
                chunk_token_limit=limit,
                chunk_overlap_tokens=chunk_overlap_tokens,
            )
            chunk_texts = [chunk.text for chunk in chunks]
        else:
            chunk_texts = self.split_text_into_token_aware_chunks(
                stripped_text,
                max_tokens=limit,
            )

        logger.info(
            "Long text split into %d chunks (chunk_token_limit=%d, chunk_overlap_tokens=%d)",
            len(chunk_texts),
            limit,
            chunk_overlap_tokens,
        )

        start_time = time.perf_counter()
        translated_chunks: list[str] = []

        for chunk_idx, chunk in enumerate(chunk_texts, start=1):
            logger.debug(
                "Translating long-text chunk %d/%d",
                chunk_idx,
                len(chunk_texts),
            )
            translated = self.translate_batch([chunk])[0]
            translated_chunks.append(translated.strip())

        elapsed = time.perf_counter() - start_time

        if chunk_overlap_tokens > 0:
            translated_text = self.chunk_merger.merge_translated_chunks(translated_chunks)
        else:
            translated_text = " ".join(
                chunk for chunk in translated_chunks if chunk
            ).strip()

        return ChunkedTranslationResult(
            translated_text=translated_text,
            chunk_count=len(chunk_texts),
            elapsed_seconds=elapsed,
        )