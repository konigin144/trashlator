from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import torch
from transformers import MarianMTModel, MarianTokenizer

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