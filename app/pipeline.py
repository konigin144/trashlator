from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from app.config import AppConfig
from app.masking import mask_emojis, unmask_emojis
from app.preprocess import is_url_like_text
from app.translator import OpusTranslator
from app.validate import validate_translation, summarize_validation

logger = logging.getLogger(__name__)


def _chunk_indices(indices: list[int], batch_size: int) -> list[list[int]]:
    return [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]


def run_pipeline(config: AppConfig) -> None:
    start_time = time.perf_counter()

    logger.info("Starting translation pipeline")
    logger.info("Input file: %s", config.input_path)
    logger.info("Output file: %s", config.output_path)
    logger.info("Model: %s", config.model_name)
    logger.info("Device: %s", config.device)
    logger.info("Batch size: %s", config.batch_size)
    logger.info("Limit: %s", config.limit)
    logger.info("max_input_length: %s", config.max_input_length)
    logger.info("chunk_token_limit: %s", config.chunk_token_limit)
    logger.info("max_new_tokens: %s", config.max_new_tokens)
    logger.info("chunk_overlap_tokens: %s", config.chunk_overlap_tokens)
    logger.info("num_beams: %s", config.num_beams)
    logger.info("skip_url_like: %s", config.skip_url_like)

    df = pd.read_csv(
        config.input_path,
        encoding=config.input_encoding,
        low_memory=False,
    )
    logger.info("Loaded %s rows from input CSV", len(df))

    expected_columns = {config.text_column, config.label_column}
    missing = expected_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if config.limit is not None:
        df = df.head(config.limit).copy()
        logger.info("Applied limit, processing %s rows", len(df))

    df[config.text_column] = df[config.text_column].astype(str)

    before_drop = len(df)
    df = df[df[config.text_column].str.strip() != ""].copy()
    dropped = before_drop - len(df)

    if dropped:
        logger.warning("Dropped %s empty rows during preprocessing", dropped)

    logger.info("Rows after preprocessing: %s", len(df))

    translator = OpusTranslator(
        model_name=config.model_name,
        device=config.device,
        max_input_length=config.max_input_length,
        max_new_tokens=config.max_new_tokens,
        num_beams=config.num_beams,
    )

    texts = df[config.text_column].tolist()

    emoji_mask_results = [mask_emojis(text) for text in texts]
    masked_texts = [result.masked_text for result in emoji_mask_results]
    emoji_replacements = [result.replacements for result in emoji_mask_results]
    contains_emoji_flags = [result.contains_emoji for result in emoji_mask_results]

    df["flag_contains_emoji"] = contains_emoji_flags

    url_like_flags = [
        is_url_like_text(text) if config.skip_url_like else False
        for text in masked_texts
    ]
    url_like_count = sum(url_like_flags)
    logger.info("Detected %d URL-like texts", url_like_count)

    length_checks = translator.check_input_lengths(masked_texts)
    token_counts = [result.token_count for result in length_checks]
    too_long_flags = [result.is_too_long for result in length_checks]
    too_long_count = sum(too_long_flags)
    logger.info("Detected %d texts exceeding model input limit", too_long_count)

    df["token_count"] = token_counts
    df["too_long_for_model"] = too_long_flags
    df["url_like_skipped"] = url_like_flags

    translated_texts: list[str | None] = [None] * len(texts)
    precomputed_statuses: list[str | None] = [None] * len(texts)
    precomputed_errors: list[str | None] = [None] * len(texts)
    was_chunked: list[bool] = [False] * len(texts)
    chunk_counts: list[int] = [0] * len(texts)

    for i, is_url_like in enumerate(url_like_flags):
        if is_url_like:
            precomputed_statuses[i] = "skipped_url_like"
            precomputed_errors[i] = "Record looks like a URL-like string and was skipped."

    normal_indices: list[int] = []
    long_indices: list[int] = []

    for i in range(len(texts)):
        if precomputed_statuses[i] is not None:
            continue

        if too_long_flags[i]:
            long_indices.append(i)
        else:
            normal_indices.append(i)

    logger.info("Texts eligible for normal translation: %d", len(normal_indices))
    logger.info("Texts eligible for chunked translation: %d", len(long_indices))

    normal_batches = _chunk_indices(normal_indices, config.batch_size)

    for batch_no, batch_indices in enumerate(normal_batches, start=1):
        batch_texts = [masked_texts[i] for i in batch_indices]
        logger.info(
            "Processing normal batch %d/%d (%d records)",
            batch_no,
            len(normal_batches),
            len(batch_indices),
        )

        try:
            batch_result = translator.translate_batch_with_metadata(batch_texts)
            logger.info(
                "Completed normal batch %d/%d in %.2fs",
                batch_no,
                len(normal_batches),
                batch_result.elapsed_seconds,
            )

            for idx, translated_text in zip(batch_indices, batch_result.translations):
                translated_texts[idx] = unmask_emojis(
                    translated_text,
                    emoji_replacements[idx],
                )

        except Exception as exc:
            logger.exception(
                "Translation failed for normal batch %d/%d",
                batch_no,
                len(normal_batches),
            )
            for idx in batch_indices:
                precomputed_statuses[idx] = "translation_error"
                precomputed_errors[idx] = f"Batch translation failed: {exc}"

    for record_no, idx in enumerate(long_indices, start=1):
        logger.info(
            "Processing long record %d/%d (row_idx=%d token_count=%d chunk_token_limit=%d)",
            record_no,
            len(long_indices),
            idx,
            token_counts[idx],
            config.chunk_token_limit,
        )

        try:
            chunked_result = translator.translate_long_text(
                masked_texts[idx],
                chunk_token_limit=config.chunk_token_limit,
                chunk_overlap_tokens=config.chunk_overlap_tokens,
            )

            translated_texts[idx] = unmask_emojis(
                chunked_result.translated_text,
                emoji_replacements[idx],
            )
            precomputed_statuses[idx] = "ok_chunked"
            was_chunked[idx] = True
            chunk_counts[idx] = chunked_result.chunk_count

            logger.info(
                "Completed long record %d/%d in %.2fs using %d chunks",
                record_no,
                len(long_indices),
                chunked_result.elapsed_seconds,
                chunked_result.chunk_count,
            )

        except Exception as exc:
            logger.exception(
                "Chunked translation failed for long record %d/%d (row_idx=%d)",
                record_no,
                len(long_indices),
                idx,
            )
            precomputed_statuses[idx] = "translation_error"
            precomputed_errors[idx] = f"Chunked translation failed: {exc}"

    validation_results = [
        validate_translation(
            source_text=texts[i],
            translated_text=translated_texts[i],
            precomputed_status=precomputed_statuses[i],
            precomputed_error_message=precomputed_errors[i],
        )
        for i in range(len(texts))
    ]

    validation_summary = summarize_validation(validation_results)
    logger.info("Validation summary: %s", validation_summary)

    df["translated_text"] = translated_texts
    df["status"] = [result.status for result in validation_results]
    df["placeholder_ok"] = [result.placeholder_ok for result in validation_results]
    df["validation_error"] = [result.error_message for result in validation_results]
    df["was_chunked"] = was_chunked
    df["chunk_count"] = chunk_counts
    df["model_name"] = config.model_name
    df["source_lang"] = config.source_lang
    df["target_lang"] = config.target_lang

    df.to_csv(
        config.output_path,
        index=False,
        encoding=config.output_encoding,
    )
    logger.info("Saved translated output to %s", config.output_path)

    elapsed = time.perf_counter() - start_time
    logger.info("Pipeline finished in %.2f seconds", elapsed)

    report = {
        "input_path": str(config.input_path),
        "output_path": str(config.output_path),
        "model_name": config.model_name,
        "device": config.device,
        "batch_size": config.batch_size,
        "limit": config.limit,
        "source_lang": config.source_lang,
        "target_lang": config.target_lang,
        "max_input_length": config.max_input_length,
        "chunk_token_limit": config.chunk_token_limit,
        "max_new_tokens": config.max_new_tokens,
        "chunk_overlap_tokens": config.chunk_overlap_tokens,
        "num_beams": config.num_beams,
        "skip_url_like": config.skip_url_like,
        "rows_loaded": before_drop,
        "rows_processed": len(df),
        "rows_dropped_empty": dropped,
        "rows_too_long_for_model": too_long_count,
        "rows_skipped_url_like": url_like_count,
        "rows_chunked": sum(was_chunked),
        "elapsed_seconds": round(elapsed, 2),
        "validation_summary": validation_summary,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
    }

    if config.report_path is not None:
        with open(config.report_path, "w", encoding="utf-8-sig") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info("Saved run report to %s", config.report_path)