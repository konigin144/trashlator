from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from app.config import AppConfig
from app.masking import mask_emojis, unmask_emojis
from app.preprocess import is_url_like_text
from app.qe import build_qe_columns
from app.translator import OpusTranslator
from app.validate import validate_translation

from qe.service import QEService

logger = logging.getLogger(__name__)

QE_COLUMN_DEFAULTS = {
    "qe_score": None,
    "qe_label": None,
    "qe_error": None,
    "qe_backend": None,
    "qe_model_name": None,
}

STATUS_SUMMARY_KEYS = (
    "ok",
    "ok_chunked",
    "empty_translation",
    "placeholder_mismatch",
    "too_long_for_model",
    "translation_error",
    "skipped_url_like",
    "skipped_max_translate_tokens",
)


@dataclass(slots=True)
class ChunkMetrics:
    rows_loaded: int
    rows_processed: int
    rows_dropped_empty: int
    rows_too_long_for_model: int
    rows_skipped_url_like: int
    rows_skipped_max_translate_tokens: int
    rows_chunked: int
    validation_summary: dict[str, int]


def _chunk_indices(indices: list[int], batch_size: int) -> list[list[int]]:
    return [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]


def _build_dynamic_batches(
    indices: list[int],
    token_counts: list[int],
    *,
    max_batch_size: int,
    batch_token_budget: int,
) -> list[list[int]]:
    if max_batch_size <= 0:
        raise ValueError("max_batch_size must be greater than 0")

    if batch_token_budget <= 0:
        raise ValueError("batch_token_budget must be greater than 0")

    batches: list[list[int]] = []
    current_batch: list[int] = []
    current_tokens = 0

    for idx in indices:
        row_tokens = token_counts[idx]
        exceeds_row_limit = len(current_batch) >= max_batch_size
        exceeds_token_budget = (
            bool(current_batch)
            and current_tokens + row_tokens > batch_token_budget
        )

        if exceeds_row_limit or exceeds_token_budget:
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0

        current_batch.append(idx)
        current_tokens += row_tokens

    if current_batch:
        batches.append(current_batch)

    return batches


def _sort_indices_by_token_count(
    indices: list[int],
    token_counts: list[int],
) -> list[int]:
    return sorted(indices, key=token_counts.__getitem__)


def _empty_validation_summary() -> dict[str, int]:
    return {key: 0 for key in STATUS_SUMMARY_KEYS}


def _merge_validation_summaries(
    left: dict[str, int],
    right: dict[str, int],
) -> dict[str, int]:
    merged = dict(left)
    for status, count in right.items():
        merged[status] = merged.get(status, 0) + count
    return merged


def _translate_single_text_with_limit(
    *,
    translator: OpusTranslator,
    text: str,
    max_input_length: int,
    chunk_token_limit: int,
    chunk_overlap_tokens: int,
) -> tuple[str, str, int]:
    original_max_input_length = translator.max_input_length
    translator.max_input_length = max_input_length

    try:
        length_check = translator.check_input_length(text)
        if length_check.is_too_long:
            chunked_result = translator.translate_long_text(
                text,
                chunk_token_limit=chunk_token_limit,
                chunk_overlap_tokens=chunk_overlap_tokens,
            )
            return (
                chunked_result.translated_text,
                "ok_chunked",
                chunked_result.chunk_count,
            )

        translated_text = translator.translate_batch([text])[0]
        return translated_text, "ok", 0
    finally:
        translator.max_input_length = original_max_input_length


def _initialize_output_columns(df: pd.DataFrame, config: AppConfig) -> None:
    df["translated_text"] = pd.NA
    df["status"] = pd.NA
    df["placeholder_ok"] = pd.NA
    df["validation_error"] = pd.NA
    df["was_chunked"] = False
    df["chunk_count"] = 0
    df["model_name"] = config.model_name
    df["source_lang"] = config.source_lang
    df["target_lang"] = config.target_lang

    for column_name, default_value in QE_COLUMN_DEFAULTS.items():
        df[column_name] = default_value


def _finalize_record(
    *,
    df: pd.DataFrame,
    row_index: int,
    qe_service: QEService,
    source_text: str,
    translated_text: str | None,
    precomputed_status: str | None,
    precomputed_error_message: str | None,
    was_chunked: bool,
    chunk_count: int,
) -> str:
    validation_result = validate_translation(
        source_text=source_text,
        translated_text=translated_text,
        precomputed_status=precomputed_status,
        precomputed_error_message=precomputed_error_message,
    )
    qe_columns = build_qe_columns(
        qe_service=qe_service,
        status=validation_result.status,
        source_text=source_text,
        translated_text=translated_text,
    )

    df.at[row_index, "translated_text"] = translated_text
    df.at[row_index, "status"] = validation_result.status
    df.at[row_index, "placeholder_ok"] = validation_result.placeholder_ok
    df.at[row_index, "validation_error"] = validation_result.error_message
    df.at[row_index, "was_chunked"] = was_chunked
    df.at[row_index, "chunk_count"] = chunk_count

    for column_name, value in qe_columns.items():
        df.at[row_index, column_name] = value

    return validation_result.status


def _resolve_and_finalize_record(
    *,
    config: AppConfig,
    df: pd.DataFrame,
    row_index: int,
    translator: OpusTranslator,
    qe_service: QEService,
    source_text: str,
    masked_text: str,
    emoji_replacements: list[str],
    translated_text: str | None,
    precomputed_status: str | None,
    precomputed_error_message: str | None,
    was_chunked: bool,
    chunk_count: int,
) -> str:
    final_translated_text = translated_text
    final_status = precomputed_status
    final_error = precomputed_error_message
    final_was_chunked = was_chunked
    final_chunk_count = chunk_count

    validation_result = validate_translation(
        source_text=source_text,
        translated_text=final_translated_text,
        precomputed_status=final_status,
        precomputed_error_message=final_error,
    )

    if config.retry_placeholder_mismatch and validation_result.status == "placeholder_mismatch":
        retry_max_input_length = max(1, int(config.max_input_length * 0.5))
        logger.info(
            "Retrying translation for row %d after placeholder_mismatch with max_input_length=%d",
            row_index,
            retry_max_input_length,
        )

        try:
            retried_text, retried_status, retried_chunk_count = _translate_single_text_with_limit(
                translator=translator,
                text=masked_text,
                max_input_length=retry_max_input_length,
                chunk_token_limit=config.chunk_token_limit,
                chunk_overlap_tokens=config.chunk_overlap_tokens,
            )
            final_translated_text = unmask_emojis(retried_text, emoji_replacements)
            final_status = retried_status
            final_error = None
            final_was_chunked = retried_status == "ok_chunked"
            final_chunk_count = retried_chunk_count
        except Exception as exc:
            logger.exception(
                "Retry translation failed for row %d after placeholder_mismatch",
                row_index,
            )
            final_translated_text = None
            final_status = "translation_error"
            final_error = (
                "Retry translation failed after placeholder_mismatch: "
                f"{exc}"
            )
            final_was_chunked = False
            final_chunk_count = 0
    elif not config.retry_placeholder_mismatch and validation_result.status == "placeholder_mismatch":
        logger.info("Placeholder mismatch retry is disabled")

    return _finalize_record(
        df=df,
        row_index=row_index,
        qe_service=qe_service,
        source_text=source_text,
        translated_text=final_translated_text,
        precomputed_status=final_status,
        precomputed_error_message=final_error,
        was_chunked=final_was_chunked,
        chunk_count=final_chunk_count,
    )


def _append_output_chunk(
    df: pd.DataFrame,
    *,
    config: AppConfig,
    write_header: bool,
) -> None:
    mode = "w" if write_header else "a"
    df.to_csv(
        config.output_path,
        mode=mode,
        header=write_header,
        index=False,
        encoding=config.output_encoding,
    )


def _load_progress_state(config: AppConfig) -> dict[str, Any] | None:
    if config.progress_path is None or not config.progress_path.exists():
        return None

    with open(config.progress_path, "r", encoding="utf-8-sig") as f:
        state = json.load(f)

    if (
        state.get("input_path") != str(config.input_path)
        or state.get("output_path") != str(config.output_path)
    ):
        logger.warning(
            "Ignoring progress file %s because it does not match the current input/output paths",
            config.progress_path,
        )
        return None

    if state.get("completed"):
        return None

    return state


def _save_progress_state(
    *,
    config: AppConfig,
    input_rows_consumed: int,
    chunks_completed: int,
    rows_loaded: int,
    rows_processed: int,
    rows_dropped_empty: int,
    rows_too_long_for_model: int,
    rows_skipped_url_like: int,
    rows_skipped_max_translate_tokens: int,
    rows_chunked: int,
    validation_summary: dict[str, int],
) -> None:
    if config.progress_path is None:
        return

    state = {
        "input_path": str(config.input_path),
        "output_path": str(config.output_path),
        "completed": False,
        "input_rows_consumed": input_rows_consumed,
        "chunks_completed": chunks_completed,
        "rows_loaded": rows_loaded,
        "rows_processed": rows_processed,
        "rows_dropped_empty": rows_dropped_empty,
        "rows_too_long_for_model": rows_too_long_for_model,
        "rows_skipped_url_like": rows_skipped_url_like,
        "rows_skipped_max_translate_tokens": rows_skipped_max_translate_tokens,
        "rows_chunked": rows_chunked,
        "validation_summary": validation_summary,
    }

    temp_path = config.progress_path.with_name(f"{config.progress_path.name}.tmp")
    with open(temp_path, "w", encoding="utf-8-sig") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    temp_path.replace(config.progress_path)


def _clear_progress_state(config: AppConfig) -> None:
    if config.progress_path is not None and config.progress_path.exists():
        config.progress_path.unlink()


def _process_chunk(
    *,
    chunk_df: pd.DataFrame,
    config: AppConfig,
    translator: OpusTranslator,
    qe_service: QEService,
    chunk_label: str,
) -> tuple[pd.DataFrame, ChunkMetrics]:
    expected_columns = {config.text_column, config.label_column}
    missing = expected_columns - set(chunk_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    rows_loaded = len(chunk_df)
    chunk_df = chunk_df.copy()
    chunk_df["input_row_number"] = chunk_df.index + 1
    chunk_df.reset_index(drop=True, inplace=True)
    chunk_df[config.text_column] = chunk_df[config.text_column].astype(str)

    before_drop = len(chunk_df)
    chunk_df = chunk_df[chunk_df[config.text_column].str.strip() != ""].copy()
    chunk_df.reset_index(drop=True, inplace=True)
    dropped = before_drop - len(chunk_df)

    if dropped:
        logger.warning("%s dropped %s empty rows during preprocessing", chunk_label, dropped)

    logger.info("%s rows after preprocessing: %s", chunk_label, len(chunk_df))

    texts = chunk_df[config.text_column].tolist()
    emoji_mask_results = [mask_emojis(text) for text in texts]
    masked_texts = [result.masked_text for result in emoji_mask_results]
    emoji_replacements = [result.replacements for result in emoji_mask_results]
    contains_emoji_flags = [result.contains_emoji for result in emoji_mask_results]

    chunk_df["flag_contains_emoji"] = contains_emoji_flags

    url_like_flags = [
        is_url_like_text(text) if config.skip_url_like else False
        for text in masked_texts
    ]
    url_like_count = sum(url_like_flags)
    logger.info("%s detected %d URL-like texts", chunk_label, url_like_count)

    length_checks = translator.check_input_lengths(masked_texts)
    token_counts = [result.token_count for result in length_checks]
    too_long_flags = [result.is_too_long for result in length_checks]
    too_long_count = sum(too_long_flags)
    logger.info(
        "%s detected %d texts exceeding model input limit",
        chunk_label,
        too_long_count,
    )

    max_translate_token_flags = [
        (
            config.max_translate_tokens is not None
            and token_count > config.max_translate_tokens
        )
        for token_count in token_counts
    ]
    skipped_max_translate_tokens_count = sum(max_translate_token_flags)
    logger.info(
        "%s detected %d texts exceeding max_translate_tokens",
        chunk_label,
        skipped_max_translate_tokens_count,
    )

    chunk_df["token_count"] = token_counts
    chunk_df["too_long_for_model"] = too_long_flags
    chunk_df["exceeds_max_translate_tokens"] = max_translate_token_flags
    chunk_df["url_like_skipped"] = url_like_flags
    _initialize_output_columns(chunk_df, config)

    normal_indices: list[int] = []
    long_indices: list[int] = []

    def finalize_row(
        *,
        row_index: int,
        translated_text: str | None,
        precomputed_status: str | None,
        precomputed_error_message: str | None,
        was_chunked: bool,
        chunk_count: int,
    ) -> None:
        _resolve_and_finalize_record(
            config=config,
            df=chunk_df,
            row_index=row_index,
            translator=translator,
            qe_service=qe_service,
            source_text=texts[row_index],
            masked_text=masked_texts[row_index],
            emoji_replacements=emoji_replacements[row_index],
            translated_text=translated_text,
            precomputed_status=precomputed_status,
            precomputed_error_message=precomputed_error_message,
            was_chunked=was_chunked,
            chunk_count=chunk_count,
        )

    for i in range(len(texts)):
        if url_like_flags[i]:
            finalize_row(
                row_index=i,
                translated_text=None,
                precomputed_status="skipped_url_like",
                precomputed_error_message="Record looks like a URL-like string and was skipped.",
                was_chunked=False,
                chunk_count=0,
            )
            continue

        if max_translate_token_flags[i]:
            finalize_row(
                row_index=i,
                translated_text=None,
                precomputed_status="skipped_max_translate_tokens",
                precomputed_error_message=(
                    "Record exceeds max_translate_tokens limit "
                    f"({token_counts[i]} > {config.max_translate_tokens})."
                ),
                was_chunked=False,
                chunk_count=0,
            )
            continue

        if too_long_flags[i]:
            long_indices.append(i)
        else:
            normal_indices.append(i)

    logger.info("%s texts eligible for normal translation: %d", chunk_label, len(normal_indices))
    logger.info("%s texts eligible for chunked translation: %d", chunk_label, len(long_indices))

    if config.sort_batches_by_length:
        normal_indices = _sort_indices_by_token_count(normal_indices, token_counts)
        logger.info("%s sorted normal translation records by token count before batching", chunk_label)

    normal_batches = _build_dynamic_batches(
        normal_indices,
        token_counts,
        max_batch_size=config.batch_size,
        batch_token_budget=config.batch_token_budget,
    )

    for batch_no, batch_indices in enumerate(normal_batches, start=1):
        batch_texts = [masked_texts[i] for i in batch_indices]
        logger.info(
            "%s processing normal batch %d/%d (%d records, %d tokens)",
            chunk_label,
            batch_no,
            len(normal_batches),
            len(batch_indices),
            sum(token_counts[i] for i in batch_indices),
        )

        try:
            batch_result = translator.translate_batch_with_metadata(batch_texts)
            logger.info(
                "%s completed normal batch %d/%d in %.2fs",
                chunk_label,
                batch_no,
                len(normal_batches),
                batch_result.elapsed_seconds,
            )

            for idx, translated_text in zip(batch_indices, batch_result.translations):
                finalize_row(
                    row_index=idx,
                    translated_text=unmask_emojis(translated_text, emoji_replacements[idx]),
                    precomputed_status=None,
                    precomputed_error_message=None,
                    was_chunked=False,
                    chunk_count=0,
                )
        except Exception as exc:
            logger.exception(
                "%s translation failed for normal batch %d/%d",
                chunk_label,
                batch_no,
                len(normal_batches),
            )
            for idx in batch_indices:
                finalize_row(
                    row_index=idx,
                    translated_text=None,
                    precomputed_status="translation_error",
                    precomputed_error_message=f"Batch translation failed: {exc}",
                    was_chunked=False,
                    chunk_count=0,
                )

    for record_no, idx in enumerate(long_indices, start=1):
        logger.info(
            (
                "%s processing long record %d/%d "
                "(row_idx=%d token_count=%d chunk_token_limit=%d)"
            ),
            chunk_label,
            record_no,
            len(long_indices),
            int(chunk_df.iloc[idx]["input_row_number"]),
            token_counts[idx],
            config.chunk_token_limit,
        )

        try:
            chunked_result = translator.translate_long_text(
                masked_texts[idx],
                chunk_token_limit=config.chunk_token_limit,
                chunk_overlap_tokens=config.chunk_overlap_tokens,
            )

            logger.info(
                "%s completed long record %d/%d in %.2fs using %d chunks",
                chunk_label,
                record_no,
                len(long_indices),
                chunked_result.elapsed_seconds,
                chunked_result.chunk_count,
            )

            finalize_row(
                row_index=idx,
                translated_text=unmask_emojis(
                    chunked_result.translated_text,
                    emoji_replacements[idx],
                ),
                precomputed_status="ok_chunked",
                precomputed_error_message=None,
                was_chunked=True,
                chunk_count=chunked_result.chunk_count,
            )
        except Exception as exc:
            logger.exception(
                "%s chunked translation failed for long record %d/%d (row_idx=%d)",
                chunk_label,
                record_no,
                len(long_indices),
                int(chunk_df.iloc[idx]["input_row_number"]),
            )
            finalize_row(
                row_index=idx,
                translated_text=None,
                precomputed_status="translation_error",
                precomputed_error_message=f"Chunked translation failed: {exc}",
                was_chunked=False,
                chunk_count=0,
            )

    validation_summary = _empty_validation_summary()
    status_counts = chunk_df["status"].dropna().value_counts()
    for status, count in status_counts.items():
        validation_summary[str(status)] = int(count)

    metrics = ChunkMetrics(
        rows_loaded=rows_loaded,
        rows_processed=len(chunk_df),
        rows_dropped_empty=dropped,
        rows_too_long_for_model=too_long_count,
        rows_skipped_url_like=url_like_count,
        rows_skipped_max_translate_tokens=skipped_max_translate_tokens_count,
        rows_chunked=int(chunk_df["was_chunked"].fillna(False).sum()),
        validation_summary=validation_summary,
    )

    return chunk_df, metrics


def run_pipeline(config: AppConfig) -> None:
    start_time = time.perf_counter()

    logger.info("Starting translation pipeline")
    logger.info("Input file: %s", config.input_path)
    logger.info("Output file: %s", config.output_path)
    logger.info("Progress file: %s", config.progress_path)
    logger.info("Model: %s", config.model_name)
    logger.info("Device: %s", config.device)
    logger.info("Batch size: %s", config.batch_size)
    logger.info("Batch token budget: %s", config.batch_token_budget)
    logger.info("Input chunk size: %s", config.input_chunk_size)
    logger.info("Limit: %s", config.limit)
    logger.info("max_input_length: %s", config.max_input_length)
    logger.info("chunk_token_limit: %s", config.chunk_token_limit)
    logger.info("max_new_tokens: %s", config.max_new_tokens)
    logger.info("chunk_overlap_tokens: %s", config.chunk_overlap_tokens)
    logger.info("num_beams: %s", config.num_beams)
    logger.info("skip_url_like: %s", config.skip_url_like)
    logger.info("sort_batches_by_length: %s", config.sort_batches_by_length)
    logger.info("retry_placeholder_mismatch: %s", config.retry_placeholder_mismatch)
    logger.info("checkpoint_interval: %s", config.checkpoint_interval)
    logger.info("max_translate_tokens: %s", config.max_translate_tokens)
    logger.info("enable_qe: %s", config.enable_qe)
    logger.info("qe_backend: %s", config.qe_backend)
    logger.info("qe_model_name: %s", config.qe_model_name)

    progress_state = _load_progress_state(config)
    resumed = progress_state is not None and config.output_path.exists()

    if resumed:
        logger.info(
            "Resuming partial run from progress file: input_rows_consumed=%d chunks_completed=%d",
            int(progress_state["input_rows_consumed"]),
            int(progress_state["chunks_completed"]),
        )
    else:
        logger.info("Starting fresh run")

    qe_service = QEService.from_config(
        enable_qe=config.enable_qe,
        qe_backend=config.qe_backend,
        qe_model_name=config.qe_model_name,
        qe_high_threshold=config.qe_high_threshold,
        qe_medium_threshold=config.qe_medium_threshold,
    )

    translator = OpusTranslator(
        model_name=config.model_name,
        device=config.device,
        max_input_length=config.max_input_length,
        max_new_tokens=config.max_new_tokens,
        num_beams=config.num_beams,
    )

    rows_loaded_total = int(progress_state["rows_loaded"]) if resumed else 0
    rows_processed_total = int(progress_state["rows_processed"]) if resumed else 0
    rows_dropped_empty_total = int(progress_state["rows_dropped_empty"]) if resumed else 0
    rows_too_long_for_model_total = (
        int(progress_state["rows_too_long_for_model"]) if resumed else 0
    )
    rows_skipped_url_like_total = (
        int(progress_state["rows_skipped_url_like"]) if resumed else 0
    )
    rows_skipped_max_translate_tokens_total = (
        int(progress_state["rows_skipped_max_translate_tokens"]) if resumed else 0
    )
    rows_chunked_total = int(progress_state["rows_chunked"]) if resumed else 0
    validation_summary_total = (
        _merge_validation_summaries(
            _empty_validation_summary(),
            progress_state["validation_summary"],
        )
        if resumed
        else _empty_validation_summary()
    )

    input_rows_consumed = int(progress_state["input_rows_consumed"]) if resumed else 0
    chunks_completed = int(progress_state["chunks_completed"]) if resumed else 0
    rows_to_skip = input_rows_consumed
    limited_rows_seen = 0
    first_chunk_written = resumed and config.output_path.exists()
    expected_columns_validated = resumed

    reader = pd.read_csv(
        config.input_path,
        encoding=config.input_encoding,
        low_memory=False,
        chunksize=config.input_chunk_size,
    )

    for raw_chunk in reader:
        current_chunk = raw_chunk

        if config.limit is not None:
            if limited_rows_seen >= config.limit:
                break

            remaining = config.limit - limited_rows_seen
            current_chunk = current_chunk.head(remaining)
            limited_rows_seen += len(current_chunk)

        if current_chunk.empty:
            continue

        if not expected_columns_validated:
            expected_columns = {config.text_column, config.label_column}
            missing = expected_columns - set(current_chunk.columns)
            if missing:
                raise ValueError(f"Missing required columns: {sorted(missing)}")
            expected_columns_validated = True

        if rows_to_skip >= len(current_chunk):
            rows_to_skip -= len(current_chunk)
            continue

        if rows_to_skip > 0:
            current_chunk = current_chunk.iloc[rows_to_skip:].copy()
            rows_to_skip = 0

        current_chunk = current_chunk.copy()
        current_chunk.index = range(
            input_rows_consumed,
            input_rows_consumed + len(current_chunk),
        )

        chunk_label = f"chunk {chunks_completed + 1}"
        logger.info("%s loaded %d input rows", chunk_label, len(current_chunk))

        processed_chunk, chunk_metrics = _process_chunk(
            chunk_df=current_chunk,
            config=config,
            translator=translator,
            qe_service=qe_service,
            chunk_label=chunk_label,
        )

        _append_output_chunk(
            processed_chunk,
            config=config,
            write_header=not first_chunk_written,
        )
        first_chunk_written = True

        rows_loaded_total += chunk_metrics.rows_loaded
        rows_processed_total += chunk_metrics.rows_processed
        rows_dropped_empty_total += chunk_metrics.rows_dropped_empty
        rows_too_long_for_model_total += chunk_metrics.rows_too_long_for_model
        rows_skipped_url_like_total += chunk_metrics.rows_skipped_url_like
        rows_skipped_max_translate_tokens_total += (
            chunk_metrics.rows_skipped_max_translate_tokens
        )
        rows_chunked_total += chunk_metrics.rows_chunked
        validation_summary_total = _merge_validation_summaries(
            validation_summary_total,
            chunk_metrics.validation_summary,
        )
        input_rows_consumed += chunk_metrics.rows_loaded
        chunks_completed += 1

        _save_progress_state(
            config=config,
            input_rows_consumed=input_rows_consumed,
            chunks_completed=chunks_completed,
            rows_loaded=rows_loaded_total,
            rows_processed=rows_processed_total,
            rows_dropped_empty=rows_dropped_empty_total,
            rows_too_long_for_model=rows_too_long_for_model_total,
            rows_skipped_url_like=rows_skipped_url_like_total,
            rows_skipped_max_translate_tokens=rows_skipped_max_translate_tokens_total,
            rows_chunked=rows_chunked_total,
            validation_summary=validation_summary_total,
        )
        logger.info(
            "Saved progress checkpoint to %s after %s (rows_written_total=%d)",
            config.progress_path,
            chunk_label,
            rows_processed_total,
        )

    elapsed = time.perf_counter() - start_time
    logger.info("Validation summary: %s", validation_summary_total)
    logger.info("Pipeline finished in %.2f seconds", elapsed)

    report = {
        "input_path": str(config.input_path),
        "output_path": str(config.output_path),
        "progress_path": str(config.progress_path) if config.progress_path is not None else None,
        "model_name": config.model_name,
        "device": config.device,
        "batch_size": config.batch_size,
        "batch_token_budget": config.batch_token_budget,
        "input_chunk_size": config.input_chunk_size,
        "limit": config.limit,
        "source_lang": config.source_lang,
        "target_lang": config.target_lang,
        "max_input_length": config.max_input_length,
        "chunk_token_limit": config.chunk_token_limit,
        "max_new_tokens": config.max_new_tokens,
        "chunk_overlap_tokens": config.chunk_overlap_tokens,
        "num_beams": config.num_beams,
        "skip_url_like": config.skip_url_like,
        "checkpoint_interval": config.checkpoint_interval,
        "max_translate_tokens": config.max_translate_tokens,
        "enable_qe": config.enable_qe,
        "qe_backend": config.qe_backend,
        "qe_model_name": config.qe_model_name,
        "resumed": resumed,
        "chunks_completed": chunks_completed,
        "input_rows_consumed": input_rows_consumed,
        "rows_loaded": rows_loaded_total,
        "rows_processed": rows_processed_total,
        "rows_dropped_empty": rows_dropped_empty_total,
        "rows_too_long_for_model": rows_too_long_for_model_total,
        "rows_skipped_url_like": rows_skipped_url_like_total,
        "rows_skipped_max_translate_tokens": rows_skipped_max_translate_tokens_total,
        "rows_chunked": rows_chunked_total,
        "elapsed_seconds": round(elapsed, 2),
        "validation_summary": validation_summary_total,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
    }

    if config.report_path is not None:
        with open(config.report_path, "w", encoding="utf-8-sig") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info("Saved run report to %s", config.report_path)

    _clear_progress_state(config)
