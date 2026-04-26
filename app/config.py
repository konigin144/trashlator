from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from qe.service import DISABLED_QE_BACKENDS, SUPPORTED_QE_BACKENDS


def _get_env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value is not None else default


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer.") from exc


def _get_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a float.") from exc


def _get_env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False

    raise ValueError(f"Environment variable {name} must be a boolean-like value.")


@dataclass(slots=True)
class AppConfig:
    input_path: Path
    output_path: Path
    model_name: str
    batch_size: int
    device: str
    log_level: str
    log_file: Path | None

    limit: int | None = None
    source_lang: str | None = None
    target_lang: str | None = None
    report_path: Path | None = None
    progress_path: Path | None = None

    max_input_length: int = 256
    chunk_token_limit: int = 160
    max_new_tokens: int = 512
    chunk_overlap_tokens: int = 0
    num_beams: int = 1
    skip_url_like: bool = True
    sort_batches_by_length: bool = False
    retry_placeholder_mismatch: bool = True
    checkpoint_interval: int = 50
    max_translate_tokens: int | None = 1024
    batch_token_budget: int = 4096
    input_chunk_size: int = 5000

    text_column: str = "message"
    label_column: str = "label"
    input_encoding: str = "utf-8-sig"
    output_encoding: str = "utf-8"

    enable_qe: bool = False
    qe_backend: str | None = None
    qe_model_name: str | None = None
    qe_high_threshold: float = 0.7
    qe_medium_threshold: float = 0.4

    @classmethod
    def from_env(cls) -> "AppConfig":
        input_path = Path(_get_env_str("INPUT_PATH", "data/input/input.csv"))
        output_path = Path(_get_env_str("OUTPUT_PATH", "data/output/output.csv"))

        model_name = os.getenv("MODEL_NAME")
        if model_name is not None:
            model_name = model_name.strip() or None

        batch_size = _get_env_int("BATCH_SIZE", 16)
        device = _get_env_str("DEVICE", "cpu")
        log_level = _get_env_str("LOG_LEVEL", "INFO").upper()

        log_file_env = os.getenv("LOG_FILE")
        log_file = Path(log_file_env) if log_file_env else None

        limit_env = os.getenv("LIMIT")
        limit = int(limit_env) if limit_env else None

        source_lang = os.getenv("SOURCE_LANG")
        if source_lang is not None:
            source_lang = source_lang.strip() or None

        target_lang = os.getenv("TARGET_LANG")
        if target_lang is not None:
            target_lang = target_lang.strip() or None

        report_path_env = os.getenv("REPORT_PATH")
        report_path = Path(report_path_env) if report_path_env else None
        progress_path_env = os.getenv("PROGRESS_PATH")
        progress_path = Path(progress_path_env) if progress_path_env else None

        max_input_length = _get_env_int("MAX_INPUT_LENGTH", 256)
        chunk_token_limit = _get_env_int("CHUNK_TOKEN_LIMIT", 160)
        max_new_tokens = _get_env_int("MAX_NEW_TOKENS", 512)
        chunk_overlap_tokens = _get_env_int("CHUNK_OVERLAP_TOKENS", 0)
        num_beams = _get_env_int("NUM_BEAMS", 1)
        skip_url_like = _get_env_bool("SKIP_URL_LIKE", True)
        sort_batches_by_length = _get_env_bool("SORT_BATCHES_BY_LENGTH", False)
        retry_placeholder_mismatch = _get_env_bool("RETRY_PLACEHOLDER_MISMATCH", True)
        checkpoint_interval = _get_env_int("CHECKPOINT_INTERVAL", 50)
        max_translate_tokens_env = os.getenv("MAX_TRANSLATE_TOKENS")
        max_translate_tokens = (
            int(max_translate_tokens_env)
            if max_translate_tokens_env
            else 1024
        )
        batch_token_budget = _get_env_int("BATCH_TOKEN_BUDGET", 4096)
        input_chunk_size = _get_env_int("INPUT_CHUNK_SIZE", 5000)

        enable_qe = _get_env_bool("ENABLE_QE", False)

        qe_backend = os.getenv("QE_BACKEND")
        if qe_backend is not None:
            qe_backend = qe_backend.strip() or None

        qe_model_name = os.getenv("QE_MODEL_NAME")
        if qe_model_name is not None:
            qe_model_name = qe_model_name.strip() or None

        qe_high_threshold = _get_env_float("QE_HIGH_THRESHOLD", 0.7)
        qe_medium_threshold = _get_env_float("QE_MEDIUM_THRESHOLD", 0.4)

        return cls(
            input_path=input_path,
            output_path=output_path,
            model_name=model_name,
            batch_size=batch_size,
            device=device,
            log_level=log_level,
            log_file=log_file,
            limit=limit,
            source_lang=source_lang,
            target_lang=target_lang,
            report_path=report_path,
            progress_path=progress_path,
            max_input_length=max_input_length,
            chunk_token_limit=chunk_token_limit,
            max_new_tokens=max_new_tokens,
            chunk_overlap_tokens=chunk_overlap_tokens,
            num_beams=num_beams,
            skip_url_like=skip_url_like,
            sort_batches_by_length=sort_batches_by_length,
            retry_placeholder_mismatch=retry_placeholder_mismatch,
            checkpoint_interval=checkpoint_interval,
            max_translate_tokens=max_translate_tokens,
            batch_token_budget=batch_token_budget,
            input_chunk_size=input_chunk_size,
            enable_qe=enable_qe,
            qe_backend=qe_backend,
            qe_model_name=qe_model_name,
            qe_high_threshold=qe_high_threshold,
            qe_medium_threshold=qe_medium_threshold,
        )

    def validate(self) -> None:
        if not self.model_name:
            raise ValueError("model_name must be provided via CLI or environment.")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than 0.")

        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be either 'cpu' or 'cuda'.")

        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file does not exist: {self.input_path}")

        if self.limit is not None and self.limit <= 0:
            raise ValueError("limit must be greater than 0 if provided.")

        if self.log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(
                "log_level must be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL"
            )

        if self.max_input_length <= 0:
            raise ValueError("max_input_length must be greater than 0.")

        if self.chunk_token_limit <= 0:
            raise ValueError("chunk_token_limit must be greater than 0.")

        if self.chunk_token_limit > self.max_input_length:
            raise ValueError("chunk_token_limit must not exceed max_input_length.")

        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than 0.")

        if self.chunk_overlap_tokens < 0:
            raise ValueError("chunk_overlap_tokens must be non-negative.")

        if self.chunk_overlap_tokens >= self.chunk_token_limit:
            raise ValueError(
                "chunk_overlap_tokens must be smaller than chunk_token_limit."
            )

        if self.num_beams <= 0:
            raise ValueError("num_beams must be greater than 0.")

        if self.enable_qe:
            if not self.qe_backend:
                raise ValueError("qe_backend must be provided when enable_qe is true.")

            if self.qe_backend in DISABLED_QE_BACKENDS:
                raise ValueError(DISABLED_QE_BACKENDS[self.qe_backend])

            if self.qe_backend not in SUPPORTED_QE_BACKENDS:
                raise ValueError(
                    f"qe_backend must be one of: {sorted(SUPPORTED_QE_BACKENDS)}"
                )

            if not self.qe_model_name:
                raise ValueError(
                    "qe_model_name must be provided when enable_qe is true."
                )

            if self.qe_medium_threshold < 0 or self.qe_high_threshold < 0:
                raise ValueError("QE thresholds must be non-negative.")

            if self.qe_medium_threshold > self.qe_high_threshold:
                raise ValueError(
                    "qe_medium_threshold must not be greater than qe_high_threshold."
                )

        if self.checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be greater than 0.")

        if self.max_translate_tokens is not None and self.max_translate_tokens <= 0:
            raise ValueError("max_translate_tokens must be greater than 0 if provided.")

        if self.batch_token_budget <= 0:
            raise ValueError("batch_token_budget must be greater than 0.")

        if self.input_chunk_size <= 0:
            raise ValueError("input_chunk_size must be greater than 0.")

    def ensure_output_dirs(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.log_file is not None:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

        if self.report_path is not None:
            self.report_path.parent.mkdir(parents=True, exist_ok=True)

        if self.progress_path is not None:
            self.progress_path.parent.mkdir(parents=True, exist_ok=True)
