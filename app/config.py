from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


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

    max_input_length: int = 512
    max_new_tokens: int = 512
    num_beams: int = 1
    skip_url_like: bool = True

    text_column: str = "message"
    label_column: str = "label"
    input_encoding: str = "utf-8-sig"
    output_encoding: str = "utf-8"

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

        max_input_length = _get_env_int("MAX_INPUT_LENGTH", 512)
        max_new_tokens = _get_env_int("MAX_NEW_TOKENS", 512)
        num_beams = _get_env_int("NUM_BEAMS", 1)
        skip_url_like = _get_env_bool("SKIP_URL_LIKE", True)

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
            max_input_length=max_input_length,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            skip_url_like=skip_url_like,
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

        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than 0.")

        if self.num_beams <= 0:
            raise ValueError("num_beams must be greater than 0.")

    def ensure_output_dirs(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.log_file is not None:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

        if self.report_path is not None:
            self.report_path.parent.mkdir(parents=True, exist_ok=True)