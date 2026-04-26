from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from app.config import AppConfig
from app.logging_config import setup_logging
from app.pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch translation pipeline for phishing/legit CSV datasets."
    )

    parser.add_argument("--input", dest="input_path", type=Path, help="Path to input CSV file.")
    parser.add_argument("--output", dest="output_path", type=Path, help="Path to output CSV file.")
    parser.add_argument("--model", dest="model_name", type=str, help="Hugging Face model name.")
    parser.add_argument("--batch-size", dest="batch_size", type=int, help="Batch size for translation.")
    parser.add_argument("--device", dest="device", type=str, choices=["cpu", "cuda"], help="Inference device.")
    parser.add_argument("--limit", dest="limit", type=int, help="Optional limit of rows to process.")
    parser.add_argument(
        "--log-level",
        dest="log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level.",
    )
    parser.add_argument("--log-file", dest="log_file", type=Path, help="Optional log file path.")
    parser.add_argument("--report", dest="report_path", type=Path, help="Optional report JSON path.")
    parser.add_argument("--progress-file", dest="progress_path", type=Path, help="Optional progress-state JSON path.")
    parser.add_argument("--source-lang", dest="source_lang", type=str, help="Source language code.")
    parser.add_argument("--target-lang", dest="target_lang", type=str, help="Target language code.")
    parser.add_argument(
        "--input-chunk-size",
        dest="input_chunk_size",
        type=int,
        help="Number of input rows to stream into memory at once.",
    )

    parser.add_argument(
        "--max-input-length",
        dest="max_input_length",
        type=int,
        help="Maximum input length in tokens.",
    )
    parser.add_argument(
        "--max-new-tokens",
        dest="max_new_tokens",
        type=int,
        help="Maximum number of output tokens to generate.",
    )
    parser.add_argument(
        "--chunk-token-limit",
        dest="chunk_token_limit",
        type=int,
        help="Token limit for chunking long texts before translation.",
    )
    parser.add_argument(
        "--chunk-overlap-tokens",
        dest="chunk_overlap_tokens",
        type=int,
        help="Token overlap between adjacent chunks for long-text translation.",
    )
    parser.add_argument(
        "--num-beams",
        dest="num_beams",
        type=int,
        help="Beam search width.",
    )
    parser.add_argument(
        "--skip-url-like",
        dest="skip_url_like",
        action="store_true",
        help="Skip translation for URL-like records.",
    )
    parser.add_argument(
        "--no-skip-url-like",
        dest="skip_url_like",
        action="store_false",
        help="Do not skip translation for URL-like records.",
    )
    parser.set_defaults(skip_url_like=None)
    parser.add_argument(
        "--sort-batches-by-length",
        dest="sort_batches_by_length",
        action="store_true",
        help="Group normal translation batches by token length.",
    )
    parser.add_argument(
        "--no-sort-batches-by-length",
        dest="sort_batches_by_length",
        action="store_false",
        help="Keep normal translation batches in input order.",
    )
    parser.set_defaults(sort_batches_by_length=None)
    parser.add_argument(
        "--retry-placeholder-mismatch",
        dest="retry_placeholder_mismatch",
        action="store_true",
        help="Retry translation with lower max_input_length after placeholder mismatch.",
    )
    parser.add_argument(
        "--no-retry-placeholder-mismatch",
        dest="retry_placeholder_mismatch",
        action="store_false",
        help="Do not retry translation after placeholder mismatch.",
    )
    parser.set_defaults(retry_placeholder_mismatch=None)
    parser.add_argument(
        "--checkpoint-interval",
        dest="checkpoint_interval",
        type=int,
        help="Number of finalized rows between output checkpoints.",
    )
    parser.add_argument(
        "--max-translate-tokens",
        dest="max_translate_tokens",
        type=int,
        help="Skip translation for rows exceeding this token count.",
    )
    parser.add_argument(
        "--batch-token-budget",
        dest="batch_token_budget",
        type=int,
        help="Maximum summed token count for a normal translation batch.",
    )

    parser.add_argument(
        "--enable-qe",
        dest="enable_qe",
        action="store_true",
        help="Enable QE scoring for translated records.",
    )
    parser.add_argument(
        "--disable-qe",
        dest="enable_qe",
        action="store_false",
        help="Disable QE scoring.",
    )
    parser.set_defaults(enable_qe=None)

    parser.add_argument(
        "--qe-backend",
        dest="qe_backend",
        type=str,
        choices=["comet"],
        help="QE backend to use.",
    )
    parser.add_argument(
        "--qe-model-name",
        dest="qe_model_name",
        type=str,
        help="Model name/path for QE backend.",
    )
    parser.add_argument(
        "--qe-high-threshold",
        dest="qe_high_threshold",
        type=float,
        help="High-confidence threshold for QE score.",
    )
    parser.add_argument(
        "--qe-medium-threshold",
        dest="qe_medium_threshold",
        type=float,
        help="Medium-confidence threshold for QE score.",
    )

    return parser


def merge_cli_with_env(args: argparse.Namespace) -> AppConfig:
    config = AppConfig.from_env()

    if args.input_path is not None:
        config.input_path = args.input_path
    if args.output_path is not None:
        config.output_path = args.output_path
    if args.model_name is not None:
        config.model_name = args.model_name
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.device is not None:
        config.device = args.device
    if args.limit is not None:
        config.limit = args.limit
    if args.log_level is not None:
        config.log_level = args.log_level
    if args.log_file is not None:
        config.log_file = args.log_file
    if args.report_path is not None:
        config.report_path = args.report_path
    if args.progress_path is not None:
        config.progress_path = args.progress_path
    if args.source_lang is not None:
        config.source_lang = args.source_lang
    if args.target_lang is not None:
        config.target_lang = args.target_lang
    if args.input_chunk_size is not None:
        config.input_chunk_size = args.input_chunk_size
    if args.max_input_length is not None:
        config.max_input_length = args.max_input_length
    if args.max_new_tokens is not None:
        config.max_new_tokens = args.max_new_tokens
    if args.chunk_token_limit is not None:
        config.chunk_token_limit = args.chunk_token_limit
    if args.chunk_overlap_tokens is not None:
        config.chunk_overlap_tokens = args.chunk_overlap_tokens
    if args.num_beams is not None:
        config.num_beams = args.num_beams
    if args.skip_url_like is not None:
        config.skip_url_like = args.skip_url_like
    if args.sort_batches_by_length is not None:
        config.sort_batches_by_length = args.sort_batches_by_length
    if args.retry_placeholder_mismatch is not None:
        config.retry_placeholder_mismatch = args.retry_placeholder_mismatch
    if args.checkpoint_interval is not None:
        config.checkpoint_interval = args.checkpoint_interval
    if args.max_translate_tokens is not None:
        config.max_translate_tokens = args.max_translate_tokens
    if args.batch_token_budget is not None:
        config.batch_token_budget = args.batch_token_budget

    if args.enable_qe is not None:
        config.enable_qe = args.enable_qe
    if args.qe_backend is not None:
        config.qe_backend = args.qe_backend
    if args.qe_model_name is not None:
        config.qe_model_name = args.qe_model_name
    if args.qe_high_threshold is not None:
        config.qe_high_threshold = args.qe_high_threshold
    if args.qe_medium_threshold is not None:
        config.qe_medium_threshold = args.qe_medium_threshold

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if config.log_file is None:
        config.log_file = Path("logs") / f"run_{timestamp}.log"

    if config.report_path is None:
        config.report_path = Path("logs") / f"run_report_{timestamp}.json"
    if config.progress_path is None:
        config.progress_path = config.output_path.with_name(
            f"{config.output_path.stem}.progress.json"
        )

    return config


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = merge_cli_with_env(args)
    config.ensure_output_dirs()
    setup_logging(log_level=config.log_level, log_file=config.log_file)
    config.validate()

    run_pipeline(config)


if __name__ == "__main__":
    main()