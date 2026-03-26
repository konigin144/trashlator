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
    parser.add_argument("--source-lang", dest="source_lang", type=str, help="Source language code.")
    parser.add_argument("--target-lang", dest="target_lang", type=str, help="Target language code.")

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
    if args.source_lang is not None:
        config.source_lang = args.source_lang
    if args.target_lang is not None:
        config.target_lang = args.target_lang
    if args.max_input_length is not None:
        config.max_input_length = args.max_input_length
    if args.max_new_tokens is not None:
        config.max_new_tokens = args.max_new_tokens
    if args.num_beams is not None:
        config.num_beams = args.num_beams
    if args.skip_url_like is not None:
        config.skip_url_like = args.skip_url_like

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if config.log_file is None:
        config.log_file = Path("logs") / f"run_{timestamp}.log"

    if config.report_path is None:
        config.report_path = Path("logs") / f"run_report_{timestamp}.json"

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