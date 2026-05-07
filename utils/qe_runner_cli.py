from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from qe.result import QEResult
from qe.service import QEService, SUPPORTED_QE_BACKENDS


QERunSummary = tuple[int, float | None]


QE_COLUMNS = [
    "qe_score",
    "qe_label",
    "qe_error",
    "qe_backend",
    "qe_model_name",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run QE scoring for paired source/translated texts from one CSV or two CSVs."
        )
    )
    parser.add_argument(
        "--input",
        dest="combined_input_path",
        type=Path,
        help="Path to one CSV file containing both source and translated texts.",
    )
    parser.add_argument(
        "--source-input",
        "--source",
        dest="source_input_path",
        type=Path,
        help="Path to the CSV file with source texts.",
    )
    parser.add_argument(
        "--translated-input",
        "--translated",
        dest="translated_input_path",
        type=Path,
        help="Path to the CSV file with translated texts.",
    )
    parser.add_argument(
        "--output",
        dest="output_path",
        type=Path,
        required=True,
        help="Path to the QE output CSV file.",
    )
    parser.add_argument(
        "--source-column",
        dest="source_column",
        default="message",
        help="Column containing source texts. Defaults to 'message'.",
    )
    parser.add_argument(
        "--translated-column",
        dest="translated_column",
        default="message",
        help="Column containing translated texts. Defaults to 'message'.",
    )
    parser.add_argument(
        "--qe-backend",
        dest="qe_backend",
        default="comet",
        choices=sorted(SUPPORTED_QE_BACKENDS),
        help="QE backend to use. Defaults to 'comet'.",
    )
    parser.add_argument(
        "--qe-model-name",
        "--model",
        dest="qe_model_name",
        required=True,
        help="Model name/path for the selected QE backend.",
    )
    parser.add_argument(
        "--qe-high-threshold",
        dest="qe_high_threshold",
        type=float,
        default=0.7,
        help="High-confidence threshold for QE score.",
    )
    parser.add_argument(
        "--qe-medium-threshold",
        dest="qe_medium_threshold",
        type=float,
        default=0.4,
        help="Medium-confidence threshold for QE score.",
    )
    parser.add_argument(
        "--limit",
        dest="limit",
        type=int,
        help="Run QE for only the first N rows.",
    )
    parser.add_argument(
        "--input-encoding",
        dest="input_encoding",
        default="utf-8-sig",
        help="Encoding for both input CSV files. Defaults to utf-8-sig.",
    )
    parser.add_argument(
        "--output-encoding",
        dest="output_encoding",
        default="utf-8",
        help="Encoding for the output CSV file. Defaults to utf-8.",
    )
    return parser


def _validate_inputs(
    *,
    combined_input_path: Path | None,
    source_input_path: Path | None,
    translated_input_path: Path | None,
    source_column: str,
    translated_column: str,
    limit: int | None,
) -> None:
    uses_combined_input = combined_input_path is not None
    uses_paired_inputs = (
        source_input_path is not None or translated_input_path is not None
    )

    if uses_combined_input and uses_paired_inputs:
        raise ValueError(
            "Use either --input for a combined CSV or --source/--translated for paired CSVs."
        )

    if not uses_combined_input and not uses_paired_inputs:
        raise ValueError(
            "Provide either --input or both --source and --translated CSV paths."
        )

    if uses_combined_input:
        if not combined_input_path.exists():
            raise FileNotFoundError(f"Input CSV not found: {combined_input_path}")
    else:
        if source_input_path is None or translated_input_path is None:
            raise ValueError(
                "Both --source and --translated are required for paired CSV mode."
            )

        if not source_input_path.exists():
            raise FileNotFoundError(f"Source input CSV not found: {source_input_path}")

        if not translated_input_path.exists():
            raise FileNotFoundError(
                f"Translated input CSV not found: {translated_input_path}"
            )

    if not source_column:
        raise ValueError("source_column must not be empty.")

    if not translated_column:
        raise ValueError("translated_column must not be empty.")

    if limit is not None and limit <= 0:
        raise ValueError("limit must be a positive integer if provided.")


def _read_limited_csv(path: Path, *, encoding: str, limit: int | None) -> pd.DataFrame:
    df = pd.read_csv(path, encoding=encoding, low_memory=False)
    if limit is not None:
        return df.head(limit).copy()
    return df


def _require_column(df: pd.DataFrame, column_name: str, file_label: str) -> None:
    if column_name not in df.columns:
        raise ValueError(
            f"{file_label} CSV is missing required column: {column_name}"
        )


def _combined_output_frame(
    source_df: pd.DataFrame,
    translated_df: pd.DataFrame,
) -> pd.DataFrame:
    source_output = source_df.reset_index(drop=True).copy()
    translated_output = translated_df.reset_index(drop=True).copy()

    overlapping_columns = set(source_output.columns).intersection(
        translated_output.columns
    )
    qe_column_conflicts = set(QE_COLUMNS)
    source_columns_to_suffix = overlapping_columns.union(
        set(source_output.columns).intersection(qe_column_conflicts)
    )
    translated_columns_to_suffix = overlapping_columns.union(
        set(translated_output.columns).intersection(qe_column_conflicts)
    )
    source_output.rename(
        columns={column: f"{column}_source" for column in source_columns_to_suffix},
        inplace=True,
    )
    translated_output.rename(
        columns={
            column: f"{column}_translated"
            for column in translated_columns_to_suffix
        },
        inplace=True,
    )

    return pd.concat([source_output, translated_output], axis=1)


def _single_output_frame(input_df: pd.DataFrame) -> pd.DataFrame:
    output = input_df.reset_index(drop=True).copy()
    qe_column_conflicts = set(output.columns).intersection(QE_COLUMNS)
    output.rename(
        columns={column: f"{column}_input" for column in qe_column_conflicts},
        inplace=True,
    )
    return output


def _qe_columns_from_result(result: QEResult) -> dict[str, Any]:
    return {
        "qe_score": result.score,
        "qe_label": result.label,
        "qe_error": result.error,
        "qe_backend": result.backend,
        "qe_model_name": result.model_name,
    }


def _qe_columns_for_skipped_status(status: str) -> dict[str, Any]:
    return {
        "qe_score": None,
        "qe_label": None,
        "qe_error": f"QE skipped: translation status is '{status}'.",
        "qe_backend": None,
        "qe_model_name": None,
    }


def _status_allows_qe(status: Any) -> bool:
    normalized = "" if pd.isna(status) else str(status).strip()
    return normalized in {"ok", "ok_chunked"}


def _statuses_need_qe(statuses: pd.Series | None) -> bool:
    if statuses is None:
        return True
    return any(_status_allows_qe(status) for status in statuses.tolist())


def _score_rows(
    *,
    qe_service: QEService | None,
    source_texts: pd.Series,
    translated_texts: pd.Series,
    statuses: pd.Series | None,
) -> list[dict[str, Any]]:
    qe_rows: list[dict[str, Any]] = []
    source_values = source_texts.fillna("").astype(str)
    translated_values = translated_texts.fillna("").astype(str)

    if statuses is None:
        status_values = [None] * len(source_values)
    else:
        status_values = statuses.tolist()

    for source_text, translated_text, status in zip(
        source_values,
        translated_values,
        status_values,
    ):
        if statuses is not None and not _status_allows_qe(status):
            status_text = "" if pd.isna(status) else str(status).strip()
            qe_rows.append(_qe_columns_for_skipped_status(status_text))
            continue

        if qe_service is None:
            raise ValueError("QE service is required for rows with ok status.")

        result = qe_service.score(
            source_text=source_text,
            translated_text=translated_text,
        )
        qe_rows.append(_qe_columns_from_result(result))

    return qe_rows


def run_qe_csv(
    *,
    output_path: Path,
    qe_model_name: str,
    combined_input_path: Path | None = None,
    source_input_path: Path | None = None,
    translated_input_path: Path | None = None,
    source_column: str = "message",
    translated_column: str = "message",
    qe_backend: str = "comet",
    qe_high_threshold: float = 0.7,
    qe_medium_threshold: float = 0.4,
    limit: int | None = None,
    input_encoding: str = "utf-8-sig",
    output_encoding: str = "utf-8",
) -> QERunSummary:
    _validate_inputs(
        combined_input_path=combined_input_path,
        source_input_path=source_input_path,
        translated_input_path=translated_input_path,
        source_column=source_column,
        translated_column=translated_column,
        limit=limit,
    )

    if combined_input_path is not None:
        input_df = _read_limited_csv(
            combined_input_path,
            encoding=input_encoding,
            limit=limit,
        )
        _require_column(input_df, source_column, "Input")
        _require_column(input_df, translated_column, "Input")

        statuses = input_df["status"] if "status" in input_df.columns else None
        qe_service = (
            QEService.from_config(
                enable_qe=True,
                qe_backend=qe_backend,
                qe_model_name=qe_model_name,
                qe_high_threshold=qe_high_threshold,
                qe_medium_threshold=qe_medium_threshold,
            )
            if _statuses_need_qe(statuses)
            else None
        )
        qe_rows = _score_rows(
            qe_service=qe_service,
            source_texts=input_df[source_column],
            translated_texts=input_df[translated_column],
            statuses=statuses,
        )
        output_df = _single_output_frame(input_df)
    else:
        assert source_input_path is not None
        assert translated_input_path is not None

        source_df = _read_limited_csv(
            source_input_path,
            encoding=input_encoding,
            limit=limit,
        )
        translated_df = _read_limited_csv(
            translated_input_path,
            encoding=input_encoding,
            limit=limit,
        )

        _require_column(source_df, source_column, "Source input")
        _require_column(translated_df, translated_column, "Translated input")

        if len(source_df) != len(translated_df):
            raise ValueError(
                "Source and translated CSV files must contain the same number of paired rows."
            )

        qe_service = QEService.from_config(
            enable_qe=True,
            qe_backend=qe_backend,
            qe_model_name=qe_model_name,
            qe_high_threshold=qe_high_threshold,
            qe_medium_threshold=qe_medium_threshold,
        )
        qe_rows = _score_rows(
            qe_service=qe_service,
            source_texts=source_df[source_column],
            translated_texts=translated_df[translated_column],
            statuses=None,
        )
        output_df = _combined_output_frame(source_df, translated_df)

    qe_df = pd.DataFrame(qe_rows, columns=QE_COLUMNS)
    output_df = pd.concat([output_df, qe_df], axis=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False, encoding=output_encoding)

    scores = [
        float(score)
        for score in qe_df["qe_score"].dropna().tolist()
    ]
    average_score = sum(scores) / len(scores) if scores else None

    return len(output_df), average_score


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    row_count, average_score = run_qe_csv(
        combined_input_path=args.combined_input_path,
        source_input_path=args.source_input_path,
        translated_input_path=args.translated_input_path,
        output_path=args.output_path,
        source_column=args.source_column,
        translated_column=args.translated_column,
        qe_backend=args.qe_backend,
        qe_model_name=args.qe_model_name,
        qe_high_threshold=args.qe_high_threshold,
        qe_medium_threshold=args.qe_medium_threshold,
        limit=args.limit,
        input_encoding=args.input_encoding,
        output_encoding=args.output_encoding,
    )
    if average_score is None:
        print("Average QE score: n/a")
    else:
        print(f"Average QE score: {average_score:.6f}")
    print(f"Wrote {row_count} QE rows to {args.output_path}")


if __name__ == "__main__":
    main()
