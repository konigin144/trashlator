from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from qe.result import QEResult
from utils.qe_runner_cli import run_qe_csv


def test_run_qe_csv_scores_limited_rows_and_combines_columns(tmp_path: Path) -> None:
    source_path = tmp_path / "source.csv"
    translated_path = tmp_path / "translated.csv"
    output_path = tmp_path / "qe.csv"

    pd.DataFrame(
        {
            "id": [1, 2, 3],
            "message": ["hello", "pay now", "ignored"],
            "source_meta": ["a", "b", "c"],
        }
    ).to_csv(source_path, index=False, encoding="utf-8")
    pd.DataFrame(
        {
            "message": ["hallo", "jetzt zahlen", "ignoriert"],
            "translation_meta": ["x", "y", "z"],
        }
    ).to_csv(translated_path, index=False, encoding="utf-8")

    qe_service = SimpleNamespace(
        score=MagicMock(
            side_effect=[
                QEResult(
                    score=0.91,
                    label="high_confidence",
                    error=None,
                    backend="comet",
                    model_name="qe-model",
                ),
                QEResult(
                    score=0.52,
                    label="medium_confidence",
                    error=None,
                    backend="comet",
                    model_name="qe-model",
                ),
            ]
        )
    )

    with patch(
        "utils.qe_runner_cli.QEService.from_config",
        return_value=qe_service,
    ) as from_config:
        row_count, average_score = run_qe_csv(
            source_input_path=source_path,
            translated_input_path=translated_path,
            output_path=output_path,
            qe_model_name="qe-model",
            limit=2,
            input_encoding="utf-8",
            output_encoding="utf-8",
        )

    assert row_count == 2
    assert average_score == pytest.approx(0.715)
    from_config.assert_called_once_with(
        enable_qe=True,
        qe_backend="comet",
        qe_model_name="qe-model",
        qe_high_threshold=0.7,
        qe_medium_threshold=0.4,
    )
    assert qe_service.score.call_args_list[0].kwargs == {
        "source_text": "hello",
        "translated_text": "hallo",
    }
    assert qe_service.score.call_args_list[1].kwargs == {
        "source_text": "pay now",
        "translated_text": "jetzt zahlen",
    }

    output_df = pd.read_csv(output_path, encoding="utf-8")
    assert output_df.columns.tolist() == [
        "id",
        "message_source",
        "source_meta",
        "message_translated",
        "translation_meta",
        "qe_score",
        "qe_label",
        "qe_error",
        "qe_backend",
        "qe_model_name",
    ]
    assert output_df["message_source"].tolist() == ["hello", "pay now"]
    assert output_df["message_translated"].tolist() == ["hallo", "jetzt zahlen"]
    assert output_df["qe_label"].tolist() == [
        "high_confidence",
        "medium_confidence",
    ]


def test_run_qe_csv_supports_custom_text_columns(tmp_path: Path) -> None:
    source_path = tmp_path / "source.csv"
    translated_path = tmp_path / "translated.csv"
    output_path = tmp_path / "qe.csv"

    pd.DataFrame({"src": ["reset password"]}).to_csv(source_path, index=False)
    pd.DataFrame({"mt": ["passwort zurucksetzen"]}).to_csv(
        translated_path,
        index=False,
    )

    qe_service = SimpleNamespace(
        score=MagicMock(
            return_value=QEResult(
                score=0.8,
                label="high_confidence",
                backend="comet",
                model_name="qe-model",
            )
        )
    )

    with patch(
        "utils.qe_runner_cli.QEService.from_config",
        return_value=qe_service,
    ):
        row_count, average_score = run_qe_csv(
            source_input_path=source_path,
            translated_input_path=translated_path,
            output_path=output_path,
            source_column="src",
            translated_column="mt",
            qe_model_name="qe-model",
        )

    assert row_count == 1
    assert average_score == pytest.approx(0.8)
    qe_service.score.assert_called_once_with(
        source_text="reset password",
        translated_text="passwort zurucksetzen",
    )


def test_run_qe_csv_supports_single_input_file_and_skips_non_ok_statuses(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "combined.csv"
    output_path = tmp_path / "qe.csv"

    pd.DataFrame(
        {
            "message": ["hello", "pay now", "broken placeholder"],
            "translated_text": ["hallo", "jetzt zahlen", "kaputt"],
            "status": ["ok", "ok_chunked", "placeholder_mismatch"],
            "label": [0, 1, 1],
        }
    ).to_csv(input_path, index=False, encoding="utf-8")

    qe_service = SimpleNamespace(
        score=MagicMock(
            side_effect=[
                QEResult(
                    score=0.9,
                    label="high_confidence",
                    backend="comet",
                    model_name="qe-model",
                ),
                QEResult(
                    score=0.6,
                    label="medium_confidence",
                    backend="comet",
                    model_name="qe-model",
                ),
            ]
        )
    )

    with patch(
        "utils.qe_runner_cli.QEService.from_config",
        return_value=qe_service,
    ):
        row_count, average_score = run_qe_csv(
            combined_input_path=input_path,
            output_path=output_path,
            source_column="message",
            translated_column="translated_text",
            qe_model_name="qe-model",
            input_encoding="utf-8",
            output_encoding="utf-8",
        )

    assert row_count == 3
    assert average_score == pytest.approx(0.75)
    assert qe_service.score.call_count == 2
    assert qe_service.score.call_args_list[0].kwargs == {
        "source_text": "hello",
        "translated_text": "hallo",
    }
    assert qe_service.score.call_args_list[1].kwargs == {
        "source_text": "pay now",
        "translated_text": "jetzt zahlen",
    }

    output_df = pd.read_csv(output_path, encoding="utf-8")
    assert output_df.columns.tolist() == [
        "message",
        "translated_text",
        "status",
        "label",
        "qe_score",
        "qe_label",
        "qe_error",
        "qe_backend",
        "qe_model_name",
    ]
    assert output_df["qe_score"].tolist()[:2] == [0.9, 0.6]
    assert pd.isna(output_df.loc[2, "qe_score"])
    assert output_df.loc[2, "qe_error"] == (
        "QE skipped: translation status is 'placeholder_mismatch'."
    )


def test_run_qe_csv_does_not_load_qe_when_single_input_statuses_are_all_skipped(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "combined.csv"
    output_path = tmp_path / "qe.csv"

    pd.DataFrame(
        {
            "message": ["hello"],
            "translated_text": ["hallo"],
            "status": ["translation_error"],
        }
    ).to_csv(input_path, index=False, encoding="utf-8")

    with patch("utils.qe_runner_cli.QEService.from_config") as from_config:
        row_count, average_score = run_qe_csv(
            combined_input_path=input_path,
            output_path=output_path,
            source_column="message",
            translated_column="translated_text",
            qe_model_name="qe-model",
            input_encoding="utf-8",
            output_encoding="utf-8",
        )

    assert row_count == 1
    assert average_score is None
    from_config.assert_not_called()

    output_df = pd.read_csv(output_path, encoding="utf-8")
    assert output_df.loc[0, "qe_error"] == (
        "QE skipped: translation status is 'translation_error'."
    )


def test_run_qe_csv_rejects_mismatched_row_counts(tmp_path: Path) -> None:
    source_path = tmp_path / "source.csv"
    translated_path = tmp_path / "translated.csv"

    pd.DataFrame({"message": ["one", "two"]}).to_csv(source_path, index=False)
    pd.DataFrame({"message": ["eins"]}).to_csv(translated_path, index=False)

    with pytest.raises(ValueError, match="same number of paired rows"):
        run_qe_csv(
            source_input_path=source_path,
            translated_input_path=translated_path,
            output_path=tmp_path / "qe.csv",
            qe_model_name="qe-model",
        )


def test_run_qe_csv_requires_positive_limit(tmp_path: Path) -> None:
    source_path = tmp_path / "source.csv"
    translated_path = tmp_path / "translated.csv"
    source_path.write_text("message\nhello\n", encoding="utf-8")
    translated_path.write_text("message\nhallo\n", encoding="utf-8")

    with pytest.raises(ValueError, match="positive integer"):
        run_qe_csv(
            source_input_path=source_path,
            translated_input_path=translated_path,
            output_path=tmp_path / "qe.csv",
            qe_model_name="qe-model",
            limit=0,
        )


def test_run_qe_csv_rejects_mixed_single_and_paired_inputs(tmp_path: Path) -> None:
    combined_path = tmp_path / "combined.csv"
    source_path = tmp_path / "source.csv"
    translated_path = tmp_path / "translated.csv"
    combined_path.write_text("message,translated_text\nhello,hallo\n", encoding="utf-8")
    source_path.write_text("message\nhello\n", encoding="utf-8")
    translated_path.write_text("message\nhallo\n", encoding="utf-8")

    with pytest.raises(ValueError, match="either --input"):
        run_qe_csv(
            combined_input_path=combined_path,
            source_input_path=source_path,
            translated_input_path=translated_path,
            output_path=tmp_path / "qe.csv",
            source_column="message",
            translated_column="translated_text",
            qe_model_name="qe-model",
        )
