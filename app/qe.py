from __future__ import annotations

from typing import Any, Dict

from qe.result import QEResult
from qe.service import QEService


def build_qe_columns(
    qe_service: QEService,
    status: str,
    source_text: str,
    translated_text: str | None,
) -> Dict[str, Any]:
    """
    Build QE-related output columns for a single record.

    Rules:
    - QE is executed only for successfully translated records.
    - For non-ok statuses, QE columns are returned as empty metadata.
    - If translated_text is empty, QE is skipped safely.
    """
    if status not in {"ok", "ok_chunked"}:
        return {
            "qe_score": None,
            "qe_label": None,
            "qe_error": None,
            "qe_backend": None,
            "qe_model_name": None,
        }

    if not translated_text:
        return {
            "qe_score": None,
            "qe_label": None,
            "qe_error": "QE skipped: translated_text is empty.",
            "qe_backend": None,
            "qe_model_name": None,
        }

    result: QEResult = qe_service.score(
        source_text=source_text,
        translated_text=translated_text,
    )

    return {
        "qe_score": result.score,
        "qe_label": result.label,
        "qe_error": result.error,
        "qe_backend": result.backend,
        "qe_model_name": result.model_name,
    }