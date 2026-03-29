from __future__ import annotations

from typing import Optional

from qe.base import QEBackend
from qe.result import QEResult
from qe.transquest_backend import TransQuestBackend
from qe.comet_backend import CometBackend


class QEService:
    def __init__(self, backend: Optional[QEBackend]) -> None:
        self.backend = backend

    @classmethod
    def from_config(
        cls,
        enable_qe: bool,
        qe_backend: Optional[str],
        qe_model_name: Optional[str],
        qe_high_threshold: float = 0.7,
        qe_medium_threshold: float = 0.4,
    ) -> "QEService":
        if not enable_qe:
            return cls(backend=None)
        
        supported_backends = {"transquest", "comet"}

        if qe_backend not in supported_backends:
            raise ValueError(
                f"Unsupported QE backend '{qe_backend}'. "
                f"Supported: {sorted(supported_backends)}"
            )

        if qe_backend == "transquest":
            if not qe_model_name:
                raise ValueError("QE is enabled, but qe_model_name is missing.")

            return cls(
                backend=TransQuestBackend(
                    model_name=qe_model_name,
                    high_threshold=qe_high_threshold,
                    medium_threshold=qe_medium_threshold,
                )
            )
        
        if qe_backend == "comet":
            if not qe_model_name:
                raise ValueError("QE model name required for COMET backend.")

            return cls(
                backend=CometBackend(
                    model_name=qe_model_name,
                    high_threshold=qe_high_threshold,
                    medium_threshold=qe_medium_threshold,
                )
            )

    def score(self, source_text: str, translated_text: str) -> QEResult:
        if self.backend is None:
            return QEResult(
                score=None,
                label=None,
                error=None,
                backend=None,
                model_name=None,
            )
        return self.backend.score(source_text, translated_text)