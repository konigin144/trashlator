from __future__ import annotations

from typing import List

from qe.base import QEBackend
from qe.result import QEResult


class CometBackend(QEBackend):
    def __init__(
        self,
        model_name: str = "Unbabel/wmt22-cometkiwi-da",
        high_threshold: float = 0.7,
        medium_threshold: float = 0.4,
    ) -> None:
        self.model_name = model_name
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold

        try:
            from comet import download_model, load_from_checkpoint

            model_path = download_model(model_name)
            self.model = load_from_checkpoint(model_path)
        except Exception as exc:
            self.model = None
            self._load_error = str(exc)

    def _label_from_score(self, score: float) -> str:
        if score >= self.high_threshold:
            return "high_confidence"
        if score >= self.medium_threshold:
            return "medium_confidence"
        return "low_confidence"

    def score(self, source_text: str, translated_text: str) -> QEResult:
        if self.model is None:
            return QEResult(
                score=None,
                label=None,
                error=f"COMET load failed: {getattr(self, '_load_error', None)}",
                backend="comet",
                model_name=self.model_name,
            )

        try:
            data = [
                {
                    "src": source_text,
                    "mt": translated_text,
                }
            ]

            output = self.model.predict(data, batch_size=1)

            # COMET zwraca listę scores
            score = float(output["scores"][0])

            return QEResult(
                score=score,
                label=self._label_from_score(score),
                error=None,
                backend="comet",
                model_name=self.model_name,
            )

        except Exception as exc:
            return QEResult(
                score=None,
                label=None,
                error=str(exc),
                backend="comet",
                model_name=self.model_name,
            )