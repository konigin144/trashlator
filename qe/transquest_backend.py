from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

from qe.base import QEBackend
from qe.result import QEResult


class TransQuestBackend(QEBackend):
    """
    ### NOT SUPPORTED ###

    POC backend for sentence-level QE using TransQuest.

    Notes:
    - This implementation uses lazy import so that TransQuest is optional.
    - The actual constructor/API of TransQuest may differ depending on package version.
    - Treat this as a safe adapter shell that can be adjusted once the exact model/API is confirmed.
    """

    def __init__(
        self,
        model_name: str,
        high_threshold: float = 0.7,
        medium_threshold: float = 0.4,
        backend_name: str = "transquest",
    ) -> None:
        self.model_name = model_name
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold
        self.backend_name = backend_name

        self._model = None
        self._load_error: Optional[str] = None

    def _load_model(self) -> None:
        if self._model is not None or self._load_error is not None:
            return

        try:
            # This import path may need adjustment depending on the exact TransQuest version.
            # Keeping it here, not globally, so users without TransQuest installed can still run trashlator.
            from transquest.algo.sentence_level.monotransquest.run_model import MonoTransQuestModel
        except Exception as exc:
            self._load_error = f"Failed to import TransQuest: {exc}"
            return

        try:
            # This constructor is a best-effort starting point for a POC.
            # You may need to tweak "model_type" or arguments depending on the package/model you use.
            self._model = MonoTransQuestModel(
                model_type="xlmroberta",
                model_name=self.model_name,
                use_cuda=False,
            )
        except Exception as exc:
            self._load_error = f"Failed to initialize TransQuest model '{self.model_name}': {exc}"

    def _label_from_score(self, score: float) -> str:
        if score >= self.high_threshold:
            return "high_confidence"
        if score >= self.medium_threshold:
            return "medium_confidence"
        return "low_confidence"

    def _predict_score(self, source_text: str, translated_text: str) -> float:
        """
        Raw call into TransQuest model.

        This may need adaptation once you verify the exact return structure of the chosen TransQuest model.
        """
        assert self._model is not None

        # Many sentence-level QE APIs expect [[source, target]]
        pairs = [[source_text, translated_text]]

        raw_output = self._model.predict(pairs)

        # Best-effort normalization for a few likely return shapes.
        if isinstance(raw_output, (list, tuple)):
            first = raw_output[0]
            if isinstance(first, (int, float)):
                return float(first)
            if isinstance(first, (list, tuple)) and first:
                return float(first[0])

        if isinstance(raw_output, (int, float)):
            return float(raw_output)

        raise ValueError(f"Unsupported TransQuest prediction output: {raw_output!r}")

    def score(self, source_text: str, translated_text: str) -> QEResult:
        if not source_text or not translated_text:
            return QEResult(
                score=None,
                label=None,
                error="Source or translated text is empty.",
                backend=self.backend_name,
                model_name=self.model_name,
            )

        self._load_model()

        if self._load_error is not None:
            return QEResult(
                score=None,
                label=None,
                error=self._load_error,
                backend=self.backend_name,
                model_name=self.model_name,
            )

        try:
            score = self._predict_score(source_text, translated_text)
            label = self._label_from_score(score)

            return QEResult(
                score=score,
                label=label,
                error=None,
                backend=self.backend_name,
                model_name=self.model_name,
            )
        except Exception as exc:
            return QEResult(
                score=None,
                label=None,
                error=f"TransQuest scoring failed: {exc}",
                backend=self.backend_name,
                model_name=self.model_name,
            )

    def score_batch(self, pairs: Iterable[Tuple[str, str]]) -> List[QEResult]:
        pairs = list(pairs)

        if not pairs:
            return []

        self._load_model()

        if self._load_error is not None:
            return [
                QEResult(
                    score=None,
                    label=None,
                    error=self._load_error,
                    backend=self.backend_name,
                    model_name=self.model_name,
                )
                for _ in pairs
            ]

        try:
            model_input = [[source, target] for source, target in pairs]
            raw_output = self._model.predict(model_input)  # type: ignore[union-attr]

            scores: List[float] = []
            if isinstance(raw_output, (list, tuple)):
                for item in raw_output:
                    if isinstance(item, (int, float)):
                        scores.append(float(item))
                    elif isinstance(item, (list, tuple)) and item:
                        scores.append(float(item[0]))
                    else:
                        raise ValueError(f"Unsupported batch item output: {item!r}")
            else:
                raise ValueError(f"Unsupported batch prediction output: {raw_output!r}")

            if len(scores) != len(pairs):
                raise ValueError(
                    f"Prediction count mismatch: got {len(scores)} scores for {len(pairs)} pairs."
                )

            return [
                QEResult(
                    score=score,
                    label=self._label_from_score(score),
                    error=None,
                    backend=self.backend_name,
                    model_name=self.model_name,
                )
                for score in scores
            ]

        except Exception as exc:
            return [
                QEResult(
                    score=None,
                    label=None,
                    error=f"TransQuest batch scoring failed: {exc}",
                    backend=self.backend_name,
                    model_name=self.model_name,
                )
                for _ in pairs
            ]