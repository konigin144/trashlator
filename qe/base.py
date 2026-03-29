from abc import ABC, abstractmethod
from typing import Iterable, List, Tuple

from qe.result import QEResult


class QEBackend(ABC):
    @abstractmethod
    def score(self, source_text: str, translated_text: str) -> QEResult:
        """
        Score a single source/translation pair.
        """
        raise NotImplementedError

    def score_batch(self, pairs: Iterable[Tuple[str, str]]) -> List[QEResult]:
        """
        Default batch scoring implementation using single-item scoring.
        Backends can override this for better performance.
        """
        return [self.score(source, target) for source, target in pairs]