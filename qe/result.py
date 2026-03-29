from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class QEResult:
    score: Optional[float]
    label: Optional[str]
    error: Optional[str] = None
    backend: Optional[str] = None
    model_name: Optional[str] = None