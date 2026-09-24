"""Recognition result models."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class MatchResult:
    member_id: Optional[str]
    name: str
    similarity: float
    is_known: bool

    @classmethod
    def unknown(cls, similarity: float = 0.0) -> "MatchResult":
        return cls(None, "陌生人", float(similarity), False)
