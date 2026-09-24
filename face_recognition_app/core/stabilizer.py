"""Multi-frame voting to reduce identity flicker."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from typing import Deque, Dict, Optional


class IdentityStabilizer:
    def __init__(self, window_size: int = 5, votes_required: int = 3) -> None:
        if window_size < 1:
            raise ValueError("window_size must be positive")
        if votes_required < 1 or votes_required > window_size:
            raise ValueError("votes_required must be within the window")
        self._window_size = window_size
        self._votes_required = votes_required
        self._history: Dict[str, Deque[Optional[str]]] = defaultdict(
            lambda: deque(maxlen=self._window_size)
        )

    def update(self, track_id: str, member_id: Optional[str]) -> Optional[str]:
        history = self._history[track_id]
        history.append(member_id)
        known_votes = Counter(value for value in history if value is not None)
        if not known_votes:
            return None
        candidate, votes = known_votes.most_common(1)[0]
        if votes >= self._votes_required:
            return candidate
        return None

    def clear(self, track_id: Optional[str] = None) -> None:
        if track_id is None:
            self._history.clear()
        else:
            self._history.pop(track_id, None)
