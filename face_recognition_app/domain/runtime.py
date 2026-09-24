"""Runtime value objects shared by workers and the user interface."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence, Tuple

import numpy as np


class AppState(Enum):
    IDLE = "idle"
    RECOGNIZING = "recognizing"
    ENROLLING = "enrolling"
    ERROR = "error"


@dataclass(frozen=True)
class FaceOverlay:
    track_id: str
    box: Tuple[float, float, float, float]
    member_id: Optional[str]
    name: str
    similarity: float
    is_known: bool


@dataclass(frozen=True)
class FrameResult:
    frame_bgr: np.ndarray
    faces: Sequence[FaceOverlay]

    def __post_init__(self) -> None:
        frame = np.asarray(self.frame_bgr)
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
            raise ValueError("frame_bgr must be a non-empty BGR image")
        copied = frame.copy()
        copied.setflags(write=False)
        object.__setattr__(self, "frame_bgr", copied)
        object.__setattr__(self, "faces", tuple(self.faces))


@dataclass(frozen=True)
class WorkerErrorInfo:
    code: str
    message: str
    recoverable: bool
