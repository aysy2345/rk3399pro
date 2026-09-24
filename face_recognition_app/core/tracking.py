"""Small IoU tracker used only to stabilize identities across nearby frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


Box = Tuple[float, float, float, float]


def intersection_over_union(first: Box, second: Box) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(
        0.0, first[3] - first[1]
    )
    second_area = max(0.0, second[2] - second[0]) * max(
        0.0, second[3] - second[1]
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


@dataclass
class _Track:
    box: Box
    missed: int = 0


class IoUTracker:
    def __init__(self, iou_threshold: float = 0.3, max_missed: int = 2) -> None:
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between 0 and 1")
        if max_missed < 0:
            raise ValueError("max_missed must be non-negative")
        self._iou_threshold = iou_threshold
        self._max_missed = max_missed
        self._tracks: Dict[str, _Track] = {}
        self._next_id = 1

    def update(self, boxes: Sequence[Box]) -> List[str]:
        normalized = [tuple(float(value) for value in box) for box in boxes]
        assignments: Dict[int, str] = {}
        used_tracks = set()
        candidates = []
        for index, box in enumerate(normalized):
            for track_id, track in self._tracks.items():
                score = intersection_over_union(box, track.box)
                if score >= self._iou_threshold:
                    candidates.append((score, index, track_id))
        for _, index, track_id in sorted(candidates, reverse=True):
            if index in assignments or track_id in used_tracks:
                continue
            assignments[index] = track_id
            used_tracks.add(track_id)

        for track_id in list(self._tracks):
            if track_id in used_tracks:
                index = next(
                    item for item, assigned in assignments.items()
                    if assigned == track_id
                )
                self._tracks[track_id] = _Track(normalized[index], 0)
            else:
                track = self._tracks[track_id]
                track.missed += 1
                if track.missed > self._max_missed:
                    del self._tracks[track_id]

        for index, box in enumerate(normalized):
            if index in assignments:
                continue
            track_id = "face-{}".format(self._next_id)
            self._next_id += 1
            self._tracks[track_id] = _Track(box, 0)
            assignments[index] = track_id
        return [assignments[index] for index in range(len(normalized))]

    def clear(self) -> None:
        self._tracks.clear()
