"""Cosine-similarity matching for a small local face database."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from face_recognition_app.domain.member import Member
from face_recognition_app.domain.recognition import MatchResult


class FeatureError(ValueError):
    """Raised when a face feature is malformed."""


def normalize_vector(feature: np.ndarray) -> np.ndarray:
    vector = np.asarray(feature, dtype=np.float32).reshape(-1)
    if vector.size == 0 or not np.all(np.isfinite(vector)):
        raise FeatureError("feature must contain finite values")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise FeatureError("feature norm must be greater than zero")
    return vector / norm


def normalize_rows(features: np.ndarray) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float32)
    if matrix.ndim != 2:
        raise FeatureError("features must be a two-dimensional matrix")
    if matrix.shape[0] == 0:
        return matrix.copy()
    if not np.all(np.isfinite(matrix)):
        raise FeatureError("features must contain finite values")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise FeatureError("every feature must have a non-zero norm")
    return matrix / norms


class FaceMatcher:
    def __init__(
        self,
        members: Sequence[Member],
        embeddings: np.ndarray,
        threshold: float,
    ) -> None:
        if threshold < -1.0 or threshold > 1.0:
            raise ValueError("threshold must be between -1 and 1")
        self._members = tuple(members)
        self._embeddings = normalize_rows(embeddings)
        self._threshold = float(threshold)
        if len(self._members) != self._embeddings.shape[0]:
            raise ValueError("member count must equal embedding row count")

    def match(self, feature: np.ndarray) -> MatchResult:
        if not self._members:
            return MatchResult.unknown()
        query = normalize_vector(feature)
        if query.shape[0] != self._embeddings.shape[1]:
            raise FeatureError(
                "feature dimension {} does not match database dimension {}".format(
                    query.shape[0], self._embeddings.shape[1]
                )
            )
        similarities = self._embeddings.dot(query)
        index = int(np.argmax(similarities))
        similarity = float(similarities[index])
        if similarity < self._threshold or not self._members[index].active:
            return MatchResult.unknown(similarity)
        member = self._members[index]
        return MatchResult(
            member_id=member.member_id,
            name=member.name,
            similarity=similarity,
            is_known=True,
        )
