"""Build a stable member template from multiple face features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np

from face_recognition_app.core.matcher import FeatureError, normalize_rows, normalize_vector


class EnrollmentError(ValueError):
    """Raised when valid enrollment samples are insufficient."""


@dataclass(frozen=True)
class EnrollmentTemplate:
    embedding: np.ndarray
    accepted_samples: int
    rejected_samples: int


def _deduplicate(features: np.ndarray, duplicate_threshold: float) -> np.ndarray:
    kept: List[np.ndarray] = []
    for feature in features:
        if not kept:
            kept.append(feature)
            continue
        similarities = np.asarray(kept).dot(feature)
        if float(np.max(similarities)) < duplicate_threshold:
            kept.append(feature)
    return np.asarray(kept, dtype=np.float32)


def build_enrollment_template(
    features: Iterable[np.ndarray],
    minimum_samples: int = 5,
    duplicate_threshold: float = 0.995,
    center_threshold: float = 0.65,
) -> EnrollmentTemplate:
    samples = [np.asarray(feature, dtype=np.float32).reshape(-1) for feature in features]
    if len(samples) < minimum_samples:
        raise EnrollmentError("not enough enrollment samples")
    dimensions = {sample.shape[0] for sample in samples}
    if len(dimensions) != 1:
        raise FeatureError("all enrollment features must have the same dimension")

    normalized = normalize_rows(np.vstack(samples))
    unique = _deduplicate(normalized, duplicate_threshold)
    if unique.shape[0] < minimum_samples:
        raise EnrollmentError("not enough distinct enrollment samples")

    provisional_center = normalize_vector(np.mean(unique, axis=0))
    similarities = unique.dot(provisional_center)
    accepted = unique[similarities >= center_threshold]
    if accepted.shape[0] < minimum_samples:
        raise EnrollmentError("not enough consistent enrollment samples")

    template = normalize_vector(np.mean(accepted, axis=0))
    return EnrollmentTemplate(
        embedding=template,
        accepted_samples=int(accepted.shape[0]),
        rejected_samples=int(normalized.shape[0] - accepted.shape[0]),
    )
