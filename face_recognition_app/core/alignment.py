"""Five-point face alignment for MobileFaceNet input."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


REFERENCE_LANDMARKS_112 = np.asarray(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def align_face(
    image: np.ndarray,
    landmarks: Sequence[Sequence[float]],
    output_size: Tuple[int, int] = (112, 112),
) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for face alignment") from exc

    source = np.asarray(landmarks, dtype=np.float32)
    if source.shape != (5, 2) or not np.all(np.isfinite(source)):
        raise ValueError("landmarks must be a finite 5x2 array")
    if output_size != (112, 112):
        scale = np.asarray(output_size, dtype=np.float32) / 112.0
        target = REFERENCE_LANDMARKS_112 * scale
    else:
        target = REFERENCE_LANDMARKS_112
    matrix, _ = cv2.estimateAffinePartial2D(source, target, method=cv2.LMEDS)
    if matrix is None:
        raise ValueError("unable to estimate face alignment transform")
    return cv2.warpAffine(
        np.asarray(image),
        matrix,
        output_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
