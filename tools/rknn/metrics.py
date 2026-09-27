"""Pure NumPy metrics for ONNX and RKNN output comparison."""

from itertools import product
from math import ceil

import numpy as np


class MetricError(ValueError):
    """Raised when model output cannot be compared safely."""


def _finite_array(value, name):
    array = np.asarray(value, dtype=np.float32)
    if not np.all(np.isfinite(array)):
        raise MetricError("{} must contain finite values".format(name))
    return array


def cosine_similarity(left, right):
    left_vector = _finite_array(left, "left vector").reshape(-1).astype(np.float64)
    right_vector = _finite_array(right, "right vector").reshape(-1).astype(np.float64)
    if left_vector.size != right_vector.size:
        raise MetricError("vectors must have the same size")
    denominator = np.linalg.norm(left_vector) * np.linalg.norm(right_vector)
    if denominator <= 1e-12:
        raise MetricError("cannot compare a zero-length vector")
    return float(np.dot(left_vector, right_vector) / denominator)


def compare_mobilefacenet(reference, candidate, minimum_cosine=0.99):
    reference_vector = _finite_array(reference, "reference embedding").reshape(-1)
    candidate_vector = _finite_array(candidate, "candidate embedding").reshape(-1)
    if reference_vector.size != 512 or candidate_vector.size != 512:
        raise MetricError("MobileFaceNet embeddings must contain 512 values")
    if not 0.0 <= minimum_cosine <= 1.0:
        raise MetricError("minimum_cosine must be between 0 and 1")
    similarity = cosine_similarity(reference_vector, candidate_vector)
    return {
        "embedding_size": 512,
        "cosine_similarity": similarity,
        "minimum_cosine": float(minimum_cosine),
        "passed": similarity >= minimum_cosine,
    }


def map_retinaface_outputs(outputs):
    mapped = {}
    for index, output in enumerate(outputs):
        array = _finite_array(output, "RetinaFace output {}".format(index))
        if array.ndim == 3 and array.shape[0] == 1:
            array = array[0]
        if array.ndim != 2:
            raise MetricError(
                "RetinaFace outputs must have shape [1,N,C] or [N,C]"
            )
        width = array.shape[1]
        if width in mapped:
            raise MetricError("RetinaFace output dimensions must be unique")
        mapped[width] = array
    if not all(width in mapped for width in (4, 2, 10)):
        raise MetricError("RetinaFace outputs must end in 4, 2 and 10 values")
    boxes, scores, landmarks = mapped[4], mapped[2], mapped[10]
    if not (len(boxes) == len(scores) == len(landmarks)):
        raise MetricError("RetinaFace outputs must have the same anchor count")
    return boxes, scores, landmarks


def _generate_priors(height, width):
    min_sizes = ((16, 32), (64, 128), (256, 512))
    steps = (8, 16, 32)
    anchors = []
    for sizes, step in zip(min_sizes, steps):
        feature_h = int(ceil(float(height) / step))
        feature_w = int(ceil(float(width) / step))
        for row, col in product(range(feature_h), range(feature_w)):
            for size in sizes:
                anchors.append(
                    (
                        (col + 0.5) * step / width,
                        (row + 0.5) * step / height,
                        float(size) / width,
                        float(size) / height,
                    )
                )
    return np.asarray(anchors, dtype=np.float32)


def _decode_boxes(locations, priors):
    centers = priors[:, :2] + locations[:, :2] * 0.1 * priors[:, 2:]
    sizes = priors[:, 2:] * np.exp(locations[:, 2:] * 0.2)
    return np.concatenate((centers - sizes / 2, centers + sizes / 2), axis=1)


def _decode_landmarks(values, priors):
    points = values.reshape((-1, 5, 2))
    centers = priors[:, np.newaxis, :2]
    sizes = priors[:, np.newaxis, 2:]
    return centers + points * 0.1 * sizes


def _nms(boxes, scores, threshold):
    if boxes.size == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[current], x1[rest])
        yy1 = np.maximum(y1[current], y1[rest])
        xx2 = np.minimum(x2[current], x2[rest])
        yy2 = np.minimum(y2[current], y2[rest])
        intersection = np.maximum(0.0, xx2 - xx1) * np.maximum(
            0.0, yy2 - yy1
        )
        union = areas[current] + areas[rest] - intersection
        overlap = intersection / np.maximum(union, 1e-12)
        order = rest[overlap <= threshold]
    return keep


def decode_retinaface(
    outputs,
    source_shape,
    input_size=(640, 640),
    confidence_threshold=0.8,
    nms_threshold=0.4,
    top_k=5000,
    keep_top_k=750,
):
    locations, confidences, landmark_values = map_retinaface_outputs(outputs)
    width, height = input_size
    priors = _generate_priors(height, width)
    if len(locations) != len(priors):
        raise MetricError("RetinaFace output count does not match priors")
    source_height, source_width = source_shape
    scores = confidences[:, 1]
    selected = np.flatnonzero(scores >= confidence_threshold)
    if not selected.size:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0, 5, 2), dtype=np.float32),
        )
    order = selected[np.argsort(scores[selected])[::-1]][:top_k]
    boxes = _decode_boxes(locations[order], priors[order])
    landmarks = _decode_landmarks(landmark_values[order], priors[order])
    kept = _nms(boxes, scores[order], nms_threshold)[:keep_top_k]
    boxes = boxes[kept]
    landmarks = landmarks[kept]
    box_scale = np.asarray(
        (source_width, source_height, source_width, source_height),
        dtype=np.float32,
    )
    point_scale = np.asarray((source_width, source_height), dtype=np.float32)
    boxes = boxes * box_scale
    boxes = np.clip(
        boxes,
        (0.0, 0.0, 0.0, 0.0),
        (
            float(source_width - 1),
            float(source_height - 1),
            float(source_width - 1),
            float(source_height - 1),
        ),
    )
    landmarks = landmarks * point_scale
    landmarks[:, :, 0] = np.clip(landmarks[:, :, 0], 0, source_width - 1)
    landmarks[:, :, 1] = np.clip(landmarks[:, :, 1], 0, source_height - 1)
    valid = np.logical_and(boxes[:, 2] > boxes[:, 0], boxes[:, 3] > boxes[:, 1])
    return boxes[valid].astype(np.float32), landmarks[valid].astype(np.float32)


def _validate_detections(boxes, landmarks, label):
    boxes = _finite_array(boxes, "{} boxes".format(label))
    landmarks = _finite_array(landmarks, "{} landmarks".format(label))
    if boxes.ndim != 2 or boxes.shape[1:] != (4,):
        raise MetricError("{} boxes must have shape [N,4]".format(label))
    if landmarks.ndim != 3 or landmarks.shape[1:] != (5, 2):
        raise MetricError("{} landmarks must have shape [N,5,2]".format(label))
    if len(boxes) != len(landmarks):
        raise MetricError("{} detection arrays must have equal length".format(label))
    return boxes, landmarks


def _box_iou(left, right):
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, float(left[2] - left[0])) * max(
        0.0, float(left[3] - left[1])
    )
    right_area = max(0.0, float(right[2] - right[0])) * max(
        0.0, float(right[3] - right[1])
    )
    return intersection / max(left_area + right_area - intersection, 1e-12)


def compare_retinaface_detections(
    reference_boxes,
    reference_landmarks,
    candidate_boxes,
    candidate_landmarks,
    box_tolerance=3.0,
    landmark_tolerance=3.0,
):
    reference_boxes, reference_landmarks = _validate_detections(
        reference_boxes, reference_landmarks, "reference"
    )
    candidate_boxes, candidate_landmarks = _validate_detections(
        candidate_boxes, candidate_landmarks, "candidate"
    )
    if box_tolerance < 0.0 or landmark_tolerance < 0.0:
        raise MetricError("detection tolerances must not be negative")
    remaining = set(range(len(candidate_boxes)))
    matches = []
    for reference_index, reference_box in enumerate(reference_boxes):
        if not remaining:
            break
        candidate_index = max(
            remaining,
            key=lambda index: _box_iou(reference_box, candidate_boxes[index]),
        )
        remaining.remove(candidate_index)
        matches.append((reference_index, candidate_index))
    if matches:
        box_error = max(
            float(
                np.max(
                    np.abs(reference_boxes[left] - candidate_boxes[right])
                )
            )
            for left, right in matches
        )
        landmark_error = max(
            float(
                np.max(
                    np.abs(
                        reference_landmarks[left] - candidate_landmarks[right]
                    )
                )
            )
            for left, right in matches
        )
    else:
        box_error = 0.0
        landmark_error = 0.0
    counts_match = len(reference_boxes) == len(candidate_boxes)
    return {
        "reference_count": int(len(reference_boxes)),
        "candidate_count": int(len(candidate_boxes)),
        "matched_count": int(len(matches)),
        "max_box_error": box_error,
        "max_landmark_error": landmark_error,
        "box_tolerance": float(box_tolerance),
        "landmark_tolerance": float(landmark_tolerance),
        "passed": (
            counts_match
            and len(matches) == len(reference_boxes)
            and box_error <= box_tolerance
            and landmark_error <= landmark_tolerance
        ),
    }
