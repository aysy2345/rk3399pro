"""RKNN Lite inference backends for RK3399Pro.

The RKNN models already contain their mean/scale preprocessing settings, so
the runtime receives resized BGR images in HWC layout.  Detection decoding and
embedding validation stay shared with the ONNX implementations.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Optional, Sequence

import cv2
import numpy as np

from face_recognition_app.inference.interfaces import InferenceError
from face_recognition_app.inference.onnx_backend import (
    OnnxMobileFaceNetEmbedder,
    OnnxRetinaFaceDetector,
    _validate_bgr_image,
)


RuntimeFactory = Callable[[], Any]


def _default_runtime_factory() -> Any:
    try:
        from rknnlite.api import RKNNLite
    except ImportError as exc:
        raise InferenceError(
            "RKNN Lite is required when runtime.backend is 'rknn'"
        ) from exc
    return RKNNLite()


class _RknnLiteSession:
    """Small ONNX-session-compatible adapter around RKNNLite."""

    def __init__(
        self,
        model_path: Path,
        output_count: int,
        runtime_factory: Optional[RuntimeFactory] = None,
    ) -> None:
        try:
            self._runtime = (runtime_factory or _default_runtime_factory)()
        except InferenceError:
            raise
        except Exception as exc:
            raise InferenceError("RKNN Lite runtime creation failed") from exc
        self._released = False
        self._inputs = [SimpleNamespace(name="input")]
        self._outputs = [
            SimpleNamespace(name="output_{}".format(index))
            for index in range(output_count)
        ]
        try:
            self._require_success(
                "load_rknn", self._runtime.load_rknn(str(Path(model_path)))
            )
            self._require_success("init_runtime", self._runtime.init_runtime())
        except InferenceError:
            self.release()
            raise
        except Exception as exc:
            self.release()
            raise InferenceError("RKNN Lite initialization failed") from exc

    @staticmethod
    def _require_success(stage: str, result: Any) -> None:
        if result not in (None, 0):
            raise InferenceError(
                "RKNN Lite {} failed with code {}".format(stage, result)
            )

    def get_inputs(self) -> Sequence[Any]:
        return self._inputs

    def get_outputs(self) -> Sequence[Any]:
        return self._outputs

    def run(
        self, output_names: Sequence[str], feed: Mapping[str, np.ndarray]
    ) -> Sequence[np.ndarray]:
        del output_names
        if self._released:
            raise InferenceError("RKNN Lite runtime has been released")
        if len(feed) != 1:
            raise InferenceError("RKNN Lite expects exactly one input")
        input_value = next(iter(feed.values()))
        try:
            outputs = self._runtime.inference(inputs=[input_value])
        except Exception as exc:
            raise InferenceError("RKNN Lite inference failed") from exc
        if not outputs:
            raise InferenceError("RKNN Lite returned no outputs")
        return outputs

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        release = getattr(self._runtime, "release", None)
        if callable(release):
            release()


class RknnMobileFaceNetEmbedder(OnnxMobileFaceNetEmbedder):
    """MobileFaceNet embedder backed by RKNN Lite."""

    def __init__(
        self,
        model_path: Path,
        runtime_factory: Optional[RuntimeFactory] = None,
        input_size=(112, 112),
    ) -> None:
        session = _RknnLiteSession(model_path, 1, runtime_factory)
        super().__init__(session=session, input_size=input_size)
        self._rknn_session = session

    def preprocess(self, aligned_face_bgr: np.ndarray) -> np.ndarray:
        _validate_bgr_image(aligned_face_bgr, "aligned face")
        return cv2.resize(
            aligned_face_bgr, self._input_size, interpolation=cv2.INTER_LINEAR
        )

    def release(self) -> None:
        self._rknn_session.release()


class RknnRetinaFaceDetector(OnnxRetinaFaceDetector):
    """RetinaFace detector backed by RKNN Lite."""

    def __init__(
        self,
        model_path: Path,
        runtime_factory: Optional[RuntimeFactory] = None,
        input_size=(640, 640),
        confidence_threshold: float = 0.8,
        nms_threshold: float = 0.4,
        top_k: int = 5000,
        keep_top_k: int = 750,
    ) -> None:
        session = _RknnLiteSession(model_path, 3, runtime_factory)
        super().__init__(
            session=session,
            input_size=input_size,
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
            top_k=top_k,
            keep_top_k=keep_top_k,
        )
        self._rknn_session = session

    def preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        _validate_bgr_image(frame_bgr, "frame")
        return cv2.resize(
            frame_bgr, self._input_size, interpolation=cv2.INTER_LINEAR
        )

    def release(self) -> None:
        self._rknn_session.release()
