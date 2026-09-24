"""Qt worker and thread host for camera-driven recognition."""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Optional

from PyQt5.QtCore import QObject, QThread, Qt, pyqtSignal, pyqtSlot

from face_recognition_app.domain.runtime import AppState, WorkerErrorInfo
from face_recognition_app.hardware.camera import (
    Camera,
    CameraError,
    CameraReadError,
)
from face_recognition_app.inference.interfaces import InferenceError


_NO_SNAPSHOT = object()


class RecognitionWorker(QObject):
    preview_ready = pyqtSignal(object)
    recognition_ready = pyqtSignal(object)
    enrollment_progress = pyqtSignal(object)
    status_changed = pyqtSignal(object)
    error_raised = pyqtSignal(object)
    finished = pyqtSignal()

    def __init__(
        self,
        camera: Camera,
        pipeline: Any,
        inference_interval_ms: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__()
        if inference_interval_ms < 0:
            raise ValueError("inference_interval_ms must be non-negative")
        self._camera = camera
        self._pipeline = pipeline
        self._interval_seconds = inference_interval_ms / 1000.0
        self._clock = clock
        self._stop_event = threading.Event()
        self._control_lock = threading.RLock()
        self._mode = "recognition"
        self._enrollment_session: Optional[Any] = None
        self._pending_snapshot: Any = _NO_SNAPSHOT
        self._running = False

    @property
    def running(self) -> bool:
        with self._control_lock:
            return self._running

    def request_stop(self) -> None:
        self._stop_event.set()

    def begin_enrollment(self, session: Any) -> None:
        with self._control_lock:
            self._enrollment_session = session
            self._mode = "enrollment"

    def cancel_enrollment(self) -> None:
        with self._control_lock:
            self._enrollment_session = None
            self._mode = "recognition"

    def refresh_store(self, snapshot: Any) -> None:
        with self._control_lock:
            self._pending_snapshot = snapshot

    def _controls(self) -> Any:
        with self._control_lock:
            mode = self._mode
            session = self._enrollment_session
            snapshot = self._pending_snapshot
            self._pending_snapshot = _NO_SNAPSHOT
        return mode, session, snapshot

    @pyqtSlot()
    def run(self) -> None:
        with self._control_lock:
            if self._running:
                return
            self._running = True
        self._stop_event.clear()
        last_inference: Optional[float] = None
        try:
            self._camera.open()
            while not self._stop_event.is_set():
                frame = self._camera.read()
                self.preview_ready.emit(frame.copy())
                mode, session, snapshot = self._controls()
                if snapshot is not _NO_SNAPSHOT:
                    self._pipeline.refresh_store(snapshot)
                if mode == "enrollment" and session is not None:
                    self.status_changed.emit(AppState.ENROLLING)
                    self.enrollment_progress.emit(session.process(frame))
                    continue
                self.status_changed.emit(AppState.RECOGNIZING)
                now = self._clock()
                due = (
                    last_inference is None
                    or self._interval_seconds == 0.0
                    or now - last_inference >= self._interval_seconds
                )
                if due:
                    self.recognition_ready.emit(self._pipeline.process(frame))
                    last_inference = now
        except CameraReadError as exc:
            self._emit_error("camera_read", str(exc), True)
        except CameraError as exc:
            self._emit_error("camera_open", str(exc), True)
        except InferenceError as exc:
            self._emit_error("inference", str(exc), False)
        except Exception as exc:
            self._emit_error("worker", str(exc), False)
        finally:
            try:
                self._camera.release()
            except Exception as exc:
                self._emit_error("camera_release", str(exc), False)
            with self._control_lock:
                self._running = False
            self.finished.emit()

    def _emit_error(self, code: str, message: str, recoverable: bool) -> None:
        self.status_changed.emit(AppState.ERROR)
        self.error_raised.emit(WorkerErrorInfo(code, message, recoverable))


class WorkerThreadHost(QObject):
    preview_ready = pyqtSignal(object)
    recognition_ready = pyqtSignal(object)
    enrollment_progress = pyqtSignal(object)
    error_raised = pyqtSignal(object)
    finished = pyqtSignal()

    def __init__(self, worker_factory: Callable[[], RecognitionWorker]) -> None:
        super().__init__()
        self._worker_factory = worker_factory
        self._worker: Optional[RecognitionWorker] = None
        self._thread: Optional[QThread] = None
        self._pending_enrollment: Optional[Any] = None
        self._pending_snapshot: Any = _NO_SNAPSHOT

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.isRunning()

    def start(self) -> None:
        if self.running:
            return
        worker = self._worker_factory()
        if self._pending_enrollment is not None:
            worker.begin_enrollment(self._pending_enrollment)
        if self._pending_snapshot is not _NO_SNAPSHOT:
            worker.refresh_store(self._pending_snapshot)
            self._pending_snapshot = _NO_SNAPSHOT
        thread = QThread()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(thread.quit, type=Qt.DirectConnection)
        worker.preview_ready.connect(self.preview_ready.emit)
        worker.recognition_ready.connect(self.recognition_ready.emit)
        worker.enrollment_progress.connect(self.enrollment_progress.emit)
        worker.error_raised.connect(self.error_raised.emit)
        thread.finished.connect(self.finished.emit)
        thread.finished.connect(
            lambda finished_thread=thread: self._clear_finished(
                finished_thread
            )
        )
        self._worker = worker
        self._thread = thread
        thread.start()

    def stop(self, wait_ms: int = 2000) -> bool:
        worker = self._worker
        thread = self._thread
        if worker is None or thread is None:
            return True
        worker.request_stop()
        return bool(thread.wait(wait_ms))

    def begin_enrollment(self, session: Any) -> None:
        self._pending_enrollment = session
        if self._worker is not None:
            self._worker.begin_enrollment(session)

    def cancel_enrollment(self) -> None:
        self._pending_enrollment = None
        if self._worker is not None:
            self._worker.cancel_enrollment()

    def refresh_store(self, snapshot: Any) -> None:
        if self._worker is None:
            self._pending_snapshot = snapshot
        else:
            self._worker.refresh_store(snapshot)

    def _clear_finished(self, finished_thread: QThread) -> None:
        if finished_thread is not self._thread:
            return
        self._worker = None
        self._thread = None
