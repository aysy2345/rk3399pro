"""Application state controller independent from concrete widgets."""

from __future__ import annotations

from typing import Any

from PyQt5.QtCore import QObject, pyqtSignal

from face_recognition_app.domain.runtime import AppState, WorkerErrorInfo


class AppController(QObject):
    state_changed = pyqtSignal(object)
    preview_ready = pyqtSignal(object)
    recognition_ready = pyqtSignal(object)
    enrollment_progress = pyqtSignal(object)
    error_raised = pyqtSignal(object)

    def __init__(self, worker_host: Any) -> None:
        super().__init__()
        self._host = worker_host
        self._state = AppState.IDLE
        self._resume_recognition = False
        self._last_error = None
        worker_host.preview_ready.connect(self.preview_ready.emit)
        worker_host.recognition_ready.connect(self.recognition_ready.emit)
        worker_host.enrollment_progress.connect(self.enrollment_progress.emit)
        worker_host.error_raised.connect(self._on_error)

    @property
    def state(self) -> AppState:
        return self._state

    @property
    def last_error(self) -> Any:
        return self._last_error

    def _set_state(self, state: AppState) -> None:
        if state == self._state:
            return
        self._state = state
        self.state_changed.emit(state)

    def start_recognition(self) -> None:
        if self._state in (AppState.RECOGNIZING, AppState.ENROLLING):
            return
        if self._state == AppState.ERROR:
            self._host.stop()
            self._host.cancel_enrollment()
        self._last_error = None
        self._host.start()
        self._set_state(AppState.RECOGNIZING)

    def stop_recognition(self) -> None:
        if self._state == AppState.IDLE:
            return
        self._host.stop()
        self._resume_recognition = False
        self._set_state(AppState.IDLE)

    def begin_enrollment(self, session: Any) -> None:
        if self._state == AppState.ENROLLING:
            return
        self._resume_recognition = self._state == AppState.RECOGNIZING
        if self._state == AppState.ERROR:
            self._host.stop()
        self._host.begin_enrollment(session)
        if not self._resume_recognition:
            self._host.start()
        self._set_state(AppState.ENROLLING)

    def cancel_enrollment(self) -> None:
        if self._state != AppState.ENROLLING:
            return
        self._host.cancel_enrollment()
        if self._resume_recognition:
            self._set_state(AppState.RECOGNIZING)
        else:
            self._host.stop()
            self._set_state(AppState.IDLE)
        self._resume_recognition = False

    def finish_enrollment(self) -> None:
        self.cancel_enrollment()

    def refresh_store(self, snapshot: Any) -> None:
        self._host.refresh_store(snapshot)

    def retry(self) -> None:
        error = self._last_error
        if self._state != AppState.ERROR or not error or not error.recoverable:
            return
        self._host.stop()
        self._host.cancel_enrollment()
        self._last_error = None
        self._host.start()
        self._set_state(AppState.RECOGNIZING)

    def close(self) -> None:
        self._host.stop()
        self._set_state(AppState.IDLE)

    def _on_error(self, error: WorkerErrorInfo) -> None:
        self._last_error = error
        self._set_state(AppState.ERROR)
        self.error_raised.emit(error)
