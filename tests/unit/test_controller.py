from PyQt5.QtCore import QObject, pyqtSignal

from face_recognition_app.app.controller import AppController
from face_recognition_app.domain.runtime import AppState, WorkerErrorInfo


class StubHost(QObject):
    preview_ready = pyqtSignal(object)
    recognition_ready = pyqtSignal(object)
    enrollment_progress = pyqtSignal(object)
    error_raised = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.start_calls = 0
        self.stop_calls = 0
        self.begin_calls = []
        self.cancel_calls = 0
        self.refresh_calls = []

    def start(self):
        self.start_calls += 1

    def stop(self, wait_ms=2000):
        self.stop_calls += 1
        return True

    def begin_enrollment(self, session):
        self.begin_calls.append(session)

    def cancel_enrollment(self):
        self.cancel_calls += 1

    def refresh_store(self, snapshot):
        self.refresh_calls.append(snapshot)


def test_controller_recognition_start_stop_are_idempotent(qtbot):
    host = StubHost()
    controller = AppController(host)
    states = []
    controller.state_changed.connect(states.append)

    controller.start_recognition()
    controller.start_recognition()
    controller.stop_recognition()
    controller.stop_recognition()

    assert host.start_calls == 1
    assert host.stop_calls == 1
    assert states == [AppState.RECOGNIZING, AppState.IDLE]
    assert controller.state == AppState.IDLE


def test_controller_restores_recognition_after_enrollment(qtbot):
    host = StubHost()
    controller = AppController(host)
    session = object()
    controller.start_recognition()

    controller.begin_enrollment(session)
    controller.cancel_enrollment()

    assert host.begin_calls == [session]
    assert host.cancel_calls == 1
    assert controller.state == AppState.RECOGNIZING


def test_controller_stops_idle_enrollment_on_cancel(qtbot):
    host = StubHost()
    controller = AppController(host)

    controller.begin_enrollment(object())
    controller.cancel_enrollment()

    assert host.start_calls == 1
    assert host.cancel_calls == 1
    assert host.stop_calls == 1
    assert controller.state == AppState.IDLE


def test_controller_enters_error_and_retry_restarts_host(qtbot):
    host = StubHost()
    controller = AppController(host)
    errors = []
    controller.error_raised.connect(errors.append)
    controller.start_recognition()

    error = WorkerErrorInfo("camera_read", "摄像头断开", True)
    host.error_raised.emit(error)
    controller.retry()

    assert errors == [error]
    assert host.stop_calls == 1
    assert host.start_calls == 2
    assert controller.state == AppState.RECOGNIZING


def test_controller_forwards_store_refresh():
    host = StubHost()
    controller = AppController(host)
    snapshot = object()

    controller.refresh_store(snapshot)

    assert host.refresh_calls == [snapshot]
