import numpy as np

from face_recognition_app.core.enrollment_session import EnrollmentProgress
from face_recognition_app.domain.runtime import FrameResult
from face_recognition_app.hardware.camera import CameraReadError
from face_recognition_app.workers.recognition_worker import (
    RecognitionWorker,
    WorkerThreadHost,
)


class FakeCamera:
    def __init__(self, frames=None, error=None):
        self.frames = list(frames or [])
        self.error = error
        self.open_calls = 0
        self.release_calls = 0

    @property
    def is_open(self):
        return self.open_calls > self.release_calls

    def open(self):
        self.open_calls += 1

    def read(self):
        if self.error:
            raise self.error
        return self.frames.pop(0).copy()

    def release(self):
        self.release_calls += 1


class StubPipeline:
    def __init__(self):
        self.calls = 0
        self.snapshots = []

    def process(self, frame):
        self.calls += 1
        return FrameResult(frame, ())

    def refresh_store(self, snapshot):
        self.snapshots.append(snapshot)


class StubEnrollment:
    def __init__(self):
        self.calls = 0

    def process(self, frame):
        self.calls += 1
        return EnrollmentProgress(True, "有效样本", 1, 15, "front", False)


def frame():
    return np.zeros((16, 16, 3), dtype=np.uint8)


def test_worker_processes_current_frame_and_releases_camera():
    camera = FakeCamera([frame()])
    pipeline = StubPipeline()
    worker = RecognitionWorker(camera, pipeline, inference_interval_ms=0)
    previews = []
    results = []
    worker.preview_ready.connect(
        lambda value: (previews.append(value), worker.request_stop())
    )
    worker.recognition_ready.connect(results.append)

    worker.run()

    assert len(previews) == 1
    assert len(results) == 1
    assert pipeline.calls == 1
    assert camera.open_calls == 1
    assert camera.release_calls == 1


def test_worker_enrollment_mode_skips_recognition_pipeline():
    camera = FakeCamera([frame()])
    pipeline = StubPipeline()
    enrollment = StubEnrollment()
    worker = RecognitionWorker(camera, pipeline, inference_interval_ms=0)
    progress = []
    worker.begin_enrollment(enrollment)
    worker.preview_ready.connect(lambda _: worker.request_stop())
    worker.enrollment_progress.connect(progress.append)

    worker.run()

    assert enrollment.calls == 1
    assert pipeline.calls == 0
    assert len(progress) == 1


def test_worker_turns_camera_error_into_structured_signal():
    camera = FakeCamera(error=CameraReadError("disconnected"))
    worker = RecognitionWorker(camera, StubPipeline(), inference_interval_ms=0)
    errors = []
    worker.error_raised.connect(errors.append)

    worker.run()

    assert errors[0].code == "camera_read"
    assert errors[0].recoverable
    assert camera.release_calls == 1


def test_worker_applies_pending_store_refresh_before_processing():
    camera = FakeCamera([frame()])
    pipeline = StubPipeline()
    worker = RecognitionWorker(camera, pipeline, inference_interval_ms=0)
    snapshot = object()
    worker.refresh_store(snapshot)
    worker.preview_ready.connect(lambda _: worker.request_stop())

    worker.run()

    assert pipeline.snapshots == [snapshot]


def test_thread_host_starts_and_stops_worker(qtbot):
    camera = FakeCamera([frame() for _ in range(20)])
    pipeline = StubPipeline()
    host = WorkerThreadHost(
        lambda: RecognitionWorker(camera, pipeline, inference_interval_ms=0)
    )
    host.preview_ready.connect(lambda _: host.stop())

    with qtbot.waitSignal(host.finished, timeout=2000):
        host.start()

    assert pipeline.calls >= 1
    assert camera.open_calls == 1
    assert camera.release_calls == 1
