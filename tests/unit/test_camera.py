import cv2
import numpy as np
import pytest

from face_recognition_app.hardware.camera import (
    CameraError,
    CameraReadError,
    OpenCVCamera,
)


class StubCapture:
    def __init__(self, opened=True, reads=None):
        self.opened = opened
        self.reads = list(reads or [])
        self.properties = {}
        self.release_calls = 0
        self.read_calls = 0

    def isOpened(self):
        return self.opened

    def set(self, prop, value):
        self.properties[prop] = value
        return True

    def read(self):
        self.read_calls += 1
        if not self.reads:
            return False, None
        return self.reads.pop(0)

    def release(self):
        self.release_calls += 1
        self.opened = False


class CaptureFactory:
    def __init__(self, captures):
        self.captures = list(captures)
        self.indices = []

    def __call__(self, index):
        self.indices.append(index)
        return self.captures.pop(0)


def make_camera(factory, retry_count=0):
    return OpenCVCamera(
        index=2,
        width=640,
        height=480,
        target_fps=30,
        retry_count=retry_count,
        capture_factory=factory,
    )


def test_open_configures_capture_and_read_returns_a_copy():
    source = np.full((4, 5, 3), 17, dtype=np.uint8)
    capture = StubCapture(reads=[(True, source)])
    factory = CaptureFactory([capture])
    camera = make_camera(factory)

    camera.open()
    frame = camera.read()
    frame[0, 0, 0] = 99

    assert factory.indices == [2]
    assert capture.properties[cv2.CAP_PROP_FRAME_WIDTH] == 640
    assert capture.properties[cv2.CAP_PROP_FRAME_HEIGHT] == 480
    assert capture.properties[cv2.CAP_PROP_FPS] == 30
    assert source[0, 0, 0] == 17
    assert camera.is_open


def test_open_retries_and_releases_failed_captures():
    captures = [StubCapture(opened=False) for _ in range(3)]
    camera = make_camera(CaptureFactory(captures), retry_count=2)

    with pytest.raises(CameraError, match="after 3 attempts"):
        camera.open()

    assert [capture.release_calls for capture in captures] == [1, 1, 1]
    assert not camera.is_open


def test_read_failure_raises_specific_error():
    camera = make_camera(CaptureFactory([StubCapture(reads=[(False, None)])]))
    camera.open()

    with pytest.raises(CameraReadError, match="unable to read"):
        camera.read()


def test_open_and_release_are_idempotent():
    capture = StubCapture(reads=[(True, np.zeros((2, 2, 3), dtype=np.uint8))])
    factory = CaptureFactory([capture])
    camera = make_camera(factory)

    camera.open()
    camera.open()
    camera.release()
    camera.release()

    assert factory.indices == [2]
    assert capture.release_calls == 1
    assert not camera.is_open


def test_context_manager_releases_camera_on_error():
    capture = StubCapture()
    camera = make_camera(CaptureFactory([capture]))

    with pytest.raises(RuntimeError, match="boom"):
        with camera:
            raise RuntimeError("boom")

    assert capture.release_calls == 1
