import numpy as np
import pytest
from PyQt5.QtCore import QObject, Qt, pyqtSignal

from face_recognition_app.domain.runtime import (
    AppState,
    FaceOverlay,
    FrameResult,
    WorkerErrorInfo,
)
from face_recognition_app.ui.main_window import MainWindow
from face_recognition_app.ui.video_widget import VideoWidget


class StubController(QObject):
    state_changed = pyqtSignal(object)
    preview_ready = pyqtSignal(object)
    recognition_ready = pyqtSignal(object)
    enrollment_progress = pyqtSignal(object)
    error_raised = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.state = AppState.IDLE
        self.start_calls = 0
        self.stop_calls = 0
        self.retry_calls = 0
        self.close_calls = 0

    def start_recognition(self):
        self.start_calls += 1

    def stop_recognition(self):
        self.stop_calls += 1

    def retry(self):
        self.retry_calls += 1

    def close(self):
        self.close_calls += 1


def make_window(qtbot, member_count=0):
    controller = StubController()
    window = MainWindow(controller, member_count=member_count)
    qtbot.addWidget(window)
    window.show()
    return controller, window


def test_main_window_starts_idle_with_expected_controls(qtbot):
    _, window = make_window(qtbot, member_count=3)

    assert window.windowTitle() == "本地人脸识别"
    assert isinstance(window.video_widget, VideoWidget)
    assert window.status_label.text() == "待机"
    assert window.member_count_label.text() == "成员：3"
    assert window.start_button.isEnabled()
    assert not window.stop_button.isEnabled()
    assert window.add_member_button.isEnabled()
    assert window.manage_members_button.isEnabled()
    assert not window.retry_button.isVisible()


def test_main_result_uses_explicit_high_contrast_text(qtbot):
    _, window = make_window(qtbot)

    assert window.result_label.objectName() == "resultLabel"
    assert "QLabel#resultLabel { color: #f8fafc;" in window.styleSheet()


def test_state_changes_update_buttons_and_status(qtbot):
    controller, window = make_window(qtbot)

    controller.state_changed.emit(AppState.RECOGNIZING)
    assert window.status_label.text() == "识别中"
    assert not window.start_button.isEnabled()
    assert window.stop_button.isEnabled()

    controller.state_changed.emit(AppState.ENROLLING)
    assert window.status_label.text() == "成员登记中"
    assert not window.stop_button.isEnabled()
    assert not window.add_member_button.isEnabled()


def test_preview_and_recognition_result_update_video_and_text(qtbot):
    controller, window = make_window(qtbot)
    image = np.zeros((100, 160, 3), dtype=np.uint8)
    controller.preview_ready.emit(image)

    face = FaceOverlay(
        track_id="face-1",
        box=(20.0, 10.0, 100.0, 90.0),
        member_id="001",
        name="张三",
        similarity=0.93,
        is_known=True,
    )
    controller.recognition_ready.emit(FrameResult(image, (face,)))

    assert window.video_widget.has_frame
    assert window.video_widget.overlay_count == 1
    assert "张三" in window.result_label.text()
    assert "93.0%" in window.result_label.text()


def test_recoverable_error_shows_retry_and_calls_controller(qtbot):
    controller, window = make_window(qtbot)
    controller.error_raised.emit(
        WorkerErrorInfo("camera_read", "摄像头已断开", True)
    )

    assert window.status_label.text() == "错误"
    assert window.error_label.text() == "摄像头已断开"
    assert window.retry_button.isVisible()

    qtbot.mouseClick(window.retry_button, Qt.LeftButton)
    assert controller.retry_calls == 1


def test_buttons_forward_actions_and_close_releases_controller(qtbot):
    controller, window = make_window(qtbot)
    add_requests = []
    manage_requests = []
    window.add_member_requested.connect(lambda: add_requests.append(True))
    window.manage_members_requested.connect(lambda: manage_requests.append(True))

    qtbot.mouseClick(window.start_button, Qt.LeftButton)
    qtbot.mouseClick(window.add_member_button, Qt.LeftButton)
    qtbot.mouseClick(window.manage_members_button, Qt.LeftButton)
    controller.state_changed.emit(AppState.RECOGNIZING)
    qtbot.mouseClick(window.stop_button, Qt.LeftButton)
    window.close()

    assert controller.start_calls == 1
    assert controller.stop_calls == 1
    assert controller.close_calls == 1
    assert add_requests == [True]
    assert manage_requests == [True]


def test_video_widget_renders_bgr_frame_and_overlay(qtbot):
    widget = VideoWidget()
    qtbot.addWidget(widget)
    widget.resize(320, 240)
    widget.show()
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    image[:, :, 2] = 255
    overlay = FaceOverlay(
        "face-1", (20.0, 20.0, 100.0, 100.0), None, "陌生人", 0.4, False
    )

    widget.set_result(FrameResult(image, (overlay,)))
    rendered = widget.grab()

    assert not rendered.isNull()
    assert widget.has_frame
    assert widget.overlay_count == 1


def test_video_widget_horizontal_flip_changes_pixels_without_mutating_source(qtbot):
    widget = VideoWidget(flip_horizontal=True)
    qtbot.addWidget(widget)
    frame = np.zeros((2, 4, 3), dtype=np.uint8)
    frame[:, :2] = (0, 0, 255)
    frame[:, 2:] = (255, 0, 0)
    original = frame.copy()

    widget.set_frame(frame)

    assert widget.flip_horizontal is True
    assert widget._image.pixelColor(0, 0).blue() == 255
    assert widget._image.pixelColor(3, 0).red() == 255
    np.testing.assert_array_equal(frame, original)


@pytest.mark.parametrize(
    ("flip_horizontal", "expected_side"),
    [(False, "left"), (True, "right")],
)
def test_video_widget_keeps_overlay_aligned_after_horizontal_flip(
    qtbot, flip_horizontal, expected_side
):
    widget = VideoWidget(flip_horizontal=flip_horizontal)
    qtbot.addWidget(widget)
    widget.resize(320, 240)
    widget.show()
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    overlay = FaceOverlay(
        "face-1", (10.0, 20.0, 50.0, 100.0), "001", "张三", 0.9, True
    )

    widget.set_result(FrameResult(image, (overlay,)))
    rendered = widget.grab().toImage()
    green_counts = {"left": 0, "right": 0}
    for y in range(rendered.height()):
        for x in range(rendered.width()):
            color = rendered.pixelColor(x, y)
            if color.green() > 120 and color.red() < 120:
                side = "left" if x < rendered.width() // 2 else "right"
                green_counts[side] += 1

    assert green_counts[expected_side] > green_counts[
        "right" if expected_side == "left" else "left"
    ]
