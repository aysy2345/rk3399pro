import time

import numpy as np
from PyQt5.QtCore import Qt

from face_recognition_app.app.bootstrap import BackendPair, build_application
from face_recognition_app.app.config import parse_config
from face_recognition_app.core.enrollment import EnrollmentTemplate
from face_recognition_app.core.enrollment_session import EnrollmentProgress
from face_recognition_app.domain.runtime import AppState
from face_recognition_app.inference.fake import FakeFaceDetector, FakeFaceEmbedder


def config_data():
    return {
        "runtime": {"backend": "fake", "inference_interval_ms": 0},
        "camera": {
            "index": 0,
            "width": 160,
            "height": 120,
            "target_fps": 30,
            "retry_count": 0,
        },
        "models": {
            "detector_path": "missing-detector.onnx",
            "recognizer_path": "missing-recognizer.onnx",
        },
        "recognition": {
            "detection_threshold": 0.8,
            "recognition_threshold": 0.6,
            "min_face_size": 20,
            "min_sharpness": 0.0,
            "window_size": 5,
            "votes_required": 3,
            "enrollment_samples": 15,
            "enrollment_interval_ms": 0,
            "save_photos": False,
        },
        "storage": {"data_dir": "face_data"},
    }


class RepeatingCamera:
    def __init__(self):
        self.open_calls = 0
        self.release_calls = 0

    @property
    def is_open(self):
        return self.open_calls > self.release_calls

    def open(self):
        self.open_calls += 1

    def read(self):
        time.sleep(0.002)
        return np.zeros((120, 160, 3), dtype=np.uint8)

    def release(self):
        self.release_calls += 1


class StaticBackendFactory:
    def create(self, config):
        return BackendPair(
            FakeFaceDetector([]),
            FakeFaceEmbedder([np.array([1.0, 0.0], dtype=np.float32)]),
        )


class StubEnrollmentSession:
    def process(self, frame):
        return EnrollmentProgress(
            False, "等待测试完成", 0, 15, "front", False
        )

    def build_template(self):
        return EnrollmentTemplate(
            np.array([1.0, 0.0], dtype=np.float32), 15, 0
        )


def test_fake_desktop_flow_start_enroll_refresh_manage_and_close(qtbot, tmp_path):
    config = parse_config(config_data(), tmp_path)
    cameras = []

    def camera_factory():
        camera = RepeatingCamera()
        cameras.append(camera)
        return camera

    bundle = build_application(
        config,
        camera_factory=camera_factory,
        backend_factory=StaticBackendFactory(),
        enrollment_session_factory=StubEnrollmentSession,
    )
    window = bundle.window
    qtbot.addWidget(window)
    window.show()

    assert bundle.controller.state == AppState.IDLE
    assert window.member_count_label.text() == "成员：0"

    qtbot.mouseClick(window.start_button, Qt.LeftButton)
    qtbot.waitUntil(lambda: window.video_widget.has_frame, timeout=2000)
    qtbot.mouseClick(window.stop_button, Qt.LeftButton)
    assert bundle.controller.state == AppState.IDLE
    assert cameras[0].release_calls == 1

    wizard = bundle.coordinator.create_enrollment_wizard()
    qtbot.addWidget(wizard)
    wizard.show()
    wizard.member_id_edit.setText("001")
    wizard.name_edit.setText("张三")
    qtbot.mouseClick(wizard.next_button, Qt.LeftButton)
    qtbot.waitUntil(
        lambda: bundle.controller.state == AppState.ENROLLING, timeout=2000
    )
    bundle.controller.enrollment_progress.emit(
        EnrollmentProgress(True, "有效样本", 15, 15, None, True)
    )
    qtbot.mouseClick(wizard.capture_next_button, Qt.LeftButton)
    qtbot.mouseClick(wizard.save_button, Qt.LeftButton)

    qtbot.waitUntil(
        lambda: bundle.controller.state == AppState.IDLE, timeout=2000
    )
    assert window.member_count_label.text() == "成员：1"
    assert bundle.store.load().members[0].name == "张三"

    manager = bundle.coordinator.create_member_manager_dialog()
    qtbot.addWidget(manager)
    manager.show()
    assert manager.table.rowCount() == 1
    assert manager.name_edits["001"].text() == "张三"

    window.close()
    assert not bundle.worker_host.running
