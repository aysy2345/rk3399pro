"""Composition root for the Phase 5 desktop application."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Optional

import numpy as np
from PyQt5.QtWidgets import QMessageBox

from face_recognition_app.app.config import (
    AppConfig,
    validate_runtime_paths,
)
from face_recognition_app.app.controller import AppController
from face_recognition_app.app.member_service import MemberService
from face_recognition_app.core.enrollment_session import EnrollmentSession
from face_recognition_app.core.pipeline import RecognitionPipeline
from face_recognition_app.hardware.camera import OpenCVCamera
from face_recognition_app.inference.fake import (
    FakeFaceDetector,
    FakeFaceEmbedder,
)
from face_recognition_app.inference.onnx_backend import (
    OnnxMobileFaceNetEmbedder,
    OnnxRetinaFaceDetector,
)
from face_recognition_app.storage.face_store import FaceStore
from face_recognition_app.ui.enrollment_wizard import EnrollmentWizard
from face_recognition_app.ui.main_window import MainWindow
from face_recognition_app.ui.member_manager_dialog import MemberManagerDialog
from face_recognition_app.workers.recognition_worker import (
    RecognitionWorker,
    WorkerThreadHost,
)


class BootstrapError(RuntimeError):
    """Raised when the selected runtime cannot be assembled."""


@dataclass(frozen=True)
class BackendPair:
    detector: Any
    embedder: Any


class BackendFactory:
    def create(self, config: AppConfig) -> BackendPair:
        backend = config.runtime.backend
        if backend == "fake":
            return BackendPair(
                FakeFaceDetector([]),
                FakeFaceEmbedder(
                    [np.ones(512, dtype=np.float32)]
                ),
            )
        if backend == "rknn":
            raise BootstrapError("RKNN 后端将在 Phase 7 板端集成阶段提供")
        if backend == "onnx":
            validate_runtime_paths(config)
            return BackendPair(
                OnnxRetinaFaceDetector(
                    model_path=config.models.detector_path,
                    confidence_threshold=(
                        config.recognition.detection_threshold
                    ),
                ),
                OnnxMobileFaceNetEmbedder(
                    model_path=config.models.recognizer_path
                ),
            )
        raise BootstrapError("不支持的推理后端：{}".format(backend))


def apply_overrides(
    config: AppConfig,
    backend: Optional[str] = None,
    camera_index: Optional[int] = None,
) -> AppConfig:
    result = config
    if backend is not None:
        if backend not in ("fake", "onnx", "rknn"):
            raise BootstrapError("不支持的推理后端：{}".format(backend))
        result = replace(
            result,
            runtime=replace(result.runtime, backend=backend),
        )
    if camera_index is not None:
        if (
            isinstance(camera_index, bool)
            or not isinstance(camera_index, int)
            or camera_index < 0
        ):
            raise BootstrapError("camera index 必须是非负整数")
        result = replace(
            result,
            camera=replace(result.camera, index=camera_index),
        )
    return result


class DesktopCoordinator:
    def __init__(
        self,
        window: MainWindow,
        controller: AppController,
        member_service: MemberService,
        enrollment_session_factory: Callable[[], Any],
        enrollment_enabled: bool = True,
        preview_flip_horizontal: bool = False,
    ) -> None:
        self.window = window
        self._controller = controller
        self._member_service = member_service
        self._enrollment_session_factory = enrollment_session_factory
        self._enrollment_enabled = enrollment_enabled
        self._preview_flip_horizontal = preview_flip_horizontal
        self._active_dialog = None
        window.add_member_requested.connect(self.open_enrollment_wizard)
        window.manage_members_requested.connect(self.open_member_manager_dialog)

    def _update_member_count(self, snapshot: Any) -> None:
        self.window.update_member_count(len(snapshot.members))

    def create_enrollment_wizard(
        self, member: Optional[Any] = None, parent: Optional[Any] = None
    ) -> EnrollmentWizard:
        if not self._enrollment_enabled:
            raise BootstrapError(
                "Fake 后端不检测真实人脸，成员采集请使用 ONNX 后端"
            )
        wizard = EnrollmentWizard(
            self._controller,
            self._member_service,
            self._enrollment_session_factory,
            parent or self.window,
            member=member,
            preview_flip_horizontal=self._preview_flip_horizontal,
        )
        wizard.member_saved.connect(self._update_member_count)
        return wizard

    def open_enrollment_wizard(self) -> None:
        if not self._enrollment_enabled:
            QMessageBox.warning(
                self.window,
                "当前模式不能采集",
                "Fake 后端不检测真实人脸，无法采集成员。"
                "请关闭程序后使用 ONNX 后端（--backend onnx）重新启动。",
            )
            return
        wizard = self.create_enrollment_wizard()
        self._active_dialog = wizard
        try:
            wizard.exec_()
        finally:
            self._active_dialog = None

    def create_member_manager_dialog(
        self, parent: Optional[Any] = None
    ) -> MemberManagerDialog:
        def reenrollment_factory(member: Any, dialog_parent: Any) -> Any:
            return self.create_enrollment_wizard(member, dialog_parent)

        dialog = MemberManagerDialog(
            self._member_service,
            reenrollment_factory if self._enrollment_enabled else None,
            parent or self.window,
        )
        dialog.members_changed.connect(self._update_member_count)
        return dialog

    def open_member_manager_dialog(self) -> None:
        dialog = self.create_member_manager_dialog()
        self._active_dialog = dialog
        try:
            dialog.exec_()
        finally:
            self._active_dialog = None


@dataclass
class ApplicationBundle:
    config: AppConfig
    backend: BackendPair
    store: FaceStore
    worker_host: WorkerThreadHost
    controller: AppController
    member_service: MemberService
    coordinator: DesktopCoordinator
    window: MainWindow


def build_application(
    config: AppConfig,
    camera_factory: Optional[Callable[[], Any]] = None,
    backend_factory: Optional[Any] = None,
    enrollment_session_factory: Optional[Callable[[], Any]] = None,
) -> ApplicationBundle:
    factory = backend_factory or BackendFactory()
    backend = factory.create(config)
    store = FaceStore(config.storage.data_dir)
    snapshot = store.load()
    pipeline = RecognitionPipeline(
        backend.detector,
        backend.embedder,
        snapshot,
        config.recognition.recognition_threshold,
        config.recognition.window_size,
        config.recognition.votes_required,
    )

    if camera_factory is None:
        camera_config = config.camera

        def camera_factory() -> OpenCVCamera:
            return OpenCVCamera(
                camera_config.index,
                camera_config.width,
                camera_config.height,
                camera_config.target_fps,
                camera_config.retry_count,
            )

    def worker_factory() -> RecognitionWorker:
        return RecognitionWorker(
            camera_factory(),
            pipeline,
            config.runtime.inference_interval_ms,
        )

    worker_host = WorkerThreadHost(worker_factory)
    controller = AppController(worker_host)
    member_service = MemberService(store, controller.refresh_store)

    custom_enrollment_factory = enrollment_session_factory is not None
    if enrollment_session_factory is None:
        recognition = config.recognition

        def enrollment_session_factory() -> EnrollmentSession:
            return EnrollmentSession(
                backend.detector,
                backend.embedder,
                recognition.enrollment_samples,
                recognition.min_face_size,
                recognition.min_sharpness,
                recognition.enrollment_interval_ms,
            )

    window = MainWindow(
        controller,
        member_count=len(snapshot.members),
        preview_flip_horizontal=config.camera.preview_flip_horizontal,
    )
    coordinator = DesktopCoordinator(
        window,
        controller,
        member_service,
        enrollment_session_factory,
        enrollment_enabled=(
            config.runtime.backend != "fake" or custom_enrollment_factory
        ),
        preview_flip_horizontal=config.camera.preview_flip_horizontal,
    )
    return ApplicationBundle(
        config=config,
        backend=backend,
        store=store,
        worker_host=worker_host,
        controller=controller,
        member_service=member_service,
        coordinator=coordinator,
        window=window,
    )
