"""Three-step member enrollment wizard."""

from __future__ import annotations

from typing import Any, Callable, Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QCloseEvent
from PyQt5.QtWidgets import (
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from face_recognition_app.app.member_service import MemberDraft, MemberServiceError
from face_recognition_app.core.enrollment_session import (
    EnrollmentProgress,
    POSE_LABELS,
)
from face_recognition_app.domain.member import Member
from face_recognition_app.ui.video_widget import VideoWidget


class EnrollmentWizard(QDialog):
    member_saved = pyqtSignal(object)

    def __init__(
        self,
        controller: Any,
        member_service: Any,
        session_factory: Callable[[], Any],
        parent: Optional[QWidget] = None,
        member: Optional[Member] = None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self._member_service = member_service
        self._session_factory = session_factory
        self._draft: Optional[MemberDraft] = None
        self._session = None
        self._template = None
        self._enrollment_active = False
        self._saving = False
        self._existing_member = member
        self.setWindowTitle(
            "重新采集成员" if member is not None else "添加新成员"
        )
        self.setModal(True)
        self.resize(820, 650)
        self._build_ui()
        controller.preview_ready.connect(self.video_widget.set_frame)
        controller.enrollment_progress.connect(self._on_progress)
        if member is not None:
            self._begin_reenrollment(member)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(12)
        self.step_label = QLabel("步骤 1/3 · 基本信息", self)
        self.step_label.setObjectName("stepLabel")
        layout.addWidget(self.step_label)

        self.same_name_label = QLabel("", self)
        self.same_name_label.setObjectName("warningLabel")
        self.same_name_label.setWordWrap(True)
        self.same_name_label.hide()
        layout.addWidget(self.same_name_label)
        self.error_label = QLabel("", self)
        self.error_label.setObjectName("errorLabel")
        self.error_label.setWordWrap(True)
        self.error_label.hide()
        layout.addWidget(self.error_label)

        self.pages = QStackedWidget(self)
        self.pages.addWidget(self._build_information_page())
        self.pages.addWidget(self._build_capture_page())
        self.pages.addWidget(self._build_confirmation_page())
        layout.addWidget(self.pages, 1)

        self.cancel_button = QPushButton("取消", self)
        self.cancel_button.clicked.connect(self.reject)
        footer = QHBoxLayout()
        footer.addStretch(1)
        footer.addWidget(self.cancel_button)
        layout.addLayout(footer)
        self.setStyleSheet(
            """
            QDialog { background: #0b1220; color: #e5edf6; }
            QLabel { color: #e5edf6; }
            QLabel#stepLabel { color: #60a5fa; font-size: 20px; font-weight: 700; }
            QLabel#poseLabel {
                background: #162235; color: #f8fafc; padding: 8px;
                border-radius: 6px; font-size: 16px; font-weight: 700;
            }
            QLabel#reasonLabel {
                color: #fbbf24; padding: 4px;
                font-size: 14px; font-weight: 600;
            }
            QLabel#warningLabel {
                background: #493716; color: #fde68a; padding: 8px;
                border-radius: 6px;
            }
            QLabel#errorLabel {
                background: #4a1d24; color: #fecaca; padding: 8px;
                border-radius: 6px;
            }
            QLineEdit {
                background: #162235; color: #f8fafc; border: 1px solid #3d5675;
                border-radius: 6px; padding: 8px;
            }
            QPushButton {
                background: #263b55; color: #f8fafc; border: 1px solid #3d5675;
                border-radius: 7px; padding: 8px 18px;
            }
            QPushButton:disabled { background: #1b2737; color: #65758a; }
            QProgressBar#captureProgress {
                background: #162235; color: #f8fafc;
                border: 1px solid #3d5675; border-radius: 5px;
                text-align: center; min-height: 20px; font-weight: 700;
            }
            QProgressBar#captureProgress::chunk {
                background: #2563eb; border-radius: 4px;
            }
            """
        )

    def _build_information_page(self) -> QWidget:
        page = QWidget(self)
        layout = QVBoxLayout(page)
        form = QFormLayout()
        self.member_id_edit = QLineEdit(page)
        self.member_id_edit.setPlaceholderText("例如：001")
        self.name_edit = QLineEdit(page)
        self.name_edit.setPlaceholderText("请输入姓名")
        form.addRow("成员编号", self.member_id_edit)
        form.addRow("姓名", self.name_edit)
        layout.addLayout(form)
        layout.addStretch(1)
        self.next_button = QPushButton("下一步：采集样本", page)
        self.next_button.clicked.connect(self._start_capture)
        layout.addWidget(self.next_button, 0, Qt.AlignRight)
        return page

    def _build_capture_page(self) -> QWidget:
        page = QWidget(self)
        layout = QVBoxLayout(page)
        self.video_widget = VideoWidget(page)
        layout.addWidget(self.video_widget, 1)
        self.pose_label = QLabel("请正视摄像头并保持稳定", page)
        self.pose_label.setObjectName("poseLabel")
        self.pose_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.pose_label)
        self.reason_label = QLabel("等待有效样本", page)
        self.reason_label.setObjectName("reasonLabel")
        self.reason_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.reason_label)
        self.capture_progress = QProgressBar(page)
        self.capture_progress.setObjectName("captureProgress")
        self.capture_progress.setRange(0, 15)
        self.capture_progress.setValue(0)
        self.capture_progress.setFormat("%v/%m")
        layout.addWidget(self.capture_progress)
        actions = QHBoxLayout()
        self.capture_back_button = QPushButton("返回", page)
        self.capture_back_button.clicked.connect(self._back_to_information)
        self.capture_next_button = QPushButton("下一步：确认保存", page)
        self.capture_next_button.setEnabled(False)
        self.capture_next_button.clicked.connect(self._show_confirmation)
        actions.addWidget(self.capture_back_button)
        actions.addStretch(1)
        actions.addWidget(self.capture_next_button)
        layout.addLayout(actions)
        return page

    def _build_confirmation_page(self) -> QWidget:
        page = QWidget(self)
        layout = QVBoxLayout(page)
        self.summary_label = QLabel("", page)
        self.summary_label.setWordWrap(True)
        self.summary_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.summary_label, 1)
        actions = QHBoxLayout()
        self.confirm_back_button = QPushButton("返回采集", page)
        self.confirm_back_button.clicked.connect(self._back_to_capture)
        self.save_button = QPushButton("确认并保存", page)
        self.save_button.clicked.connect(self._save)
        actions.addWidget(self.confirm_back_button)
        actions.addStretch(1)
        actions.addWidget(self.save_button)
        layout.addLayout(actions)
        return page

    def _clear_error(self) -> None:
        self.error_label.clear()
        self.error_label.hide()

    def _show_error(self, message: str) -> None:
        self.error_label.setText(message)
        self.error_label.show()

    def _start_capture(self) -> None:
        self._clear_error()
        try:
            draft = self._member_service.validate_new_member(
                self.member_id_edit.text(), self.name_edit.text()
            )
            session = self._session_factory()
        except (MemberServiceError, ValueError, RuntimeError) as exc:
            self._show_error(str(exc))
            return
        self._draft = draft
        if draft.same_name_warning:
            self.same_name_label.setText(
                "已有同名成员；编号不同仍可继续，请确认没有选错人员。"
            )
            self.same_name_label.show()
        else:
            self.same_name_label.hide()
        self._begin_capture(session)
        self.pages.setCurrentIndex(1)
        self.step_label.setText("步骤 2/3 · 自动采集")

    def _begin_reenrollment(self, member: Member) -> None:
        self._draft = MemberDraft(member.member_id, member.name, False)
        self.member_id_edit.setText(member.member_id)
        self.name_edit.setText(member.name)
        self.capture_back_button.hide()
        try:
            session = self._session_factory()
        except (ValueError, RuntimeError) as exc:
            self._show_error("无法开始重新采集：{}".format(exc))
            return
        self._begin_capture(session)
        self.pages.setCurrentIndex(1)
        self.step_label.setText("步骤 1/2 · 自动采集")

    def _begin_capture(self, session: Any) -> None:
        self._session = session
        self._template = None
        self.capture_progress.setRange(0, 15)
        self.capture_progress.setValue(0)
        self.capture_next_button.setEnabled(False)
        self.pose_label.setText("请正视摄像头并保持稳定")
        self.reason_label.setText("等待有效样本")
        self._controller.begin_enrollment(session)
        self._enrollment_active = True

    def _on_progress(self, progress: EnrollmentProgress) -> None:
        if not self._enrollment_active or self.pages.currentIndex() != 1:
            return
        self.capture_progress.setRange(0, progress.target_count)
        self.capture_progress.setValue(progress.accepted_count)
        self.reason_label.setText(progress.reason)
        if progress.complete:
            self.pose_label.setText("采集完成")
            self.capture_next_button.setEnabled(True)
        elif progress.required_pose:
            pose = POSE_LABELS.get(progress.required_pose, progress.required_pose)
            self.pose_label.setText("请{}并保持稳定".format(pose))

    def _show_confirmation(self) -> None:
        if not self.capture_next_button.isEnabled() or self._session is None:
            return
        self._clear_error()
        try:
            self._template = self._session.build_template()
        except (ValueError, RuntimeError) as exc:
            self._show_error("无法生成成员特征：{}".format(exc))
            return
        action = "重新采集" if self._existing_member is not None else "添加成员"
        self.summary_label.setText(
            "{}\n成员编号：{}\n姓名：{}\n有效样本：{}\n\n"
            "确认后才会写入本地成员库。".format(
                action,
                self._draft.member_id,
                self._draft.name,
                self._template.accepted_samples,
            )
        )
        self.pages.setCurrentIndex(2)
        self.step_label.setText(
            "步骤 2/2 · 确认保存"
            if self._existing_member is not None
            else "步骤 3/3 · 确认保存"
        )

    def _back_to_information(self) -> None:
        if self._existing_member is not None:
            return
        if self._enrollment_active:
            self._controller.cancel_enrollment()
            self._enrollment_active = False
        self._session = None
        self._template = None
        self.pages.setCurrentIndex(0)
        self.step_label.setText("步骤 1/3 · 基本信息")

    def _back_to_capture(self) -> None:
        self._clear_error()
        self.pages.setCurrentIndex(1)
        self.step_label.setText(
            "步骤 1/2 · 自动采集"
            if self._existing_member is not None
            else "步骤 2/3 · 自动采集"
        )

    def _save(self) -> None:
        if self._saving or self._draft is None or self._template is None:
            return
        self._saving = True
        self.save_button.setEnabled(False)
        self.cancel_button.setEnabled(False)
        self._clear_error()
        try:
            if self._existing_member is not None:
                snapshot = self._member_service.replace_member_embedding(
                    self._draft.member_id,
                    self._template.embedding,
                )
            else:
                snapshot = self._member_service.add_member(
                    self._draft.member_id,
                    self._draft.name,
                    self._template.embedding,
                )
        except (MemberServiceError, ValueError, RuntimeError) as exc:
            self._show_error(str(exc))
            self._saving = False
            self.save_button.setEnabled(True)
            self.cancel_button.setEnabled(True)
            return
        self._controller.finish_enrollment()
        self._enrollment_active = False
        self._saving = False
        self.member_saved.emit(snapshot)
        self.accept()

    def reject(self) -> None:
        if self._saving:
            return
        if self._enrollment_active:
            self._controller.cancel_enrollment()
            self._enrollment_active = False
        super().reject()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._saving:
            event.ignore()
            return
        super().closeEvent(event)
