"""Main desktop window for local face recognition."""

from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QCloseEvent
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from face_recognition_app.domain.runtime import AppState, FrameResult, WorkerErrorInfo
from face_recognition_app.ui.video_widget import VideoWidget


class MainWindow(QMainWindow):
    add_member_requested = pyqtSignal()
    manage_members_requested = pyqtSignal()

    def __init__(self, controller, member_count: int = 0, parent=None) -> None:
        super().__init__(parent)
        self._controller = controller
        self.setWindowTitle("本地人脸识别")
        self.resize(1100, 760)
        self._build_ui(member_count)
        self._connect_controller()
        self._apply_state(controller.state)

    def _build_ui(self, member_count: int) -> None:
        root = QWidget(self)
        root.setObjectName("root")
        layout = QVBoxLayout(root)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(14)

        self.video_widget = VideoWidget(root)
        layout.addWidget(self.video_widget, 1)

        panel = QFrame(root)
        panel.setObjectName("controlPanel")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(18, 14, 18, 14)
        panel_layout.setSpacing(10)

        info_row = QHBoxLayout()
        self.status_label = QLabel("待机", panel)
        self.status_label.setObjectName("statusLabel")
        self.result_label = QLabel("尚未开始识别", panel)
        self.result_label.setObjectName("resultLabel")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.member_count_label = QLabel("成员：{}".format(member_count), panel)
        self.member_count_label.setObjectName("memberCountLabel")
        self.member_count_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        info_row.addWidget(self.status_label)
        info_row.addWidget(self.result_label, 1)
        info_row.addWidget(self.member_count_label)
        panel_layout.addLayout(info_row)

        self.error_label = QLabel("", panel)
        self.error_label.setObjectName("errorLabel")
        self.error_label.setWordWrap(True)
        self.error_label.hide()
        panel_layout.addWidget(self.error_label)

        actions = QHBoxLayout()
        actions.setSpacing(10)
        self.start_button = QPushButton("开始识别", panel)
        self.stop_button = QPushButton("停止识别", panel)
        self.add_member_button = QPushButton("添加成员", panel)
        self.manage_members_button = QPushButton("成员管理", panel)
        self.retry_button = QPushButton("重试", panel)
        self.retry_button.hide()
        for button in (
            self.start_button,
            self.stop_button,
            self.add_member_button,
            self.manage_members_button,
            self.retry_button,
        ):
            button.setMinimumHeight(40)
            actions.addWidget(button)
        panel_layout.addLayout(actions)
        layout.addWidget(panel)
        self.setCentralWidget(root)

        self.setStyleSheet(
            """
            QWidget#root { background: #0b1220; color: #e5edf6; }
            QFrame#controlPanel {
                background: #162235;
                border: 1px solid #2a3b52;
                border-radius: 10px;
            }
            QLabel#statusLabel { color: #60a5fa; font-size: 17px; font-weight: 700; }
            QLabel#resultLabel { font-size: 19px; font-weight: 700; }
            QLabel#memberCountLabel { color: #b6c5d6; font-size: 15px; }
            QLabel#errorLabel {
                background: #4a1d24; color: #fecaca; padding: 8px;
                border-radius: 6px;
            }
            QPushButton {
                background: #263b55; color: #f8fafc; border: 1px solid #3d5675;
                border-radius: 7px; padding: 8px 18px; font-size: 15px;
            }
            QPushButton:hover { background: #315174; }
            QPushButton:disabled { background: #1b2737; color: #65758a; }
            """
        )

        self.start_button.clicked.connect(self._controller.start_recognition)
        self.stop_button.clicked.connect(self._controller.stop_recognition)
        self.retry_button.clicked.connect(self._controller.retry)
        self.add_member_button.clicked.connect(self.add_member_requested.emit)
        self.manage_members_button.clicked.connect(
            self.manage_members_requested.emit
        )

    def _connect_controller(self) -> None:
        self._controller.state_changed.connect(self._apply_state)
        self._controller.preview_ready.connect(self.video_widget.set_frame)
        self._controller.recognition_ready.connect(self._show_recognition)
        self._controller.error_raised.connect(self._show_error)

    def _apply_state(self, state: AppState) -> None:
        labels = {
            AppState.IDLE: "待机",
            AppState.RECOGNIZING: "识别中",
            AppState.ENROLLING: "成员登记中",
            AppState.ERROR: "错误",
        }
        self.status_label.setText(labels[state])
        idle = state == AppState.IDLE
        recognizing = state == AppState.RECOGNIZING
        enrolling = state == AppState.ENROLLING
        self.start_button.setEnabled(idle)
        self.stop_button.setEnabled(recognizing)
        self.add_member_button.setEnabled(not enrolling)
        self.manage_members_button.setEnabled(not enrolling)
        if state != AppState.ERROR:
            self.retry_button.hide()
            self.error_label.hide()
        if idle:
            self.result_label.setText("尚未开始识别")

    def _show_recognition(self, result: FrameResult) -> None:
        self.video_widget.set_result(result)
        if not result.faces:
            self.result_label.setText("未检测到人脸")
            return
        labels = []
        for face in result.faces:
            if face.is_known:
                labels.append(
                    "{} {:.1f}%".format(face.name, face.similarity * 100.0)
                )
            else:
                labels.append("陌生人")
        self.result_label.setText(" · ".join(labels))

    def _show_error(self, error: WorkerErrorInfo) -> None:
        self._apply_state(AppState.ERROR)
        self.error_label.setText(error.message)
        self.error_label.show()
        self.retry_button.setVisible(error.recoverable)

    def update_member_count(self, count: int) -> None:
        if count < 0:
            raise ValueError("member count must be non-negative")
        self.member_count_label.setText("成员：{}".format(count))

    def closeEvent(self, event: QCloseEvent) -> None:
        self._controller.close()
        event.accept()
