"""Searchable table for local member management."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from face_recognition_app.app.member_service import MemberServiceError
from face_recognition_app.domain.member import Member


class MemberManagerDialog(QDialog):
    members_changed = pyqtSignal(object)

    def __init__(
        self,
        member_service: Any,
        reenrollment_factory: Optional[Callable[[Member, QWidget], Any]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._member_service = member_service
        self._reenrollment_factory = reenrollment_factory
        self._members: Dict[str, Member] = {}
        self.name_edits: Dict[str, QLineEdit] = {}
        self.rename_buttons: Dict[str, QPushButton] = {}
        self.reenroll_buttons: Dict[str, QPushButton] = {}
        self.delete_buttons: Dict[str, QPushButton] = {}
        self.setWindowTitle("成员管理")
        self.setModal(True)
        self.resize(980, 620)
        self._build_ui()
        self.refresh()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(12)

        header = QHBoxLayout()
        title = QLabel("成员管理", self)
        title.setObjectName("titleLabel")
        self.search_edit = QLineEdit(self)
        self.search_edit.setPlaceholderText("搜索成员编号或姓名")
        self.search_edit.setClearButtonEnabled(True)
        self.count_label = QLabel("共 0 人", self)
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(self.search_edit, 1)
        header.addWidget(self.count_label)
        layout.addLayout(header)

        self.error_label = QLabel("", self)
        self.error_label.setObjectName("errorLabel")
        self.error_label.setWordWrap(True)
        self.error_label.hide()
        layout.addWidget(self.error_label)

        self.table = QTableWidget(0, 4, self)
        self.table.setHorizontalHeaderLabels(
            ["成员编号", "姓名", "登记时间", "操作"]
        )
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        table_header = self.table.horizontalHeader()
        table_header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        table_header.setSectionResizeMode(1, QHeaderView.Stretch)
        table_header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        table_header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        layout.addWidget(self.table, 1)

        close_button = QPushButton("关闭", self)
        close_button.clicked.connect(self.accept)
        footer = QHBoxLayout()
        footer.addStretch(1)
        footer.addWidget(close_button)
        layout.addLayout(footer)

        self.search_edit.textChanged.connect(self._apply_filter)
        self.setStyleSheet(
            """
            QDialog { background: #0b1220; color: #e5edf6; }
            QLabel#titleLabel { color: #60a5fa; font-size: 20px; font-weight: 700; }
            QLabel#errorLabel {
                background: #4a1d24; color: #fecaca; padding: 8px;
                border-radius: 6px;
            }
            QLineEdit, QTableWidget {
                background: #162235; color: #f8fafc; border: 1px solid #3d5675;
                border-radius: 6px; padding: 6px;
            }
            QHeaderView::section {
                background: #263b55; color: #f8fafc; padding: 8px;
                border: 0; border-right: 1px solid #3d5675;
            }
            QPushButton {
                background: #263b55; color: #f8fafc; border: 1px solid #3d5675;
                border-radius: 6px; padding: 7px 12px;
            }
            QPushButton:hover { background: #315174; }
            """
        )

    def _show_error(self, message: str) -> None:
        self.error_label.setText(message)
        self.error_label.show()

    def _clear_error(self) -> None:
        self.error_label.clear()
        self.error_label.hide()

    def refresh(self) -> None:
        self._clear_error()
        try:
            snapshot = self._member_service.list_members()
        except (MemberServiceError, ValueError, RuntimeError) as exc:
            self._show_error(str(exc))
            return
        self._render(snapshot)

    def _render(self, snapshot: Any) -> None:
        members = tuple(snapshot.members)
        self._members = {member.member_id: member for member in members}
        self.name_edits.clear()
        self.rename_buttons.clear()
        self.reenroll_buttons.clear()
        self.delete_buttons.clear()
        self.table.setRowCount(len(members))
        for row, member in enumerate(members):
            identifier = QTableWidgetItem(member.member_id)
            identifier.setData(Qt.UserRole, member.member_id)
            self.table.setItem(row, 0, identifier)

            name_edit = QLineEdit(member.name, self.table)
            self.table.setCellWidget(row, 1, name_edit)
            self.name_edits[member.member_id] = name_edit

            registered = member.registered_at.replace("T", " ")[:19]
            self.table.setItem(row, 2, QTableWidgetItem(registered))

            actions = QWidget(self.table)
            action_layout = QHBoxLayout(actions)
            action_layout.setContentsMargins(2, 2, 2, 2)
            action_layout.setSpacing(5)
            rename_button = QPushButton("保存姓名", actions)
            reenroll_button = QPushButton("重新采集", actions)
            delete_button = QPushButton("删除", actions)
            reenroll_button.setEnabled(self._reenrollment_factory is not None)
            rename_button.clicked.connect(
                lambda checked=False, value=member.member_id: self._rename(value)
            )
            reenroll_button.clicked.connect(
                lambda checked=False, value=member.member_id: self._reenroll(value)
            )
            delete_button.clicked.connect(
                lambda checked=False, value=member.member_id: self._delete(value)
            )
            for button in (rename_button, reenroll_button, delete_button):
                action_layout.addWidget(button)
            self.table.setCellWidget(row, 3, actions)
            self.rename_buttons[member.member_id] = rename_button
            self.reenroll_buttons[member.member_id] = reenroll_button
            self.delete_buttons[member.member_id] = delete_button
        self.count_label.setText("共 {} 人".format(len(members)))
        self._apply_filter(self.search_edit.text())

    def _apply_filter(self, text: str) -> None:
        query = text.strip().casefold()
        for row in range(self.table.rowCount()):
            identifier_item = self.table.item(row, 0)
            member_id = identifier_item.text()
            name = self.name_edits[member_id].text()
            visible = not query or query in member_id.casefold() or query in name.casefold()
            self.table.setRowHidden(row, not visible)

    def _rename(self, member_id: str) -> None:
        self._clear_error()
        try:
            snapshot = self._member_service.rename_member(
                member_id, self.name_edits[member_id].text()
            )
        except (MemberServiceError, ValueError, RuntimeError) as exc:
            self._show_error(str(exc))
            return
        self._render(snapshot)
        self.members_changed.emit(snapshot)

    def _delete(self, member_id: str) -> None:
        member = self._members[member_id]
        answer = QMessageBox.question(
            self,
            "确认删除",
            "确定删除成员 {}（{}）吗？此操作不可撤销。".format(
                member.name, member.member_id
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        self._clear_error()
        try:
            snapshot = self._member_service.delete_member(member_id)
        except (MemberServiceError, ValueError, RuntimeError) as exc:
            self._show_error(str(exc))
            return
        self._render(snapshot)
        self.members_changed.emit(snapshot)

    def _reenroll(self, member_id: str) -> None:
        if self._reenrollment_factory is None:
            return
        wizard = self._reenrollment_factory(self._members[member_id], self)
        wizard.member_saved.connect(self._on_reenrolled)
        wizard.exec_()

    def _on_reenrolled(self, snapshot: Any) -> None:
        self._render(snapshot)
        self.members_changed.emit(snapshot)
