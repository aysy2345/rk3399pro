import numpy as np
from PyQt5.QtCore import QObject, Qt, pyqtSignal
from PyQt5.QtWidgets import QDialog, QMessageBox

from face_recognition_app.domain.member import Member
from face_recognition_app.storage.face_store import FaceStoreSnapshot
from face_recognition_app.ui.member_manager_dialog import MemberManagerDialog


def make_snapshot(items):
    members = tuple(Member.create(member_id, name) for member_id, name in items)
    matrix = np.eye(len(members), dtype=np.float32)
    return FaceStoreSnapshot(members, matrix)


class StubMemberService:
    def __init__(self):
        self.snapshot = make_snapshot(
            [("001", "张三"), ("002", "李四"), ("003", "王五")]
        )
        self.rename_calls = []
        self.delete_calls = []
        self.error = None

    def list_members(self):
        if self.error:
            raise self.error
        return self.snapshot

    def rename_member(self, member_id, name):
        if self.error:
            raise self.error
        self.rename_calls.append((member_id, name))
        items = [
            (member.member_id, name if member.member_id == member_id else member.name)
            for member in self.snapshot.members
        ]
        self.snapshot = make_snapshot(items)
        return self.snapshot

    def delete_member(self, member_id):
        if self.error:
            raise self.error
        self.delete_calls.append(member_id)
        items = [
            (member.member_id, member.name)
            for member in self.snapshot.members
            if member.member_id != member_id
        ]
        self.snapshot = make_snapshot(items)
        return self.snapshot


class StubReenrollmentDialog(QObject):
    member_saved = pyqtSignal(object)

    def __init__(self, snapshot):
        super().__init__()
        self.snapshot = snapshot
        self.exec_calls = 0

    def exec_(self):
        self.exec_calls += 1
        self.member_saved.emit(self.snapshot)
        return QDialog.Accepted


def make_dialog(qtbot):
    service = StubMemberService()
    opened = []

    def factory(member, parent):
        child = StubReenrollmentDialog(service.snapshot)
        opened.append((member, parent, child))
        return child

    dialog = MemberManagerDialog(service, factory)
    qtbot.addWidget(dialog)
    dialog.show()
    return service, opened, dialog


def test_dialog_lists_members_and_filters_by_identifier_or_name(qtbot):
    _, _, dialog = make_dialog(qtbot)

    assert dialog.table.rowCount() == 3
    assert dialog.count_label.text() == "共 3 人"

    dialog.search_edit.setText("002")
    assert [dialog.table.isRowHidden(row) for row in range(3)] == [
        True,
        False,
        True,
    ]
    dialog.search_edit.setText("王五")
    assert [dialog.table.isRowHidden(row) for row in range(3)] == [
        True,
        True,
        False,
    ]


def test_inline_rename_refreshes_table_and_emits_snapshot(qtbot):
    service, _, dialog = make_dialog(qtbot)
    changed = []
    dialog.members_changed.connect(changed.append)
    dialog.name_edits["001"].setText("张小三")

    qtbot.mouseClick(dialog.rename_buttons["001"], Qt.LeftButton)

    assert service.rename_calls == [("001", "张小三")]
    assert dialog.name_edits["001"].text() == "张小三"
    assert changed == [service.snapshot]


def test_delete_requires_confirmation_then_refreshes(qtbot, monkeypatch):
    service, _, dialog = make_dialog(qtbot)
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args, **kwargs: QMessageBox.Yes,
    )

    qtbot.mouseClick(dialog.delete_buttons["002"], Qt.LeftButton)

    assert service.delete_calls == ["002"]
    assert dialog.table.rowCount() == 2
    assert "002" not in dialog.name_edits


def test_reenrollment_uses_injected_wizard_and_refreshes(qtbot):
    service, opened, dialog = make_dialog(qtbot)
    changed = []
    dialog.members_changed.connect(changed.append)

    qtbot.mouseClick(dialog.reenroll_buttons["003"], Qt.LeftButton)

    assert opened[0][0].member_id == "003"
    assert opened[0][1] is dialog
    assert opened[0][2].exec_calls == 1
    assert changed == [service.snapshot]
