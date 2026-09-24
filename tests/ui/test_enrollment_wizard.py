import numpy as np
from PyQt5.QtCore import QObject, Qt, pyqtSignal
from PyQt5.QtWidgets import QDialog

from face_recognition_app.app.member_service import MemberDraft, MemberServiceError
from face_recognition_app.core.enrollment import EnrollmentTemplate
from face_recognition_app.core.enrollment_session import EnrollmentProgress
from face_recognition_app.ui.enrollment_wizard import EnrollmentWizard


class StubController(QObject):
    preview_ready = pyqtSignal(object)
    enrollment_progress = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.begin_calls = []
        self.cancel_calls = 0
        self.finish_calls = 0

    def begin_enrollment(self, session):
        self.begin_calls.append(session)

    def cancel_enrollment(self):
        self.cancel_calls += 1

    def finish_enrollment(self):
        self.finish_calls += 1


class StubSession:
    def __init__(self):
        self.template = EnrollmentTemplate(
            embedding=np.array([1.0, 0.0], dtype=np.float32),
            accepted_samples=15,
            rejected_samples=0,
        )

    def build_template(self):
        return self.template


class StubMemberService:
    def __init__(self):
        self.add_calls = []
        self.validation_error = None
        self.save_error = None
        self.same_name_warning = False

    def validate_new_member(self, member_id, name):
        if self.validation_error:
            raise self.validation_error
        return MemberDraft(
            member_id.strip(), name.strip(), self.same_name_warning
        )

    def add_member(self, member_id, name, embedding):
        if self.save_error:
            raise self.save_error
        self.add_calls.append((member_id, name, embedding.copy()))
        return object()


def make_wizard(qtbot):
    controller = StubController()
    service = StubMemberService()
    session = StubSession()
    wizard = EnrollmentWizard(controller, service, lambda: session)
    qtbot.addWidget(wizard)
    wizard.show()
    return controller, service, session, wizard


def fill_information(wizard):
    wizard.member_id_edit.setText("001")
    wizard.name_edit.setText("张三")


def complete_capture(controller):
    controller.enrollment_progress.emit(
        EnrollmentProgress(
            accepted=True,
            reason="有效样本",
            accepted_count=15,
            target_count=15,
            required_pose=None,
            complete=True,
        )
    )


def test_wizard_starts_on_information_page_and_blocks_invalid_input(qtbot):
    controller, service, _, wizard = make_wizard(qtbot)
    service.validation_error = MemberServiceError("成员编号不能为空")

    qtbot.mouseClick(wizard.next_button, Qt.LeftButton)

    assert wizard.pages.currentIndex() == 0
    assert wizard.error_label.text() == "成员编号不能为空"
    assert controller.begin_calls == []


def test_wizard_warns_for_same_name_but_starts_capture(qtbot):
    controller, service, session, wizard = make_wizard(qtbot)
    service.same_name_warning = True
    fill_information(wizard)

    qtbot.mouseClick(wizard.next_button, Qt.LeftButton)

    assert wizard.pages.currentIndex() == 1
    assert wizard.same_name_label.isVisible()
    assert "同名" in wizard.same_name_label.text()
    assert controller.begin_calls == [session]
    assert not wizard.capture_next_button.isEnabled()


def test_wizard_saves_only_after_complete_capture_and_confirmation(qtbot):
    controller, service, session, wizard = make_wizard(qtbot)
    fill_information(wizard)
    saved = []
    wizard.member_saved.connect(saved.append)

    qtbot.mouseClick(wizard.next_button, Qt.LeftButton)
    assert service.add_calls == []
    complete_capture(controller)
    assert wizard.capture_next_button.isEnabled()

    qtbot.mouseClick(wizard.capture_next_button, Qt.LeftButton)
    assert wizard.pages.currentIndex() == 2
    assert "001" in wizard.summary_label.text()
    assert "15" in wizard.summary_label.text()
    assert service.add_calls == []

    qtbot.mouseClick(wizard.save_button, Qt.LeftButton)

    assert len(service.add_calls) == 1
    assert service.add_calls[0][0:2] == ("001", "张三")
    np.testing.assert_allclose(service.add_calls[0][2], session.template.embedding)
    assert controller.finish_calls == 1
    assert controller.cancel_calls == 0
    assert len(saved) == 1
    assert wizard.result() == QDialog.Accepted


def test_wizard_cancel_during_capture_does_not_save(qtbot):
    controller, service, _, wizard = make_wizard(qtbot)
    fill_information(wizard)
    qtbot.mouseClick(wizard.next_button, Qt.LeftButton)

    qtbot.mouseClick(wizard.cancel_button, Qt.LeftButton)

    assert service.add_calls == []
    assert controller.cancel_calls == 1
    assert wizard.result() == QDialog.Rejected


def test_wizard_save_failure_stays_open_and_can_retry(qtbot):
    controller, service, _, wizard = make_wizard(qtbot)
    service.save_error = MemberServiceError("保存成员失败：disk full")
    fill_information(wizard)
    qtbot.mouseClick(wizard.next_button, Qt.LeftButton)
    complete_capture(controller)
    qtbot.mouseClick(wizard.capture_next_button, Qt.LeftButton)

    qtbot.mouseClick(wizard.save_button, Qt.LeftButton)

    assert wizard.pages.currentIndex() == 2
    assert "保存成员失败" in wizard.error_label.text()
    assert controller.finish_calls == 0
    assert controller.cancel_calls == 0
    assert wizard.save_button.isEnabled()
