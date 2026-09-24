"""PyQt5 desktop user interface."""

from .enrollment_wizard import EnrollmentWizard
from .main_window import MainWindow
from .member_manager_dialog import MemberManagerDialog
from .video_widget import VideoWidget

__all__ = [
    "EnrollmentWizard",
    "MainWindow",
    "MemberManagerDialog",
    "VideoWidget",
]
