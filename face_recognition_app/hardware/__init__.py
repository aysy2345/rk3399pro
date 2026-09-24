"""Hardware adapters."""

from .camera import Camera, CameraError, CameraReadError, OpenCVCamera

__all__ = ["Camera", "CameraError", "CameraReadError", "OpenCVCamera"]
