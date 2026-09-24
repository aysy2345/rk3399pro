"""Inference backends for face detection and feature extraction."""

from .interfaces import FaceDetection, FaceDetector, FaceEmbedder, InferenceError

__all__ = ["FaceDetection", "FaceDetector", "FaceEmbedder", "InferenceError"]
