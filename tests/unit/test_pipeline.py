import numpy as np

from face_recognition_app.core.pipeline import RecognitionPipeline
from face_recognition_app.domain.member import Member
from face_recognition_app.inference.fake import FakeFaceDetector, FakeFaceEmbedder
from face_recognition_app.inference.interfaces import FaceDetection
from face_recognition_app.storage.face_store import FaceStoreSnapshot


def detection(box=(10.0, 10.0, 90.0, 90.0)):
    return FaceDetection(
        box=box,
        score=0.95,
        landmarks=np.asarray(
            [[30, 40], [70, 40], [50, 58], [35, 75], [65, 75]],
            dtype=np.float32,
        ),
    )


def snapshot(member=True):
    if not member:
        return FaceStoreSnapshot((), np.empty((0, 0), dtype=np.float32))
    return FaceStoreSnapshot(
        (Member.create("001", "张三"),),
        np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
    )


def test_pipeline_confirms_known_identity_after_required_votes():
    detector = FakeFaceDetector([[detection()]] * 3)
    embedder = FakeFaceEmbedder([np.asarray([1.0, 0.0, 0.0])])
    pipeline = RecognitionPipeline(
        detector,
        embedder,
        snapshot(),
        threshold=0.8,
        window_size=5,
        votes_required=3,
    )
    frame = np.zeros((112, 112, 3), dtype=np.uint8)

    first = pipeline.process(frame)
    second = pipeline.process(frame)
    third = pipeline.process(frame)

    assert not first.faces[0].is_known
    assert not second.faces[0].is_known
    assert third.faces[0].is_known
    assert third.faces[0].member_id == "001"
    assert third.faces[0].name == "张三"
    assert third.faces[0].track_id == "face-1"


def test_pipeline_handles_multiple_faces_and_store_refresh():
    detector = FakeFaceDetector(
        [[detection(), detection((120.0, 10.0, 200.0, 90.0))], [detection()]]
    )
    embedder = FakeFaceEmbedder([np.asarray([1.0, 0.0, 0.0])])
    pipeline = RecognitionPipeline(
        detector,
        embedder,
        snapshot(),
        threshold=0.8,
        window_size=1,
        votes_required=1,
    )
    frame = np.zeros((220, 220, 3), dtype=np.uint8)

    result = pipeline.process(frame)
    pipeline.refresh_store(snapshot(member=False))
    refreshed = pipeline.process(frame)

    assert [face.track_id for face in result.faces] == ["face-1", "face-2"]
    assert all(face.is_known for face in result.faces)
    assert not refreshed.faces[0].is_known
    assert refreshed.faces[0].name == "陌生人"
    assert refreshed.frame_bgr is not frame
