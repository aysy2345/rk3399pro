from face_recognition_app.core.tracking import IoUTracker


def test_tracker_keeps_identity_for_overlapping_box():
    tracker = IoUTracker(iou_threshold=0.3, max_missed=1)

    first = tracker.update([(10, 10, 50, 50)])
    second = tracker.update([(12, 11, 52, 51)])

    assert first == ["face-1"]
    assert second == ["face-1"]


def test_tracker_separates_faces_and_expires_missing_tracks():
    tracker = IoUTracker(iou_threshold=0.3, max_missed=1)

    assert tracker.update([(0, 0, 20, 20), (50, 0, 70, 20)]) == [
        "face-1",
        "face-2",
    ]
    tracker.update([])
    tracker.update([])

    assert tracker.update([(0, 0, 20, 20)]) == ["face-3"]
