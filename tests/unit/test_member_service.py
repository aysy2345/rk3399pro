import numpy as np
import pytest

from face_recognition_app.app.member_service import (
    MemberService,
    MemberServiceError,
)
from face_recognition_app.storage.face_store import FaceStore, FaceStoreError


def test_validate_new_member_trims_text_and_warns_for_same_name(tmp_path):
    store = FaceStore(tmp_path)
    service = MemberService(store)
    service.add_member("001", "张三", np.array([1.0, 0.0], dtype=np.float32))

    draft = service.validate_new_member(" 002 ", " 张三 ")

    assert draft.member_id == "002"
    assert draft.name == "张三"
    assert draft.same_name_warning


@pytest.mark.parametrize(
    ("member_id", "name", "message"),
    [
        ("", "张三", "成员编号不能为空"),
        ("  ", "张三", "成员编号不能为空"),
        ("001", "", "姓名不能为空"),
        ("001", "  ", "姓名不能为空"),
    ],
)
def test_validate_new_member_rejects_empty_information(
    tmp_path, member_id, name, message
):
    service = MemberService(FaceStore(tmp_path))

    with pytest.raises(MemberServiceError, match=message):
        service.validate_new_member(member_id, name)


def test_validate_new_member_rejects_duplicate_identifier(tmp_path):
    service = MemberService(FaceStore(tmp_path))
    service.add_member("001", "张三", np.array([1.0, 0.0], dtype=np.float32))

    with pytest.raises(MemberServiceError, match="成员编号已存在"):
        service.validate_new_member("001", "李四")


def test_add_member_notifies_snapshot_once_after_success(tmp_path):
    snapshots = []
    service = MemberService(FaceStore(tmp_path), snapshots.append)

    snapshot = service.add_member(
        " 001 ", " 张三 ", np.array([3.0, 4.0], dtype=np.float32)
    )

    assert len(snapshots) == 1
    assert snapshots[0] is snapshot
    assert [member.member_id for member in snapshot.members] == ["001"]
    assert [member.name for member in snapshot.members] == ["张三"]
    np.testing.assert_allclose(snapshot.embeddings[0], [0.6, 0.8])


def test_add_member_does_not_notify_when_store_save_fails():
    class FailingStore:
        def load(self):
            return type("Snapshot", (), {"members": tuple()})()

        def add(self, member, embedding):
            raise FaceStoreError("disk full")

    snapshots = []
    service = MemberService(FailingStore(), snapshots.append)

    with pytest.raises(MemberServiceError, match="保存成员失败"):
        service.add_member("001", "张三", np.array([1.0, 0.0]))

    assert snapshots == []
