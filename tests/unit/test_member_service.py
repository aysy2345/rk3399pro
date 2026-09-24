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


def test_list_rename_replace_and_delete_members(tmp_path):
    snapshots = []
    service = MemberService(FaceStore(tmp_path), snapshots.append)
    service.add_member("001", "张三", np.array([1.0, 0.0], dtype=np.float32))
    service.add_member("002", "李四", np.array([0.0, 1.0], dtype=np.float32))
    snapshots.clear()

    loaded = service.list_members()
    assert [member.member_id for member in loaded.members] == ["001", "002"]

    renamed = service.rename_member("001", " 张小三 ")
    assert renamed.members[0].name == "张小三"
    replaced = service.replace_member_embedding(
        "001", np.array([1.0, 1.0], dtype=np.float32)
    )
    np.testing.assert_allclose(
        replaced.embeddings[0],
        np.array([1.0, 1.0]) / np.sqrt(2.0),
    )
    deleted = service.delete_member("002")
    assert [member.member_id for member in deleted.members] == ["001"]
    assert snapshots == [renamed, replaced, deleted]


def test_member_updates_validate_input_and_wrap_store_errors():
    class FailingStore:
        def load(self):
            raise FaceStoreError("broken")

        def rename(self, member_id, name):
            raise FaceStoreError("disk full")

        def replace_embedding(self, member_id, embedding):
            raise FaceStoreError("disk full")

        def delete(self, member_id):
            raise FaceStoreError("disk full")

    snapshots = []
    service = MemberService(FailingStore(), snapshots.append)

    with pytest.raises(MemberServiceError, match="读取成员库失败"):
        service.list_members()
    with pytest.raises(MemberServiceError, match="姓名不能为空"):
        service.rename_member("001", " ")
    with pytest.raises(MemberServiceError, match="修改姓名失败"):
        service.rename_member("001", "张三")
    with pytest.raises(MemberServiceError, match="更新人脸特征失败"):
        service.replace_member_embedding("001", np.array([1.0, 0.0]))
    with pytest.raises(MemberServiceError, match="删除成员失败"):
        service.delete_member("001")
    assert snapshots == []
