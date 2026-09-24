import os
from pathlib import Path

import numpy as np
import pytest

from face_recognition_app.domain.member import Member
from face_recognition_app.storage.face_store import (
    DuplicateMemberError,
    FaceStore,
    FaceStoreError,
)


def test_empty_store(tmp_path):
    snapshot = FaceStore(tmp_path).load()

    assert snapshot.members == ()
    assert snapshot.embeddings.shape == (0, 0)


def test_add_rename_replace_delete_round_trip(tmp_path):
    store = FaceStore(tmp_path)
    first = Member.create("001", "张三")
    second = Member.create("002", "李四")

    store.add(first, np.asarray([3.0, 4.0]))
    store.add(second, np.asarray([0.0, 2.0]))
    renamed = store.rename("002", "李小四")
    replaced = store.replace_embedding("001", np.asarray([1.0, 0.0]))

    assert [member.name for member in renamed.members] == ["张三", "李小四"]
    assert replaced.embeddings[0].tolist() == pytest.approx([1.0, 0.0])

    final = store.delete("002")
    reloaded = store.load()
    assert [member.member_id for member in final.members] == ["001"]
    assert [member.member_id for member in reloaded.members] == ["001"]
    assert reloaded.embeddings.shape == (1, 2)


def test_duplicate_identifier_is_rejected(tmp_path):
    store = FaceStore(tmp_path)
    store.add(Member.create("001", "张三"), np.asarray([1.0, 0.0]))

    with pytest.raises(DuplicateMemberError):
        store.add(Member.create("001", "另一个人"), np.asarray([0.0, 1.0]))


def test_partial_store_is_rejected(tmp_path):
    (tmp_path / "members.json").write_text(
        '{"version": 1, "feature_dim": 0, "members": []}',
        encoding="utf-8",
    )

    with pytest.raises(FaceStoreError, match="must both exist"):
        FaceStore(tmp_path).load()


def test_failed_second_replace_restores_original_store(tmp_path, monkeypatch):
    store = FaceStore(tmp_path)
    original = Member.create("001", "张三")
    store.add(original, np.asarray([1.0, 0.0]))
    real_replace = os.replace
    failed = {"value": False}

    def fail_metadata_once(source, target):
        source_path = Path(source)
        target_path = Path(target)
        if (
            not failed["value"]
            and target_path.name == "members.json"
            and source_path.suffix == ".tmp"
        ):
            failed["value"] = True
            raise OSError("simulated metadata replace failure")
        return real_replace(source, target)

    monkeypatch.setattr(os, "replace", fail_metadata_once)
    with pytest.raises(FaceStoreError, match="unable to save"):
        store.add(Member.create("002", "李四"), np.asarray([0.0, 1.0]))

    snapshot = store.load()
    assert [member.member_id for member in snapshot.members] == ["001"]
    np.testing.assert_allclose(
        snapshot.embeddings,
        np.asarray([[1.0, 0.0]], dtype=np.float32),
    )
