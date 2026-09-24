"""Member enrollment business rules and face-store coordination."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from face_recognition_app.domain.member import Member
from face_recognition_app.storage.face_store import (
    DuplicateMemberError,
    FaceStore,
    FaceStoreError,
    FaceStoreSnapshot,
)


class MemberServiceError(RuntimeError):
    """Raised when a member operation cannot be completed."""


@dataclass(frozen=True)
class MemberDraft:
    member_id: str
    name: str
    same_name_warning: bool


class MemberService:
    def __init__(
        self,
        store: FaceStore,
        on_snapshot: Optional[Callable[[FaceStoreSnapshot], None]] = None,
    ) -> None:
        self._store = store
        self._on_snapshot = on_snapshot

    def validate_new_member(self, member_id: str, name: str) -> MemberDraft:
        clean_id = member_id.strip() if isinstance(member_id, str) else ""
        clean_name = name.strip() if isinstance(name, str) else ""
        if not clean_id:
            raise MemberServiceError("成员编号不能为空")
        if not clean_name:
            raise MemberServiceError("姓名不能为空")
        try:
            snapshot = self._store.load()
        except FaceStoreError as exc:
            raise MemberServiceError("读取成员库失败：{}".format(exc)) from exc
        if any(member.member_id == clean_id for member in snapshot.members):
            raise MemberServiceError("成员编号已存在")
        same_name = any(member.name == clean_name for member in snapshot.members)
        return MemberDraft(clean_id, clean_name, same_name)

    def add_member(
        self, member_id: str, name: str, embedding: np.ndarray
    ) -> FaceStoreSnapshot:
        draft = self.validate_new_member(member_id, name)
        member = Member.create(draft.member_id, draft.name)
        try:
            snapshot = self._store.add(member, embedding)
        except DuplicateMemberError as exc:
            raise MemberServiceError("成员编号已存在") from exc
        except (FaceStoreError, ValueError) as exc:
            raise MemberServiceError("保存成员失败：{}".format(exc)) from exc
        if self._on_snapshot is not None:
            self._on_snapshot(snapshot)
        return snapshot
