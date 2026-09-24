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

    @staticmethod
    def _clean_member_id(member_id: str) -> str:
        clean_id = member_id.strip() if isinstance(member_id, str) else ""
        if not clean_id:
            raise MemberServiceError("成员编号不能为空")
        return clean_id

    @staticmethod
    def _clean_name(name: str) -> str:
        clean_name = name.strip() if isinstance(name, str) else ""
        if not clean_name:
            raise MemberServiceError("姓名不能为空")
        return clean_name

    def _notify(self, snapshot: FaceStoreSnapshot) -> FaceStoreSnapshot:
        if self._on_snapshot is not None:
            self._on_snapshot(snapshot)
        return snapshot

    def list_members(self) -> FaceStoreSnapshot:
        try:
            return self._store.load()
        except FaceStoreError as exc:
            raise MemberServiceError("读取成员库失败：{}".format(exc)) from exc

    def validate_new_member(self, member_id: str, name: str) -> MemberDraft:
        clean_id = self._clean_member_id(member_id)
        clean_name = self._clean_name(name)
        snapshot = self.list_members()
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
        return self._notify(snapshot)

    def rename_member(self, member_id: str, name: str) -> FaceStoreSnapshot:
        clean_id = self._clean_member_id(member_id)
        clean_name = self._clean_name(name)
        try:
            snapshot = self._store.rename(clean_id, clean_name)
        except (FaceStoreError, ValueError) as exc:
            raise MemberServiceError("修改姓名失败：{}".format(exc)) from exc
        return self._notify(snapshot)

    def replace_member_embedding(
        self, member_id: str, embedding: np.ndarray
    ) -> FaceStoreSnapshot:
        clean_id = self._clean_member_id(member_id)
        try:
            snapshot = self._store.replace_embedding(clean_id, embedding)
        except (FaceStoreError, ValueError) as exc:
            raise MemberServiceError("更新人脸特征失败：{}".format(exc)) from exc
        return self._notify(snapshot)

    def delete_member(self, member_id: str) -> FaceStoreSnapshot:
        clean_id = self._clean_member_id(member_id)
        try:
            snapshot = self._store.delete(clean_id)
        except (FaceStoreError, ValueError) as exc:
            raise MemberServiceError("删除成员失败：{}".format(exc)) from exc
        return self._notify(snapshot)
