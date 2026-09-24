"""Atomic local member and embedding storage."""

from __future__ import annotations

import json
import os
import shutil
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np

from face_recognition_app.core.matcher import normalize_rows, normalize_vector
from face_recognition_app.domain.member import Member


class FaceStoreError(RuntimeError):
    """Raised when the local face store is missing or inconsistent."""


class DuplicateMemberError(FaceStoreError):
    """Raised when a member identifier already exists."""


@dataclass(frozen=True)
class FaceStoreSnapshot:
    members: Sequence[Member]
    embeddings: np.ndarray


class FaceStore:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        self.members_path = self.data_dir / "members.json"
        self.embeddings_path = self.data_dir / "embeddings.npy"
        self._lock = threading.RLock()

    def load(self) -> FaceStoreSnapshot:
        with self._lock:
            metadata_exists = self.members_path.is_file()
            embeddings_exist = self.embeddings_path.is_file()
            if not metadata_exists and not embeddings_exist:
                return FaceStoreSnapshot(tuple(), np.empty((0, 0), dtype=np.float32))
            if metadata_exists != embeddings_exist:
                raise FaceStoreError("members.json and embeddings.npy must both exist")
            try:
                with self.members_path.open("r", encoding="utf-8") as handle:
                    metadata = json.load(handle)
                embeddings = np.load(str(self.embeddings_path), allow_pickle=False)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                raise FaceStoreError("unable to load face store: {}".format(exc)) from exc
            return self._validate_loaded(metadata, embeddings)

    def _validate_loaded(
        self, metadata: object, embeddings: np.ndarray
    ) -> FaceStoreSnapshot:
        if not isinstance(metadata, dict) or metadata.get("version") != 1:
            raise FaceStoreError("unsupported or missing face-store version")
        raw_members = metadata.get("members")
        if not isinstance(raw_members, list):
            raise FaceStoreError("members must be a list")
        try:
            members = tuple(Member.from_dict(item) for item in raw_members)
            matrix = normalize_rows(embeddings)
        except ValueError as exc:
            raise FaceStoreError(str(exc)) from exc
        identifiers = [member.member_id for member in members]
        if len(set(identifiers)) != len(identifiers):
            raise FaceStoreError("member identifiers must be unique")
        if matrix.shape[0] != len(members):
            raise FaceStoreError("member count and embedding count differ")
        feature_dim = metadata.get("feature_dim", 0)
        if not isinstance(feature_dim, int) or feature_dim < 0:
            raise FaceStoreError("feature_dim must be a non-negative integer")
        if members and matrix.shape[1] != feature_dim:
            raise FaceStoreError("feature dimension does not match metadata")
        if not members and matrix.shape != (0, 0):
            raise FaceStoreError("an empty face store must use a 0x0 embedding matrix")
        return FaceStoreSnapshot(members, matrix)

    def save(self, members: Sequence[Member], embeddings: np.ndarray) -> None:
        with self._lock:
            member_list = list(members)
            identifiers = [member.member_id for member in member_list]
            if len(set(identifiers)) != len(identifiers):
                raise DuplicateMemberError("member identifiers must be unique")
            if member_list:
                matrix = normalize_rows(embeddings)
                if matrix.shape[0] != len(member_list):
                    raise FaceStoreError("member count and embedding count differ")
            else:
                matrix = np.empty((0, 0), dtype=np.float32)

            metadata = {
                "version": 1,
                "feature_dim": int(matrix.shape[1]) if member_list else 0,
                "members": [member.to_dict() for member in member_list],
            }
            self._atomic_write(metadata, matrix)

    def _atomic_write(self, metadata: dict, embeddings: np.ndarray) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        metadata_had_original = self.members_path.exists()
        embeddings_had_original = self.embeddings_path.exists()
        token = uuid.uuid4().hex
        metadata_tmp = self.data_dir / ("members.{}.tmp".format(token))
        embeddings_tmp = self.data_dir / ("embeddings.{}.tmp".format(token))
        metadata_backup = self.data_dir / ("members.{}.bak".format(token))
        embeddings_backup = self.data_dir / ("embeddings.{}.bak".format(token))

        try:
            with metadata_tmp.open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(metadata, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            with embeddings_tmp.open("wb") as handle:
                np.save(handle, embeddings, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())

            if self.members_path.exists():
                shutil.copy2(str(self.members_path), str(metadata_backup))
            if self.embeddings_path.exists():
                shutil.copy2(str(self.embeddings_path), str(embeddings_backup))

            os.replace(str(embeddings_tmp), str(self.embeddings_path))
            os.replace(str(metadata_tmp), str(self.members_path))
        except Exception as exc:
            self._restore_backup(
                metadata_backup, self.members_path, metadata_had_original
            )
            self._restore_backup(
                embeddings_backup, self.embeddings_path, embeddings_had_original
            )
            raise FaceStoreError("unable to save face store: {}".format(exc)) from exc
        finally:
            for path in (
                metadata_tmp,
                embeddings_tmp,
                metadata_backup,
                embeddings_backup,
            ):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass

    @staticmethod
    def _restore_backup(backup: Path, target: Path, had_original: bool) -> None:
        if backup.exists():
            os.replace(str(backup), str(target))
        elif not had_original:
            try:
                target.unlink()
            except FileNotFoundError:
                pass

    def add(self, member: Member, embedding: np.ndarray) -> FaceStoreSnapshot:
        with self._lock:
            snapshot = self.load()
            if any(item.member_id == member.member_id for item in snapshot.members):
                raise DuplicateMemberError(member.member_id)
            vector = normalize_vector(embedding)
            if snapshot.members and vector.shape[0] != snapshot.embeddings.shape[1]:
                raise FaceStoreError("feature dimension does not match existing store")
            matrix = (
                vector.reshape(1, -1)
                if not snapshot.members
                else np.vstack([snapshot.embeddings, vector])
            )
            members = list(snapshot.members) + [member]
            self.save(members, matrix)
            return FaceStoreSnapshot(tuple(members), normalize_rows(matrix))

    def rename(self, member_id: str, name: str) -> FaceStoreSnapshot:
        with self._lock:
            snapshot = self.load()
            found = False
            members: List[Member] = []
            for member in snapshot.members:
                if member.member_id == member_id:
                    members.append(member.renamed(name))
                    found = True
                else:
                    members.append(member)
            if not found:
                raise FaceStoreError("member not found: {}".format(member_id))
            self.save(members, snapshot.embeddings)
            return FaceStoreSnapshot(tuple(members), snapshot.embeddings.copy())

    def replace_embedding(
        self, member_id: str, embedding: np.ndarray
    ) -> FaceStoreSnapshot:
        with self._lock:
            snapshot = self.load()
            indices = [
                index
                for index, member in enumerate(snapshot.members)
                if member.member_id == member_id
            ]
            if not indices:
                raise FaceStoreError("member not found: {}".format(member_id))
            vector = normalize_vector(embedding)
            if vector.shape[0] != snapshot.embeddings.shape[1]:
                raise FaceStoreError("feature dimension does not match existing store")
            matrix = snapshot.embeddings.copy()
            matrix[indices[0]] = vector
            self.save(snapshot.members, matrix)
            return FaceStoreSnapshot(snapshot.members, matrix)

    def delete(self, member_id: str) -> FaceStoreSnapshot:
        with self._lock:
            snapshot = self.load()
            keep = [
                index
                for index, member in enumerate(snapshot.members)
                if member.member_id != member_id
            ]
            if len(keep) == len(snapshot.members):
                raise FaceStoreError("member not found: {}".format(member_id))
            members = [snapshot.members[index] for index in keep]
            matrix = (
                snapshot.embeddings[keep]
                if members
                else np.empty((0, 0), dtype=np.float32)
            )
            self.save(members, matrix)
            return FaceStoreSnapshot(tuple(members), normalize_rows(matrix))
