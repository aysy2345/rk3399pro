"""Member domain model."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Mapping


class MemberValidationError(ValueError):
    """Raised when member data is invalid."""


def _clean_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MemberValidationError("{} must be non-empty text".format(field))
    return value.strip()


@dataclass(frozen=True)
class Member:
    member_id: str
    name: str
    registered_at: str
    active: bool = True

    @classmethod
    def create(cls, member_id: str, name: str) -> "Member":
        return cls(
            member_id=_clean_text(member_id, "member_id"),
            name=_clean_text(name, "name"),
            registered_at=datetime.now(timezone.utc).isoformat(),
            active=True,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Member":
        if not isinstance(data, Mapping):
            raise MemberValidationError("member must be an object")
        member_id = _clean_text(data.get("member_id"), "member_id")
        name = _clean_text(data.get("name"), "name")
        registered_at = _clean_text(data.get("registered_at"), "registered_at")
        try:
            datetime.fromisoformat(registered_at)
        except ValueError as exc:
            raise MemberValidationError("registered_at must be ISO-8601") from exc
        active = data.get("active", True)
        if not isinstance(active, bool):
            raise MemberValidationError("active must be true or false")
        return cls(member_id, name, registered_at, active)

    def renamed(self, name: str) -> "Member":
        return Member(
            member_id=self.member_id,
            name=_clean_text(name, "name"),
            registered_at=self.registered_at,
            active=self.active,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "member_id": self.member_id,
            "name": self.name,
            "registered_at": self.registered_at,
            "active": self.active,
        }
