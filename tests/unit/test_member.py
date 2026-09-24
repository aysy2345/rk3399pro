import pytest

from face_recognition_app.domain.member import Member, MemberValidationError


def test_member_round_trip_and_rename():
    member = Member.create("001", "张三")
    restored = Member.from_dict(member.to_dict())

    assert restored == member
    assert restored.renamed("李四").name == "李四"
    assert restored.renamed("李四").member_id == "001"


def test_member_rejects_blank_identifier():
    with pytest.raises(MemberValidationError, match="member_id"):
        Member.create(" ", "张三")
