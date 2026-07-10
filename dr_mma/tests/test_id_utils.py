"""UUIDv7 and ID utilities tests."""

import time
import uuid
import pytest
from dr_mma.engine.id_utils import (
    uuid7, uuid7_hex, uuid7_str,
    make_id, make_short_id, make_long_id,
    is_legacy_id,
)


class TestUUID7:
    def test_uuid7_returns_uuid_object(self):
        uid = uuid7()
        assert isinstance(uid, uuid.UUID)

    def test_uuid7_version_is_7(self):
        uid = uuid7()
        assert uid.version == 7

    def test_uuid7_variant_is_rfc(self):
        uid = uuid7()
        assert uid.variant == uuid.RFC_4122

    def test_uuid7_hex_length(self):
        h = uuid7_hex()
        assert len(h) == 32

    def test_uuid7_str_has_dashes(self):
        s = uuid7_str()
        assert "-" in s

    def test_uuid7_monotonic(self):
        """UUIDv7s generated in sequence should be strictly monotonic."""
        ids = [uuid7() for _ in range(100)]
        sorted_ids = sorted(ids)
        assert ids == sorted_ids


class TestMakeID:
    def test_make_id_default(self):
        iid = make_id()
        assert iid.startswith("ID-")
        assert len(iid) > 10

    def test_make_id_custom_prefix(self):
        iid = make_id("SES")
        assert iid.startswith("SES-")

    def test_make_id_contains_timestamp(self):
        before = int(time.time() * 1000)
        iid = make_id("T")
        after = int(time.time() * 1000)
        # UUIDv7 hex first 12 chars encode the timestamp (48 bits)
        ts_hex = iid.split("-")[1][:12]
        ts = int(ts_hex, 16)
        assert before <= ts <= after + 100  # allow slight skew

    def test_make_id_sortable(self):
        """IDs generated in sequence should sort chronologically."""
        ids = [make_id() for _ in range(50)]
        sorted_ids = sorted(ids)
        assert ids == sorted_ids

    def test_make_short_id(self):
        iid = make_short_id("TASK")
        assert iid.startswith("TASK-")

    def test_make_long_id(self):
        iid = make_long_id("ART")
        assert iid.startswith("ART-")
        # Long IDs use full UUIDv7 hex (32 chars after prefix)
        hex_part = iid.split("-")[1]
        assert len(hex_part) == 32


class TestIsLegacyID:
    def test_legacy_format_short(self):
        assert is_legacy_id("SES-a1b2c3d4")
        assert is_legacy_id("T-xyz7890")

    def test_new_format_full_hex(self):
        assert not is_legacy_id("SES-0000018a5b3fa1b2c3d4e5f6a7b8c9d0")
        assert not is_legacy_id("TASK-0000018a5b3fa1b2c3d4e5f6a7b8c9d0")

    def test_empty_string(self):
        assert is_legacy_id("")
