"""
ID Utilities — UUIDv7 time-ordered IDs and prefixed entity IDs.

UUIDv7 generates time-sortable unique identifiers:
  - First 48 bits: Unix timestamp in milliseconds
  - Next 4 bits: Version (0111 = 7)
  - Remaining 76 bits: Random/counter with 2-bit variant (10 = RFC 4122)

UUIDs generated within the same millisecond use an incrementing counter
in the lower bits to guarantee monotonic order.

All entity IDs follow the pattern: PREFIX-UUIDv7_HEX
This ensures lexicographic sorting matches chronological order.
"""

import os
import threading
import time
import uuid


# ── Monotonic UUIDv7 with thread-safe counter ───────────────────────────────

_last_ts: int = 0
_seq_counter: int = 0
_seq_lock = threading.Lock()


def uuid7() -> uuid.UUID:
    """
    Generate a UUIDv7 (time-ordered) UUID.
    
    Monotonic within the same millisecond: uses an incrementing
    12-bit sequence counter to guarantee sort order.
    
    Format:
      - 48 bits: Unix timestamp in milliseconds
      -  4 bits: Version (0111 = 7)
      - 12 bits: Sequence counter (0-4095, resets on timestamp change)
      - 62 bits: Random
      -  2 bits: Variant (10 = RFC 4122)
    Total: 128 bits
    """
    global _last_ts, _seq_counter
    
    timestamp_ms = int(time.time() * 1000)
    
    with _seq_lock:
        if timestamp_ms == _last_ts:
            _seq_counter = (_seq_counter + 1) & 0xFFF  # 12-bit counter
        else:
            _seq_counter = 0
            _last_ts = timestamp_ms
        seq = _seq_counter
    
    random_bytes = os.urandom(8)
    
    # Build the 16 bytes
    bytes_arr = bytearray(16)
    
    # Bytes 0-5: timestamp (48 bits, big-endian)
    ts = timestamp_ms
    for i in range(5, -1, -1):
        bytes_arr[i] = ts & 0xFF
        ts >>= 8
    
    # Byte 6: 4-bit version (0111 = 7) + 4 high bits of sequence counter
    bytes_arr[6] = (0x70 | ((seq >> 8) & 0x0F))
    
    # Byte 7: 8 low bits of sequence counter (part of the 12-bit counter)
    bytes_arr[7] = seq & 0xFF
    
    # Byte 8: 2-bit variant (10 = RFC 4122) + 6 bits random
    bytes_arr[8] = (0x80 | (random_bytes[0] & 0x3F))
    
    # Bytes 9-15: random (56 bits)
    for i in range(9, 16):
        bytes_arr[i] = random_bytes[i - 8]
    
    return uuid.UUID(bytes=bytes(bytes_arr))


def uuid7_hex() -> str:
    """Return UUIDv7 hex string (32 hex chars, no dashes)."""
    return uuid7().hex


def uuid7_str() -> str:
    """Return UUIDv7 string with dashes (standard UUID format)."""
    return str(uuid7())


# ── Random suffix helper ───────────────────────────────────────────────────

def _random_suffix(length: int = 8) -> str:
    """Return a random hex suffix of given length."""
    return os.urandom(length // 2 + 1).hex()[:length]


# ── Prefixed Entity IDs ────────────────────────────────────────────────────

def make_id(prefix: str = "ID", hex_length: int = 32) -> str:
    """
    Generate a time-sortable prefixed ID.
    
    Format: {PREFIX}-{UUIDv7_HEX[:hex_len]}
    Example: SES-0000018a5b3fa1b2c3d4e5f6
    
    UUIDv7 hex already contains the timestamp in its first 12 characters,
    so no separate timestamp prefix is needed. IDs are strictly monotonic
    even within the same millisecond. Sorting alphabetically == sorting
    chronologically.
    
    Args:
        prefix: Entity type prefix (e.g. "SES", "MSG", "TASK", "ART")
        hex_length: Hex chars after prefix (default 32 = full UUIDv7)
    
    Returns:
        Prefixed ID string
    """
    uid_hex = uuid7_hex()
    max_hex = min(hex_length, 32)
    return f"{prefix}-{uid_hex[:max_hex]}"


def make_short_id(prefix: str = "ID") -> str:
    """Generate a compact time-sortable ID (20 chars total)."""
    return make_id(prefix, hex_length=16)


def make_long_id(prefix: str = "ID") -> str:
    """Generate a full UUIDv7-based prefixed ID (37 chars total)."""
    return f"{prefix}-{uuid7_hex()}"


# ── Backward-compatible ID detection ────────────────────────────────────────

def is_legacy_id(entity_id: str) -> bool:
    """
    Check if an ID uses the legacy format.
    
    Legacy short: PREFIX-XXXXXXXX (prefix + 8 hex chars, no UUIDv7 structure)
    New format:   PREFIX-{32 hex chars of UUIDv7}
    
    Detects legacy by checking hex part length < 24 (new is always 32).
    """
    if "-" not in entity_id:
        return True
    hex_part = entity_id.split("-", 1)[1]
    return len(hex_part) < 24
