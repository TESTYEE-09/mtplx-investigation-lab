"""Persistent SessionBank cold-tier primitives.

The cache-bank package deliberately sits below the serving layer and above raw
disk I/O. It serializes only committed SessionBank snapshots into immutable
bytes, then lets a single background writer persist them. No live MLX arrays
are ever handed to the writer thread.
"""

from .cold_tier import (
    COLD_TIER_FORMAT_VERSION,
    DEFAULT_COLD_TIER_DIR,
    DEFAULT_COLD_TIER_MAX_BYTES,
    DEFAULT_COLD_TIER_MIN_PREFIX_TOKENS,
    SessionBankColdTier,
    parse_size_bytes,
)

__all__ = [
    "COLD_TIER_FORMAT_VERSION",
    "DEFAULT_COLD_TIER_DIR",
    "DEFAULT_COLD_TIER_MAX_BYTES",
    "DEFAULT_COLD_TIER_MIN_PREFIX_TOKENS",
    "SessionBankColdTier",
    "parse_size_bytes",
]
