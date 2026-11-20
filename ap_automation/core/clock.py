"""
A replaceable clock.

Production code asks `now()` for the current time. Replay experiments install a simulated clock so that
"evidence age" and write-back timestamps follow the transaction dates of the data being replayed instead of the
wall clock. The default is the real clock, so nothing changes unless a caller installs another one.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone

_clock: Callable[[], datetime] | None = None


def now() -> datetime:
    return _clock() if _clock is not None else datetime.now(timezone.utc)


def set_clock(fn: Callable[[], datetime] | None) -> None:
    """Install a clock (a no-arg callable returning an aware datetime). None restores the real clock."""
    global _clock
    _clock = fn


def is_simulated() -> bool:
    return _clock is not None
