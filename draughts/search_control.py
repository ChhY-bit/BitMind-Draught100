from __future__ import annotations

from threading import Event
from typing import Optional


class SearchCancelled(Exception):
    """A caller stopped a search; this is not an agent failure."""


def check_cancelled(cancel_event: Optional[Event]) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise SearchCancelled()
