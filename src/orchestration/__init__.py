from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .turn import (
    TurnCancelledEvent,
    TurnCitation,
    TurnCitationEvent,
    TurnConversationEvent,
    TurnErrorEvent,
    TurnOutputEvent,
    TurnRequest,
    TurnTextEvent,
    TurnToolActivity,
    TurnToolActivityEvent,
    TurnToolStatus,
)

if TYPE_CHECKING:
    from .orchestrator import Orchestrator

__all__ = [
    "Orchestrator",
    "TurnCancelledEvent",
    "TurnCitation",
    "TurnCitationEvent",
    "TurnConversationEvent",
    "TurnErrorEvent",
    "TurnOutputEvent",
    "TurnRequest",
    "TurnTextEvent",
    "TurnToolActivity",
    "TurnToolActivityEvent",
    "TurnToolStatus",
]


def __getattr__(name: str) -> Any:
    if name == "Orchestrator":
        from .orchestrator import Orchestrator

        return Orchestrator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")