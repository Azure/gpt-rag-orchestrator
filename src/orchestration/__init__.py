"""Orchestration package.

Package-level imports are **lazy** so that ``import orchestration.turn`` (and
by extension any import of the lightweight typed contract) does *not* eagerly
pull in the heavy ``Orchestrator`` class with its Azure SDK, Cosmos, and
OpenTelemetry transitive dependencies.

Consumers that only need the input/output contracts::

    from orchestration.turn import TurnRequest, TurnEvent   # zero heavy deps

Consumers that need the full orchestrator::

    from orchestration.orchestrator import Orchestrator     # explicit heavy import
    from orchestration import Orchestrator                   # lazy, deferred until accessed
"""
from __future__ import annotations

__all__ = [
    "Orchestrator",
    "TurnRequest",
    "TurnEvent",
]


def __getattr__(name: str):  # PEP 562 lazy attribute
    if name == "Orchestrator":
        from .orchestrator import Orchestrator
        return Orchestrator
    if name in ("TurnRequest", "TurnEvent"):
        from . import turn as _turn
        return getattr(_turn, name)
    raise AttributeError(f"module 'orchestration' has no attribute {name!r}")