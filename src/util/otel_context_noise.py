"""Suppress the benign OpenTelemetry context-detach noise produced by streaming.

``opentelemetry.context.detach`` logs ``Failed to detach context`` at ERROR --
with a full traceback -- whenever a context token is detached from a different
asyncio task than the one that attached it.

Instrumented async generators hit that condition by design. The generator
attaches the context on whichever task pulls a chunk, but it is closed either by
the garbage collector or by the ASGI server from a different task, so the detach
lands in a foreign context. Streaming a response therefore emits at least one of
these records per turn.

The record carries no actionable signal: ``detach`` swallows the underlying
exception, so nothing is lost and the request still succeeds. It only drowns
real errors in the log. The noise also originates inside ``agent_framework``'s
own GenAI instrumentation, which wraps ``yield`` statements in
``trace.use_span``, so it cannot be fixed at our call sites.

Only that one message is dropped -- every other ``opentelemetry.context`` record
still propagates.
"""

from __future__ import annotations

import logging

_CONTEXT_LOGGER_NAME = "opentelemetry.context"
_DETACH_FAILURE_MESSAGE = "failed to detach context"


class AsyncGeneratorContextDetachFilter(logging.Filter):
    """Drop ``Failed to detach context`` records, keeping every other record."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # pragma: no cover - defensive: never break logging
            return True
        return _DETACH_FAILURE_MESSAGE not in message.lower()


def silence_context_detach_noise() -> bool:
    """Install the filter on the ``opentelemetry.context`` logger.

    Idempotent, so it is safe to call from every observability entry point.
    Returns ``True`` when the filter was installed by this call.
    """
    context_logger = logging.getLogger(_CONTEXT_LOGGER_NAME)
    already_installed = any(
        isinstance(existing, AsyncGeneratorContextDetachFilter)
        for existing in context_logger.filters
    )
    if already_installed:
        return False
    context_logger.addFilter(AsyncGeneratorContextDetachFilter())
    return True
