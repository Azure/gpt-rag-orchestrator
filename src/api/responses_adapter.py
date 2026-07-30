"""Azure AI Foundry Responses API SSE serialization for typed turn events.

This module translates the runtime-neutral :class:`TurnOutputEvent` types into
the Foundry Responses API server-sent event (SSE) wire format so that the
hosted agent entrypoint can stream a standards-compliant response without
knowing about the internal orchestration details.

Reference event types follow the OpenAI Responses API streaming specification
as adopted by Azure AI Foundry.
"""

from __future__ import annotations

import json
from typing import assert_never

from orchestration.turn import (
    TurnCancelledEvent,
    TurnCitationEvent,
    TurnConversationEvent,
    TurnErrorEvent,
    TurnOutputEvent,
    TurnTextEvent,
    TurnToolActivityEvent,
    TurnToolStatus,
)


def _sse(event_type: str, data: dict) -> str:
    """Format one SSE frame from an event name and a JSON-serialisable dict."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


def serialize_responses_events(
    event: TurnOutputEvent,
    *,
    response_id: str,
    item_id: str,
    output_index: int = 0,
    content_index: int = 0,
) -> list[str]:
    """Translate one typed turn event into Foundry Responses API SSE frames.

    Returns a *list* because some events expand to multiple SSE frames.
    For example, :class:`TurnConversationEvent` opens the response envelope,
    the output item, and the text content part in one shot.

    A ``None`` return is never used here — callers that want to suppress an
    event should simply discard the returned empty list.
    """
    if isinstance(event, TurnConversationEvent):
        return [
            _sse(
                "response.created",
                {
                    "type": "response.created",
                    "response": {
                        "id": response_id,
                        "object": "response",
                        "status": "in_progress",
                        "conversation_id": event.conversation_id,
                        "output": [],
                    },
                },
            ),
            _sse(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "response_id": response_id,
                    "output_index": output_index,
                    "item": {
                        "id": item_id,
                        "type": "message",
                        "role": "assistant",
                        "status": "in_progress",
                        "content": [],
                    },
                },
            ),
            _sse(
                "response.content_part.added",
                {
                    "type": "response.content_part.added",
                    "item_id": item_id,
                    "output_index": output_index,
                    "content_index": content_index,
                    "part": {"type": "output_text", "text": ""},
                },
            ),
        ]

    if isinstance(event, TurnTextEvent):
        return [
            _sse(
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "item_id": item_id,
                    "output_index": output_index,
                    "content_index": content_index,
                    "delta": event.text,
                },
            )
        ]

    if isinstance(event, TurnCitationEvent):
        c = event.citation
        annotation: dict = {
            "type": "url_citation",
            "citation_id": c.citation_id,
        }
        if c.title is not None:
            annotation["title"] = c.title
        if c.url is not None:
            annotation["url"] = c.url
        if c.snippet is not None:
            annotation["snippet"] = c.snippet
        return [
            _sse(
                "response.output_text.annotation.added",
                {
                    "type": "response.output_text.annotation.added",
                    "item_id": item_id,
                    "output_index": output_index,
                    "content_index": content_index,
                    "annotation": annotation,
                },
            )
        ]

    if isinstance(event, TurnToolActivityEvent):
        a = event.activity
        status = a.status.value  # "started" | "completed" | "failed"
        if a.status == TurnToolStatus.STARTED:
            return [
                _sse(
                    "response.function_call_arguments.delta",
                    {
                        "type": "response.function_call_arguments.delta",
                        "call_id": a.call_id or "",
                        "name": a.tool_name,
                        "delta": "",
                        "status": status,
                    },
                )
            ]
        return [
            _sse(
                "response.function_call_arguments.done",
                {
                    "type": "response.function_call_arguments.done",
                    "call_id": a.call_id or "",
                    "name": a.tool_name,
                    "status": status,
                    **({"message": a.message} if a.message else {}),
                },
            )
        ]

    if isinstance(event, TurnErrorEvent):
        return [
            _sse(
                "error",
                {
                    "type": "error",
                    "code": event.code,
                    "message": event.message,
                    "retryable": event.retryable,
                },
            )
        ]

    if isinstance(event, TurnCancelledEvent):
        return [
            _sse(
                "response.cancelled",
                {
                    "type": "response.cancelled",
                    "reason": event.reason,
                },
            )
        ]

    assert_never(event)


def responses_terminal_events(
    *,
    response_id: str,
    item_id: str,
    full_text: str,
    output_index: int = 0,
    content_index: int = 0,
) -> list[str]:
    """Return the closing SSE frames emitted after all deltas have been sent.

    These frames close the content part, the output item, and the response
    envelope in order.  The caller accumulates the full text from
    :class:`TurnTextEvent` deltas and passes it here.
    """
    return [
        _sse(
            "response.output_text.done",
            {
                "type": "response.output_text.done",
                "item_id": item_id,
                "output_index": output_index,
                "content_index": content_index,
                "text": full_text,
            },
        ),
        _sse(
            "response.content_part.done",
            {
                "type": "response.content_part.done",
                "item_id": item_id,
                "output_index": output_index,
                "content_index": content_index,
                "part": {"type": "output_text", "text": full_text},
            },
        ),
        _sse(
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "response_id": response_id,
                "output_index": output_index,
                "item": {
                    "id": item_id,
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": full_text}],
                },
            },
        ),
        _sse(
            "response.completed",
            {
                "type": "response.completed",
                "response": {
                    "id": response_id,
                    "object": "response",
                    "status": "completed",
                },
            },
        ),
    ]
