"""Hosted agent entrypoint for Azure AI Foundry.

This FastAPI application wires the runtime-neutral orchestration core to the
Foundry Responses API streaming format.  It is intentionally separate from the
classic ``main.py`` so that:

- No Cosmos DB or orchestrator Container Apps dependency exists in the hosted
  execution path (conversation history is managed by Foundry Conversations).
- Only ADR-eligible strategies are admitted; unsupported strategies fail
  explicitly rather than silently falling back.
- The hosted image is immutable: the ``VERSION`` file is read once at startup
  and served on the ``GET /health`` endpoint.

Usage::

    uvicorn api.hosted_entrypoint:app --host 0.0.0.0 --port 8001

Or via Docker by overriding the entrypoint command.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Optional, Sequence

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from api.responses_adapter import responses_terminal_events, serialize_responses_events
from dependencies import get_config
from orchestration.turn import (
    TurnCancelledEvent,
    TurnCitationEvent,
    TurnConversationEvent,
    TurnErrorEvent,
    TurnOutputEvent,
    TurnRequest,
    TurnTextEvent,
    TurnToolActivityEvent,
)
from strategies.agent_strategy_factory import AgentStrategyFactory
from strategies.hosted_strategies import (
    HOSTED_ELIGIBLE_STRATEGIES,
    HostedConversationMessage,
    build_hosted_conversation,
    guard_hosted_strategy,
)

logger = logging.getLogger(__name__)

# ── Immutable version (read once; never changes at runtime) ──────────────────
_VERSION_FILE = Path(__file__).resolve().parents[2] / "VERSION"
try:
    _APP_VERSION: str = _VERSION_FILE.read_text().strip()
except FileNotFoundError:
    _APP_VERSION = "0.0.0"


# ── Pydantic schemas for the Foundry invocation contract ────────────────────

class InvocationMessage(BaseModel):
    """One message turn in the inbound invocation request."""

    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message text content")


class InvocationRequest(BaseModel):
    """Inbound invocation from the Foundry runtime to the hosted agent.

    The Foundry runtime provides at least one user message and an optional
    ``conversation_id`` that identifies the Foundry-managed Conversation thread.
    When ``conversation_id`` is absent the agent generates a transient id for
    the duration of the response.
    """

    messages: list[InvocationMessage] = Field(
        ...,
        min_length=1,
        description="Ordered message history; the last user message is the current ask.",
    )
    conversation_id: Optional[str] = Field(
        None,
        description="Foundry-managed conversation/thread id (optional).",
    )
    metadata: Optional[dict[str, Any]] = Field(
        default_factory=dict,
        description=(
            "Request metadata such as correlation_id and question_id. "
            "Caller-provided identity fields are not trusted by hosted mode."
        ),
    )


class HealthResponse(BaseModel):
    """Liveness/readiness response surfacing the immutable image version."""

    status: str = "ok"
    version: str
    eligible_strategies: list[str]


# ── Hosted execution path (no Cosmos) ───────────────────────────────────────

async def _hosted_stream(
    turn: TurnRequest,
    strategy_key: str,
    history: Sequence[HostedConversationMessage] = (),
) -> AsyncIterator[TurnOutputEvent]:
    """Run one turn with managed history and no Cosmos-backed runtime state.

    The caller supplies the complete ordered prior history from Foundry managed
    Conversations. Per-user profile memory remains disabled because the hosted
    invocation contract does not yet expose an authenticated Foundry identity.

    For Responses-backed server-thread strategies (``maf_agent_service``,
    ``single_agent_rag``), the Foundry-managed ``conversation_id`` is forwarded
    as the stable server-side thread id when it was provided by the Foundry
    runtime.  When absent, ``thread_id`` is intentionally left unset so each
    strategy can create a real Foundry conversation object on first use rather
    than receiving a synthesised UUID that is not a valid Foundry conversation.

    Raises :class:`ValueError` for unsupported strategies — never silently
    falls back.
    """
    guard_hosted_strategy(strategy_key)
    strategy = await AgentStrategyFactory.get_strategy(
        strategy_key,
        hosted_runtime=True,
    )

    external_conversation_id = turn.conversation_id is not None
    conversation_id = turn.conversation_id or str(uuid.uuid4())

    if hasattr(strategy, "set_context"):
        strategy.set_context(conversation_id)

    strategy.user_context = {}
    strategy.conversation = build_hosted_conversation(
        strategy_key,
        conversation_id,
        history,
        external_conversation_id=external_conversation_id,
    )

    yield TurnConversationEvent(conversation_id=conversation_id)

    try:
        async for chunk in strategy.initiate_agent_flow(turn.ask):
            if isinstance(chunk, (TurnCitationEvent, TurnToolActivityEvent)):
                yield chunk
            elif isinstance(chunk, str):
                yield TurnTextEvent(text=chunk)
            else:
                raise TypeError(
                    f"Strategy emitted unsupported chunk type: {type(chunk).__name__}"
                )
    except asyncio.CancelledError:
        yield TurnCancelledEvent()
        raise
    except Exception:
        yield TurnErrorEvent()
        raise


def _sse_generator(
    turn: TurnRequest,
    strategy_key: str,
    response_id: str,
    item_id: str,
    history: Sequence[HostedConversationMessage] = (),
) -> AsyncIterator[str]:
    """Wrap ``_hosted_stream`` and serialize events to Responses API SSE."""

    async def _gen() -> AsyncIterator[str]:
        full_text: list[str] = []
        error_emitted = False

        try:
            async for event in _hosted_stream(turn, strategy_key, history):
                frames = serialize_responses_events(
                    event,
                    response_id=response_id,
                    item_id=item_id,
                )
                if isinstance(event, TurnErrorEvent):
                    error_emitted = True
                if isinstance(event, TurnTextEvent):
                    full_text.append(event.text)
                for frame in frames:
                    yield frame

            # Emit closing frames only when no error occurred.
            if not error_emitted:
                for frame in responses_terminal_events(
                    response_id=response_id,
                    item_id=item_id,
                    full_text="".join(full_text),
                ):
                    yield frame

        except Exception:
            logger.exception("[hosted] Unhandled error in SSE generator")
            if not error_emitted:
                from api.responses_adapter import _sse
                yield _sse(
                    "error",
                    {
                        "type": "error",
                        "code": "internal_error",
                        "message": "An internal server error occurred.",
                        "retryable": False,
                    },
                )

    return _gen()


# ── FastAPI application ──────────────────────────────────────────────────────

app = FastAPI(
    title="GPT-RAG Hosted Agent",
    description=(
        "Foundry-hosted execution of the GPT-RAG orchestration core "
        "over the Responses API streaming contract."
    ),
    version=_APP_VERSION,
)


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness check",
    description=(
        "Returns the immutable image version and the set of strategies "
        "eligible for this hosted runtime.  Use for container readiness probes."
    ),
)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=_APP_VERSION,
        eligible_strategies=sorted(HOSTED_ELIGIBLE_STRATEGIES),
    )


@app.post(
    "/invocations",
    summary="Handle a Foundry invocation",
    description=(
        "Accepts one conversation turn from the Foundry runtime and streams "
        "a response in the Azure AI Foundry Responses API SSE format.  "
        "Only hosted-eligible strategies are admitted; unsupported strategies "
        "return HTTP 422."
    ),
    responses={
        200: {
            "description": "OK — Responses API SSE stream",
            "content": {"text/event-stream": {}},
        },
        422: {
            "description": "Unsupported strategy or validation error",
            "content": {
                "application/json": {
                    "example": {"detail": "Strategy 'multimodal' is not supported in the hosted runtime."}
                }
            },
        },
    },
)
async def invocations(body: InvocationRequest) -> StreamingResponse:
    """Translate a Foundry invocation into a Responses API SSE stream.

    The last ``user`` message in ``body.messages`` is used as the current ask.
    The ``conversation_id`` is passed through to the strategy as the Foundry
    Conversations thread id; when absent a transient id is generated.
    """
    # Derive the current ask and ordered prior history without duplicating the
    # current user message inside strategy input.
    # The Foundry invocation contract requires the last message to be the current
    # user turn; an assistant message (or any other role) at the end indicates
    # a malformed invocation — reject explicitly rather than silently discarding.
    if body.messages[-1].role != "user":
        raise HTTPException(
            status_code=422,
            detail=(
                "The last message must have role 'user'. "
                "The Foundry invocation contract requires the current user turn "
                "to be the final message in the list."
            ),
        )
    ask = body.messages[-1].content.strip()
    if not ask:
        raise HTTPException(
            status_code=422,
            detail="The last user message must not be empty.",
        )
    ask_index = len(body.messages) - 1

    # Resolve and guard the strategy.
    cfg = get_config()
    strategy_key = cfg.get("AGENT_STRATEGY", "maf_lite")

    try:
        guard_hosted_strategy(strategy_key)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # Build the transport-neutral turn request.
    metadata = body.metadata or {}
    history: list[HostedConversationMessage] = [
        {"role": message.role, "text": message.content}
        for message in body.messages[:ask_index]
    ]
    turn = TurnRequest(
        ask=ask,
        conversation_id=body.conversation_id,
        question_id=metadata.get("question_id"),
        user_context={},
        correlation_id=metadata.get("correlation_id"),
    )

    response_id = f"resp_{uuid.uuid4().hex}"
    item_id = f"item_{uuid.uuid4().hex}"

    logger.info(
        "[hosted] invocation: strategy=%s conversation_id=%s response_id=%s",
        strategy_key,
        body.conversation_id or "∅",
        response_id,
    )

    return StreamingResponse(
        _sse_generator(turn, strategy_key, response_id, item_id, history),
        media_type="text/event-stream",
        headers={"X-Response-ID": response_id},
    )
