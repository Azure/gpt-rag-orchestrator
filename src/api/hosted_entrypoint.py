"""Hosted agent entrypoint for Azure AI Foundry.

This FastAPI application wires the runtime-neutral orchestration core to the
Foundry Responses API streaming format.  It is intentionally separate from the
classic ``main.py`` so that:

- No Cosmos DB reads or writes occur in the hosted execution path.
  Conversation history comes from Foundry Conversations (the ordered
  ``messages`` list in each invocation request) and is injected directly into
  the strategy's ``conversation["messages"]`` on every turn.  Per-user profile
  memory (which the classic path persists to Cosmos) is disabled via the
  ``"hosted_mode": True`` sentinel in the conversation dict; strategies skip
  profile load and save when this flag is set.
- For ``maf_agent_service``, server-side thread continuity within a process is
  maintained via an in-memory mapping of ``conversation_id → thread_id``
  (``_maf_thread_cache``).  The cache is scoped to a single process and is lost
  on restart; Foundry Conversations continues to own the authoritative message
  history.
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
from typing import Any, Optional

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
from strategies.hosted_strategies import HOSTED_ELIGIBLE_STRATEGIES, guard_hosted_strategy

logger = logging.getLogger(__name__)

# In-process thread-id cache for maf_agent_service continuity.
# Maps conversation_id → Foundry Agent Service thread_id so that consecutive
# turns within the same conversation reuse the same server-side thread.
# The cache is local to this process; Foundry Conversations owns the
# authoritative message history and is not affected by a process restart.
_maf_thread_cache: dict[str, str] = {}

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
        description="Arbitrary pass-through metadata (e.g. correlation_id, run_id).",
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
    history_messages: list[dict] | None = None,
) -> AsyncIterator[TurnOutputEvent]:
    """Run one turn via the strategy without any Cosmos DB dependency.

    Conversation history is injected from *history_messages* (the ordered prior
    turns from the Foundry invocation request) directly into
    ``strategy.conversation["messages"]``.  The ``"hosted_mode": True`` sentinel
    tells history-owning strategies (``maf_lite``, ``maf_agent_service``) to skip
    per-user profile load and save so no Cosmos DB operations occur.

    For ``maf_agent_service`` the server-side thread id is restored from the
    in-process ``_maf_thread_cache`` (keyed by *conversation_id*) so consecutive
    turns within the same Foundry Conversation reuse the same agent thread.

    Raises :class:`ValueError` for unsupported strategies — never silently
    falls back.
    """
    guard_hosted_strategy(strategy_key)
    strategy = await AgentStrategyFactory.get_strategy(strategy_key)

    conversation_id = turn.conversation_id or str(uuid.uuid4())

    if hasattr(strategy, "set_context"):
        strategy.set_context(conversation_id)

    strategy.user_context = turn.user_context or {}
    strategy.conversation = {
        "id": conversation_id,
        "hosted_mode": True,
        "messages": list(history_messages or []),
    }

    # Restore the Agent Service server-side thread so maf_agent_service resumes
    # the same thread for every turn in the same Foundry Conversation.
    if strategy_key == "maf_agent_service" and conversation_id in _maf_thread_cache:
        strategy.conversation["thread_id"] = _maf_thread_cache[conversation_id]

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
    finally:
        # Persist thread_id for maf_agent_service so the next turn in the same
        # Foundry Conversation can resume the existing server-side thread.
        if strategy_key == "maf_agent_service":
            thread_id = strategy.conversation.get("thread_id")
            if thread_id:
                _maf_thread_cache[conversation_id] = thread_id


def _sse_generator(
    turn: TurnRequest,
    strategy_key: str,
    response_id: str,
    item_id: str,
    history_messages: list[dict] | None = None,
) -> AsyncIterator[str]:
    """Wrap ``_hosted_stream`` and serialize events to Responses API SSE."""

    async def _gen() -> AsyncIterator[str]:
        full_text: list[str] = []
        error_emitted = False

        try:
            async for event in _hosted_stream(turn, strategy_key, history_messages):
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

    All messages in ``body.messages`` before the last user message are treated
    as the prior conversation history and injected into
    ``strategy.conversation["messages"]`` so that history-owning strategies
    (``maf_lite``, ``single_agent_rag``) restore multi-turn context.  The last
    ``user`` message is the current ask.  ``conversation_id`` is passed through
    to the strategy; when absent a transient id is generated.
    """
    # Derive the current ask from the last user message.
    user_messages = [m for m in body.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(
            status_code=422,
            detail="At least one message with role='user' is required.",
        )
    ask = user_messages[-1].content.strip()
    if not ask:
        raise HTTPException(
            status_code=422,
            detail="The last user message must not be empty.",
        )

    # Resolve and guard the strategy.
    cfg = get_config()
    strategy_key = cfg.get("AGENT_STRATEGY", "maf_lite")

    try:
        guard_hosted_strategy(strategy_key)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # Build prior-turn history: all messages before the last user message.
    # This maps the ordered Foundry Conversations history into the exact
    # ``conversation["messages"]`` dict format consumed by maf_lite and
    # single_agent_rag (both accept either "content" or "text" key).
    last_user_idx = max(i for i, m in enumerate(body.messages) if m.role == "user")
    history_messages = [
        {"role": m.role, "content": m.content}
        for m in body.messages[:last_user_idx]
    ]

    # Build the transport-neutral turn request.
    metadata = body.metadata or {}
    turn = TurnRequest(
        ask=ask,
        conversation_id=body.conversation_id,
        question_id=metadata.get("question_id"),
        user_context=metadata.get("user_context") or {},
        correlation_id=metadata.get("correlation_id"),
    )

    response_id = f"resp_{uuid.uuid4().hex}"
    item_id = f"item_{uuid.uuid4().hex}"

    logger.info(
        "[hosted] invocation: strategy=%s conversation_id=%s history=%d response_id=%s",
        strategy_key,
        body.conversation_id or "∅",
        len(history_messages),
        response_id,
    )

    return StreamingResponse(
        _sse_generator(turn, strategy_key, response_id, item_id, history_messages),
        media_type="text/event-stream",
        headers={"X-Response-ID": response_id},
    )
