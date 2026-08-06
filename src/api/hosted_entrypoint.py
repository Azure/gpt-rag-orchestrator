"""Hosted agent entrypoint for Azure AI Foundry.

This FastAPI application wires the runtime-neutral orchestration core to the
Foundry Responses API streaming format.  It is intentionally separate from the
classic ``main.py`` so that:

- No Cosmos DB or orchestrator Container Apps dependency exists in the hosted
  execution path (conversation history is managed by Foundry Conversations).
- Only ADR-eligible strategies are admitted; unsupported strategies fail
  explicitly rather than silently falling back.
- The hosted image is immutable: the ``VERSION`` file is read once at startup
  and served on the ``GET /readiness`` and compatibility ``GET /health``
  endpoints.

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
from typing import Any, Literal, Optional, Sequence

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from api.responses_adapter import responses_terminal_events, serialize_responses_events
from connectors.foundry_conversations import resolve_managed_conversation_id
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
    HOSTED_SERVER_THREAD_STRATEGIES,
    HOSTED_TOOLBOX_STRATEGIES,
    HostedConversationMessage,
    build_hosted_conversation,
    guard_hosted_strategy,
)
from util.foundry_platform import (
    MISSING_CALL_CONTEXT_MESSAGE,
    MissingFoundryCallContextError,
    require_foundry_call_id,
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

    role: Literal["user", "assistant"] = Field(
        ...,
        description="Message role: 'user' or 'assistant'",
    )
    content: str = Field(..., description="Message text content")


class InvocationRequest(BaseModel):
    """Inbound invocation from the Foundry runtime to the hosted agent.

    The final message is the current user ask. An optional ``conversation_id``
    identifies the Foundry-managed Conversation. Responses-backed strategies
    validate a supplied id or create a real managed Conversation when it is
    absent.
    """

    messages: list[InvocationMessage] = Field(
        ...,
        min_length=1,
        description="Ordered message history; the final message is the current user ask.",
    )
    conversation_id: Optional[str] = Field(
        None,
        description=(
            "Foundry-managed Conversation id. Responses-backed strategies create "
            "one through the Foundry SDK when it is omitted."
        ),
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
    Conversations. Responses-backed strategies validate or create the backing
    managed Conversation through the Foundry SDK before using its id as a
    service thread. Per-user profile memory remains disabled because the hosted
    invocation contract does not yet expose an authenticated Foundry identity.

    Raises :class:`ValueError` for unsupported strategies — never silently
    falls back. Raises :class:`~util.foundry_platform.MissingFoundryCallContextError`
    (a :class:`ValueError` subclass) when *strategy_key* is Toolbox-integrated
    and the turn carries no validated Foundry call id — hosted retrieval must
    fail closed rather than fall back to service identity or a manual filter.
    """
    guard_hosted_strategy(strategy_key)
    if strategy_key in HOSTED_TOOLBOX_STRATEGIES and not turn.foundry_call_id:
        raise MissingFoundryCallContextError(MISSING_CALL_CONTEXT_MESSAGE)
    strategy = await AgentStrategyFactory.get_strategy(
        strategy_key,
        hosted_runtime=True,
    )

    if strategy_key in HOSTED_SERVER_THREAD_STRATEGIES:
        conversation_id = await resolve_managed_conversation_id(
            strategy.project_client,
            turn.conversation_id,
        )
    else:
        conversation_id = turn.conversation_id or str(uuid.uuid4())

    if hasattr(strategy, "set_context"):
        strategy.set_context(conversation_id)

    strategy.user_context = {}
    strategy.foundry_call_id = turn.foundry_call_id
    strategy.conversation = build_hosted_conversation(
        strategy_key,
        conversation_id,
        history,
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

        except MissingFoundryCallContextError as exc:
            # Defense-in-depth only: the normal ``/invocations`` path already
            # rejects a missing/malformed call id with an HTTP 401 *before*
            # ``StreamingResponse`` is constructed, so this branch cannot be
            # reached from a real HTTP request today. It exists solely to
            # correctly classify the error if some future/internal caller
            # ever invokes ``_hosted_stream``/``_sse_generator`` directly and
            # bypasses that precheck. By the time this generator body is
            # running, SSE response headers (HTTP 200) are already committed
            # to the client -- there is no way to retroactively send a 401
            # here. We can only emit a distinctly-coded SSE error frame
            # instead of silently downgrading it to a generic
            # ``internal_error`` frame via the broad ``except Exception``
            # below.
            logger.warning(
                "[hosted] Missing Foundry call context reached SSE generator "
                "directly (bypassed /invocations precheck) for strategy=%s",
                strategy_key,
            )
            if not error_emitted:
                from api.responses_adapter import _sse
                yield _sse(
                    "error",
                    {
                        "type": "error",
                        "code": "missing_call_context",
                        "message": str(exc),
                        "retryable": False,
                    },
                )

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
@app.get(
    "/readiness",
    response_model=HealthResponse,
    summary="Readiness check",
    description=(
        "Returns the immutable image version and the set of strategies "
        "eligible for this hosted runtime."
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
        401: {
            "description": (
                "Missing or malformed platform call context — hosted "
                "retrieval failed closed rather than falling back to "
                "service identity or a manual filter."
            ),
            "content": {
                "application/json": {
                    "example": {
                        "detail": (
                            "Missing or malformed platform call context "
                            "('x-agent-foundry-call-id')."
                        )
                    }
                }
            },
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
async def invocations(request: Request, body: InvocationRequest) -> StreamingResponse:
    """Translate a Foundry invocation into a Responses API SSE stream.

    The final message must be the current user ask. Messages after the current
    ask are rejected rather than silently discarded. Responses-backed
    strategies validate a supplied ``conversation_id`` or create a real managed
    Conversation when it is absent.

    Toolbox-integrated strategies additionally require the platform-injected
    ``x-agent-foundry-call-id`` header (Foundry hosted-agent container
    protocol 2.0). This container never reads or forwards ``Authorization``,
    and never trusts caller/model identity fields or ``x-client`` group
    claims — the opaque call id is the only supported identity-passthrough
    correlation, echoed to Toolbox so it can resolve the signed-in user and
    supply per-user credentials. When the header is absent or malformed,
    hosted retrieval fails closed with HTTP 401 rather than falling back to
    service identity or a manual ``metadata_security_id`` filter.
    """
    current_message = body.messages[-1]
    if current_message.role != "user":
        raise HTTPException(
            status_code=422,
            detail=(
                "The final message must have role='user'; messages after the "
                "current ask are not allowed."
            ),
        )
    ask = current_message.content.strip()
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

    # Toolbox-integrated strategies must carry a validated, platform-injected
    # call id (ADR-0001, Azure/GPT-RAG#591). Never trust Authorization,
    # caller/model identity fields, or x-client group claims here.
    foundry_call_id: Optional[str] = None
    if strategy_key in HOSTED_TOOLBOX_STRATEGIES:
        try:
            foundry_call_id = require_foundry_call_id(request.headers)
        except MissingFoundryCallContextError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc

    # Build the transport-neutral turn request.
    metadata = body.metadata or {}
    history: list[HostedConversationMessage] = [
        {"role": message.role, "text": message.content}
        for message in body.messages[:-1]
    ]
    turn = TurnRequest(
        ask=ask,
        conversation_id=body.conversation_id,
        question_id=metadata.get("question_id"),
        user_context={},
        correlation_id=metadata.get("correlation_id"),
        foundry_call_id=foundry_call_id,
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
