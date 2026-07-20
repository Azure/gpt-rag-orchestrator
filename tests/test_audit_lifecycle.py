import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from orchestration.orchestrator import Orchestrator
from strategies.agent_strategy_factory import AgentStrategyFactory
from telemetry.audit import AuditEmitter
from telemetry.audit_contract import AuditSettings


class CaptureHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        if hasattr(record, "event_type"):
            self.records.append(record)


class Strategy:
    def __init__(self, name, flow):
        self.strategy_type = SimpleNamespace(value=name)
        self.conversation = {}
        self._flow = flow

    def set_context(self, _conversation_id):
        return None

    def initiate_agent_flow(self, _ask):
        return self._flow()


def configure_emitter(*, sensitive_fields=frozenset()):
    emitter = AuditEmitter(
        AuditSettings(
            enabled=True,
            sensitive_content_enabled=bool(sensitive_fields),
            sensitive_content_fields=sensitive_fields,
            actor_pseudonym_enabled=False,
            source_event_limit=25,
            hmac_key_id="v1",
            hmac_key=b"k" * 32,
            additional_redacted_keys=frozenset(),
        ),
        service_name="gpt-rag-orchestrator",
        service_version="3.7.0",
        environment="test",
    )
    AuditEmitter._default = emitter
    return emitter


def build_orchestrator(strategy):
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.conversation_id = "conversation"
    orchestrator.principal_id = "anonymous"
    orchestrator.correlation_id = "req_" + ("1" * 32)
    orchestrator.user_context = {}
    orchestrator.agentic_strategy = strategy
    orchestrator.database_client = SimpleNamespace(
        get_document=AsyncMock(
            return_value={
                "id": "conversation",
                "principal_id": "anonymous-conversation",
            }
        ),
        update_document=AsyncMock(),
    )
    orchestrator.database_container = "conversations"
    orchestrator.conversation_compaction_config = SimpleNamespace(
        enabled=False,
        max_bytes=0,
        max_messages=0,
        max_questions=0,
        preserve_recent_messages=0,
        preserve_recent_questions=0,
    )
    orchestrator._prepare_conversation_for_persistence = lambda value: value
    return orchestrator


@pytest.fixture()
def audit_capture():
    configure_emitter()
    logger = logging.getLogger("gptrag.audit")
    previous = (list(logger.handlers), logger.propagate, logger.level)
    capture = CaptureHandler()
    logger.handlers = [capture]
    logger.propagate = False
    logger.setLevel(logging.INFO)
    try:
        yield capture
    finally:
        logger.handlers, logger.propagate, logger.level = previous


def types(capture):
    return [record.event_type for record in capture.records]


def test_registry_guard_covers_every_active_strategy():
    assert AgentStrategyFactory.registered_strategy_names() == {
        "maf_lite",
        "maf_agent_service",
        "multimodal",
        "single_agent_rag",
        "nl2sql",
        "mcp",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "strategy_name",
    sorted(AgentStrategyFactory.registered_strategy_names()),
)
async def test_request_lifecycle_seam_covers_every_registered_strategy(
    strategy_name, audit_capture
):
    async def success_flow():
        yield "answer"

    orchestrator = build_orchestrator(Strategy(strategy_name, success_flow))
    chunks = [
        chunk
        async for chunk in orchestrator.stream_response("question", "question-id")
    ]
    await asyncio.sleep(0)

    assert chunks == ["conversation ", "answer"]
    assert types(audit_capture) == [
        "request.started",
        "route.selected",
        "outcome.produced",
        "request.completed",
    ]
    assert (
        orchestrator.agentic_strategy.conversation["questions"][0][
            "correlation_id"
        ]
        == orchestrator.correlation_id
    )


@pytest.mark.asyncio
async def test_failed_stream_rejects_outcome_and_emits_terminal_failure(
    audit_capture,
):
    async def failed_flow():
        yield "partial"
        raise RuntimeError("do not export this")

    orchestrator = build_orchestrator(Strategy("maf_lite", failed_flow))
    with pytest.raises(RuntimeError):
        _ = [
            chunk
            async for chunk in orchestrator.stream_response(
                "question", "question-id"
            )
        ]

    assert types(audit_capture)[-2:] == [
        "outcome.rejected",
        "request.failed",
    ]
    assert audit_capture.records[-1].partial_output is True
    assert "do not export this" not in str(
        [record.__dict__ for record in audit_capture.records]
    )


@pytest.mark.asyncio
async def test_successful_stream_captures_only_bounded_allowlisted_response(
    audit_capture,
):
    configure_emitter(sensitive_fields=frozenset({"response"}))

    async def success_flow():
        yield "a" * 1500
        yield "b" * 1500

    orchestrator = build_orchestrator(Strategy("maf_lite", success_flow))
    _ = [
        chunk
        async for chunk in orchestrator.stream_response(
            "question", "question-id"
        )
    ]

    outcome = next(
        record
        for record in audit_capture.records
        if record.event_type == "outcome.produced"
    )
    assert len(outcome.response) == 2048
    assert outcome.response.startswith("a" * 1500)


@pytest.mark.asyncio
async def test_sse_disconnect_cancels_partial_stream_and_persists(
    audit_capture,
):
    first_chunk = asyncio.Event()
    keep_streaming = asyncio.Event()

    async def slow_flow():
        yield "partial"
        first_chunk.set()
        await keep_streaming.wait()

    orchestrator = build_orchestrator(Strategy("mcp", slow_flow))

    async def consume():
        async for _chunk in orchestrator.stream_response(
            "question", "question-id"
        ):
            pass

    task = asyncio.create_task(consume())
    await first_chunk.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)

    assert types(audit_capture)[-1] == "request.cancelled"
    assert audit_capture.records[-1].partial_output is True
    orchestrator.database_client.update_document.assert_awaited_once()
