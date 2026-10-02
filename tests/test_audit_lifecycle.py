import asyncio
import importlib
import logging
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Response
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from starlette.requests import Request

from orchestration.orchestrator import Orchestrator
from strategies import agent_provider_v2
from strategies.agent_strategy_factory import AgentStrategyFactory
from strategies.maf_agent_service_strategy import MafAgentServiceStrategy
from strategies.maf_lite_strategy import MafLiteStrategy
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
@pytest.mark.parametrize("strategy_class", [MafLiteStrategy, MafAgentServiceStrategy])
@pytest.mark.parametrize(
    ("outcome", "partial_output"),
    [
        ("success", False), ("success", True),
        ("failure", False), ("failure", True),
        ("cancellation", False), ("cancellation", True),
        ("initialization_failure", False),
    ],
)
async def test_real_maf_turn_sse_chain_preserves_safe_terminal_outcomes(
    strategy_class, outcome, partial_output, patch_dependencies, mock_config,
    audit_capture, caplog,
):
    """Exercise real strategy, orchestration, audit, SSE and exported spans."""
    with patch(f"{strategy_class.__module__}.get_config", return_value=mock_config):
        strategy = strategy_class()
    strategy.profile_memory_enabled = False
    strategy._create_search_provider = AsyncMock(return_value=None)
    strategy._read_prompt = AsyncMock(return_value="Test instructions")
    strategy._cached_instructions = "Test instructions"
    strategy._chat_client = MagicMock()
    strategy._chat_client._client.chat.completions.create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="GREETING"))]))
    failure_type = asyncio.CancelledError if outcome == "cancellation" else RuntimeError
    marker = "synthetic-sensitive-upstream-detail"
    failure = failure_type(marker)
    if outcome == "initialization_failure" and strategy_class is MafLiteStrategy:
        strategy._get_or_create_chat_client = MagicMock(side_effect=failure)

    async def model_stream(*_args, **_kwargs):
        if partial_output:
            yield SimpleNamespace(text="partial answer", contents=[])
        if outcome != "success":
            raise failure
        yield SimpleNamespace(text="complete answer", contents=[])

    agent = MagicMock()
    agent.run_stream = model_stream
    agent.__aenter__ = AsyncMock(return_value=agent)
    agent.__aexit__ = AsyncMock(return_value=False)
    provider = MagicMock()
    provider.as_agent.return_value = agent
    exporter = InMemorySpanExporter()
    trace_provider = TracerProvider()
    trace_provider.add_span_processor(SimpleSpanProcessor(exporter))
    orchestrator = build_orchestrator(strategy)
    body = SimpleNamespace(
        type="ask", ask="question", question=None, conversation_id="conversation",
        question_id="question-id", user_context={})
    request = Request({
        "type": "http", "method": "POST", "path": "/orchestrator", "headers": [],
        "client": ("127.0.0.1", 1234), "server": ("test", 80),
        "scheme": "http", "query_string": b"",
    })
    previous_main = sys.modules.pop("main", None)
    try:
        with (
            patch("dotenv.load_dotenv", return_value=False),
            patch("telemetry.Telemetry.configure_basic"),
            patch("telemetry.Telemetry.log_log_level_diagnostics"),
        ):
            main = importlib.import_module("main")
        with (
            patch("strategies.maf_lite_strategy.ChatAgent", return_value=agent),
            patch.object(agent_provider_v2, "get_provider",
                         new=AsyncMock(
                             return_value=provider,
                             side_effect=failure if outcome == "initialization_failure" else None,
                         )),
            patch.object(agent_provider_v2, "get_or_create_agent_details",
                         new=AsyncMock(return_value=MagicMock())),
            patch.object(agent_provider_v2, "stream_agent_run", new=model_stream),
            patch("orchestration.orchestrator.tracer",
                  trace_provider.get_tracer("test.maf.failure")),
            patch.object(main.Orchestrator, "from_turn_request",
                         new=AsyncMock(return_value=orchestrator)),
        ):
            streaming = await main.orchestrator_endpoint(
                request=request, response=Response(), body=body,
                x_api_key=None, dapr_api_token=None, authorization=None)
            chunks = []
            if outcome == "cancellation":
                with pytest.raises(asyncio.CancelledError) as caught:
                    async for chunk in streaming.body_iterator:
                        chunks.append(chunk)
                assert caught.value is failure
            else:
                chunks = [chunk async for chunk in streaming.body_iterator]
            await asyncio.sleep(0)
    finally:
        sys.modules.pop("main", None)
        if previous_main is not None:
            sys.modules["main"] = previous_main
        trace_provider.shutdown()

    expected_chunks = ["conversation "]
    if partial_output:
        expected_chunks.append("partial answer")
    audit_data = str([record.__dict__ for record in audit_capture.records])
    assert marker not in audit_data
    assert marker not in "".join(chunks)
    assert marker not in caplog.text
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "stream_response"
    assert marker not in spans[0].to_json()
    if outcome == "cancellation":
        assert types(audit_capture)[-1] == "request.cancelled"
        assert spans[0].status.status_code != StatusCode.ERROR
    elif outcome in {"failure", "initialization_failure"}:
        expected_chunks.append(
            "event: error\ndata: An internal server error occurred.\n\n"
        )
        assert types(audit_capture)[-2:] == ["outcome.rejected", "request.failed"]
        assert audit_capture.records[-1].partial_output is partial_output
        assert spans[0].status.status_code == StatusCode.ERROR
    else:
        expected_chunks.append("complete answer")
        assert types(audit_capture)[-2:] == ["outcome.produced", "request.completed"]
        assert spans[0].status.status_code != StatusCode.ERROR
    assert chunks == expected_chunks
    messages = strategy.conversation.get("messages", [])
    if outcome == "success":
        assert messages == [
            {"role": "user", "text": "question"},
            {"role": "assistant", "text": "".join(expected_chunks[1:])},
        ]
    else:
        assert messages == []
        assert "request.completed" not in types(audit_capture)
        assert "outcome.produced" not in types(audit_capture)
    if outcome in {"failure", "cancellation"}:
        assert agent.__aexit__.await_args.args[1] is failure
    elif outcome == "initialization_failure":
        agent.__aenter__.assert_not_awaited()
    orchestrator.database_client.update_document.assert_awaited_once()


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
