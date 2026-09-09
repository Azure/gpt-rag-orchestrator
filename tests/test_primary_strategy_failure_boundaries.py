"""Real primary strategy -> turn -> audit/SSE failure contracts."""

import asyncio
import importlib
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

from plugins.nl2sql.nl2sql_types import ExecuteQueryResult, ValidateSQLQueryResult
from strategies.maf_plugins.user_profile_memory import UserProfile, UserProfileMemory
from strategies.multimodal_strategy import MultimodalStrategy
from strategies.nl2sql_strategy import NL2SQLStrategy
from test_audit_lifecycle import audit_capture, build_orchestrator, types as audit_types


@pytest.fixture
def sql_strategy(patch_dependencies):
    plugin = SimpleNamespace(
        get_all_datasources_info=AsyncMock(return_value={"datasources": []}),
        tables_retrieval=AsyncMock(return_value={"tables": []}),
        validate_sql_query=AsyncMock(return_value=ValidateSQLQueryResult(is_valid=True)),
        execute_sql_query=AsyncMock(return_value=ExecuteQueryResult(results=[{"value": 1}])),
    )
    with patch("strategies.nl2sql_strategy.NL2SQLPlugin", return_value=plugin):
        strategy = NL2SQLStrategy()
    strategy._triage_prompt = "Triage"
    strategy._sqlquery_prompt = "SQL"
    strategy._syntetizer_prompt = "Synthesis"
    strategy._chat_client = MagicMock()
    strategy._run_agent = AsyncMock(side_effect=[
        '{"datasource_name":"test-source","datasource_type":"sql_database"}',
        '{"sql_query":"SELECT 1"}',
    ])
    strategy._collect_schema_context = AsyncMock(return_value={
        "all_tables": [], "table_candidates": [], "schemas": [], "unavailable_schemas": [],
        "similar_queries": [],
    })
    return strategy


@pytest.fixture
def vision_strategy(patch_dependencies, mock_config):
    with patch("strategies.multimodal_strategy.get_config", return_value=mock_config):
        strategy = MultimodalStrategy()
    strategy._chat_client = MagicMock()
    strategy._user_memory = UserProfileMemory(chat_client=strategy._chat_client)
    strategy._create_search_provider = AsyncMock(return_value=None)
    strategy._classify_intent = AsyncMock(return_value="greeting")
    strategy._cached_instructions = "Test instructions"
    strategy._post_flow_cleanup = AsyncMock()
    return strategy


@pytest.fixture
def main_module(patch_dependencies):
    previous = sys.modules.pop("main", None)
    try:
        with (
            patch("dotenv.load_dotenv", return_value=False),
            patch("telemetry.Telemetry.configure_basic"),
            patch("telemetry.Telemetry.log_log_level_diagnostics"),
            patch("opentelemetry.instrumentation.fastapi.FastAPIInstrumentor.instrument_app"),
            patch("opentelemetry.instrumentation.httpx.HTTPXClientInstrumentor.instrument"),
        ):
            yield importlib.import_module("main")
    finally:
        sys.modules.pop("main", None)
        if previous is not None:
            sys.modules["main"] = previous


async def _run_http_turn(main, strategy, cancellation=None, *, profile_user_id=None):
    orchestrator = build_orchestrator(strategy)
    if profile_user_id is not None or isinstance(strategy, MultimodalStrategy):
        # This fixture explicitly exercises an existing profile/welcome path.
        # Missing user_id is covered separately as an ordinary no-memory turn.
        orchestrator.database_client.get_document.return_value["user_id"] = (
            profile_user_id if profile_user_id is not None else "existing-profile-user"
        )
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    request = Request({
        "type": "http", "method": "POST", "path": "/orchestrator", "headers": [],
        "client": ("127.0.0.1", 1234), "server": ("test", 80),
        "scheme": "http", "query_string": b"",
    })
    body = SimpleNamespace(
        type="ask", ask="question", question=None, conversation_id="conversation",
        question_id="question-id", user_context={},
    )
    chunks = []
    try:
        with (
            patch("orchestration.orchestrator.tracer", provider.get_tracer("test.primary.failure")),
            patch.object(main.Orchestrator, "from_turn_request",
                         new=AsyncMock(return_value=orchestrator)),
        ):
            response = await main.orchestrator_endpoint(
                request=request, response=Response(), body=body,
                x_api_key=None, dapr_api_token=None, authorization=None,
            )
            if cancellation is not None:
                with pytest.raises(asyncio.CancelledError) as raised:
                    async for chunk in response.body_iterator:
                        chunks.append(chunk)
                assert raised.value is cancellation
            else:
                chunks = [chunk async for chunk in response.body_iterator]
            await asyncio.sleep(0)
    finally:
        provider.shutdown()
    return chunks, exporter.get_finished_spans(), orchestrator


@pytest.mark.parametrize("variant", ["sql", "vision", "vision-welcome"])
@pytest.mark.parametrize("outcome,partial", [
    ("success", False), ("success", True),
    ("failure", False), ("failure", True),
    ("cancelled", False), ("cancelled", True),
    ("initialization", False),
])
async def test_primary_failure_chain_preserves_buffering_prefix_history_and_audit(
    variant, outcome, partial, sql_strategy, vision_strategy, main_module, audit_capture, caplog,
):
    strategy = sql_strategy if variant == "sql" else vision_strategy
    marker = "synthetic-private-primary-provider-detail"
    failure = asyncio.CancelledError(marker) if outcome == "cancelled" else RuntimeError(marker)
    welcome = ""
    if variant == "vision-welcome":
        strategy._user_memory.user_profile = UserProfile(name="Test User")
        # Cached legacy profile must not produce a welcome without owner binding.
        welcome = ""
    if outcome == "initialization":
        if variant == "sql":
            strategy._load_prompts = AsyncMock(side_effect=failure)
        else:
            strategy._get_or_create_chat_client = MagicMock(side_effect=failure)
        welcome = ""

    async def model_stream(*_args, **_kwargs):
        if partial:
            yield SimpleNamespace(text="partial answer")
        if outcome != "success":
            raise failure
        yield SimpleNamespace(text="complete answer")

    agent = MagicMock()
    agent.run_stream = model_stream
    agent.__aenter__ = AsyncMock(return_value=agent)
    agent.__aexit__ = AsyncMock(return_value=False)
    with patch(f"{strategy.__module__}.ChatAgent", return_value=agent):
        chunks, spans, orchestrator = await _run_http_turn(
            main_module, strategy, failure if outcome == "cancelled" else None,
        )

    expected = ["conversation "] + ([welcome] if welcome else [])
    if variant == "sql" and partial:
        expected.append("partial answer")
    if outcome == "success":
        expected.append(("partial answer" if partial and variant != "sql" else "") + "complete answer")
    elif outcome != "cancelled":
        expected.append("event: error\ndata: An internal server error occurred.\n\n")
    assert chunks == expected
    assert marker not in caplog.text
    assert marker not in str([record.__dict__ for record in audit_capture.records])
    assert len(spans) == 1
    assert marker not in spans[0].to_json()
    messages = strategy.conversation.get("messages", [])
    if outcome == "success":
        answer = ("partial answer" if partial else "") + "complete answer"
        assert messages == [
            {"role": "user", "text": "question"}, {"role": "assistant", "text": answer},
        ]
        assert audit_types(audit_capture)[-2:] == ["outcome.produced", "request.completed"]
        assert spans[0].status.status_code != StatusCode.ERROR
    else:
        assert messages == []
        assert "request.completed" not in audit_types(audit_capture)
        assert "outcome.produced" not in audit_types(audit_capture)
        if outcome == "cancelled":
            assert audit_types(audit_capture)[-1] == "request.cancelled"
            assert spans[0].status.status_code != StatusCode.ERROR
        else:
            assert audit_types(audit_capture)[-2:] == ["outcome.rejected", "request.failed"]
            assert audit_capture.records[-1].partial_output is bool(
                welcome or (partial and variant == "sql"))
            assert spans[0].status.status_code == StatusCode.ERROR
    orchestrator.database_client.update_document.assert_awaited_once()
    if outcome == "initialization":
        agent.__aenter__.assert_not_awaited()
    else:
        agent.__aexit__.assert_awaited_once()
        if outcome != "success":
            assert agent.__aexit__.await_args.args[1] is failure
    if variant != "sql":
        strategy._post_flow_cleanup.assert_not_awaited()


@pytest.mark.parametrize("branch,answer", [
    ("answered", "Direct answer"),
    ("missing", "I could not identify a configured SQL datasource that matches this question."),
    ("unsupported", "The selected datasource 'test-source' is type 'semantic_model', which is not currently supported by the SQL execution path."),
    ("no_sql", "I could not generate a valid SQL query for this question."),
    ("validation", "I generated a SQL query, but it did not pass validation: rejected query"),
    ("execution", "The SQL query could not be executed: provider result error"),
], ids=["answered", "missing", "unsupported", "no_sql", "validation", "execution"])
async def test_explicit_nl2sql_result_answers_keep_completed_contract(
    branch, answer, sql_strategy, main_module, audit_capture,
):
    triage = '{"datasource_name":"test-source","datasource_type":"sql_database"}'
    if branch == "answered":
        triage = "QUESTION_ANSWERED Direct answer"
    elif branch == "missing":
        triage = "{}"
    elif branch == "unsupported":
        triage = '{"datasource_name":"test-source","datasource_type":"semantic_model"}'
    sql_strategy._run_agent.side_effect = [triage, "{}" if branch == "no_sql" else '{"sql_query":"SELECT 1"}']
    if branch == "validation":
        sql_strategy._nl2sql_plugin.validate_sql_query.return_value = ValidateSQLQueryResult(
            is_valid=False, error="rejected query")
    elif branch == "execution":
        sql_strategy._nl2sql_plugin.execute_sql_query.return_value = ExecuteQueryResult(
            error="provider result error")
    chunks, spans, _ = await _run_http_turn(main_module, sql_strategy)
    assert chunks == ["conversation ", answer]
    assert sql_strategy.conversation["messages"] == [
        {"role": "user", "text": "question"}, {"role": "assistant", "text": answer},
    ]
    assert audit_types(audit_capture)[-2:] == ["outcome.produced", "request.completed"]
    assert spans[0].status.status_code != StatusCode.ERROR
    if branch != "execution":
        sql_strategy._nl2sql_plugin.execute_sql_query.assert_not_awaited()


async def test_multimodal_buffer_is_deduplicated_then_validated_before_emission(
    vision_strategy, main_module, audit_capture,
):
    async def model_stream(*_args, **_kwargs):
        yield SimpleNamespace(text="![figure](image.png)\n")
        yield SimpleNamespace(text="![figure](image.png)\nanswer")

    agent = MagicMock()
    agent.run_stream = model_stream
    agent.__aenter__ = AsyncMock(return_value=agent)
    agent.__aexit__ = AsyncMock(return_value=False)
    vision_strategy.validate_response_images = True
    vision_strategy._validate_response_images = AsyncMock(return_value="validated answer")
    with patch("strategies.multimodal_strategy.ChatAgent", return_value=agent):
        chunks, _, _ = await _run_http_turn(main_module, vision_strategy)
    assert chunks == ["conversation ", "validated answer"]
    vision_strategy._validate_response_images.assert_awaited_once()
    buffered, question = vision_strategy._validate_response_images.await_args.args
    assert buffered.count("![figure](image.png)") == 1
    assert question == "question"
    assert vision_strategy.conversation["messages"][-1]["text"] == "validated answer"
    assert audit_types(audit_capture)[-1] == "request.completed"
