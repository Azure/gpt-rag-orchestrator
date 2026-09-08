"""Bounded P5 persistence evidence through maintained orchestration/SDK paths."""

import asyncio
import json
import logging
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from azure.core.exceptions import AzureError
from openai import AsyncOpenAI, BadRequestError

from orchestration import orchestrator as orchestration
from orchestration.turn import TurnRequest, TurnTextEvent
from strategies import agent_provider_v2 as provider
from connectors.cosmosdb import CosmosDBClient
from test_audit_lifecycle import Strategy
from test_legacy_runtime_boundary_dispositions import strategy as strategy


MARKER = "synthetic-private-persistence-detail"


@pytest.fixture
async def pending_writes():
    before = set(orchestration._persistence_tasks)
    yield
    # Test cleanup only, not a production shutdown/drain guarantee.
    await asyncio.gather(
        *(orchestration._persistence_tasks - before), return_exceptions=True
    )
    await asyncio.sleep(0)
    assert orchestration._persistence_tasks == before


@pytest.fixture
def make_orchestrator(patch_dependencies, mock_config, mock_cosmos, monkeypatch):
    async def make(flow, *, conversation_id="conversation", principal="principal"):
        value = Strategy("mcp", flow)
        monkeypatch.setattr(orchestration, "get_config", lambda: mock_config)
        monkeypatch.setattr(orchestration, "get_cosmosdb_client", lambda: mock_cosmos)
        monkeypatch.setattr(
            orchestration.AgentStrategyFactory, "get_strategy",
            AsyncMock(return_value=value),
        )
        mock_cosmos.update_document = AsyncMock(return_value={"id": "conversation"})
        return await orchestration.Orchestrator.create(
            conversation_id=conversation_id,
            user_context={"principal_id": principal},
        )
    return make


@pytest.mark.parametrize("state", ["new", "missing", "existing"])
@pytest.mark.parametrize("principal", ["principal", "anonymous"])
async def test_classic_order_snapshots_and_task_ownership(
    make_orchestrator, mock_cosmos, pending_writes, caplog, state, principal,
):
    caplog.set_level(logging.INFO)
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = []
    created = []
    updated = []

    async def create(container, key, body, *, partition_key):
        calls.append("create")
        assert container == "conversations"
        assert key == body["id"] == instance.conversation_id
        expected_partition = f"anonymous-{key}" if principal == "anonymous" else principal
        assert partition_key == body["principal_id"] == expected_partition
        created.append(deepcopy(body))
        body["connector_mutation"] = True
        entered.set()
        await release.wait()
        calls.append("created")
        return body

    async def update(container, body):
        calls.append("update")
        updated.append(deepcopy(body))
        body["messages"][0]["text"] = "connector mutation"
        return body

    async def flow():
        instance.agentic_strategy.conversation["messages"] = [
            {"role": "assistant", "text": "answer"}
        ]
        if state != "existing":
            await entered.wait()
        yield "answer"

    instance = await make_orchestrator(
        flow, conversation_id=None if state == "new" else "conversation",
        principal=principal,
    )
    if state == "existing":
        mock_cosmos.get_document.return_value = {
            "id": "conversation", "principal_id": principal,
        }
    mock_cosmos.create_document.side_effect = create
    mock_cosmos.update_document.side_effect = update
    request = TurnRequest(ask="question", question_id="question-id")
    events = await asyncio.wait_for(
        _collect(instance.stream_turn(request)), timeout=2
    )
    assert [e.text for e in events if isinstance(e, TurnTextEvent)] == ["answer"]
    if state != "existing":
        assert calls == ["create"]  # Successful stream did not wait for storage.
        assert len(orchestration._persistence_tasks) == 2
        assert "conversation_persist_async_done" not in caplog.text
        mock_cosmos.update_document.assert_not_awaited()
    # Mutations after scheduling must not change the pending update snapshot.
    live = instance.agentic_strategy.conversation
    live["messages"][0]["text"] = "later turn"
    release.set()
    await asyncio.gather(*orchestration._persistence_tasks)
    assert calls == (["update"] if state == "existing" else ["create", "created", "update"])
    assert updated[0]["messages"] == [{"role": "assistant", "text": "answer"}]
    assert updated[0]["questions"][0]["question_id"] == "question-id"
    assert live["messages"][0]["text"] == "later turn"
    assert "connector_mutation" not in live
    assert "connector_mutation" not in updated[0]
    if created:
        assert "messages" not in created[0]
        assert "questions" not in created[0]
    assert "conversation_persist_async_done" in caplog.text
    assert not orchestration._persistence_tasks


async def _collect(stream):
    return [event async for event in stream]

@pytest.mark.parametrize("primary", ["error", "cancel"])
async def test_classic_primary_failure_does_not_abandon_pending_create(
    make_orchestrator, mock_cosmos, pending_writes, caplog, primary,
):
    entered = asyncio.Event()
    release = asyncio.Event()
    failure = asyncio.CancelledError(MARKER) if primary == "cancel" else RuntimeError(MARKER)

    async def create(*args, **kwargs):
        entered.set()
        await release.wait()
        raise ValueError(MARKER)

    async def flow():
        await entered.wait()
        yield "partial"
        raise failure

    instance = await make_orchestrator(flow, conversation_id=None)
    mock_cosmos.create_document.side_effect = create
    stream = instance.stream_response("question")
    await anext(stream)
    assert await anext(stream) == "partial"
    with pytest.raises(type(failure)) as caught:
        await anext(stream)
    assert caught.value is failure
    assert len(orchestration._persistence_tasks) == 2
    release.set()
    assert await asyncio.gather(*orchestration._persistence_tasks) == [False, False]
    mock_cosmos.update_document.assert_not_awaited()
    assert "Error asynchronously persisting" in caplog.text
    assert "create unconfirmed" in caplog.text
    assert MARKER not in caplog.text


@pytest.mark.parametrize("write", ["create", "update"])
@pytest.mark.parametrize("outcome", ["none", "error", "cancel"])
async def test_classic_real_cosmos_unconfirmed_write_does_not_certify_stream(
    make_orchestrator, pending_writes, caplog, write, outcome,
):
    caplog.set_level(logging.INFO)

    async def flow():
        yield "answer"

    instance = await make_orchestrator(flow, conversation_id=None)
    # Exercise the real connector's SDK-error -> None mapping and its mutations.
    cosmos = CosmosDBClient.__new__(CosmosDBClient)
    container = SimpleNamespace(
        create_item=AsyncMock(return_value={"id": "conversation"}),
        replace_item=AsyncMock(return_value={"id": "conversation"}),
    )
    cosmos._get_container = lambda _name: container
    instance.database_client = cosmos
    failure = (
        AzureError(MARKER) if outcome == "none" else
        RuntimeError(MARKER) if outcome == "error" else asyncio.CancelledError(MARKER)
    )
    boundary = container.create_item if write == "create" else container.replace_item
    boundary.side_effect = failure
    assert [x async for x in instance.stream_response("question")][-1] == "answer"
    results = await asyncio.gather(
        *orchestration._persistence_tasks, return_exceptions=True
    )
    container.create_item.assert_awaited_once()
    if write == "create":
        container.replace_item.assert_not_awaited()
    else:
        container.replace_item.assert_awaited_once()
    if outcome == "none":
        assert f"Conversation {write} unconfirmed" in caplog.text
    elif outcome == "error":
        assert "Error asynchronously persisting" in caplog.text
    else:
        assert any(isinstance(result, asyncio.CancelledError) for result in results)
    assert "conversation_persist_async_done" not in caplog.text
    assert MARKER not in caplog.text
    assert not orchestration._persistence_tasks


@pytest.mark.parametrize("primary", ["success", "error", "cancel", "close"])
@pytest.mark.parametrize("secondary", ["snapshot", "write", "schedule"])
async def test_classic_cleanup_preserves_primary_outcome(
    make_orchestrator, mock_cosmos, pending_writes, monkeypatch, caplog,
    primary, secondary,
):
    failure = asyncio.CancelledError(MARKER) if primary == "cancel" else RuntimeError(MARKER)

    class BadSnapshot:
        def __deepcopy__(self, memo):
            raise ValueError(MARKER)

    async def flow():
        if secondary == "snapshot":
            instance.agentic_strategy.conversation["bad"] = BadSnapshot()
        yield "partial"
        if primary in {"error", "cancel"}:
            raise failure

    instance = await make_orchestrator(flow)
    mock_cosmos.get_document.return_value = {"id": "conversation", "principal_id": "principal"}
    mock_cosmos.update_document.side_effect = ValueError(MARKER)
    stream = instance.stream_response("question")
    assert await anext(stream) == "conversation "
    assert await anext(stream) == "partial"
    if secondary == "schedule":
        monkeypatch.setattr(
            asyncio, "create_task", MagicMock(side_effect=ValueError(MARKER))
        )
    if primary == "close":
        await stream.aclose()
    elif primary == "success":
        with pytest.raises(StopAsyncIteration):
            await anext(stream)
    else:
        with pytest.raises(type(failure)) as caught:
            await anext(stream)
        assert caught.value is failure
    await asyncio.gather(*orchestration._persistence_tasks)
    if secondary == "write":
        mock_cosmos.update_document.assert_awaited_once()
        assert "Error asynchronously persisting" in caplog.text
    else:
        mock_cosmos.update_document.assert_not_awaited()
        assert "Could not schedule conversation persistence" in caplog.text
    assert "conversation_persist_async_done" not in caplog.text
    assert MARKER not in caplog.text


@pytest.mark.parametrize("primary", ["success", "error", "cancel", "close"])
@pytest.mark.parametrize("diagnostic", ["ordinary", "cancel"])
async def test_cleanup_diagnostic_failure_always_resets_audit_context(
    make_orchestrator, mock_cosmos, pending_writes, monkeypatch, primary, diagnostic,
):
    from telemetry.audit import AuditEmitter
    from test_audit_lifecycle import configure_emitter

    monkeypatch.setattr(AuditEmitter, "_default", None)
    configure_emitter()
    end_audit = MagicMock(wraps=orchestration.end_audit_request)
    monkeypatch.setattr(orchestration, "end_audit_request", end_audit)
    primary_failure = (
        asyncio.CancelledError(MARKER) if primary == "cancel" else RuntimeError(MARKER)
    )
    diagnostic_failure = (
        asyncio.CancelledError("diagnostic cancellation")
        if diagnostic == "cancel" else OSError("diagnostic unavailable")
    )

    class BadSnapshot:
        def __deepcopy__(self, memo):
            raise ValueError(MARKER)

    class FailingHandler(logging.Handler):
        def emit(self, record):
            if "Could not schedule conversation persistence" in record.getMessage():
                raise diagnostic_failure

    async def flow():
        instance.agentic_strategy.conversation["bad"] = BadSnapshot()
        yield "partial"
        if primary in {"error", "cancel"}:
            raise primary_failure

    instance = await make_orchestrator(flow)
    mock_cosmos.get_document.return_value = {
        "id": "conversation", "principal_id": "principal",
    }
    stream = instance.stream_response("question")
    assert await anext(stream) == "conversation "
    assert await anext(stream) == "partial"
    handler = FailingHandler()
    logger = logging.getLogger()
    logger.addHandler(handler)
    try:
        operation = stream.aclose() if primary == "close" else anext(stream)
        if diagnostic == "cancel":
            with pytest.raises(asyncio.CancelledError) as caught:
                await operation
            assert caught.value is diagnostic_failure
        elif primary == "close":
            await operation
        elif primary == "success":
            with pytest.raises(StopAsyncIteration):
                await operation
        else:
            with pytest.raises(type(primary_failure)) as caught:
                await operation
            assert caught.value is primary_failure
    finally:
        logger.removeHandler(handler)
    end_audit.assert_called_once()
    mock_cosmos.update_document.assert_not_awaited()


@pytest.mark.parametrize("tail", ["current", "old_identical", "concurrent", "missing_id"])
async def test_managed_strategy_reconciliation_uses_submitted_sdk_identity(
    strategy, monkeypatch, caplog, tail,
):
    """Real strategy -> persistence -> installed SDK serialization and parsing."""
    calls = []
    submitted = []

    def transport(request):
        calls.append(request.method)
        if request.method == "POST":
            submitted.extend(json.loads(request.content)["items"])
            return httpx.Response(400, json={"error": {"message": MARKER}})
        assistant = deepcopy(submitted[1])
        user = {**submitted[0], "id": "msg_user", "status": "completed"}
        if tail in {"old_identical", "concurrent"}:
            assistant["id"] = "msg_another_turn"
        if tail == "concurrent":
            assistant["content"][0]["text"] = "another answer"
        if tail == "missing_id":
            del assistant["id"]
        return httpx.Response(200, json={
            "object": "list", "data": [assistant, user],
            "first_id": "msg_first", "last_id": "msg_user", "has_more": False,
        })

    agent = MagicMock()
    agent.__aenter__ = AsyncMock(return_value=agent)
    agent.__aexit__ = AsyncMock(return_value=False)
    agent.get_new_thread.return_value = MagicMock()
    runtime_provider = MagicMock()
    runtime_provider.as_agent.return_value = agent

    async def run(*args, **kwargs):
        yield SimpleNamespace(text="answer")

    monkeypatch.setattr(provider, "get_provider", AsyncMock(return_value=runtime_provider))
    monkeypatch.setattr(provider, "get_or_create_agent_details", AsyncMock(return_value=MagicMock()))
    monkeypatch.setattr(provider, "stream_agent_run", run)
    strategy.conversation.update(
        thread_id="conv_stable", agent_backend=provider.AGENT_BACKEND_TAG
    )
    async with AsyncOpenAI(
        api_key="synthetic-test-key", base_url="https://sdk.invalid/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(transport)),
    ) as sdk:
        monkeypatch.setattr(provider, "_get_openai_client", AsyncMock(return_value=sdk))
        stream = strategy._stream_agent("question")
        assert await anext(stream) == "answer"
        if tail == "current":
            with pytest.raises(StopAsyncIteration):
                await anext(stream)
            assert strategy.conversation["messages"][-1]["text"] == "answer"
        else:
            with pytest.raises(BadRequestError) as caught:
                await anext(stream)
            assert caught.value.response.request.method == "POST"
            assert strategy.conversation["messages"] == []
            assert "treating it as persisted" not in caplog.text
    assert calls == ["POST", "GET"]  # No duplicate write or application retry.
    assert submitted[1]["id"].startswith("msg_")
    agent.__aexit__.assert_awaited_once()
    assert MARKER not in caplog.text


@pytest.mark.parametrize("content", [None, [], [SimpleNamespace(text=42)]])
async def test_managed_empty_text_requires_readable_evidence(monkeypatch, caplog, content):
    failure = RuntimeError(MARKER)
    monkeypatch.setattr(provider.uuid, "uuid4", lambda: SimpleNamespace(hex="current"))
    sdk = MagicMock()
    sdk.conversations.items.create = AsyncMock(side_effect=failure)
    sdk.conversations.items.list = AsyncMock(return_value=SimpleNamespace(data=[
        SimpleNamespace(id="msg_current", role="assistant", content=content),
        SimpleNamespace(role="user", content=[SimpleNamespace(text="")]),
    ]))
    monkeypatch.setattr(provider, "_get_openai_client", AsyncMock(return_value=sdk))
    with pytest.raises(RuntimeError) as caught:
        await provider.persist_conversation_turn("conv", "", "")
    assert caught.value is failure
    sdk.conversations.items.create.assert_awaited_once()
    assert "Failed to reconcile" in caplog.text
    assert "treating it as persisted" not in caplog.text
    assert MARKER not in caplog.text
