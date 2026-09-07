"""Legacy runtime failure, persistence and optional-boundary characterizations."""

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from strategies import agent_provider_v2 as provider
from strategies import single_agent_rag_strategy_v2 as single
from orchestration.orchestrator import Orchestrator
from test_audit_lifecycle import Strategy, build_orchestrator
from test_primary_strategy_failure_boundaries import main_module, _run_http_turn, audit_capture


MARKER = "synthetic-private-legacy-detail"


@pytest.mark.parametrize("scenario", ["retry", "partial", "no_options", "unrelated", "retry_failure", "cancel"])
async def test_provider_retry_keeps_input_thread_and_non_token_options(scenario, caplog):
    calls = []
    failure = asyncio.CancelledError(MARKER) if scenario == "cancel" else RuntimeError(
        MARKER if scenario == "unrelated" else f"invalid_payload {MARKER}",
    )
    thread = object()
    message = [{"role": "user", "content": "test"}]
    options = {} if scenario == "no_options" else {"max_tokens": 10, "store": False}

    async def run(user_message, **kwargs):
        calls.append((user_message, kwargs))
        if len(calls) == 1:
            if scenario == "partial":
                yield SimpleNamespace(text="partial")
            raise failure
        if scenario == "retry_failure":
            raise failure
        yield SimpleNamespace(text="ok")

    output = []
    if scenario == "retry":
        output = [chunk.text async for chunk in provider.stream_agent_run(
            SimpleNamespace(run_stream=run), message, thread=thread, options=options,
        )]
        assert output == ["ok"]
    else:
        with pytest.raises(type(failure)) as caught:
            async for chunk in provider.stream_agent_run(
                SimpleNamespace(run_stream=run), message, thread=thread, options=options,
            ):
                output.append(chunk.text)
        assert caught.value is failure
    assert output == (["partial"] if scenario == "partial" else ["ok"] if scenario == "retry" else [])
    assert len(calls) == (2 if scenario in {"retry", "retry_failure"} else 1)
    assert all(item[0] is message and item[1]["thread"] is thread for item in calls)
    if len(calls) == 2:
        assert calls[1][1]["options"] == {"store": False}
    assert options == ({} if scenario == "no_options" else {"max_tokens": 10, "store": False})
    assert MARKER not in caplog.text


@pytest.mark.parametrize("scenario", ["success", "confirmed", "different", "unavailable", "cancel_create", "cancel_read"])
async def test_managed_persistence_only_reconciles_exact_tail(scenario, monkeypatch, caplog):
    failure = RuntimeError(MARKER)
    cancelled = asyncio.CancelledError(MARKER)
    sdk = MagicMock()
    sdk.conversations.items.create = AsyncMock(side_effect=(
        None if scenario == "success" else cancelled if scenario == "cancel_create" else failure
    ))
    tail = [
        SimpleNamespace(role="assistant", content=[SimpleNamespace(text="answer")]),
        SimpleNamespace(role="user", content=[SimpleNamespace(text="question")]),
    ]
    if scenario == "different":
        tail[0].content[0].text = "other answer"
    sdk.conversations.items.list = AsyncMock(
        return_value=SimpleNamespace(data=tail),
        side_effect=failure if scenario == "unavailable" else cancelled if scenario == "cancel_read" else None,
    )
    monkeypatch.setattr(provider, "_get_openai_client", AsyncMock(return_value=sdk))
    if scenario in {"success", "confirmed"}:
        await provider.persist_conversation_turn("conv", "question", "answer")
    else:
        expected = cancelled if scenario.startswith("cancel") else failure
        with pytest.raises(type(expected)) as caught:
            await provider.persist_conversation_turn("conv", "question", "answer")
        assert caught.value is expected
    sdk.conversations.items.create.assert_awaited_once()
    if scenario in {"success", "cancel_create"}:
        sdk.conversations.items.list.assert_not_awaited()
    else:
        sdk.conversations.items.list.assert_awaited_once_with("conv", limit=2, order="desc")
    assert MARKER not in caplog.text


@pytest.fixture
def strategy(patch_dependencies, mock_config):
    with (
        patch.object(single, "get_config", return_value=mock_config),
        patch.object(single, "get_search_client", return_value=MagicMock()),
        patch.object(single, "get_genai_client", return_value=MagicMock()),
    ):
        value = single.SingleAgentRAGStrategyV2()
    value.conversation = {"id": "conversation", "messages": []}
    value._read_prompt = AsyncMock(return_value="System")
    return value


@pytest.mark.parametrize("cancelled", [False, True])
async def test_prewarm_creation_is_optional_without_raw_diagnostics(
    patch_dependencies, mock_config, monkeypatch, caplog, cancelled,
):
    failure = asyncio.CancelledError(MARKER) if cancelled else RuntimeError(MARKER)
    monkeypatch.setattr(single, "get_config", lambda: mock_config)
    monkeypatch.setattr(provider, "get_provider", AsyncMock(return_value=MagicMock()))
    monkeypatch.setattr(provider, "get_or_create_agent_details", AsyncMock(side_effect=failure))
    if cancelled:
        with pytest.raises(asyncio.CancelledError) as caught:
            await single.prewarm_agents_client()
        assert caught.value is failure
    else:
        await single.prewarm_agents_client()
        assert "Could not pre-create" in caplog.text
    assert MARKER not in caplog.text


@pytest.mark.parametrize("cancelled", [False, True])
def test_search_initialization_failure_propagates_without_raw_log(
    patch_dependencies, mock_config, monkeypatch, caplog, cancelled,
):
    failure = asyncio.CancelledError(MARKER) if cancelled else RuntimeError(MARKER)
    monkeypatch.setattr(single, "get_config", lambda: mock_config)
    monkeypatch.setattr(single, "get_search_client", MagicMock(side_effect=failure))
    with pytest.raises(type(failure)) as caught:
        single.SingleAgentRAGStrategyV2()
    assert caught.value is failure
    assert MARKER not in caplog.text


@pytest.mark.parametrize("cancelled", [False, True])
async def test_legacy_search_context_failure_is_explicit_not_new_identity_enforcement(strategy, caplog, cancelled):
    failure = asyncio.CancelledError(MARKER) if cancelled else RuntimeError(MARKER)
    strategy.search_client.set_request_context.side_effect = failure
    strategy.search_client.search_knowledge_base = AsyncMock(return_value={"documents": []})
    tool = strategy._build_search_tool()
    if cancelled:
        with pytest.raises(asyncio.CancelledError) as caught:
            await tool("question")
        assert caught.value is failure
        strategy.search_client.search_knowledge_base.assert_not_awaited()
    else:
        await tool("question")
        strategy.search_client.search_knowledge_base.assert_awaited_once()
        assert "request context" in caplog.text
    assert MARKER not in caplog.text


@pytest.mark.parametrize("cancelled", [False, True])
@pytest.mark.parametrize("partial", [False, True])
async def test_real_direct_stream_failure_reaches_safe_turn_chain(
    strategy, main_module, audit_capture, caplog, cancelled, partial,
):
    failure = asyncio.CancelledError(MARKER) if cancelled else RuntimeError(MARKER)
    strategy.search_client.is_index_empty = AsyncMock(return_value=True)
    original_get = strategy.cfg.get
    strategy.cfg = SimpleNamespace(get=lambda key, *a, **k: False if key == "BING_RETRIEVAL_ENABLED" else original_get(key, *a, **k))

    async def chunks():
        if partial:
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="partial"))])
        raise failure

    strategy.llm_client.openai_client.chat.completions.create = AsyncMock(return_value=chunks())
    output, _, _ = await _run_http_turn(main_module, strategy, failure if cancelled else None)
    assert MARKER not in "".join(output) + caplog.text
    assert strategy.conversation.get("messages", []) == []
    if not cancelled:
        assert "event: error" in "".join(output)


@pytest.mark.parametrize("stage", ["stream", "persist"])
@pytest.mark.parametrize("hosted", [False, True])
@pytest.mark.parametrize("cancelled", [False, True])
async def test_agent_stream_and_persistence_preserve_hosted_separation(
    strategy, monkeypatch, caplog, stage, hosted, cancelled,
):
    strategy.hosted_runtime = hosted
    strategy.project_endpoint = "https://example.invalid"
    strategy.credential = MagicMock()
    strategy.model_name = "chat"
    failure = asyncio.CancelledError(MARKER) if cancelled else RuntimeError(MARKER)
    agent = MagicMock()
    agent.__aenter__ = AsyncMock(return_value=agent)
    agent.__aexit__ = AsyncMock(return_value=False)
    agent.get_new_thread.return_value = object()
    shared = SimpleNamespace(as_agent=MagicMock(return_value=agent))
    monkeypatch.setattr(provider, "get_provider", AsyncMock(return_value=shared))
    monkeypatch.setattr(provider, "get_or_create_agent_details", AsyncMock(return_value=object()))
    ensure = AsyncMock(return_value="conv")
    monkeypatch.setattr(provider, "ensure_conversation_id", ensure)
    persist = AsyncMock(side_effect=failure)
    monkeypatch.setattr(strategy, "_persist_managed_turn", persist)
    calls = []

    async def stream(_agent, message, **kwargs):
        calls.append((message, kwargs))
        yield SimpleNamespace(text="partial")
        if stage == "stream":
            raise failure

    monkeypatch.setattr(provider, "stream_agent_run", stream)
    if stage == "persist" and hosted:
        assert [chunk async for chunk in strategy._stream_agent("question")] == ["partial"]
        persist.assert_not_awaited()
    else:
        with pytest.raises(type(failure)) as caught:
            async for _ in strategy._stream_agent("question"):
                pass
        assert caught.value is failure
    if hosted:
        ensure.assert_not_awaited()
        assert calls[0][1]["options"]["store"] is False
        assert "service_thread_id" not in agent.get_new_thread.call_args.kwargs
    else:
        ensure.assert_awaited_once()
    assert MARKER not in caplog.text


@pytest.mark.parametrize("cancelled", [False, True])
async def test_legacy_strategy_token_setter_failure_has_bounded_diagnostic(
    patch_dependencies, mock_config, mock_cosmos, caplog, cancelled,
):
    failure = asyncio.CancelledError(MARKER) if cancelled else RuntimeError(MARKER)

    class RejectToken:
        @property
        def request_access_token(self):
            return None

        @request_access_token.setter
        def request_access_token(self, value):
            raise failure

    with (
        patch("orchestration.orchestrator.AgentStrategyFactory.get_strategy", AsyncMock(return_value=RejectToken())),
        patch("orchestration.orchestrator.get_config", return_value=mock_config),
        patch("orchestration.orchestrator.get_cosmosdb_client", return_value=mock_cosmos),
    ):
        if cancelled:
            with pytest.raises(asyncio.CancelledError):
                await Orchestrator.create(user_context={"principal_id": "principal"}, request_access_token="token")
        else:
            value = await Orchestrator.create(user_context={"principal_id": "principal"}, request_access_token="token")
            assert value.request_access_token == "token"
            assert "strategy token context" in caplog.text
    assert MARKER not in caplog.text


@pytest.mark.parametrize("kind", ["lifecycle", "persist", "cancel"])
async def test_orchestrator_background_and_optional_logs_remain_bounded(monkeypatch, caplog, kind):
    async def flow():
        yield "answer"

    instance = build_orchestrator(Strategy("mcp", flow))
    tasks = []
    create = asyncio.create_task
    info = logging.info

    def spawn(coro):
        task = create(coro)
        tasks.append(task)
        return task

    def log(message, *args, **kwargs):
        if kind == "lifecycle" and message.startswith("[Conversation] Started:"):
            raise RuntimeError(MARKER)
        return info(message, *args, **kwargs)

    monkeypatch.setattr(asyncio, "create_task", spawn)
    monkeypatch.setattr(logging, "info", log)
    if kind != "lifecycle":
        instance.database_client.update_document.side_effect = (
            asyncio.CancelledError(MARKER) if kind == "cancel" else RuntimeError(MARKER)
        )
    assert [chunk async for chunk in instance.stream_response("question")] == ["conversation ", "answer"]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    if kind == "cancel":
        assert any(isinstance(result, asyncio.CancelledError) for result in results)
    else:
        assert "Failed to render conversation" in caplog.text if kind == "lifecycle" else "persisting conversation" in caplog.text
    assert MARKER not in caplog.text


@pytest.mark.parametrize("cancelled", [False, True])
async def test_feedback_resolution_failure_preserves_partition_and_optional_write(caplog, cancelled):
    async def flow():
        yield "answer"

    instance = build_orchestrator(Strategy("mcp", flow))
    failure = asyncio.CancelledError(MARKER) if cancelled else RuntimeError(MARKER)

    class BadQuestion(dict):
        def get(self, *args):
            raise failure

    conversation = {"id": "conversation", "questions": [BadQuestion()]}
    instance.database_client.get_document.return_value = conversation
    if cancelled:
        with pytest.raises(asyncio.CancelledError):
            await instance.save_feedback({"text": "question"})
        instance.database_client.update_document.assert_not_awaited()
    else:
        await instance.save_feedback({"text": "question"})
        instance.database_client.update_document.assert_awaited_once()
        assert conversation["feedback"] == [{"text": "question"}]
    assert instance.database_client.get_document.await_args.kwargs["partition_key"] == "anonymous-conversation"
    assert MARKER not in caplog.text
