"""P4 negative eligibility and optional persistence through maintained callers.

The model/Foundry and Cosmos boundaries are mocked, not profile selection,
framework context hooks, memory extraction, or the direct-model adapter.
"""

import asyncio
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from agent_framework import ChatAgent

from connectors.openai_chat_client import OpenAIChatClient
from orchestration import orchestrator as orchestration
from strategies import agent_provider_v2
from strategies import maf_agent_service_strategy as service
from strategies import maf_lite_strategy as lite
from strategies import multimodal_strategy as vision
from strategies.base_agent_strategy import hosted_runtime_construction
from strategies.maf_plugins.user_profile_memory import UserProfile, UserProfileMemory


MARKER = "synthetic-private-profile-extraction"


@pytest.fixture(params=[lite, service, vision], ids=["lite", "service", "vision"])
async def caller_factory(request, patch_dependencies, mock_config, mock_cosmos, monkeypatch):
    module = request.param
    cls = {
        lite: lite.MafLiteStrategy,
        service: service.MafAgentServiceStrategy,
        vision: vision.MultimodalStrategy,
    }[module]
    monkeypatch.setattr(module, "get_config", lambda: mock_config)
    monkeypatch.setattr("connectors.openai_chat_client.get_bearer_token_provider", MagicMock())
    tasks = []
    schedule = asyncio.create_task

    def create_task(coroutine, *args, **kwargs):
        task = schedule(coroutine, *args, **kwargs)
        tasks.append((coroutine.__name__, task))
        return task

    monkeypatch.setattr(asyncio, "create_task", create_task)

    def make(*, hosted=False):
        state = SimpleNamespace(stream_failure=None, extraction=None, tasks=tasks)

        async def chunks():
            if state.stream_failure is not None:
                raise state.stream_failure
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="ordinary answer"))],
                id="test-response", model="test-model",
            )

        async def completion(**kwargs):
            if kwargs.get("stream"):
                return chunks()
            if state.extraction is not None:
                await state.extraction()
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=f'{{"name":"{MARKER}"}}'))],
                id="test-extraction", model="test-model",
            )

        sdk = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(side_effect=completion))),
            close=AsyncMock(),
        )
        monkeypatch.setattr("connectors.openai_chat_client.AsyncAzureOpenAI", lambda **kwargs: sdk)
        with hosted_runtime_construction() if hosted else nullcontext():
            strategy = cls()
        strategy.conversation = {"id": "conversation", "principal_id": "partition-principal"}
        strategy.user_context = {"user_id": "request-user", "principal_id": "request-principal"}
        strategy._read_prompt = AsyncMock(return_value="Test instructions")
        strategy._classify_intent = AsyncMock(return_value="greeting")
        strategy._create_search_provider = AsyncMock(return_value=None)
        strategy._cached_instructions = "Test instructions"
        mock_cosmos.update_document = AsyncMock(return_value={"id": "profile"})
        mock_cosmos.create_document.return_value = {"id": "profile"}
        if module is service:
            # The maintained service caller chooses its memory adapter itself.
            # Only the remote prompt-agent boundary is replaced with a local
            # real framework agent, preserving the actual provider hooks.
            model_client = OpenAIChatClient(
                azure_endpoint="https://model.invalid", model_deployment_name="test-model",
                credential=MagicMock(),
            )
            runtime_provider = SimpleNamespace(
                as_agent=lambda details, **kwargs: ChatAgent(
                    chat_client=model_client,
                    instructions="Test instructions", **kwargs,
                ),
            )
            monkeypatch.setattr(agent_provider_v2, "get_provider", AsyncMock(return_value=runtime_provider))
            monkeypatch.setattr(agent_provider_v2, "get_or_create_agent_details", AsyncMock(return_value=object()))
        state.strategy = strategy
        state.cosmos = mock_cosmos
        state.sdk = sdk
        state.module = module
        return state

    yield make
    await asyncio.gather(*(task for _, task in tasks), return_exceptions=True)


async def drain(caller):
    return await asyncio.gather(*(task for _, task in caller.tasks), return_exceptions=True)


@pytest.mark.parametrize("identity", [
    "missing", "none", "blank", "whitespace", "non_string", "default", "padded_default", "existing",
])
async def test_strategy_skips_unusable_profile_identity_without_rekeying(caller_factory, identity):
    caller = caller_factory()
    strategy = caller.strategy
    values = {
        "none": None, "blank": "", "whitespace": " \t", "non_string": 42,
        "default": "default_user", "padded_default": " default_user ",
        "existing": " legacy-user ",
    }
    if identity != "missing":
        strategy.conversation["user_id"] = values[identity]
    original = UserProfile(name="Existing", notes=["retained"])
    caller.cosmos.get_document.return_value = {"profile_data": original.model_dump_json()}
    if caller.module is not service and identity != "existing":
        # Ineligible turns must not reuse a previously cached profile provider.
        strategy._user_memory = UserProfileMemory(
            chat_client=MagicMock(), user_profile=original,
        )
    output = [chunk async for chunk in strategy.initiate_agent_flow("hello")]
    await drain(caller)
    assert output[-1] == "ordinary answer"
    assert strategy.conversation["messages"] == [
        {"role": "user", "text": "hello"},
        {"role": "assistant", "text": "ordinary answer"},
    ]
    if identity == "existing":
        assert output[0].startswith("Welcome back!")
        assert all(call.args[1] == "user_profile_ legacy-user " for call in caller.cosmos.get_document.await_args_list)
        saved = caller.cosmos.update_document.await_args.args[1]
        assert saved["id"] == "user_profile_ legacy-user "
        assert UserProfile.model_validate_json(saved["profile_data"]) == original
        # The current adapter mismatch must remain ineffective, even when the
        # underlying model returns JSON that could populate a profile.
        extraction_calls = [
            call for call in caller.sdk.chat.completions.create.await_args_list
            if not call.kwargs.get("stream")
        ]
        assert len(extraction_calls) == 1
        assert "response_format" not in extraction_calls[0].kwargs
    else:
        assert output == ["ordinary answer"]
        caller.cosmos.get_document.assert_not_awaited()
        caller.cosmos.create_document.assert_not_awaited()
        caller.cosmos.update_document.assert_not_awaited()
        assert not any(name in {"_extract_and_update_profile", "_post_flow_cleanup"} for name, _ in caller.tasks)
        assert all(call.kwargs.get("stream") for call in caller.sdk.chat.completions.create.await_args_list)
        if caller.module is service:
            assert strategy._memory_chat_client is None


async def test_classic_request_principal_does_not_become_profile_identity(
    caller_factory, mock_config, monkeypatch,
):
    caller = caller_factory()
    monkeypatch.setattr(orchestration, "get_config", lambda: mock_config)
    monkeypatch.setattr(orchestration, "get_cosmosdb_client", lambda: caller.cosmos)
    monkeypatch.setattr(orchestration.AgentStrategyFactory, "get_strategy", AsyncMock(return_value=caller.strategy))
    instance = await orchestration.Orchestrator.create(
        user_context={"principal_id": "authenticated-principal", "user_id": "request-user"},
    )
    output = [chunk async for chunk in instance.stream_response("hello")]
    await drain(caller)
    assert output == [f"{instance.conversation_id} ", "ordinary answer"]
    assert caller.strategy.conversation["principal_id"] == "authenticated-principal"
    assert "user_id" not in caller.strategy.conversation
    caller.cosmos.get_document.assert_not_awaited()
    caller.cosmos.create_document.assert_awaited_once()
    caller.cosmos.update_document.assert_awaited_once()
    assert not any(name in {"_extract_and_update_profile", "_post_flow_cleanup"} for name, _ in caller.tasks)
    if caller.module is service:
        assert caller.strategy._memory_chat_client is None


async def test_hosted_caller_does_not_enable_profile_memory(caller_factory):
    caller = caller_factory(hosted=True)
    caller.strategy.conversation["user_id"] = "existing-but-hosted"
    output = [chunk async for chunk in caller.strategy.initiate_agent_flow("hello")]
    await drain(caller)
    assert output == ["ordinary answer"]
    assert caller.strategy.cosmos is None
    caller.cosmos.get_document.assert_not_awaited()
    caller.cosmos.create_document.assert_not_awaited()
    caller.cosmos.update_document.assert_not_awaited()
    assert not any(name in {"_extract_and_update_profile", "_post_flow_cleanup"} for name, _ in caller.tasks)


@pytest.mark.parametrize("existing", [False, True], ids=["create", "update"])
@pytest.mark.parametrize("outcome", ["confirmed", "none", "write_failure", "read_failure"])
async def test_profile_save_truth_reaches_strategy_caller(caller_factory, existing, outcome, caplog):
    caplog.set_level("INFO")
    caller = caller_factory()
    caller.strategy.conversation["user_id"] = "legacy-user"
    old_profile = UserProfile(notes=["unchanged"])
    caller.cosmos.get_document.return_value = {"profile_data": old_profile.model_dump_json()} if existing else None
    write = caller.cosmos.update_document if existing else caller.cosmos.create_document
    if outcome == "none":
        write.return_value = None
    elif outcome == "write_failure":
        write.side_effect = RuntimeError(MARKER)
    elif outcome == "read_failure":
        caller.cosmos.get_document.side_effect = RuntimeError(MARKER)
    assert [chunk async for chunk in caller.strategy.initiate_agent_flow("hello")] == ["ordinary answer"]
    await drain(caller)
    assert ("Saved user profile" in caplog.text) is (outcome == "confirmed")
    assert "post_flow_profile_save" not in caplog.text
    if outcome == "none":
        assert "write not confirmed" in caplog.text
    if outcome == "read_failure":
        write.assert_not_awaited()
    else:
        write.assert_awaited_once()
        doc = write.await_args.args[1] if existing else write.await_args.kwargs["body"]
        assert UserProfile.model_validate_json(doc["profile_data"]) == (old_profile if existing else UserProfile())
    assert MARKER not in caplog.text


@pytest.mark.parametrize("outcome", ["error", "cancelled"])
async def test_ineligible_profile_preserves_primary_stream_failure(caller_factory, outcome):
    caller = caller_factory()
    failure = asyncio.CancelledError("cancelled") if outcome == "cancelled" else RuntimeError("model failed")
    caller.stream_failure = failure
    with pytest.raises(type(failure)) as caught:
        _ = [chunk async for chunk in caller.strategy.initiate_agent_flow("hello")]
    assert caught.value is failure
    await drain(caller)
    assert not caller.strategy.conversation.get("messages")
    caller.cosmos.get_document.assert_not_awaited()
    caller.cosmos.create_document.assert_not_awaited()
    caller.cosmos.update_document.assert_not_awaited()


async def test_profile_flush_cancellation_from_strategy_caller_prevents_save(caller_factory):
    caller = caller_factory()
    caller.strategy.conversation["user_id"] = "legacy-user"
    started = asyncio.Event()

    async def pending_extraction():
        started.set()
        await asyncio.Event().wait()

    caller.extraction = pending_extraction

    async def consume():
        return [chunk async for chunk in caller.strategy.initiate_agent_flow("hello")]

    primary = asyncio.create_task(consume())
    await asyncio.wait_for(started.wait(), timeout=2)
    if caller.module is service:
        cleanup = primary
    else:
        assert await primary == ["ordinary answer"]
        cleanup = next(task for name, task in caller.tasks if name == "_post_flow_cleanup")
    await asyncio.sleep(0)
    cleanup.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cleanup
    await drain(caller)
    caller.cosmos.create_document.assert_not_awaited()
    caller.cosmos.update_document.assert_not_awaited()


@pytest.mark.parametrize("boundary", ["load", "save"])
async def test_profile_io_cancellation_keeps_existing_caller_semantics(caller_factory, boundary, caplog):
    caller = caller_factory()
    caller.strategy.conversation["user_id"] = "legacy-user"
    failure = asyncio.CancelledError(MARKER)
    if boundary == "load":
        caller.cosmos.get_document.side_effect = failure
    else:
        caller.cosmos.create_document.side_effect = failure
    if boundary == "load" or caller.module is service:
        with pytest.raises(asyncio.CancelledError) as caught:
            _ = [chunk async for chunk in caller.strategy.initiate_agent_flow("hello")]
        assert caught.value is failure
    else:
        assert [chunk async for chunk in caller.strategy.initiate_agent_flow("hello")] == ["ordinary answer"]
        cleanup = next(task for name, task in caller.tasks if name == "_post_flow_cleanup")
        with pytest.raises(asyncio.CancelledError) as caught:
            await cleanup
        assert caught.value is failure
    await drain(caller)
    if boundary == "load":
        caller.cosmos.create_document.assert_not_awaited()
        assert not caller.strategy.conversation.get("messages")
    else:
        caller.cosmos.create_document.assert_awaited_once()
        assert caller.strategy.conversation["messages"][-1]["text"] == "ordinary answer"
    caller.cosmos.update_document.assert_not_awaited()
    assert "Saved user profile" not in caplog.text
    assert MARKER not in caplog.text
