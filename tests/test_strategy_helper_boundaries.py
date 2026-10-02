"""Exact legacy helper outcomes; fallback evidence is not approval."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from openai import BadRequestError

from strategies import maf_agent_service_strategy as service
from strategies import maf_lite_strategy as lite
from strategies import multimodal_strategy as vision
from strategies.maf_plugins import UserProfileMemory


@pytest.fixture(params=[lite, service, vision], ids=["lite", "service", "vision"])
def strategy(request, patch_dependencies, mock_config):
    module = request.param
    cls = {
        lite: lite.MafLiteStrategy,
        service: service.MafAgentServiceStrategy,
        vision: vision.MultimodalStrategy,
    }[module]
    with patch.object(module, "get_config", return_value=mock_config):
        instance = cls()
    instance.search_endpoint = "https://search.invalid"
    instance.search_index_name = "documents"
    instance.embedding_deployment = None
    instance._chat_client = MagicMock()
    instance._cached_instructions = "Test instructions"
    instance.profile_memory_enabled = False
    instance.conversation = {"messages": [], "session_initialized": True}
    return module, instance


@pytest.mark.parametrize("backend", ["ai_search", "foundry_iq"])
@pytest.mark.parametrize("mode", ["failure", "cancelled", "success"])
async def test_provider_construction_propagates_configured_failure(strategy, backend, mode, caplog):
    module, instance = strategy
    marker = "synthetic-private-construction-detail"
    failure = asyncio.CancelledError(marker) if mode == "cancelled" else RuntimeError(marker)
    name = "FoundryIQContextProvider" if backend == "foundry_iq" else (
        "MultimodalSearchContextProvider" if module is vision else "SearchContextProvider"
    )
    provider = object()
    with (
        patch.object(module, "get_retrieval_backend", return_value=backend),
        patch.object(module, name, return_value=provider,
                     side_effect=None if mode == "success" else failure) as construct,
    ):
        if mode != "success":
            with pytest.raises(type(failure)) as raised:
                await instance._create_search_provider()
            assert raised.value is failure
        else:
            result = await instance._create_search_provider()
            assert result is provider
    assert construct.call_count == 1
    assert construct.call_args.kwargs["conversation_id"] == instance.conversation_id
    assert marker not in caplog.text
    assert not any(record.exc_info for record in caplog.records)


async def test_real_flow_interrupts_after_constructor_failure(strategy, caplog):
    module, instance = strategy
    marker = "synthetic-private-construction-detail"
    agent = MagicMock()
    agent.__aenter__ = AsyncMock(return_value=agent)
    agent.__aexit__ = AsyncMock(return_value=False)
    agent.get_new_thread.return_value.service_thread_id = None

    async def chunks(*args, **kwargs):
        yield SimpleNamespace(text="ungrounded answer")

    agent.run_stream = chunks
    provider = SimpleNamespace(as_agent=MagicMock(return_value=agent))
    instance._classify_intent = AsyncMock(return_value="question")
    instance._user_memory = UserProfileMemory(chat_client=instance._chat_client)
    instance._post_flow_cleanup = AsyncMock()
    instance._read_prompt = AsyncMock(return_value="Test instructions")
    name = "MultimodalSearchContextProvider" if module is vision else "SearchContextProvider"
    with (
        patch.object(module, "get_retrieval_backend", return_value="ai_search"),
        patch.object(module, name, side_effect=RuntimeError(marker)),
        patch.object(lite, "ChatAgent", return_value=agent),
        patch.object(vision, "ChatAgent", return_value=agent),
        patch.object(service.agent_provider_v2, "get_provider", AsyncMock(return_value=provider)),
        patch.object(service.agent_provider_v2, "get_or_create_agent_details", AsyncMock(return_value=object())),
    ):
        with pytest.raises(RuntimeError):
            _ = [chunk async for chunk in instance.initiate_agent_flow("question")]
        await asyncio.sleep(0)
    assert instance._search_provider is None
    assert instance.conversation["messages"] == []
    agent.__aenter__.assert_not_awaited()
    assert marker not in caplog.text


@pytest.mark.parametrize("module,cls", [(lite, lite.MafLiteStrategy), (vision, vision.MultimodalStrategy)],
                         ids=["lite", "vision"])
@pytest.mark.parametrize("mode", ["failure", "cancelled", "retry-success", "retry-failure", "success", "empty"])
async def test_intent_failure_defaults_to_retrieval_not_skipping_it(
    patch_dependencies, mock_config, module, cls, mode, caplog,
):
    marker = "synthetic-private-intent-detail"
    failure = asyncio.CancelledError(marker) if mode == "cancelled" else RuntimeError(marker)
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="" if mode == "empty" else "QUESTION"))])
    bad_request = BadRequestError(
        marker, response=httpx.Response(400, request=httpx.Request("POST", "https://model.invalid")),
        body=None,
    )
    create = AsyncMock(return_value=response)
    if mode.startswith("retry-"):
        create.side_effect = [bad_request, response if mode == "retry-success" else failure]
    elif mode in {"failure", "cancelled"}:
        create.side_effect = failure
    with patch.object(module, "get_config", return_value=mock_config):
        instance = cls()
    instance._chat_client = SimpleNamespace(_client=SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    ))
    if mode == "cancelled":
        with pytest.raises(asyncio.CancelledError) as raised:
            await instance._classify_intent("question")
        assert raised.value is failure
    else:
        assert await instance._classify_intent("question") == "question"
    assert create.await_count == (2 if mode.startswith("retry-") else 1)
    if mode.startswith("retry-"):
        retry = create.await_args_list[1].kwargs
        if module is lite:
            assert "reasoning_effort" not in retry
            assert retry["max_completion_tokens"] == 200
        else:
            assert "max_completion_tokens" not in retry
            assert retry["max_tokens"] == 200
    assert marker not in caplog.text
    assert not any(record.exc_info for record in caplog.records)


@pytest.mark.parametrize("module,cls", [(lite, lite.MafLiteStrategy), (vision, vision.MultimodalStrategy)],
                         ids=["lite", "vision"])
@pytest.mark.parametrize("mode", ["flush-failure", "save-failure", "cancelled", "success", "absent"])
async def test_optional_post_flow_cleanup_preserves_order_and_cancellation(
    patch_dependencies, mock_config, module, cls, mode, caplog,
):
    marker = "synthetic-private-cleanup-detail"
    failure = asyncio.CancelledError(marker) if mode == "cancelled" else RuntimeError(marker)
    with patch.object(module, "get_config", return_value=mock_config):
        instance = cls()
    instance.profile_memory_enabled = True
    profile = object()
    flush = AsyncMock(side_effect=failure if mode in {"flush-failure", "cancelled"} else None)
    instance._user_memory = None if mode == "absent" else SimpleNamespace(flush=flush, user_profile=profile)
    instance._save_user_profile = AsyncMock(side_effect=failure if mode == "save-failure" else None)
    await instance._post_flow_cleanup("test-user")
    instance._save_user_profile.assert_not_awaited()
    flush.assert_not_awaited()
    assert instance._user_memory is None
    assert marker not in caplog.text
    assert not any(record.exc_info for record in caplog.records)


@pytest.mark.parametrize("mode", ["completed-failure", "completed-success", "absent", "failure", "cancelled"])
async def test_multimodal_flow_owns_optional_cleanup(
    patch_dependencies, mock_config, monkeypatch, caplog, mode,
):
    """Exercise the actual flow's scheduling, with a real memory worker."""
    marker = "synthetic-private-extraction-detail"
    with patch.object(vision, "get_config", return_value=mock_config):
        instance = vision.MultimodalStrategy()
    chat = SimpleNamespace(get_response=AsyncMock(
        side_effect=RuntimeError(marker) if mode == "completed-failure" else None,
        return_value=SimpleNamespace(value=None),
    ))
    memory = UserProfileMemory(chat_client=chat)
    instance._chat_client = chat
    instance._user_memory = memory
    instance._cached_instructions = "Test instructions"
    instance.conversation = {"messages": [], "session_initialized": True, "user_id": "test-user"}
    instance._classify_intent = AsyncMock(return_value="greeting")
    instance._save_user_profile = AsyncMock()
    warning = vision.logging.warning

    def warn(message, *args, **kwargs):
        if "Failed to extract user info" in message:
            raise RuntimeError(marker)
        return warning(message, *args, **kwargs)

    monkeypatch.setattr(vision.logging, "warning", warn)
    agent = MagicMock()
    agent.__aenter__ = AsyncMock(return_value=agent)
    agent.__aexit__ = AsyncMock(return_value=False)
    primary = asyncio.CancelledError(marker) if mode == "cancelled" else RuntimeError(marker)
    extraction = None

    async def chunks(messages, **kwargs):
        nonlocal extraction
        if mode in {"failure", "cancelled"}:
            raise primary
        await memory.invoked(messages)
        extraction = memory._pending_task
        assert extraction is None
        yield SimpleNamespace(text="answer")

    agent.run_stream = chunks
    tasks = []
    create_task = asyncio.create_task

    def schedule(coroutine):
        task = create_task(coroutine)
        tasks.append(task)
        return task

    with patch.object(vision, "ChatAgent", return_value=agent), patch.object(vision.asyncio, "create_task", schedule):
        flow = instance.initiate_agent_flow("hello")
        if mode in {"failure", "cancelled"}:
            with pytest.raises(type(primary)) as raised:
                await anext(flow)
            assert raised.value is primary
            assert tasks == []
            assert instance.conversation["messages"] == []
        else:
            assert await anext(flow) == "answer"
            if mode == "absent":
                # Session cleared between emitted answer and background cleanup.
                await instance.clear_session()
            with pytest.raises(StopAsyncIteration):
                await anext(flow)
            assert tasks == []
            instance._save_user_profile.assert_not_awaited()
            assert memory._pending_task is None
    agent.__aexit__.assert_awaited_once()
    chat.get_response.assert_not_awaited()
    assert not any(
            record.levelno >= vision.logging.WARNING
            and (
                "post_flow_cleanup failed" in record.getMessage()
                or "Failed to finish extraction" in record.getMessage()
    )
            for record in caplog.records
        )
    assert marker not in caplog.text
    assert not any(record.exc_info for record in caplog.records)


async def test_real_profile_flush_cancellation_prevents_cleanup_save(patch_dependencies, mock_config):
    started = asyncio.Event()

    async def response(**kwargs):
        started.set()
        await asyncio.Event().wait()

    with patch.object(vision, "get_config", return_value=mock_config):
        instance = vision.MultimodalStrategy()
    memory = UserProfileMemory(chat_client=SimpleNamespace(get_response=AsyncMock(side_effect=response)))
    instance._user_memory = memory
    instance._save_user_profile = AsyncMock()
    await memory.invoked(vision.ChatMessage(role="user", text="hello"))
    await instance._post_flow_cleanup("test-user")
    assert not started.is_set()
    memory._chat_client.get_response.assert_not_awaited()
    assert memory._pending_task is None
    instance._save_user_profile.assert_not_awaited()


@pytest.mark.parametrize("mode", ["failure", "cancelled", "success", "invalid", "empty", "missing", "timeout"])
async def test_image_validation_failure_strips_only_image_not_answer(
    patch_dependencies, mock_config, mode, caplog,
):
    caplog.set_level("INFO")
    marker = "synthetic-private-image-detail"
    failure = asyncio.CancelledError(marker) if mode == "cancelled" else (
        TimeoutError(marker) if mode == "timeout" else RuntimeError(marker)
    )
    with patch.object(vision, "get_config", return_value=mock_config):
        instance = vision.MultimodalStrategy()
    path = "https://blob.invalid/image.png?sig=synthetic-private-image-detail"
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
        content={"success": "VALID", "empty": ""}.get(mode, "INVALID"),
    ))])
    create = AsyncMock(return_value=response, side_effect=failure if mode in {"failure", "cancelled", "timeout"} else None)
    instance._chat_client = SimpleNamespace(_client=SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    ))
    instance._search_provider = SimpleNamespace(image_data={} if mode == "missing" else {path: "aW1hZ2U="})
    answer = f"before ![figure]({path}) after"
    if mode == "cancelled":
        with pytest.raises(asyncio.CancelledError) as raised:
            await instance._validate_response_images(answer, "question")
        assert raised.value is failure
    else:
        result = await instance._validate_response_images(answer, "question")
        assert result == (answer if mode == "success" else "before  after")
    assert marker not in caplog.text
    assert not any(record.exc_info for record in caplog.records)
