"""Profile collection is explicitly suspended, not dependent on adapter failure."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agent_framework import ChatMessage, Role

from connectors.openai_chat_client import OpenAIChatClient
from strategies.maf_plugins.user_profile_memory import ExtractedUserInfo, UserProfile, UserProfileMemory


@pytest.mark.parametrize("mode", ["success", "no-result", "failure", "unsupported", "cancelled"])
async def test_invoked_flush_preserves_optional_profile_contract(mode, caplog):
    marker = "synthetic-private-profile-detail"
    failure = {
        "failure": RuntimeError(marker), "unsupported": AttributeError(marker),
        "cancelled": asyncio.CancelledError(marker),
    }.get(mode)
    chat = SimpleNamespace(get_response=AsyncMock(
        return_value=SimpleNamespace(value=None if mode == "no-result" else ExtractedUserInfo(name=marker)),
        side_effect=failure,
    ))
    original = UserProfile(name="Original", preferences=["existing"], notes=["retained"])
    memory = UserProfileMemory(chat_client=chat, user_profile=original.model_copy(deep=True))
    message = ChatMessage(role=Role.USER, text=marker)
    for _ in range(2):
        await memory.invoked(message)
        await memory._extract_and_update_profile([message])
        await memory.flush()
    assert memory._pending_task is None
    chat.get_response.assert_not_awaited()
    assert memory.user_profile == original
    context = await memory.invoking(message)
    assert not context.instructions
    assert not context.messages
    assert UserProfileMemory.deserialize(memory.serialize(), chat).user_profile == original
    assert marker not in caplog.text


@pytest.mark.parametrize("reason", ["no-user-message", "unsupported-client"])
async def test_ineligible_profile_extraction_never_schedules(reason):
    class AzureAIAgentClient:
        get_response = AsyncMock()

    chat = AzureAIAgentClient() if reason == "unsupported-client" else SimpleNamespace(get_response=AsyncMock())
    memory = UserProfileMemory(chat_client=chat)
    message = ChatMessage(role=Role.SYSTEM if reason == "no-user-message" else Role.USER, text="context")
    await memory.invoked(message)
    await memory.flush()
    assert memory._pending_task is None
    chat.get_response.assert_not_awaited()


@pytest.mark.parametrize("mode", ["response", "failure"])
async def test_current_direct_adapter_does_not_populate_extracted_profile_value(caplog, mode):
    """Even perfect extraction JSON must not cause a model request or mutation."""
    marker = "synthetic-private-profile-detail"
    sdk = MagicMock()
    sdk.chat.completions.create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=f'{{"name":"{marker}"}}'))],
            model="test-model", id="test-response",
        ),
        side_effect=RuntimeError(marker) if mode == "failure" else None,
    )
    with (
        patch("connectors.openai_chat_client.get_bearer_token_provider"),
        patch("connectors.openai_chat_client.AsyncAzureOpenAI", return_value=sdk),
    ):
        client = OpenAIChatClient(
            azure_endpoint="https://example.invalid", model_deployment_name="test-model", credential=MagicMock(),
        )
    memory = UserProfileMemory(chat_client=client, user_profile=UserProfile(name="Original"))
    message = ChatMessage(role=Role.USER, text=marker)
    await memory.invoked(message)
    await memory._extract_and_update_profile([message])
    await memory.flush()
    sdk.chat.completions.create.assert_not_awaited()
    assert memory.user_profile.name == "Original"
    assert marker not in caplog.text


async def test_cancellation_of_caller_is_not_suppressed():
    memory = UserProfileMemory(chat_client=SimpleNamespace(get_response=AsyncMock()))
    entered = asyncio.Event()

    async def caller():
        await memory.invoked(ChatMessage(role=Role.USER, text="profile"))
        await memory.flush()
        entered.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(caller())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert memory._pending_task is None
    memory._chat_client.get_response.assert_not_awaited()
