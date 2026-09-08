"""Optional background profile extraction keeps primary work and diagnostics separate."""

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agent_framework import ChatMessage, Role

from connectors.openai_chat_client import OpenAIChatClient
from strategies.maf_plugins.user_profile_memory import ExtractedUserInfo, UserProfile, UserProfileMemory


@pytest.mark.parametrize("mode", ["success", "no-result", "failure", "unsupported", "cancelled", "logging-error"])
async def test_invoked_flush_preserves_optional_profile_contract(mode, monkeypatch, caplog):
    marker = "synthetic-private-profile-detail"
    caplog.set_level(logging.DEBUG)
    profile = UserProfile(name="Original", preferences=["existing"], notes=["existing"])
    extracted = ExtractedUserInfo(
        name=marker, role="Analyst", company="Example", preferences=["compact", "compact"], notes=["new", "new"],
    )
    failure = {
        "failure": RuntimeError(marker),
        "unsupported": AttributeError(f"conversation_id {marker}"),
        "cancelled": asyncio.CancelledError(marker),
        "logging-error": RuntimeError(marker),
    }.get(mode)
    chat = SimpleNamespace(get_response=AsyncMock(
        return_value=SimpleNamespace(value=None if mode == "no-result" else extracted), side_effect=failure,
    ))
    memory = UserProfileMemory(chat_client=chat, user_profile=profile)
    if mode == "logging-error":
        warning = logging.warning

        def warn(message, *args, **kwargs):
            if "Failed to extract user info" in message:
                raise RuntimeError(marker)
            return warning(message, *args, **kwargs)

        monkeypatch.setattr(logging, "warning", warn)
    message = ChatMessage(role=Role.USER, text="Profile information")
    await memory.invoked(message)
    task = memory._pending_task
    assert task is not None
    await memory.flush()
    assert task.done()
    assert memory._pending_task is None
    chat.get_response.assert_awaited_once()
    sent = chat.get_response.await_args.kwargs
    assert sent["messages"] == [message]
    assert sent["chat_options"]["response_format"] is ExtractedUserInfo
    if mode == "success":
        assert memory.user_profile.model_dump() == {
            "name": marker, "role": "Analyst", "company": "Example",
            "preferences": ["existing", "compact"], "notes": ["existing", "new"],
        }
        assert marker in (await memory.invoking(message)).instructions
        assert UserProfileMemory.deserialize(memory.serialize(), chat).user_profile == memory.user_profile
    else:
        assert memory.user_profile.name == "Original"
        assert memory.user_profile.preferences == ["existing"]
    if mode == "logging-error":
        assert any(record.levelno == logging.WARNING for record in caplog.records)
    if mode == "cancelled":
        assert not task.cancelled()
        assert any("extraction cancelled" in record.getMessage() for record in caplog.records)
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


async def test_superseded_extraction_is_cancelled_before_new_profile_is_flushed():
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def response(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            started.set()
            await release.wait()
        return SimpleNamespace(value=ExtractedUserInfo(name="Current"))

    chat = SimpleNamespace(get_response=AsyncMock(side_effect=response))
    memory = UserProfileMemory(chat_client=chat)
    message = ChatMessage(role=Role.USER, text="profile")
    await memory.invoked(message)
    first = memory._pending_task
    await asyncio.wait_for(started.wait(), timeout=1)
    await memory.invoked(message)
    await memory.flush()
    await first
    assert first.done()
    assert memory.user_profile.name == "Current"
    assert memory._pending_task is None


@pytest.mark.parametrize("mode", ["response", "failure"])
async def test_current_direct_adapter_does_not_populate_extracted_profile_value(caplog, mode):
    """Characterization of an unresolved interop gap, not approval of memory loss."""
    marker = "synthetic-private-profile-detail"
    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=f'{{"name":"{marker}"}}'))],
        model="test-model", id="test-response",
    )
    sdk = MagicMock()
    sdk.chat.completions.create = AsyncMock(
        return_value=completion, side_effect=RuntimeError(marker) if mode == "failure" else None,
    )
    with (
        patch("connectors.openai_chat_client.get_bearer_token_provider"),
        patch("connectors.openai_chat_client.AsyncAzureOpenAI", return_value=sdk),
    ):
        client = OpenAIChatClient(
            azure_endpoint="https://example.invalid", model_deployment_name="test-model", credential=MagicMock(),
        )
    memory = UserProfileMemory(chat_client=client, user_profile=UserProfile(name="Original"))
    await memory.invoked(ChatMessage(role=Role.USER, text="Profile information"))
    await memory.flush()
    sdk.chat.completions.create.assert_awaited_once()
    assert "response_format" not in sdk.chat.completions.create.await_args.kwargs
    assert memory.user_profile.name == "Original"
    assert marker not in caplog.text


async def test_flush_does_not_clear_replacement_task():
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def response(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            started.set()
            await asyncio.Event().wait()
        await release.wait()
        return SimpleNamespace(value=ExtractedUserInfo(name="Current"))

    memory = UserProfileMemory(chat_client=SimpleNamespace(get_response=AsyncMock(side_effect=response)))
    message = ChatMessage(role=Role.USER, text="profile")
    await memory.invoked(message)
    await started.wait()
    flush = asyncio.create_task(memory.flush())
    await asyncio.sleep(0)
    await memory.invoked(message)
    replacement = memory._pending_task
    try:
        await flush
        assert memory._pending_task is replacement
    finally:
        release.set()
        await replacement
    await memory.flush()
    assert memory.user_profile.name == "Current"
    assert memory._pending_task is None
