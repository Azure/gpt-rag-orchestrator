"""Configured retrieval must reach the maintained turn/audit/SSE boundary."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agent_framework import ChatMessage, Context, Role

from strategies import foundry_iq_context_provider as foundry
from strategies import multimodal_search_context_provider as multimodal
from strategies import search_context_provider as search
from strategies.maf_plugins import UserProfileMemory
from test_audit_lifecycle import audit_capture, types as audit_types
from test_context_provider_boundary_dispositions import _sdk_search
from test_primary_strategy_failure_boundaries import _run_http_turn, main_module
from test_strategy_helper_boundaries import lite, service, strategy, vision


@pytest.mark.parametrize("backend", ["ai_search", "foundry_iq"])
@pytest.mark.parametrize("mode", [
    "construction", "retrieval", "cancelled", "empty", "success", "optional",
    "disabled-endpoint", "disabled-index", "greeting", "no_retrieval",
])
async def test_required_retrieval_reaches_safe_turn(
    strategy, backend, mode, main_module, audit_capture, caplog,
):
    module, instance = strategy
    if module is service and mode in {"greeting", "no_retrieval"}:
        pytest.skip("Agent Service has no intent-based retrieval opt-out")
    marker = "synthetic-private-retrieval-failure"
    failure = asyncio.CancelledError(marker) if mode == "cancelled" else RuntimeError(marker)
    bypass = mode.startswith("disabled-") or mode in {"greeting", "no_retrieval"}
    if mode == "disabled-endpoint":
        instance.search_endpoint = None
    if mode == "disabled-index":
        instance.search_index_name = None
    instance._classify_intent = AsyncMock(
        return_value=mode if mode in {"greeting", "no_retrieval"} else "question")
    instance._user_memory = UserProfileMemory(chat_client=instance._chat_client)
    instance._user_memory.invoking = AsyncMock(
        return_value=Context(instructions="Independent optional context"),
        side_effect=RuntimeError(marker) if mode == "optional" else None,
    )
    if module is service:
        instance._create_user_memory = AsyncMock(return_value=instance._user_memory)
    instance._post_flow_cleanup = AsyncMock()
    instance._read_prompt = AsyncMock(return_value="Test instructions")
    documents = [] if mode == "empty" else [
        {"id": "one", "title": "Document", "content": "grounding"},
    ]
    sdk = _sdk_search(documents, lambda _: failure if mode in {"retrieval", "cancelled"} else None)
    retrieve = AsyncMock(
        return_value=documents,
        side_effect=failure if mode in {"retrieval", "cancelled"} else None,
    )
    agent = MagicMock()
    agent.__aenter__ = AsyncMock(return_value=agent)
    agent.__aexit__ = AsyncMock(return_value=False)
    agent.get_new_thread.return_value.service_thread_id = None
    captured = {}
    model_calls = []

    def build_agent(*args, **kwargs):
        captured["context"] = kwargs.get("context_provider")
        return agent

    async def stream(*args, **kwargs):
        context = captured["context"]
        if context is not None:
            captured["result"] = await context.invoking(
                ChatMessage(role=Role.USER, text="question"))
        model_calls.append(True)
        yield SimpleNamespace(text="ordinary answer")

    agent.run_stream = stream
    provider_name = "FoundryIQContextProvider" if backend == "foundry_iq" else (
        "MultimodalSearchContextProvider" if module is vision else "SearchContextProvider")
    real_provider = getattr(module, provider_name)
    with (
        patch.object(module, "get_retrieval_backend", return_value=backend),
        patch.object(module, provider_name, side_effect=(
            failure if mode == "construction" or bypass else real_provider)) as construct,
        patch.object(search, "SearchClient", return_value=sdk),
        patch.object(multimodal, "SearchClient", return_value=sdk),
        patch.object(foundry, "get_foundry_iq_client", return_value=SimpleNamespace(
            mcp_config=SimpleNamespace(enabled=False), retrieve=retrieve)),
        patch.object(lite, "ChatAgent", side_effect=build_agent),
        patch.object(vision, "ChatAgent", side_effect=build_agent),
        patch.object(service.agent_provider_v2, "get_provider", AsyncMock(
            return_value=SimpleNamespace(as_agent=build_agent))),
        patch.object(service.agent_provider_v2, "get_or_create_agent_details",
                     AsyncMock(return_value=object())),
    ):
        chunks, spans, _ = await _run_http_turn(
            main_module, instance, failure if mode == "cancelled" else None)

    failed = mode in {"construction", "retrieval", "cancelled"}
    if failed:
        assert not model_calls
        assert chunks == ["conversation "] + (
            [] if mode == "cancelled"
            else ["event: error\ndata: An internal server error occurred.\n\n"])
        assert not instance.conversation.get("messages")
        assert audit_types(audit_capture)[-1] == (
            "request.cancelled" if mode == "cancelled" else "request.failed")
        assert "request.completed" not in audit_types(audit_capture)
    else:
        assert model_calls == [True]
        assert chunks == ["conversation ", "ordinary answer"]
        assert audit_types(audit_capture)[-1] == "request.completed"
        if mode in {"success", "optional"}:
            assert "grounding" in "".join(message.text for message in captured["result"].messages)
    if mode == "optional":
        instance._user_memory.invoking.assert_awaited_once()
    if bypass:
        construct.assert_not_called()
        sdk.search.assert_not_awaited()
        retrieve.assert_not_awaited()
    assert marker not in caplog.text
    assert marker not in str([record.__dict__ for record in audit_capture.records])
    assert all(marker not in span.to_json() for span in spans)
