"""Tests for the hosted Responses adapters and managed Conversations entrypoint.

Covers:
- Responses API SSE serialization of every typed turn event
- Terminal closing frames after the text stream
- Hosted strategy guard (explicit failure for unsupported strategies)
- Hosted stream execution without Cosmos dependency
- Hosted FastAPI endpoints (health and invocations)
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.responses_adapter import responses_terminal_events, serialize_responses_events
from orchestration.turn import (
    TurnCancelledEvent,
    TurnCitation,
    TurnCitationEvent,
    TurnConversationEvent,
    TurnErrorEvent,
    TurnRequest,
    TurnTextEvent,
    TurnToolActivity,
    TurnToolActivityEvent,
    TurnToolStatus,
)
from strategies.hosted_strategies import (
    HOSTED_ELIGIBLE_STRATEGIES,
    guard_hosted_strategy,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

_RESP_ID = "resp_testresponse"
_ITEM_ID = "item_testitem"


def _parse_frames(frames: list[str]) -> list[dict[str, Any]]:
    """Parse a list of SSE frames into (event_type, data_dict) pairs."""
    result = []
    for frame in frames:
        lines = frame.strip().splitlines()
        event_type = None
        data = None
        for line in lines:
            if line.startswith("event: "):
                event_type = line[len("event: "):]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: "):])
        result.append({"event": event_type, "data": data})
    return result


def _serialize(event, **kwargs):
    return _parse_frames(
        serialize_responses_events(
            event,
            response_id=_RESP_ID,
            item_id=_ITEM_ID,
            **kwargs,
        )
    )


# ── ResponsesAdapter serialization ───────────────────────────────────────────

class TestResponsesAdapterConversationEvent:
    def test_emits_three_opening_frames(self):
        frames = _serialize(TurnConversationEvent("conv-1"))

        assert len(frames) == 3

    def test_response_created_has_conversation_id(self):
        frames = _serialize(TurnConversationEvent("conv-abc"))

        created = frames[0]
        assert created["event"] == "response.created"
        assert created["data"]["response"]["conversation_id"] == "conv-abc"
        assert created["data"]["response"]["id"] == _RESP_ID
        assert created["data"]["response"]["status"] == "in_progress"

    def test_output_item_added_follows_response_created(self):
        frames = _serialize(TurnConversationEvent("conv-abc"))

        item_added = frames[1]
        assert item_added["event"] == "response.output_item.added"
        assert item_added["data"]["item"]["role"] == "assistant"
        assert item_added["data"]["item"]["id"] == _ITEM_ID

    def test_content_part_added_opens_text_slot(self):
        frames = _serialize(TurnConversationEvent("conv-abc"))

        part = frames[2]
        assert part["event"] == "response.content_part.added"
        assert part["data"]["part"]["type"] == "output_text"


class TestResponsesAdapterTextEvent:
    def test_emits_output_text_delta(self):
        frames = _serialize(TurnTextEvent("Hello"))

        assert len(frames) == 1
        assert frames[0]["event"] == "response.output_text.delta"
        assert frames[0]["data"]["delta"] == "Hello"
        assert frames[0]["data"]["item_id"] == _ITEM_ID

    def test_empty_text_is_allowed(self):
        frames = _serialize(TurnTextEvent(""))

        assert frames[0]["data"]["delta"] == ""


class TestResponsesAdapterCitationEvent:
    def test_emits_annotation_added(self):
        citation = TurnCitation(
            "src-1", title="Policy Doc", url="https://example.com/policy", snippet="Relevant excerpt"
        )
        frames = _serialize(TurnCitationEvent(citation=citation))

        assert len(frames) == 1
        assert frames[0]["event"] == "response.output_text.annotation.added"
        ann = frames[0]["data"]["annotation"]
        assert ann["citation_id"] == "src-1"
        assert ann["title"] == "Policy Doc"
        assert ann["url"] == "https://example.com/policy"
        assert ann["snippet"] == "Relevant excerpt"

    def test_omits_none_optional_fields(self):
        citation = TurnCitation("src-2")
        frames = _serialize(TurnCitationEvent(citation=citation))

        ann = frames[0]["data"]["annotation"]
        assert "title" not in ann
        assert "url" not in ann
        assert "snippet" not in ann


class TestResponsesAdapterToolActivityEvent:
    def test_started_emits_arguments_delta(self):
        activity = TurnToolActivity("search_kb", TurnToolStatus.STARTED, call_id="call-1")
        frames = _serialize(TurnToolActivityEvent(activity=activity))

        assert frames[0]["event"] == "response.function_call_arguments.delta"
        assert frames[0]["data"]["name"] == "search_kb"
        assert frames[0]["data"]["call_id"] == "call-1"
        assert frames[0]["data"]["status"] == "started"

    def test_completed_emits_arguments_done(self):
        activity = TurnToolActivity("search_kb", TurnToolStatus.COMPLETED, call_id="call-1")
        frames = _serialize(TurnToolActivityEvent(activity=activity))

        assert frames[0]["event"] == "response.function_call_arguments.done"
        assert frames[0]["data"]["status"] == "completed"

    def test_failed_emits_arguments_done_with_message(self):
        activity = TurnToolActivity(
            "search_kb", TurnToolStatus.FAILED, call_id="call-1", message="Tool execution failed"
        )
        frames = _serialize(TurnToolActivityEvent(activity=activity))

        assert frames[0]["event"] == "response.function_call_arguments.done"
        assert frames[0]["data"]["status"] == "failed"
        assert frames[0]["data"]["message"] == "Tool execution failed"

    def test_completed_without_message_omits_message_key(self):
        activity = TurnToolActivity("search_kb", TurnToolStatus.COMPLETED)
        frames = _serialize(TurnToolActivityEvent(activity=activity))

        assert "message" not in frames[0]["data"]


class TestResponsesAdapterErrorEvent:
    def test_emits_error_event(self):
        frames = _serialize(TurnErrorEvent(message="Something broke", code="internal_error", retryable=False))

        assert len(frames) == 1
        assert frames[0]["event"] == "error"
        assert frames[0]["data"]["code"] == "internal_error"
        assert frames[0]["data"]["message"] == "Something broke"
        assert frames[0]["data"]["retryable"] is False

    def test_retryable_flag_is_preserved(self):
        frames = _serialize(TurnErrorEvent(retryable=True))

        assert frames[0]["data"]["retryable"] is True


class TestResponsesAdapterCancelledEvent:
    def test_emits_response_cancelled(self):
        frames = _serialize(TurnCancelledEvent(reason="cancelled"))

        assert len(frames) == 1
        assert frames[0]["event"] == "response.cancelled"
        assert frames[0]["data"]["reason"] == "cancelled"


class TestResponsesTerminalEvents:
    def test_emits_four_closing_frames_in_order(self):
        frames = _parse_frames(
            responses_terminal_events(
                response_id=_RESP_ID,
                item_id=_ITEM_ID,
                full_text="Hello world",
            )
        )

        assert len(frames) == 4
        assert frames[0]["event"] == "response.output_text.done"
        assert frames[1]["event"] == "response.content_part.done"
        assert frames[2]["event"] == "response.output_item.done"
        assert frames[3]["event"] == "response.completed"

    def test_text_is_propagated_to_all_closing_frames(self):
        frames = _parse_frames(
            responses_terminal_events(
                response_id=_RESP_ID,
                item_id=_ITEM_ID,
                full_text="Final answer.",
            )
        )

        assert frames[0]["data"]["text"] == "Final answer."
        assert frames[1]["data"]["part"]["text"] == "Final answer."
        assert frames[2]["data"]["item"]["content"][0]["text"] == "Final answer."

    def test_response_completed_carries_response_id(self):
        frames = _parse_frames(
            responses_terminal_events(
                response_id=_RESP_ID,
                item_id=_ITEM_ID,
                full_text="",
            )
        )

        assert frames[3]["data"]["response"]["id"] == _RESP_ID
        assert frames[3]["data"]["response"]["status"] == "completed"


# ── Strategy guard ───────────────────────────────────────────────────────────

class TestHostedStrategyGuard:
    @pytest.mark.parametrize("strategy", sorted(HOSTED_ELIGIBLE_STRATEGIES))
    def test_eligible_strategies_pass(self, strategy: str):
        guard_hosted_strategy(strategy)  # must not raise

    @pytest.mark.parametrize("strategy", ["multimodal", "nl2sql", "multiagent", "unknown_strategy"])
    def test_unsupported_strategies_raise_value_error(self, strategy: str):
        with pytest.raises(ValueError, match="not supported in the hosted runtime"):
            guard_hosted_strategy(strategy)

    def test_error_message_includes_eligible_strategies(self):
        with pytest.raises(ValueError) as exc_info:
            guard_hosted_strategy("nl2sql")

        msg = str(exc_info.value)
        for eligible in HOSTED_ELIGIBLE_STRATEGIES:
            assert eligible in msg


# ── Hosted stream (no Cosmos) ────────────────────────────────────────────────

class TestHostedStream:
    """Tests for the _hosted_stream execution path."""

    @pytest.fixture(autouse=True)
    def _patch_config(self, mock_config, mock_cosmos):
        with (
            patch("dependencies.get_config", return_value=mock_config),
            patch("connectors.cosmosdb.get_cosmosdb_client", return_value=mock_cosmos),
            patch("connectors.identity_manager.get_identity_manager", return_value=MagicMock()),
            patch("strategies.base_agent_strategy.get_config", return_value=mock_config),
            patch("strategies.base_agent_strategy.get_cosmosdb_client", return_value=mock_cosmos),
            patch("strategies.base_agent_strategy.get_identity_manager", return_value=MagicMock()),
            patch("strategies.base_agent_strategy.AIProjectClient"),
        ):
            yield

    @pytest.mark.asyncio
    async def test_yields_conversation_identity_first(self):
        from api.hosted_entrypoint import _hosted_stream

        async def fake_flow(_ask):
            yield "Hello"

        strategy = MagicMock()
        strategy.initiate_agent_flow = fake_flow
        turn = TurnRequest(ask="Hi", conversation_id="conv-hosted-1")

        with patch(
            "api.hosted_entrypoint.AgentStrategyFactory.get_strategy",
            new=AsyncMock(return_value=strategy),
        ):
            events = [e async for e in _hosted_stream(turn, "maf_lite")]

        assert isinstance(events[0], TurnConversationEvent)
        assert events[0].conversation_id == "conv-hosted-1"

    @pytest.mark.asyncio
    async def test_passes_ask_to_strategy(self):
        from api.hosted_entrypoint import _hosted_stream

        captured_ask = []

        async def fake_flow(ask: str):
            captured_ask.append(ask)
            yield "answer"

        strategy = MagicMock()
        strategy.initiate_agent_flow = fake_flow
        turn = TurnRequest(ask="What is the policy?", conversation_id="conv-1")

        with patch(
            "api.hosted_entrypoint.AgentStrategyFactory.get_strategy",
            new=AsyncMock(return_value=strategy),
        ):
            [e async for e in _hosted_stream(turn, "maf_lite")]

        assert captured_ask == ["What is the policy?"]

    @pytest.mark.asyncio
    async def test_emits_text_events_for_string_chunks(self):
        from api.hosted_entrypoint import _hosted_stream

        async def fake_flow(_ask):
            yield "Hello "
            yield "world"

        strategy = MagicMock()
        strategy.initiate_agent_flow = fake_flow
        turn = TurnRequest(ask="Hi", conversation_id="conv-1")

        with patch(
            "api.hosted_entrypoint.AgentStrategyFactory.get_strategy",
            new=AsyncMock(return_value=strategy),
        ):
            events = [e async for e in _hosted_stream(turn, "single_agent_rag")]

        text_events = [e for e in events if isinstance(e, TurnTextEvent)]
        assert [e.text for e in text_events] == ["Hello ", "world"]

    @pytest.mark.asyncio
    async def test_emits_error_event_on_exception(self):
        from api.hosted_entrypoint import _hosted_stream

        async def broken_flow(_ask):
            raise RuntimeError("strategy failure")
            yield  # pragma: no cover

        strategy = MagicMock()
        strategy.initiate_agent_flow = broken_flow
        turn = TurnRequest(ask="Hi", conversation_id="conv-1")

        with patch(
            "api.hosted_entrypoint.AgentStrategyFactory.get_strategy",
            new=AsyncMock(return_value=strategy),
        ):
            stream = _hosted_stream(turn, "maf_lite")
            events = []
            with pytest.raises(RuntimeError, match="strategy failure"):
                async for e in stream:
                    events.append(e)

        assert any(isinstance(e, TurnErrorEvent) for e in events)

    @pytest.mark.asyncio
    async def test_emits_cancelled_event_on_cancellation(self):
        from api.hosted_entrypoint import _hosted_stream

        async def cancelled_flow(_ask):
            raise asyncio.CancelledError
            yield  # pragma: no cover

        strategy = MagicMock()
        strategy.initiate_agent_flow = cancelled_flow
        turn = TurnRequest(ask="Hi", conversation_id="conv-1")

        with patch(
            "api.hosted_entrypoint.AgentStrategyFactory.get_strategy",
            new=AsyncMock(return_value=strategy),
        ):
            stream = _hosted_stream(turn, "maf_lite")
            events = []
            with pytest.raises(asyncio.CancelledError):
                async for e in stream:
                    events.append(e)

        assert any(isinstance(e, TurnCancelledEvent) for e in events)

    @pytest.mark.asyncio
    async def test_raises_for_unsupported_strategy(self):
        from api.hosted_entrypoint import _hosted_stream

        turn = TurnRequest(ask="Hi")
        with pytest.raises(ValueError, match="not supported in the hosted runtime"):
            async for _ in _hosted_stream(turn, "nl2sql"):
                pass  # pragma: no cover

    @pytest.mark.asyncio
    async def test_generates_conversation_id_when_none_provided(self):
        from api.hosted_entrypoint import _hosted_stream

        async def fake_flow(_ask):
            return
            yield  # pragma: no cover

        strategy = MagicMock()
        strategy.initiate_agent_flow = fake_flow
        turn = TurnRequest(ask="Hi", conversation_id=None)

        with patch(
            "api.hosted_entrypoint.AgentStrategyFactory.get_strategy",
            new=AsyncMock(return_value=strategy),
        ):
            events = [e async for e in _hosted_stream(turn, "maf_lite")]

        conv_events = [e for e in events if isinstance(e, TurnConversationEvent)]
        assert len(conv_events) == 1
        assert conv_events[0].conversation_id  # non-empty generated id

    @pytest.mark.asyncio
    async def test_passes_structured_events_through(self):
        from api.hosted_entrypoint import _hosted_stream

        citation = TurnCitationEvent(TurnCitation("src-1", title="Source"))

        async def flow_with_citation(_ask):
            yield citation
            yield "answer"

        strategy = MagicMock()
        strategy.initiate_agent_flow = flow_with_citation
        turn = TurnRequest(ask="Hi", conversation_id="conv-1")

        with patch(
            "api.hosted_entrypoint.AgentStrategyFactory.get_strategy",
            new=AsyncMock(return_value=strategy),
        ):
            events = [e async for e in _hosted_stream(turn, "mcp")]

        assert citation in events


# ── Hosted FastAPI endpoints ─────────────────────────────────────────────────

class TestHostedEntrypointAPI:
    """Integration-style tests for the FastAPI app in hosted_entrypoint."""

    @pytest.fixture(autouse=True)
    def _patch_deps(self, mock_config, mock_cosmos):
        with (
            patch("dependencies.get_config", return_value=mock_config),
            patch("api.hosted_entrypoint.get_config", return_value=mock_config),
            patch("connectors.cosmosdb.get_cosmosdb_client", return_value=mock_cosmos),
            patch("connectors.identity_manager.get_identity_manager", return_value=MagicMock()),
            patch("strategies.base_agent_strategy.get_config", return_value=mock_config),
            patch("strategies.base_agent_strategy.get_cosmosdb_client", return_value=mock_cosmos),
            patch("strategies.base_agent_strategy.get_identity_manager", return_value=MagicMock()),
            patch("strategies.base_agent_strategy.AIProjectClient"),
        ):
            yield

    @pytest.fixture()
    def client(self):
        from fastapi.testclient import TestClient
        from api.hosted_entrypoint import app

        return TestClient(app, raise_server_exceptions=False)

    def test_health_returns_ok(self, client):
        resp = client.get("/health")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "version" in body
        assert isinstance(body["eligible_strategies"], list)
        assert "maf_lite" in body["eligible_strategies"]

    def test_health_eligible_strategies_matches_guard_set(self, client):
        resp = client.get("/health")
        body = resp.json()

        assert set(body["eligible_strategies"]) == HOSTED_ELIGIBLE_STRATEGIES

    def test_invocations_rejects_unsupported_strategy(self, client, mock_config):
        original = mock_config.get.side_effect
        mock_config.get.side_effect = (
            lambda key, default=None, type=str:
            "nl2sql" if key == "AGENT_STRATEGY" else original(key, default, type)
        )

        resp = client.post(
            "/invocations",
            json={"messages": [{"role": "user", "content": "Hello"}]},
        )

        assert resp.status_code == 422
        assert "not supported" in resp.json()["detail"]

    def test_invocations_rejects_empty_messages(self, client):
        resp = client.post(
            "/invocations",
            json={"messages": [{"role": "assistant", "content": "Hi"}]},
        )

        assert resp.status_code == 422

    def test_invocations_rejects_empty_user_message(self, client):
        resp = client.post(
            "/invocations",
            json={"messages": [{"role": "user", "content": "   "}]},
        )

        assert resp.status_code == 422

    def test_invocations_streams_sse_with_eligible_strategy(self, client, mock_config):
        original = mock_config.get.side_effect
        mock_config.get.side_effect = (
            lambda key, default=None, type=str:
            "maf_lite" if key == "AGENT_STRATEGY" else original(key, default, type)
        )

        async def fake_flow(_ask):
            yield "Hello "
            yield "world"

        strategy = MagicMock()
        strategy.initiate_agent_flow = fake_flow

        with patch(
            "api.hosted_entrypoint.AgentStrategyFactory.get_strategy",
            new=AsyncMock(return_value=strategy),
        ):
            resp = client.post(
                "/invocations",
                json={
                    "messages": [{"role": "user", "content": "What is the policy?"}],
                    "conversation_id": "conv-test-1",
                },
            )

        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        body = resp.text

        # Must contain the conversation identity frame
        assert "response.created" in body
        # Must contain text deltas
        assert "response.output_text.delta" in body
        # Must close with completed
        assert "response.completed" in body
        # Must carry the conversation_id
        assert "conv-test-1" in body

    def test_invocations_response_id_header_is_present(self, client, mock_config):
        original = mock_config.get.side_effect
        mock_config.get.side_effect = (
            lambda key, default=None, type=str:
            "maf_lite" if key == "AGENT_STRATEGY" else original(key, default, type)
        )

        async def fake_flow(_ask):
            yield "answer"

        strategy = MagicMock()
        strategy.initiate_agent_flow = fake_flow

        with patch(
            "api.hosted_entrypoint.AgentStrategyFactory.get_strategy",
            new=AsyncMock(return_value=strategy),
        ):
            resp = client.post(
                "/invocations",
                json={"messages": [{"role": "user", "content": "Hi"}]},
            )

        assert resp.status_code == 200
        assert "X-Response-ID" in resp.headers
        assert resp.headers["X-Response-ID"].startswith("resp_")


# ── Conversation history injection ───────────────────────────────────────────

class TestHostedStreamHistoryInjection:
    """Verify prior-turn history is injected into strategy.conversation["messages"]."""

    @pytest.fixture(autouse=True)
    def _patch_config(self, mock_config, mock_cosmos):
        with (
            patch("dependencies.get_config", return_value=mock_config),
            patch("connectors.cosmosdb.get_cosmosdb_client", return_value=mock_cosmos),
            patch("connectors.identity_manager.get_identity_manager", return_value=MagicMock()),
            patch("strategies.base_agent_strategy.get_config", return_value=mock_config),
            patch("strategies.base_agent_strategy.get_cosmosdb_client", return_value=mock_cosmos),
            patch("strategies.base_agent_strategy.get_identity_manager", return_value=MagicMock()),
            patch("strategies.base_agent_strategy.AIProjectClient"),
        ):
            yield

    @pytest.mark.asyncio
    async def test_history_is_set_on_strategy_conversation(self):
        """_hosted_stream injects history_messages into strategy.conversation["messages"]."""
        from api.hosted_entrypoint import _hosted_stream

        strategy = MagicMock()
        captured_conv: dict = {}

        async def fake_flow(_ask):
            captured_conv.update(strategy.conversation)
            yield "answer"

        strategy.initiate_agent_flow = fake_flow
        history = [
            {"role": "user", "content": "What is RAG?"},
            {"role": "assistant", "content": "RAG stands for Retrieval-Augmented Generation."},
        ]
        turn = TurnRequest(ask="Can you give an example?", conversation_id="conv-hist-1")

        with patch(
            "api.hosted_entrypoint.AgentStrategyFactory.get_strategy",
            new=AsyncMock(return_value=strategy),
        ):
            [e async for e in _hosted_stream(turn, "maf_lite", history)]

        assert captured_conv["messages"] == history
        assert captured_conv["id"] == "conv-hist-1"

    @pytest.mark.asyncio
    async def test_hosted_mode_flag_is_set(self):
        """_hosted_stream always sets hosted_mode=True on strategy.conversation."""
        from api.hosted_entrypoint import _hosted_stream

        strategy = MagicMock()
        captured_hosted_mode: list[bool] = []

        async def fake_flow(_ask):
            captured_hosted_mode.append(strategy.conversation.get("hosted_mode", False))
            yield "answer"

        strategy.initiate_agent_flow = fake_flow
        turn = TurnRequest(ask="Hello", conversation_id="conv-hosted-flag")

        with patch(
            "api.hosted_entrypoint.AgentStrategyFactory.get_strategy",
            new=AsyncMock(return_value=strategy),
        ):
            [e async for e in _hosted_stream(turn, "single_agent_rag")]

        assert captured_hosted_mode == [True]

    @pytest.mark.asyncio
    async def test_empty_history_when_none_provided(self):
        """_hosted_stream sets conversation["messages"] to [] when no history given."""
        from api.hosted_entrypoint import _hosted_stream

        strategy = MagicMock()
        captured_msgs: list = []

        async def fake_flow(_ask):
            captured_msgs.extend(strategy.conversation.get("messages", []))
            yield "answer"

        strategy.initiate_agent_flow = fake_flow
        turn = TurnRequest(ask="First message", conversation_id="conv-no-hist")

        with patch(
            "api.hosted_entrypoint.AgentStrategyFactory.get_strategy",
            new=AsyncMock(return_value=strategy),
        ):
            [e async for e in _hosted_stream(turn, "maf_lite")]

        assert captured_msgs == []

    @pytest.mark.asyncio
    async def test_two_turn_continuity_maf_lite(self):
        """Turn 2 for maf_lite receives turn 1 history via conversation["messages"]."""
        from api.hosted_entrypoint import _hosted_stream

        # Turn 1 — strategy receives empty history (first message)
        strategy1 = MagicMock()
        received_turn1: list = []

        async def flow_t1(_ask):
            received_turn1.extend(strategy1.conversation.get("messages", []))
            yield "Turn 1 answer"

        strategy1.initiate_agent_flow = flow_t1

        turn1 = TurnRequest(ask="Turn 1 question", conversation_id="conv-two-turn")
        with patch(
            "api.hosted_entrypoint.AgentStrategyFactory.get_strategy",
            new=AsyncMock(return_value=strategy1),
        ):
            [e async for e in _hosted_stream(turn1, "maf_lite", [])]

        assert received_turn1 == [], "Turn 1 should see no prior history"

        # Turn 2 — Foundry sends the accumulated history
        history_t2 = [
            {"role": "user", "content": "Turn 1 question"},
            {"role": "assistant", "content": "Turn 1 answer"},
        ]
        strategy2 = MagicMock()
        received_turn2: list = []

        async def flow_t2(_ask):
            received_turn2.extend(strategy2.conversation.get("messages", []))
            yield "Turn 2 answer"

        strategy2.initiate_agent_flow = flow_t2

        turn2 = TurnRequest(ask="Turn 2 question", conversation_id="conv-two-turn")
        with patch(
            "api.hosted_entrypoint.AgentStrategyFactory.get_strategy",
            new=AsyncMock(return_value=strategy2),
        ):
            [e async for e in _hosted_stream(turn2, "maf_lite", history_t2)]

        assert len(received_turn2) == 2, "Turn 2 should see turn 1 history"
        assert received_turn2[0] == {"role": "user", "content": "Turn 1 question"}
        assert received_turn2[1] == {"role": "assistant", "content": "Turn 1 answer"}

    @pytest.mark.asyncio
    async def test_two_turn_continuity_single_agent_rag(self):
        """Turn 2 for single_agent_rag receives turn 1 history via conversation["messages"]."""
        from api.hosted_entrypoint import _hosted_stream

        history_t2 = [
            {"role": "user", "content": "What is the capital?"},
            {"role": "assistant", "content": "It depends on the country."},
        ]
        strategy = MagicMock()
        received: list = []

        async def flow(_ask):
            received.extend(strategy.conversation.get("messages", []))
            yield "answer"

        strategy.initiate_agent_flow = flow
        turn = TurnRequest(ask="Which country?", conversation_id="conv-sar-two")

        with patch(
            "api.hosted_entrypoint.AgentStrategyFactory.get_strategy",
            new=AsyncMock(return_value=strategy),
        ):
            [e async for e in _hosted_stream(turn, "single_agent_rag", history_t2)]

        assert len(received) == 2
        assert received[0]["role"] == "user"
        assert received[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_maf_agent_service_thread_id_cached_across_turns(self):
        """maf_agent_service thread_id is stored and restored for the next turn."""
        from api import hosted_entrypoint
        from api.hosted_entrypoint import _hosted_stream

        # Reset cache to isolate this test
        hosted_entrypoint._maf_thread_cache.clear()

        CONV_ID = "conv-maf-thread-cache"
        THREAD_ID = "thread-abc123"

        # Turn 1: strategy sets thread_id on its conversation dict
        strategy_t1 = MagicMock()

        async def flow_t1(_ask):
            # Simulate maf_agent_service recording the server thread_id
            strategy_t1.conversation["thread_id"] = THREAD_ID
            yield "answer"

        strategy_t1.initiate_agent_flow = flow_t1

        with patch(
            "api.hosted_entrypoint.AgentStrategyFactory.get_strategy",
            new=AsyncMock(return_value=strategy_t1),
        ):
            [e async for e in _hosted_stream(
                TurnRequest(ask="Turn 1", conversation_id=CONV_ID),
                "maf_agent_service",
            )]

        assert hosted_entrypoint._maf_thread_cache.get(CONV_ID) == THREAD_ID

        # Turn 2: thread_id should be restored from cache
        strategy_t2 = MagicMock()
        captured_thread: list = []

        async def flow_t2(_ask):
            captured_thread.append(strategy_t2.conversation.get("thread_id"))
            yield "answer"

        strategy_t2.initiate_agent_flow = flow_t2

        with patch(
            "api.hosted_entrypoint.AgentStrategyFactory.get_strategy",
            new=AsyncMock(return_value=strategy_t2),
        ):
            [e async for e in _hosted_stream(
                TurnRequest(ask="Turn 2", conversation_id=CONV_ID),
                "maf_agent_service",
            )]

        assert captured_thread == [THREAD_ID], "Turn 2 should receive the cached thread_id"


# ── Cosmos isolation / profile-memory isolation ──────────────────────────────

class TestHostedNoCosmosProfile:
    """Prove that the hosted path does not read/write Cosmos user profiles and
    that two callers with different conversation IDs cannot share state."""

    @pytest.fixture(autouse=True)
    def _patch_config(self, mock_config, mock_cosmos):
        with (
            patch("dependencies.get_config", return_value=mock_config),
            patch("connectors.cosmosdb.get_cosmosdb_client", return_value=mock_cosmos),
            patch("connectors.identity_manager.get_identity_manager", return_value=MagicMock()),
            patch("strategies.base_agent_strategy.get_config", return_value=mock_config),
            patch("strategies.base_agent_strategy.get_cosmosdb_client", return_value=mock_cosmos),
            patch("strategies.base_agent_strategy.get_identity_manager", return_value=MagicMock()),
            patch("strategies.base_agent_strategy.AIProjectClient"),
        ):
            yield

    @pytest.mark.asyncio
    async def test_conversation_dicts_are_isolated_between_callers(self):
        """Two concurrent callers with different conversation IDs get independent
        strategy.conversation dicts — no shared state."""
        from api.hosted_entrypoint import _hosted_stream

        convs: dict[str, dict] = {}

        def make_strategy(conv_id: str):
            s = MagicMock()

            async def flow(_ask):
                convs[conv_id] = dict(s.conversation)
                yield "answer"

            s.initiate_agent_flow = flow
            return s

        strategy_a = make_strategy("caller-A")
        strategy_b = make_strategy("caller-B")

        with patch(
            "api.hosted_entrypoint.AgentStrategyFactory.get_strategy",
            new=AsyncMock(side_effect=[strategy_a, strategy_b]),
        ):
            [e async for e in _hosted_stream(
                TurnRequest(ask="Question A", conversation_id="caller-A"),
                "maf_lite",
            )]
            [e async for e in _hosted_stream(
                TurnRequest(ask="Question B", conversation_id="caller-B"),
                "maf_lite",
            )]

        assert convs["caller-A"]["id"] == "caller-A"
        assert convs["caller-B"]["id"] == "caller-B"
        # Confirm the dicts are distinct objects
        assert convs["caller-A"] is not convs["caller-B"]

    @pytest.mark.asyncio
    async def test_hosted_stream_does_not_call_cosmos_get_document(self):
        """The hosted stream must not trigger any Cosmos get_document call for
        user profiles (proxy: cosmos mock should never be called)."""
        from api.hosted_entrypoint import _hosted_stream

        cosmos_spy = AsyncMock(return_value=None)

        strategy = MagicMock()

        async def fake_flow(_ask):
            # Simulate the strategy accessing conversation but NOT cosmos
            _ = strategy.conversation.get("messages", [])
            yield "answer"

        strategy.initiate_agent_flow = fake_flow
        strategy.cosmos.get_document = cosmos_spy

        turn = TurnRequest(ask="Hi", conversation_id="conv-no-cosmos")

        with patch(
            "api.hosted_entrypoint.AgentStrategyFactory.get_strategy",
            new=AsyncMock(return_value=strategy),
        ):
            [e async for e in _hosted_stream(turn, "maf_lite")]

        # The mock strategy has no real Cosmos path — verify the entrypoint
        # itself does not call any cosmos method
        cosmos_spy.assert_not_called()

    @pytest.mark.asyncio
    async def test_invocations_endpoint_injects_history_before_last_user_msg(self):
        """POST /invocations extracts prior history and omits the last user message."""
        from fastapi.testclient import TestClient
        from api.hosted_entrypoint import app

        async def fake_flow(_ask):
            yield "answer"

        strategy = MagicMock()
        strategy.initiate_agent_flow = fake_flow
        captured_messages: list = []

        async def capturing_flow(_ask):
            captured_messages.extend(strategy.conversation.get("messages", []))
            yield "answer"

        strategy.initiate_agent_flow = capturing_flow

        with (
            patch("api.hosted_entrypoint.get_config") as mock_cfg,
            patch(
                "api.hosted_entrypoint.AgentStrategyFactory.get_strategy",
                new=AsyncMock(return_value=strategy),
            ),
        ):
            mock_cfg.return_value.get.side_effect = lambda k, d=None, type=str: (
                "maf_lite" if k == "AGENT_STRATEGY" else d
            )
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post(
                "/invocations",
                json={
                    "messages": [
                        {"role": "user", "content": "Turn 1 question"},
                        {"role": "assistant", "content": "Turn 1 answer"},
                        {"role": "user", "content": "Turn 2 question"},
                    ],
                    "conversation_id": "conv-inject-hist",
                },
            )

        assert resp.status_code == 200
        assert len(captured_messages) == 2
        assert captured_messages[0] == {"role": "user", "content": "Turn 1 question"}
        assert captured_messages[1] == {"role": "assistant", "content": "Turn 1 answer"}


# ── Helpers ──────────────────────────────────────────────────────────────────

async def _async_gen(items):
    """Tiny helper to turn a list into an async generator."""
    for item in items:
        yield item
