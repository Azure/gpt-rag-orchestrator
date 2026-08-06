"""Tests for the hosted Responses adapters and managed Conversations entrypoint.

Covers:
- Responses API SSE serialization of every typed turn event
- Terminal closing frames after the text stream
- Hosted strategy guard (explicit failure for unsupported strategies)
- Hosted stream execution without Cosmos dependency
- Hosted FastAPI endpoints (health, readiness, responses, and invocations)
"""

from __future__ import annotations

import asyncio
import json
import logging
from copy import deepcopy
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import omit
from openai.lib.streaming.responses import ResponseStreamState
from openai.types.responses import ResponseStreamEvent
from pydantic import TypeAdapter

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
_CONVERSATION_ID = "conv-testconversation"
_CREATED_AT = 1_786_012_345.0
_MODEL = "gpt-4o"
_RESPONSE_EVENT_ADAPTER = TypeAdapter(ResponseStreamEvent)


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
            created_at=_CREATED_AT,
            model=_MODEL,
            **kwargs,
        )
    )


# ── ResponsesAdapter serialization ───────────────────────────────────────────

class TestResponsesAdapterConversationEvent:
    def test_emits_three_opening_frames(self):
        frames = _serialize(TurnConversationEvent("conv-1"))

        assert len(frames) == 3

    def test_response_created_has_standard_required_response_fields(self):
        frames = _serialize(TurnConversationEvent("conv-abc"))

        created = frames[0]
        response = created["data"]["response"]
        assert created["event"] == "response.created"
        assert response["conversation"] == {"id": "conv-abc"}
        assert "conversation_id" not in response
        assert response["id"] == _RESP_ID
        assert response["status"] == "in_progress"
        assert response["created_at"] == _CREATED_AT
        assert response["model"] == _MODEL
        assert response["tools"] == []
        assert response["tool_choice"] == "none"
        assert response["parallel_tool_calls"] is False

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
        assert part["data"]["part"]["logprobs"] == []


class TestResponsesAdapterTextEvent:
    def test_emits_output_text_delta(self):
        frames = _serialize(TurnTextEvent("Hello"))

        assert len(frames) == 1
        assert frames[0]["event"] == "response.output_text.delta"
        assert frames[0]["data"]["delta"] == "Hello"
        assert frames[0]["data"]["item_id"] == _ITEM_ID
        assert frames[0]["data"]["logprobs"] == []

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
    @pytest.mark.parametrize("status", list(TurnToolStatus))
    def test_internal_tool_progress_is_not_emitted_as_function_call_output(
        self,
        status: TurnToolStatus,
    ):
        activity = TurnToolActivity("search_kb", status, call_id="call-1")

        assert _serialize(TurnToolActivityEvent(activity=activity)) == []


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
    def test_emits_standard_error_event(self):
        frames = _serialize(TurnCancelledEvent(reason="cancelled"))

        assert len(frames) == 1
        assert frames[0]["event"] == "error"
        assert frames[0]["data"]["code"] == "cancelled"
        assert frames[0]["data"]["message"] == "cancelled"


class TestResponsesAdapterSDKModels:
    def test_every_emitted_event_validates_against_pinned_sdk_models(self):
        events = [
            TurnConversationEvent(_CONVERSATION_ID),
            TurnTextEvent("Hello"),
            TurnCitationEvent(
                TurnCitation(
                    "src-1",
                    title="Policy",
                    url="https://example.com/policy",
                )
            ),
            TurnToolActivityEvent(
                TurnToolActivity(
                    "search_kb",
                    TurnToolStatus.STARTED,
                    call_id="call-1",
                )
            ),
            TurnToolActivityEvent(
                TurnToolActivity(
                    "search_kb",
                    TurnToolStatus.COMPLETED,
                    call_id="call-1",
                )
            ),
            TurnToolActivityEvent(
                TurnToolActivity(
                    "search_kb",
                    TurnToolStatus.FAILED,
                    call_id="call-1",
                    message="Tool execution failed",
                )
            ),
            TurnErrorEvent(),
            TurnCancelledEvent(),
        ]
        frames = [
            frame
            for event in events
            for frame in _serialize(event)
        ]
        frames.extend(
            _parse_frames(
                responses_terminal_events(
                    response_id=_RESP_ID,
                    item_id=_ITEM_ID,
                    conversation_id=_CONVERSATION_ID,
                    full_text="Hello",
                    created_at=_CREATED_AT,
                    model=_MODEL,
                )
            )
        )

        for frame in frames:
            validated = _RESPONSE_EVENT_ADAPTER.validate_python(frame["data"])
            assert validated.type == frame["event"]


class TestResponsesTerminalEvents:
    def test_emits_four_closing_frames_in_order(self):
        frames = _parse_frames(
            responses_terminal_events(
                response_id=_RESP_ID,
                item_id=_ITEM_ID,
                conversation_id=_CONVERSATION_ID,
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
                conversation_id=_CONVERSATION_ID,
                full_text="Final answer.",
            )
        )

        assert frames[0]["data"]["text"] == "Final answer."
        assert frames[1]["data"]["part"]["text"] == "Final answer."
        assert frames[2]["data"]["item"]["content"][0]["text"] == "Final answer."
        assert frames[0]["data"]["logprobs"] == []
        assert frames[1]["data"]["part"]["logprobs"] == []
        assert frames[2]["data"]["item"]["content"][0]["logprobs"] == []

    def test_response_completed_carries_response_id(self):
        frames = _parse_frames(
            responses_terminal_events(
                response_id=_RESP_ID,
                item_id=_ITEM_ID,
                conversation_id=_CONVERSATION_ID,
                full_text="",
                created_at=_CREATED_AT,
                model=_MODEL,
            )
        )

        response = frames[3]["data"]["response"]
        assert response["id"] == _RESP_ID
        assert response["status"] == "completed"
        assert response["conversation"] == {"id": _CONVERSATION_ID}
        assert response["created_at"] == _CREATED_AT
        assert response["model"] == _MODEL
        assert response["tools"] == []
        assert response["tool_choice"] == "none"
        assert response["parallel_tool_calls"] is False


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

        with (
            patch(
                "api.hosted_entrypoint.AgentStrategyFactory.get_strategy",
                new=AsyncMock(return_value=strategy),
            ),
            patch(
                "api.hosted_entrypoint.resolve_managed_conversation_id",
                new=AsyncMock(return_value="conv-1"),
            ),
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
    async def test_toolbox_strategy_fails_closed_without_foundry_call_id(self):
        """ADR-0001: the mcp/Toolbox path must refuse to run without a
        validated platform call id rather than silently using service
        identity or a manual metadata filter."""
        from api.hosted_entrypoint import _hosted_stream
        from util.foundry_platform import MissingFoundryCallContextError

        turn = TurnRequest(ask="Hi", conversation_id="conv-1")  # no foundry_call_id
        factory = AsyncMock()

        with patch(
            "api.hosted_entrypoint.AgentStrategyFactory.get_strategy",
            new=factory,
        ):
            with pytest.raises(MissingFoundryCallContextError):
                async for _ in _hosted_stream(turn, "mcp"):
                    pass  # pragma: no cover

        # No fallback: the strategy must never even be constructed.
        factory.assert_not_called()

    @pytest.mark.asyncio
    async def test_toolbox_strategy_propagates_validated_call_id(self):
        from api.hosted_entrypoint import _hosted_stream

        async def fake_flow(_ask):
            yield "answer"

        strategy = MagicMock()
        strategy.initiate_agent_flow = fake_flow
        turn = TurnRequest(ask="Hi", conversation_id="conv-1", foundry_call_id="call-abc")

        with patch(
            "api.hosted_entrypoint.AgentStrategyFactory.get_strategy",
            new=AsyncMock(return_value=strategy),
        ):
            [e async for e in _hosted_stream(turn, "mcp")]

        assert strategy.foundry_call_id == "call-abc"

    @pytest.mark.asyncio
    async def test_non_toolbox_strategy_does_not_require_foundry_call_id(self):
        """Classic hosted-eligible strategies that don't call Toolbox (e.g.
        maf_lite, which uses the OBO-based Foundry IQ path) are unaffected
        by the Toolbox call-id guard."""
        from api.hosted_entrypoint import _hosted_stream

        async def fake_flow(_ask):
            yield "answer"

        strategy = MagicMock()
        strategy.initiate_agent_flow = fake_flow
        turn = TurnRequest(ask="Hi", conversation_id="conv-1")  # no foundry_call_id

        with patch(
            "api.hosted_entrypoint.AgentStrategyFactory.get_strategy",
            new=AsyncMock(return_value=strategy),
        ):
            events = [e async for e in _hosted_stream(turn, "maf_lite")]

        assert any(isinstance(e, TurnTextEvent) for e in events)
        assert strategy.foundry_call_id is None

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
        turn = TurnRequest(
            ask="Hi", conversation_id="conv-1", foundry_call_id="call-1"
        )

        with patch(
            "api.hosted_entrypoint.AgentStrategyFactory.get_strategy",
            new=AsyncMock(return_value=strategy),
        ):
            events = [e async for e in _hosted_stream(turn, "mcp")]

        assert citation in events

    @pytest.mark.parametrize(
        "strategy_key",
        ["maf_lite", "single_agent_rag", "mcp"],
    )
    @pytest.mark.asyncio
    async def test_two_turn_history_reaches_history_owning_strategy(
        self,
        strategy_key: str,
    ):
        """Turn two receives the complete ordered turn-one exchange."""
        from api.hosted_entrypoint import _hosted_stream

        received_conversations = []

        class RecordingStrategy:
            project_client = MagicMock()

            def set_context(self, _conversation_id):
                pass

            async def initiate_agent_flow(self, ask):
                received_conversations.append(deepcopy(self.conversation))
                self.conversation["messages"].extend([
                    {"role": "user", "text": ask},
                    {"role": "assistant", "text": f"answer to {ask}"},
                ])
                yield f"answer to {ask}"

        strategies = [RecordingStrategy(), RecordingStrategy()]
        factory = AsyncMock(side_effect=strategies)

        with (
            patch(
                "api.hosted_entrypoint.AgentStrategyFactory.get_strategy",
                new=factory,
            ),
            patch(
                "api.hosted_entrypoint.resolve_managed_conversation_id",
                new=AsyncMock(return_value="conv-1"),
            ),
        ):
            first_turn = TurnRequest(
                ask="first question",
                conversation_id="conv-1",
                foundry_call_id="call-1",
            )
            [event async for event in _hosted_stream(first_turn, strategy_key)]

            prior_history = [
                {"role": "user", "text": "first question"},
                {"role": "assistant", "text": "answer to first question"},
            ]
            second_turn = TurnRequest(
                ask="follow-up",
                conversation_id="conv-1",
                foundry_call_id="call-1",
            )
            [event async for event in _hosted_stream(
                second_turn,
                strategy_key,
                prior_history,
            )]

        assert [
            {
                "id": conversation["id"],
                "messages": conversation["messages"],
            }
            for conversation in received_conversations
        ] == [
            {"id": "conv-1", "messages": []},
            {"id": "conv-1", "messages": prior_history},
        ]
        if strategy_key == "single_agent_rag":
            assert [
                conversation["thread_id"]
                for conversation in received_conversations
            ] == ["conv-1", "conv-1"]
        assert factory.await_args_list[0].kwargs == {"hosted_runtime": True}
        assert factory.await_args_list[1].kwargs == {"hosted_runtime": True}

    @pytest.mark.asyncio
    async def test_two_turn_agent_service_reuses_managed_conversation_thread(self):
        from api.hosted_entrypoint import _hosted_stream
        from strategies.agent_provider_v2 import AGENT_BACKEND_TAG

        received_conversations = []

        class RecordingStrategy:
            project_client = MagicMock()

            def set_context(self, _conversation_id):
                pass

            async def initiate_agent_flow(self, _ask):
                received_conversations.append(deepcopy(self.conversation))
                yield "answer"

        factory = AsyncMock(
            side_effect=[RecordingStrategy(), RecordingStrategy()],
        )
        with (
            patch(
                "api.hosted_entrypoint.AgentStrategyFactory.get_strategy",
                new=factory,
            ),
            patch(
                "api.hosted_entrypoint.resolve_managed_conversation_id",
                new=AsyncMock(return_value="conv-thread"),
            ),
        ):
            first_turn = TurnRequest(ask="first", conversation_id="conv-thread")
            [event async for event in _hosted_stream(
                first_turn,
                "maf_agent_service",
            )]
            second_turn = TurnRequest(ask="second", conversation_id="conv-thread")
            [event async for event in _hosted_stream(
                second_turn,
                "maf_agent_service",
                [
                    {"role": "user", "text": "first"},
                    {"role": "assistant", "text": "answer"},
                ],
            )]

        assert [conversation["thread_id"] for conversation in received_conversations] == [
            "conv-thread",
            "conv-thread",
        ]
        assert all(
            conversation["agent_backend"] == AGENT_BACKEND_TAG
            for conversation in received_conversations
        )

    @pytest.mark.asyncio
    async def test_agent_service_validates_supplied_managed_conversation(self):
        from api.hosted_entrypoint import _hosted_stream

        openai_client = MagicMock()
        openai_client.__aenter__ = AsyncMock(return_value=openai_client)
        openai_client.__aexit__ = AsyncMock(return_value=False)
        openai_client.conversations.retrieve = AsyncMock(
            return_value=SimpleNamespace(id="conv_existing"),
        )
        openai_client.conversations.create = AsyncMock()

        strategy = MagicMock()
        strategy.project_client.get_openai_client.return_value = openai_client
        received = []

        async def fake_flow(_ask):
            received.append(deepcopy(strategy.conversation))
            yield "answer"

        strategy.initiate_agent_flow = fake_flow
        with patch(
            "api.hosted_entrypoint.AgentStrategyFactory.get_strategy",
            new=AsyncMock(return_value=strategy),
        ):
            events = [
                event
                async for event in _hosted_stream(
                    TurnRequest(ask="hello", conversation_id="conv_existing"),
                    "maf_agent_service",
                )
            ]

        openai_client.conversations.retrieve.assert_awaited_once_with(
            "conv_existing"
        )
        openai_client.conversations.create.assert_not_awaited()
        assert received[0]["thread_id"] == "conv_existing"
        assert events[0].conversation_id == "conv_existing"

    @pytest.mark.asyncio
    async def test_agent_service_creates_managed_conversation_when_id_absent(self):
        from api.hosted_entrypoint import _hosted_stream

        openai_client = MagicMock()
        openai_client.__aenter__ = AsyncMock(return_value=openai_client)
        openai_client.__aexit__ = AsyncMock(return_value=False)
        openai_client.conversations.retrieve = AsyncMock()
        openai_client.conversations.create = AsyncMock(
            return_value=SimpleNamespace(id="conv_created"),
        )

        strategy = MagicMock()
        strategy.project_client.get_openai_client.return_value = openai_client
        received = []

        async def fake_flow(_ask):
            received.append(deepcopy(strategy.conversation))
            yield "answer"

        strategy.initiate_agent_flow = fake_flow
        with patch(
            "api.hosted_entrypoint.AgentStrategyFactory.get_strategy",
            new=AsyncMock(return_value=strategy),
        ):
            events = [
                event
                async for event in _hosted_stream(
                    TurnRequest(ask="hello", conversation_id=None),
                    "maf_agent_service",
                )
            ]

        openai_client.conversations.create.assert_awaited_once_with()
        openai_client.conversations.retrieve.assert_not_awaited()
        assert received[0]["thread_id"] == "conv_created"
        assert events[0].conversation_id == "conv_created"

    @pytest.mark.asyncio
    async def test_agent_service_rejects_failed_conversation_validation(self):
        from api.hosted_entrypoint import _hosted_stream

        openai_client = MagicMock()
        openai_client.__aenter__ = AsyncMock(return_value=openai_client)
        openai_client.__aexit__ = AsyncMock(return_value=False)
        openai_client.conversations.retrieve = AsyncMock(
            side_effect=RuntimeError("conversation not found"),
        )

        strategy = MagicMock()
        strategy.project_client.get_openai_client.return_value = openai_client
        strategy.initiate_agent_flow = AsyncMock()

        with patch(
            "api.hosted_entrypoint.AgentStrategyFactory.get_strategy",
            new=AsyncMock(return_value=strategy),
        ):
            with pytest.raises(RuntimeError, match="conversation not found"):
                _ = [
                    event
                    async for event in _hosted_stream(
                        TurnRequest(
                            ask="hello",
                            conversation_id="conv_missing",
                        ),
                        "maf_agent_service",
                    )
                ]

        strategy.initiate_agent_flow.assert_not_called()

    @pytest.mark.asyncio
    async def test_separate_conversations_and_untrusted_callers_share_no_state(self):
        from api.hosted_entrypoint import _hosted_stream

        received = []

        class MutatingStrategy:
            def set_context(self, _conversation_id):
                pass

            async def initiate_agent_flow(self, ask):
                received.append({
                    "conversation": deepcopy(self.conversation),
                    "user_context": deepcopy(self.user_context),
                })
                self.conversation["messages"].append(
                    {"role": "assistant", "text": f"private {ask}"}
                )
                self.user_context["mutated"] = True
                yield "answer"

        histories = [
            [{"role": "user", "text": "caller A secret"}],
            [{"role": "user", "text": "caller B secret"}],
        ]
        factory = AsyncMock(
            side_effect=[MutatingStrategy(), MutatingStrategy()],
        )
        with patch(
            "api.hosted_entrypoint.AgentStrategyFactory.get_strategy",
            new=factory,
        ):
            for index, conversation_id in enumerate(("conv-a", "conv-b")):
                turn = TurnRequest(
                    ask=f"ask-{index}",
                    conversation_id=conversation_id,
                    user_context={
                        "principal_id": f"untrusted-caller-{index}",
                        "groups": ["untrusted"],
                    },
                )
                [event async for event in _hosted_stream(
                    turn,
                    "maf_lite",
                    histories[index],
                )]

        assert received == [
            {
                "conversation": {
                    "id": "conv-a",
                    "messages": [{"role": "user", "text": "caller A secret"}],
                },
                "user_context": {},
            },
            {
                "conversation": {
                    "id": "conv-b",
                    "messages": [{"role": "user", "text": "caller B secret"}],
                },
                "user_context": {},
            },
        ]
        assert histories == [
            [{"role": "user", "text": "caller A secret"}],
            [{"role": "user", "text": "caller B secret"}],
        ]


class TestSseGeneratorErrorClassification:
    """Regression coverage for the SSE error-classification gap: a
    ``MissingFoundryCallContextError`` raised inside ``_hosted_stream`` must
    be reported with a distinct, non-generic SSE error code -- it must never
    be silently downgraded to ``internal_error`` by ``_sse_generator``'s
    broad ``except Exception`` clause.

    Exercises ``_sse_generator`` directly (not just ``_hosted_stream``),
    bypassing the ``/invocations`` HTTP handler's precheck entirely, to
    simulate a hypothetical future/internal caller that reaches the
    generator without that HTTP-level 401 guard ever running."""

    @pytest.mark.asyncio
    async def test_missing_call_context_is_not_downgraded_to_internal_error(self):
        from api.hosted_entrypoint import _sse_generator
        from util.foundry_platform import MISSING_CALL_CONTEXT_MESSAGE

        turn = TurnRequest(ask="Hi", conversation_id="conv-1")  # no foundry_call_id
        factory = AsyncMock()

        with patch(
            "api.hosted_entrypoint.AgentStrategyFactory.get_strategy",
            new=factory,
        ):
            frames = [
                frame
                async for frame in _sse_generator(turn, "mcp", _RESP_ID, _ITEM_ID)
            ]

        # Fail closed before any strategy/Toolbox call is even attempted.
        factory.assert_not_called()

        parsed = _parse_frames(frames)
        error_frames = [frame for frame in parsed if frame["event"] == "error"]
        assert len(error_frames) == 1
        error_data = error_frames[0]["data"]
        assert error_data["code"] == "missing_call_context"
        assert error_data["code"] != "internal_error"
        assert error_data["message"] == MISSING_CALL_CONTEXT_MESSAGE

        # Nothing else was emitted: no partial content, no terminal/closing
        # frames after the error (consistent with error_emitted semantics).
        assert parsed == error_frames


class TestSseGeneratorLifecycle:
    @pytest.mark.asyncio
    async def test_sequence_numbers_increase_across_complete_stream(self):
        from api.hosted_entrypoint import _sse_generator

        async def fake_stream(*_args):
            yield TurnConversationEvent("conv-managed")
            yield TurnTextEvent("Hello ")
            yield TurnCitationEvent(TurnCitation("src-1"))
            yield TurnTextEvent("world")

        with patch("api.hosted_entrypoint._hosted_stream", new=fake_stream):
            frames = [
                frame
                async for frame in _sse_generator(
                    TurnRequest(ask="Hi"),
                    "maf_lite",
                    _RESP_ID,
                    _ITEM_ID,
                    model=_MODEL,
                )
            ]

        parsed = _parse_frames(frames)
        assert [frame["data"]["sequence_number"] for frame in parsed] == list(
            range(len(parsed))
        )

    @pytest.mark.asyncio
    async def test_completed_response_recovers_managed_conversation_from_stream(self):
        from api.hosted_entrypoint import _sse_generator

        async def fake_stream(*_args):
            yield TurnConversationEvent("conv-created-by-foundry")
            yield TurnTextEvent("answer")

        with patch("api.hosted_entrypoint._hosted_stream", new=fake_stream):
            frames = [
                frame
                async for frame in _sse_generator(
                    TurnRequest(ask="Hi", conversation_id=None),
                    "maf_agent_service",
                    _RESP_ID,
                    _ITEM_ID,
                    model=_MODEL,
                )
            ]

        parsed = _parse_frames(frames)
        created = parsed[0]["data"]["response"]
        completed = parsed[-1]["data"]["response"]
        assert created["conversation"] == {"id": "conv-created-by-foundry"}
        assert completed["conversation"] == created["conversation"]
        assert completed["created_at"] == created["created_at"]

    @pytest.mark.asyncio
    async def test_complete_stream_is_accepted_by_pinned_sdk_state(self):
        from api.hosted_entrypoint import _sse_generator

        async def fake_stream(*_args):
            yield TurnConversationEvent("conv-managed")
            yield TurnToolActivityEvent(
                TurnToolActivity(
                    "search_kb",
                    TurnToolStatus.STARTED,
                    call_id="call-1",
                )
            )
            yield TurnTextEvent("answer")
            yield TurnToolActivityEvent(
                TurnToolActivity(
                    "search_kb",
                    TurnToolStatus.COMPLETED,
                    call_id="call-1",
                )
            )

        with patch("api.hosted_entrypoint._hosted_stream", new=fake_stream):
            frames = [
                frame
                async for frame in _sse_generator(
                    TurnRequest(ask="Hi"),
                    "maf_lite",
                    _RESP_ID,
                    _ITEM_ID,
                    model=_MODEL,
                )
            ]

        state = ResponseStreamState(input_tools=omit, text_format=omit)
        for frame in _parse_frames(frames):
            event = _RESPONSE_EVENT_ADAPTER.validate_python(frame["data"])
            state.handle_event(event)


class TestHostedConstruction:
    @pytest.mark.parametrize("strategy_key", sorted(HOSTED_ELIGIBLE_STRATEGIES))
    @pytest.mark.asyncio
    async def test_eligible_strategy_construction_never_creates_cosmos_client(
        self,
        strategy_key: str,
        mock_config,
        mock_identity_manager,
    ):
        from strategies.agent_strategy_factory import AgentStrategyFactory
        from strategies.base_agent_strategy import BaseAgentStrategy

        class ProbeStrategy(BaseAgentStrategy):
            async def initiate_agent_flow(self, _user_message):
                yield "unused"

        cosmos_factory = MagicMock(
            side_effect=AssertionError("Cosmos must not be constructed"),
        )
        with (
            patch.dict(
                AgentStrategyFactory._REGISTRY,
                {strategy_key: lambda: ProbeStrategy()},
            ),
            patch(
                "strategies.base_agent_strategy.get_config",
                return_value=mock_config,
            ),
            patch(
                "strategies.base_agent_strategy.get_identity_manager",
                return_value=mock_identity_manager,
            ),
            patch(
                "strategies.base_agent_strategy.get_cosmosdb_client",
                new=cosmos_factory,
            ),
            patch("strategies.base_agent_strategy.AIProjectClient"),
        ):
            strategy = await AgentStrategyFactory.get_strategy(
                strategy_key,
                hosted_runtime=True,
            )

        cosmos_factory.assert_not_called()
        assert strategy.cosmos is None
        assert strategy.hosted_runtime is True
        assert strategy.profile_memory_enabled is False

    @pytest.mark.asyncio
    async def test_classic_strategy_construction_still_uses_cosmos(
        self,
        mock_config,
        mock_identity_manager,
        mock_cosmos,
    ):
        from strategies.agent_strategy_factory import AgentStrategyFactory
        from strategies.base_agent_strategy import BaseAgentStrategy

        class ProbeStrategy(BaseAgentStrategy):
            async def initiate_agent_flow(self, _user_message):
                yield "unused"

        cosmos_factory = MagicMock(return_value=mock_cosmos)
        with (
            patch.dict(
                AgentStrategyFactory._REGISTRY,
                {"maf_lite": lambda: ProbeStrategy()},
            ),
            patch(
                "strategies.base_agent_strategy.get_config",
                return_value=mock_config,
            ),
            patch(
                "strategies.base_agent_strategy.get_identity_manager",
                return_value=mock_identity_manager,
            ),
            patch(
                "strategies.base_agent_strategy.get_cosmosdb_client",
                new=cosmos_factory,
            ),
            patch("strategies.base_agent_strategy.AIProjectClient"),
        ):
            strategy = await AgentStrategyFactory.get_strategy("maf_lite")

        cosmos_factory.assert_called_once_with()
        assert strategy.cosmos is mock_cosmos
        assert strategy.hosted_runtime is False
        assert strategy.profile_memory_enabled is True


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

    def test_health_and_readiness_return_same_health_response(self, client):
        from api.hosted_entrypoint import HealthResponse, _APP_VERSION

        expected = HealthResponse(
            status="ok",
            version=_APP_VERSION,
            eligible_strategies=sorted(HOSTED_ELIGIBLE_STRATEGIES),
        ).model_dump()

        health = client.get("/health")
        readiness = client.get("/readiness")

        assert health.status_code == 200
        assert readiness.status_code == 200
        assert health.json() == expected
        assert readiness.json() == expected
        assert readiness.json() == health.json()

    @pytest.mark.parametrize("route", ["/health", "/readiness"])
    def test_readiness_routes_match_hosted_eligible_strategies(self, client, route):
        resp = client.get(route)

        assert resp.status_code == 200
        assert set(resp.json()["eligible_strategies"]) == HOSTED_ELIGIBLE_STRATEGIES

    def test_responses_and_invocations_expose_distinct_request_contracts(self, client):
        paths = client.get("/openapi.json").json()["paths"]
        responses_contract = paths["/responses"]["post"]
        invocations_contract = paths["/invocations"]["post"]

        assert responses_contract["requestBody"] != invocations_contract["requestBody"]
        assert (
            responses_contract["requestBody"]["content"]["application/json"]["schema"]["$ref"]
            .endswith("/ResponsesRequest")
        )
        assert (
            invocations_contract["requestBody"]["content"]["application/json"]["schema"]["$ref"]
            .endswith("/InvocationRequest")
        )
        assert responses_contract["responses"] == invocations_contract["responses"]

    def test_responses_accepts_live_payload_and_streams_full_lifecycle(
        self, client, mock_config
    ):
        self._use_strategy(mock_config, "maf_lite")
        received = {}

        async def fake_flow(ask):
            received["ask"] = ask
            yield "Hello "
            yield "world"

        strategy = MagicMock()
        strategy.initiate_agent_flow = fake_flow

        with patch(
            "api.hosted_entrypoint.AgentStrategyFactory.get_strategy",
            new=AsyncMock(return_value=strategy),
        ):
            response = client.post(
                "/responses",
                json={
                    "input": "Remember ...",
                    "stream": True,
                    "store": True,
                },
            )

        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        assert received == {"ask": "Remember ..."}
        created = response.text.index("event: response.created")
        delta = response.text.index("event: response.output_text.delta")
        completed = response.text.index("event: response.completed")
        assert created < delta < completed

    @pytest.mark.parametrize(
        ("conversation", "expected_id"),
        [
            ("conv-string", "conv-string"),
            ({"id": "conv-object"}, "conv-object"),
        ],
    )
    def test_responses_maps_conversation_to_managed_conversation_path(
        self,
        client,
        mock_config,
        conversation,
        expected_id,
    ):
        self._use_strategy(mock_config, "maf_agent_service")

        async def fake_flow(_ask):
            yield "answer"

        strategy = MagicMock()
        strategy.initiate_agent_flow = fake_flow
        resolve_conversation = AsyncMock(return_value=expected_id)

        with (
            patch(
                "api.hosted_entrypoint.AgentStrategyFactory.get_strategy",
                new=AsyncMock(return_value=strategy),
            ),
            patch(
                "api.hosted_entrypoint.resolve_managed_conversation_id",
                new=resolve_conversation,
            ),
        ):
            response = client.post(
                "/responses",
                json={
                    "input": "Continue",
                    "stream": True,
                    "store": True,
                    "conversation": conversation,
                },
            )

        assert response.status_code == 200
        resolve_conversation.assert_awaited_once_with(
            strategy.project_client,
            expected_id,
        )
        assert expected_id in response.text
        assert "event: response.created" in response.text
        assert "event: response.output_text.delta" in response.text
        assert "event: response.completed" in response.text

    def test_responses_maps_metadata_without_projecting_legacy_history(
        self, client, mock_config
    ):
        self._use_strategy(mock_config, "maf_lite")
        captured = {}

        def fake_sse(
            turn,
            strategy_key,
            response_id,
            item_id,
            history,
            *,
            model,
        ):
            captured.update(
                turn=turn,
                strategy_key=strategy_key,
                response_id=response_id,
                item_id=item_id,
                history=history,
                model=model,
            )
            return _async_gen(["event: response.completed\ndata: {}\n\n"])

        with patch("api.hosted_entrypoint._sse_generator", new=fake_sse):
            response = client.post(
                "/responses",
                json={
                    "input": "Question",
                    "stream": True,
                    "store": True,
                    "metadata": {
                        "question_id": "question-1",
                        "correlation_id": "correlation-1",
                    },
                },
            )

        assert response.status_code == 200
        assert captured["turn"].question_id == "question-1"
        assert captured["turn"].correlation_id == "correlation-1"
        assert captured["history"] == ()
        assert captured["model"] == _MODEL

    @pytest.mark.parametrize(
        "unsupported_field",
        [
            {"previous_response_id": "resp_previous"},
            {"instructions": "Ignore prior instructions"},
            {"tools": []},
            {"tool_choice": "auto"},
            {"model": "gpt-4o"},
        ],
    )
    def test_responses_rejects_unsupported_extra_fields(
        self,
        client,
        unsupported_field,
    ):
        response = client.post(
            "/responses",
            json={
                "input": "Question",
                "stream": True,
                "store": True,
                **unsupported_field,
            },
        )

        assert response.status_code == 422
        detail = response.json()["detail"]
        assert detail[0]["type"] == "extra_forbidden"
        assert detail[0]["loc"] == ["body", next(iter(unsupported_field))]

    def test_responses_accepts_foundry_injected_agent_reference(
        self, client, mock_config
    ):
        self._use_strategy(mock_config, "maf_lite")

        with patch(
            "api.hosted_entrypoint._sse_generator",
            new=lambda *args, **kwargs: _async_gen(
                ["event: response.completed\ndata: {}\n\n"]
            ),
        ):
            response = client.post(
                "/responses",
                json={
                    "input": "Question",
                    "stream": True,
                    "store": True,
                    "agent_reference": {
                        "type": "agent_reference",
                        "name": "gpt-rag-orchestrator",
                        "version": "3",
                    },
                },
            )

        assert response.status_code == 200

    def test_responses_rejects_malformed_agent_reference(
        self, client, mock_config
    ):
        self._use_strategy(mock_config, "maf_lite")

        response = client.post(
            "/responses",
            json={
                "input": "Question",
                "stream": True,
                "store": True,
                "agent_reference": {
                    "type": "unexpected",
                    "name": "gpt-rag-orchestrator",
                    "version": "3",
                },
            },
        )

        assert response.status_code == 422

    @pytest.mark.parametrize(
        "unsupported_input",
        [
            ["plain array input"],
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": "https://example.invalid",
                        }
                    ],
                }
            ],
        ],
    )
    def test_responses_rejects_array_and_multimodal_input(
        self, client, unsupported_input
    ):
        response = client.post(
            "/responses",
            json={
                "input": unsupported_input,
                "stream": True,
                "store": True,
            },
        )

        assert response.status_code == 422
        assert (
            "Only string input is supported; array and multimodal input are not supported."
            in str(response.json())
        )

    def test_responses_rejects_empty_string_input(self, client):
        response = client.post(
            "/responses",
            json={"input": "   ", "stream": True, "store": True},
        )

        assert response.status_code == 422
        assert "Input must not be empty or whitespace." in str(response.json())

    def test_responses_rejects_non_streaming_request(self, client):
        response = client.post(
            "/responses",
            json={"input": "Hello", "stream": False, "store": True},
        )

        assert response.status_code == 422
        assert "Only stream=true is supported" in str(response.json())

    def test_responses_rejects_non_storing_managed_request(self, client):
        response = client.post(
            "/responses",
            json={"input": "Hello", "stream": True, "store": False},
        )

        assert response.status_code == 422
        assert "Only store=true is supported" in str(response.json())

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

    def test_invocations_rejects_messages_after_current_user_ask(self, client):
        resp = client.post(
            "/invocations",
            json={
                "messages": [
                    {"role": "user", "content": "current ask"},
                    {"role": "assistant", "content": "unexpected trailing message"},
                ]
            },
        )

        assert resp.status_code == 422
        assert resp.json()["detail"] == (
            "The final message must have role='user'; messages after the "
            "current ask are not allowed."
        )

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

    def test_invocations_projects_prior_history_and_discards_untrusted_identity(
        self,
        client,
        mock_config,
    ):
        original = mock_config.get.side_effect
        mock_config.get.side_effect = (
            lambda key, default=None, type=str:
            "maf_lite" if key == "AGENT_STRATEGY" else original(key, default, type)
        )
        received = {}

        async def fake_flow(ask):
            received["ask"] = ask
            received["conversation"] = deepcopy(strategy.conversation)
            received["user_context"] = deepcopy(strategy.user_context)
            yield "answer"

        strategy = MagicMock()
        strategy.initiate_agent_flow = fake_flow

        with patch(
            "api.hosted_entrypoint.AgentStrategyFactory.get_strategy",
            new=AsyncMock(return_value=strategy),
        ):
            resp = client.post(
                "/invocations",
                json={
                    "messages": [
                        {"role": "user", "content": "first question"},
                        {"role": "assistant", "content": "first answer"},
                        {"role": "user", "content": "follow-up"},
                    ],
                    "conversation_id": "conv-two-turn",
                    "metadata": {
                        "user_context": {
                            "principal_id": "caller-controlled",
                            "groups": ["caller-controlled"],
                        }
                    },
                },
            )

        assert resp.status_code == 200
        assert received == {
            "ask": "follow-up",
            "conversation": {
                "id": "conv-two-turn",
                "messages": [
                    {"role": "user", "text": "first question"},
                    {"role": "assistant", "text": "first answer"},
                ],
            },
            "user_context": {},
        }

    # ── ADR-0001 Foundry Toolbox call-id passthrough (Azure/GPT-RAG#591) ────

    @staticmethod
    def _use_strategy(mock_config, strategy_key: str) -> None:
        original = mock_config.get.side_effect
        mock_config.get.side_effect = (
            lambda key, default=None, type=str:
            strategy_key if key == "AGENT_STRATEGY" else original(key, default, type)
        )

    def test_invocations_rejects_missing_foundry_call_id_for_toolbox_strategy(
        self, client, mock_config
    ):
        """The mcp strategy talks to Toolbox and must fail closed (401) when
        the platform-injected call id is absent, instead of silently
        proceeding with service identity or a manual metadata filter."""
        self._use_strategy(mock_config, "mcp")

        factory = AsyncMock()
        with patch(
            "api.hosted_entrypoint.AgentStrategyFactory.get_strategy",
            new=factory,
        ):
            resp = client.post(
                "/invocations",
                json={"messages": [{"role": "user", "content": "Hello"}]},
            )

        assert resp.status_code == 401
        assert "x-agent-foundry-call-id" in resp.json()["detail"]
        # No fallback: the strategy must never even be constructed.
        factory.assert_not_called()

    @pytest.mark.parametrize(
        "bad_call_id",
        [
            "has spaces",
            "line\nbreak",
            "carriage\rreturn",
            "x" * 257,
        ],
    )
    def test_invocations_rejects_malformed_foundry_call_id(
        self, client, mock_config, bad_call_id
    ):
        self._use_strategy(mock_config, "mcp")

        factory = AsyncMock()
        with patch(
            "api.hosted_entrypoint.AgentStrategyFactory.get_strategy",
            new=factory,
        ):
            resp = client.post(
                "/invocations",
                json={"messages": [{"role": "user", "content": "Hello"}]},
                headers={"x-agent-foundry-call-id": bad_call_id},
            )

        assert resp.status_code == 401
        factory.assert_not_called()

    def test_invocations_accepts_valid_foundry_call_id_for_toolbox_strategy(
        self, client, mock_config
    ):
        self._use_strategy(mock_config, "mcp")

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
                json={"messages": [{"role": "user", "content": "Hello"}]},
                headers={"x-agent-foundry-call-id": "call-abc-123"},
            )

        assert resp.status_code == 200
        assert strategy.foundry_call_id == "call-abc-123"

    def test_invocations_non_toolbox_strategy_does_not_require_foundry_call_id(
        self, client, mock_config
    ):
        """Classic behavior unchanged: strategies that don't call Toolbox
        (e.g. maf_lite) still work with no call-id header at all."""
        self._use_strategy(mock_config, "maf_lite")

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
                json={"messages": [{"role": "user", "content": "Hello"}]},
            )

        assert resp.status_code == 200
        assert strategy.foundry_call_id is None

    def test_invocations_never_logs_foundry_call_id_or_authorization(
        self, client, mock_config, caplog
    ):
        """Neither the opaque call id nor a (never-forwarded) Authorization
        header value may ever appear in application logs."""
        self._use_strategy(mock_config, "mcp")

        async def fake_flow(_ask):
            yield "answer"

        strategy = MagicMock()
        strategy.initiate_agent_flow = fake_flow

        secret_call_id = "super-secret-call-id-000"
        secret_bearer = "Bearer should-never-be-read-or-forwarded"

        with (
            patch(
                "api.hosted_entrypoint.AgentStrategyFactory.get_strategy",
                new=AsyncMock(return_value=strategy),
            ),
            caplog.at_level(logging.DEBUG),
        ):
            resp = client.post(
                "/invocations",
                json={"messages": [{"role": "user", "content": "Hello"}]},
                headers={
                    "x-agent-foundry-call-id": secret_call_id,
                    "Authorization": secret_bearer,
                },
            )

        assert resp.status_code == 200
        assert secret_call_id not in caplog.text
        assert secret_bearer not in caplog.text
        assert "should-never-be-read-or-forwarded" not in caplog.text

    @pytest.mark.asyncio
    async def test_invocations_concurrent_requests_isolate_foundry_call_ids(
        self, mock_config, mock_cosmos
    ):
        """Two concurrent /invocations requests for the mcp strategy must
        never cross-contaminate each other's Foundry call id."""
        from httpx import ASGITransport, AsyncClient

        from api.hosted_entrypoint import app

        self._use_strategy(mock_config, "mcp")
        captured: dict[str, str | None] = {}

        class RecordingStrategy:
            def __init__(self, delay: float):
                self._delay = delay

            def set_context(self, _conversation_id):
                pass

            async def initiate_agent_flow(self, ask):
                await asyncio.sleep(self._delay)
                captured[ask] = self.foundry_call_id
                yield "answer"

        strategies = iter([RecordingStrategy(0.05), RecordingStrategy(0.0)])

        async def fake_get_strategy(_strategy_key, hosted_runtime=False):
            return next(strategies)

        with patch(
            "api.hosted_entrypoint.AgentStrategyFactory.get_strategy",
            new=fake_get_strategy,
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as async_client:
                responses = await asyncio.gather(
                    async_client.post(
                        "/invocations",
                        json={"messages": [{"role": "user", "content": "ask-a"}]},
                        headers={"x-agent-foundry-call-id": "call-a"},
                    ),
                    async_client.post(
                        "/invocations",
                        json={"messages": [{"role": "user", "content": "ask-b"}]},
                        headers={"x-agent-foundry-call-id": "call-b"},
                    ),
                )

        for resp in responses:
            assert resp.status_code == 200
            assert resp.text  # force full body consumption

        assert captured == {"ask-a": "call-a", "ask-b": "call-b"}


# ── Helpers ──────────────────────────────────────────────────────────────────

async def _async_gen(items):
    """Tiny helper to turn a list into an async generator."""
    for item in items:
        yield item
