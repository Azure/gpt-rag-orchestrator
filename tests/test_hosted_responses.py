"""Tests for the hosted Responses adapters and managed Conversations entrypoint.

Covers:
- Responses API SSE serialization of every typed turn event
- Terminal closing frames after the text stream
- Hosted strategy guard (explicit failure for unsupported strategies)
- Hosted stream execution without Cosmos dependency
- Hosted ASGI endpoints (health, readiness, responses, and invocations)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from copy import deepcopy
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


def _parse_sse_body(body: str) -> list[dict[str, Any]]:
    return _parse_frames(
        [f"{frame}\n\n" for frame in body.strip().split("\n\n") if frame]
    )


def _parse_sse_stream(body: str) -> list[dict[str, Any]]:
    """Parse a complete SSE response body into typed frame dictionaries."""
    return _parse_sse_body(body)


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


@pytest.fixture(scope="module", autouse=True)
def _disable_test_otel_exporters():
    previous = os.environ.get("OTEL_SDK_DISABLED")
    os.environ["OTEL_SDK_DISABLED"] = "true"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("OTEL_SDK_DISABLED", None)
        else:
            os.environ["OTEL_SDK_DISABLED"] = previous


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
    @pytest.mark.parametrize(
        "citation",
        [
            TurnCitation(
                "src-1",
                title="Policy Doc",
                url="https://example.com/policy",
                snippet="Relevant excerpt",
            ),
            TurnCitation("src-2"),
        ],
    )
    def test_internal_citation_without_text_offsets_is_not_emitted(self, citation):
        assert _serialize(TurnCitationEvent(citation=citation)) == []


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
        assert frames[0]["data"]["param"] is None
        assert "retryable" not in frames[0]["data"]

    def test_internal_retryable_flag_is_not_added_to_standard_error(self):
        frames = _serialize(TurnErrorEvent(retryable=True))

        assert "retryable" not in frames[0]["data"]


class TestResponsesAdapterCancelledEvent:
    def test_emits_standard_error_event(self):
        frames = _serialize(TurnCancelledEvent(reason="cancelled"))

        assert len(frames) == 1
        assert frames[0]["event"] == "error"
        assert frames[0]["data"]["code"] == "cancelled"
        assert frames[0]["data"]["message"] == "cancelled"
        assert frames[0]["data"]["param"] is None


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

    def test_response_completed_echoes_canonical_metadata(self):
        frames = _parse_frames(
            responses_terminal_events(
                response_id=_RESP_ID,
                item_id=_ITEM_ID,
                conversation_id=_CONVERSATION_ID,
                full_text="",
                response_metadata={"question_id": "question-1"},
            )
        )

        assert frames[-1]["data"]["response"]["metadata"] == {
            "question_id": "question-1"
        }


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

        with patch(
            "api.hosted_entrypoint.AgentStrategyFactory.get_strategy",
            new=factory,
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

        assert received_conversations == [
            {"id": "conv-1", "messages": []},
            {"id": "conv-1", "messages": prior_history},
        ]
        assert factory.await_args_list[0].kwargs == {"hosted_runtime": True}
        assert factory.await_args_list[1].kwargs == {"hosted_runtime": True}

    @pytest.mark.parametrize(
        "strategy_key",
        sorted(HOSTED_ELIGIBLE_STRATEGIES),
    )
    @pytest.mark.asyncio
    async def test_hosted_stream_performs_zero_conversations_data_plane_operations(
        self,
        strategy_key: str,
    ):
        """Security regression: the hosted runtime must never construct a
        Conversations client or call create/read/append/delete on a managed
        Conversation, regardless of whether the caller supplies a
        conversation id, and regardless of strategy. The turn's conversation
        id is only ever an opaque echoed/generated label -- it can never
        select or recreate access to a service-managed identity."""
        from api.hosted_entrypoint import _hosted_stream

        class RecordingStrategy:
            def __init__(self):
                self.project_client = MagicMock()

            def set_context(self, _conversation_id):
                pass

            async def initiate_agent_flow(self, _ask):
                yield "answer"

        strategy = RecordingStrategy()
        with patch(
            "api.hosted_entrypoint.AgentStrategyFactory.get_strategy",
            new=AsyncMock(return_value=strategy),
        ):
            turn = TurnRequest(
                ask="hello",
                conversation_id="caller-supplied-conv-id",
                foundry_call_id="call-1",
            )
            [event async for event in _hosted_stream(turn, strategy_key)]

        # No Conversations client accessor was ever touched.
        strategy.project_client.get_openai_client.assert_not_called()
        strategy.project_client.assert_not_called()
        # No leftover thread/backend tagging: the strategy conversation dict
        # never carries a service-managed thread identity.
        assert "thread_id" not in strategy.conversation
        assert "agent_backend" not in strategy.conversation

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


class TestCancelAwareEvents:
    @pytest.mark.asyncio
    async def test_shutdown_stops_and_closes_blocked_strategy_iterator(self):
        from api.hosted_entrypoint import _cancel_aware_events

        started = asyncio.Event()
        closed = asyncio.Event()
        release = asyncio.Event()

        async def source():
            try:
                started.set()
                await release.wait()
                yield TurnTextEvent("late")
            finally:
                closed.set()

        cancellation = asyncio.Event()
        shutdown = asyncio.Event()
        consumer = asyncio.create_task(
            anext(_cancel_aware_events(source(), cancellation, shutdown))
        )
        await started.wait()

        shutdown.set()

        with pytest.raises(StopAsyncIteration):
            await asyncio.wait_for(consumer, timeout=1)
        await asyncio.wait_for(closed.wait(), timeout=1)

    @pytest.mark.asyncio
    async def test_outer_cancellation_closes_blocked_strategy_iterator(self):
        from api.hosted_entrypoint import _cancel_aware_events

        started = asyncio.Event()
        closed = asyncio.Event()
        release = asyncio.Event()

        async def source():
            try:
                started.set()
                await release.wait()
                yield TurnTextEvent("late")
            finally:
                closed.set()

        consumer = asyncio.create_task(
            anext(
                _cancel_aware_events(
                    source(),
                    asyncio.Event(),
                    asyncio.Event(),
                )
            )
        )
        await started.wait()

        consumer.cancel()

        with pytest.raises(asyncio.CancelledError):
            await consumer
        await asyncio.wait_for(closed.wait(), timeout=1)


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
        assert not any(
            frame["event"] == "response.output_text.annotation.added"
            for frame in parsed
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
            yield TurnCitationEvent(
                TurnCitation(
                    "src-internal",
                    title="Internal source",
                    url="https://example.invalid/source",
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
    """Integration-style tests for the hosted multi-protocol app."""

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
        from azure.ai.agentserver.responses import InMemoryResponseProvider
        from fastapi.testclient import TestClient
        from api.hosted_entrypoint import create_app

        app = create_app(
            store=InMemoryResponseProvider(),
            configure_observability=None,
        )
        with TestClient(app, raise_server_exceptions=False) as client:
            yield client

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

    def test_responses_and_invocations_publish_distinct_route_contracts(self, client):
        routes = {
            (route.path, method)
            for route in client.app.routes
            for method in (route.methods or set())
        }

        assert ("/responses", "POST") in routes
        assert ("/responses/{response_id}", "GET") in routes
        assert ("/responses/{response_id}", "DELETE") in routes
        assert ("/responses/{response_id}/cancel", "POST") in routes
        assert ("/responses/{response_id}/input_items", "GET") in routes
        assert ("/invocations", "POST") in routes

    def test_responses_rejects_invocations_payload(self, client):
        resp = client.post(
            "/responses",
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )

        assert resp.status_code == 422
        assert resp.json()["detail"] == (
            "The Responses request requires an input field."
        )

    def test_responses_rejects_malformed_json(self, client):
        response = client.post(
            "/responses",
            content="{",
            headers={"content-type": "application/json"},
        )

        assert response.status_code == 400
        assert response.json()["detail"] == (
            "The Responses request body must be valid JSON."
        )

    def test_responses_accepts_live_standard_payload(self, client, mock_config):
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
                "/responses",
                json={"input": "Hello", "stream": True, "store": True},
            )

        assert resp.status_code == 200
        events = _parse_sse_body(resp.text)
        event_types = [event["event"] for event in events]
        sequence_numbers = [event["data"]["sequence_number"] for event in events]
        response_ids = {
            event["data"]["response"]["id"]
            for event in events
            if isinstance(event["data"].get("response"), dict)
        }

        assert event_types[:2] == ["response.created", "response.in_progress"]
        assert "response.output_text.delta" in event_types
        assert event_types[-1] == "response.completed"
        assert sequence_numbers == list(range(len(events)))
        assert len(response_ids) == 1

    def test_responses_accepts_live_payload_and_streams_full_lifecycle(
        self, client, mock_config
    ):
        self._use_strategy(mock_config, "maf_lite")
        received = {}

        async def fake_flow(ask):
            received["ask"] = ask
            yield "Hello "
            yield TurnCitationEvent(
                TurnCitation(
                    "src-internal",
                    title="Internal source",
                    url="https://example.invalid/source",
                )
            )
            yield TurnToolActivityEvent(
                TurnToolActivity(
                    "search_kb",
                    TurnToolStatus.STARTED,
                    call_id="call-internal",
                )
            )
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
        frames = _parse_sse_stream(response.text)
        assert [frame["data"]["sequence_number"] for frame in frames] == list(
            range(len(frames))
        )
        assert not any(
            frame["event"] in {
                "response.output_text.annotation.added",
                "response.function_call_arguments.delta",
                "response.function_call_arguments.done",
            }
            for frame in frames
        )
        for frame in frames:
            _RESPONSE_EVENT_ADAPTER.validate_python(frame["data"])

    def test_responses_rejects_conversation_field(self, client):
        """Security regression: ``conversation`` is a server-side state
        selector that would let a caller-selected id drive Foundry
        Conversations create/read on the hosted container. It must be
        rejected outright, not resolved."""
        response = client.post(
            "/responses",
            json={
                "input": "Continue",
                "stream": True,
                "store": True,
                "conversation": "conv-explicit",
            },
        )

        assert response.status_code == 422
        assert response.json()["detail"] == (
            "Unsupported Responses request fields: conversation."
        )

    def test_responses_rejects_previous_response_id_field(self, client):
        """Security regression: ``previous_response_id`` is a server-side
        state selector that could recreate history access. It must be
        rejected outright; the caller must send the complete ordered input
        instead."""
        response = client.post(
            "/responses",
            json={
                "input": "Continue",
                "stream": True,
                "store": True,
                "previous_response_id": "resp_previous",
            },
        )

        assert response.status_code == 422
        assert response.json()["detail"] == (
            "Unsupported Responses request fields: previous_response_id."
        )

    def test_responses_maps_metadata_without_projecting_legacy_history(
        self, client, mock_config
    ):
        self._use_strategy(mock_config, "maf_lite")
        captured = {}

        async def fake_stream(turn, strategy_key, history):
            captured.update(
                turn=turn,
                strategy_key=strategy_key,
                history=history,
            )
            yield TurnConversationEvent("conv-metadata")
            yield TurnTextEvent("answer")

        with patch("api.hosted_entrypoint._hosted_stream", new=fake_stream):
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
        assert captured["history"] == []
        completed = _parse_sse_body(response.text)[-1]["data"]["response"]
        assert completed["metadata"] == {
            "question_id": "question-1",
            "correlation_id": "correlation-1",
        }

    def test_responses_rejects_non_string_metadata_values(self, client):
        response = client.post(
            "/responses",
            json={
                "input": "Question",
                "stream": True,
                "store": True,
                "metadata": {"nested": {"unsupported": True}},
            },
        )

        assert response.status_code == 422
        assert response.json()["detail"] == (
            "Responses metadata must contain only string keys and values."
        )

    @pytest.mark.parametrize(
        "unsupported_field",
        [
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
        field = next(iter(unsupported_field))
        assert response.json()["detail"] == (
            f"Unsupported Responses request fields: {field}."
        )

    def test_responses_accepts_foundry_injected_agent_reference(
        self, client, mock_config
    ):
        self._use_strategy(mock_config, "maf_lite")

        async def fake_flow(_ask):
            yield "answer"

        strategy = MagicMock()
        strategy.initiate_agent_flow = fake_flow
        with patch(
            "api.hosted_entrypoint.AgentStrategyFactory.get_strategy",
            new=AsyncMock(return_value=strategy),
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

        assert response.status_code == 400

    @pytest.mark.parametrize(
        ("unsupported_input", "expected_detail"),
        [
            (
                ["plain array input"],
                "Each Responses input array item must be a role/content message object.",
            ),
            (
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
                "This hosted adapter supports text-only Responses input content.",
            ),
        ],
    )
    def test_responses_rejects_array_and_multimodal_input(
        self, client, unsupported_input, expected_detail
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
        assert response.json()["detail"] == expected_detail

    def test_responses_rejects_empty_string_input(self, client):
        response = client.post(
            "/responses",
            json={"input": "   ", "stream": True, "store": True},
        )

        assert response.status_code == 422
        assert response.json()["detail"] == "The Responses input must not be empty."

    @pytest.mark.parametrize(
        ("setting", "expected"),
        [
            (None, False),
            ("false", False),
            ("0", False),
            ("unexpected", False),
            ("true", True),
            (" 1 ", True),
        ],
    )
    def test_host_observability_disables_content_capture_by_default(
        self,
        monkeypatch,
        setting,
        expected,
    ):
        from api.hosted_entrypoint import _configure_host_observability

        if setting is None:
            monkeypatch.delenv(
                "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT",
                raising=False,
            )
        else:
            monkeypatch.setenv(
                "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT",
                setting,
            )

        configure = MagicMock()
        with patch(
            "api.hosted_entrypoint.configure_agentserver_observability",
            new=configure,
        ):
            _configure_host_observability(
                connection_string="InstrumentationKey=test",
                log_level="INFO",
                enable_sensitive_data=True,
            )

        configure.assert_called_once_with(
            connection_string="InstrumentationKey=test",
            log_level="INFO",
            enable_sensitive_data=expected,
        )

    def test_responses_accepts_non_streaming_request(self, client, mock_config):
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
                "/responses",
                json={"input": "Hello", "stream": False, "store": True},
            )

        assert resp.status_code == 200
        assert resp.json()["status"] == "completed"
        assert resp.json()["output"][0]["content"][0]["text"] == "answer"

    def test_responses_store_false_is_not_retrievable(
        self,
        client,
        mock_config,
    ):
        self._use_strategy(mock_config, "maf_lite")

        async def fake_flow(_ask):
            yield "ephemeral answer"

        strategy = MagicMock()
        strategy.initiate_agent_flow = fake_flow

        with patch(
            "api.hosted_entrypoint.AgentStrategyFactory.get_strategy",
            new=AsyncMock(return_value=strategy),
        ):
            created = client.post(
                "/responses",
                json={"input": "Ephemeral question", "store": False},
            )

        assert created.status_code == 200
        assert client.get(f"/responses/{created.json()['id']}").status_code == 404

    @pytest.mark.parametrize(
        "store_field",
        [
            pytest.param({"store": True}, id="explicit-store-true"),
            pytest.param({}, id="omitted-store-defaults-true"),
        ],
    )
    def test_responses_ignores_caller_store_and_fails_closed(
        self,
        client,
        mock_config,
        store_field,
    ):
        """ADR-0004: the hosted container has zero managed-Conversations RBAC.

        A caller asking for (or defaulting to, since the Responses contract
        treats an omitted ``store`` as ``true``) managed persistence must never
        reach the SDK's auto-activated, network-bound Foundry storage
        provider. The container overrides ``store`` to ``False`` for every
        create call, so the response is created successfully but is never
        retrievable, listable, or deletable afterward — regardless of what
        the caller sent or omitted. This is the fail-closed override that
        replaces caller-controlled persistence (formerly proven by a
        ``store: True`` round trip through GET/DELETE, which this test
        supersedes: that round trip must no longer succeed).
        """
        self._use_strategy(mock_config, "maf_lite")

        async def fake_flow(_ask):
            yield "stored answer"

        strategy = MagicMock()
        strategy.initiate_agent_flow = fake_flow

        with patch(
            "api.hosted_entrypoint.AgentStrategyFactory.get_strategy",
            new=AsyncMock(return_value=strategy),
        ):
            created = client.post(
                "/responses",
                json={"input": "Stored question", **store_field},
            )

        assert created.status_code == 200
        response_id = created.json()["id"]

        assert client.get(f"/responses/{response_id}").status_code == 404
        assert (
            client.get(f"/responses/{response_id}/input_items").status_code == 404
        )
        assert client.delete(f"/responses/{response_id}").status_code == 404

    def test_responses_stream_ignores_caller_store_and_fails_closed(
        self,
        client,
        mock_config,
    ):
        """The override applies uniformly to streaming create calls too."""
        self._use_strategy(mock_config, "maf_lite")

        async def fake_flow(_ask):
            yield "streamed answer"

        strategy = MagicMock()
        strategy.initiate_agent_flow = fake_flow

        with patch(
            "api.hosted_entrypoint.AgentStrategyFactory.get_strategy",
            new=AsyncMock(return_value=strategy),
        ):
            created = client.post(
                "/responses",
                json={"input": "Stored question", "stream": True, "store": True},
            )

        assert created.status_code == 200
        events = _parse_sse_body(created.text)
        response_ids = {
            event["data"]["response"]["id"]
            for event in events
            if isinstance(event["data"].get("response"), dict)
        }
        assert len(response_ids) == 1
        response_id = next(iter(response_ids))

        assert client.get(f"/responses/{response_id}").status_code == 404

    def test_responses_honors_platform_response_id_header(
        self,
        client,
        mock_config,
    ):
        from azure.ai.agentserver.responses._id_generator import IdGenerator

        self._use_strategy(mock_config, "maf_lite")

        async def fake_flow(_ask):
            yield "answer"

        strategy = MagicMock()
        strategy.initiate_agent_flow = fake_flow
        response_id = IdGenerator.new_response_id()

        with patch(
            "api.hosted_entrypoint.AgentStrategyFactory.get_strategy",
            new=AsyncMock(return_value=strategy),
        ):
            response = client.post(
                "/responses",
                headers={"x-agent-response-id": response_id},
                json={"input": "Question", "stream": True, "store": True},
            )

        events = _parse_sse_body(response.text)
        assert response.status_code == 200
        assert events[0]["data"]["response"]["id"] == response_id
        assert events[-1]["data"]["response"]["id"] == response_id

    def test_responses_accepts_ordered_message_array_input(
        self, client, mock_config
    ):
        """Canonical hosted path: the caller supplies the complete, bounded,
        ordered history directly as the ``input`` array. Prior turns are
        replayed as strategy history and the final ``user`` item becomes the
        current ask -- with no ``conversation``/``previous_response_id``
        state dependency at all."""
        self._use_strategy(mock_config, "maf_lite")
        captured = {}

        async def fake_stream(turn, strategy_key, history):
            captured["turn"] = turn
            captured["strategy_key"] = strategy_key
            captured["history"] = history
            yield TurnConversationEvent("conv-list-input")
            yield TurnTextEvent("second answer")

        with patch("api.hosted_entrypoint._hosted_stream", new=fake_stream):
            response = client.post(
                "/responses",
                json={
                    "input": [
                        {"role": "user", "content": "first question"},
                        {"role": "assistant", "content": "first answer"},
                        {"role": "user", "content": "follow-up question"},
                    ],
                    "store": True,
                },
            )

        assert response.status_code == 200
        assert captured["turn"].ask == "follow-up question"
        assert captured["turn"].conversation_id is None
        assert captured["history"] == [
            {"role": "user", "text": "first question"},
            {"role": "assistant", "text": "first answer"},
        ]

    def test_responses_rejects_message_array_input_not_ending_in_user(
        self, client
    ):
        response = client.post(
            "/responses",
            json={
                "input": [
                    {"role": "user", "content": "first question"},
                    {"role": "assistant", "content": "first answer"},
                ],
                "store": True,
            },
        )

        assert response.status_code == 422
        assert response.json()["detail"] == (
            "The final Responses input item must have role='user'."
        )

    def test_responses_rejects_empty_message_array_input(self, client):
        response = client.post(
            "/responses",
            json={"input": [], "store": True},
        )

        assert response.status_code == 422
        assert response.json()["detail"] == (
            "The Responses input array must not be empty."
        )

    def test_responses_rejects_message_array_input_with_empty_final_ask(
        self, client
    ):
        """The final ``user`` item must resolve to non-empty text; otherwise
        an intermediate item could be silently misinterpreted as the current
        ask instead of failing explicitly."""
        response = client.post(
            "/responses",
            json={
                "input": [
                    {"role": "user", "content": "first question"},
                    {"role": "assistant", "content": "first answer"},
                    {"role": "user", "content": [{"type": "input_text", "text": "   "}]},
                ],
                "store": True,
            },
        )

        assert response.status_code == 422
        assert response.json()["detail"] == "The Responses input must not be empty."

    def test_responses_rejects_background_mode(
        self,
        client,
        mock_config,
    ):
        """ADR-0004: the hosted runtime forces store=False on every request,
        and the pinned SDK requires store=true whenever background=true is
        requested (a queued/polled response is meaningless if it can never
        be retrieved). background=true must therefore be rejected outright
        with a self-documenting error instead of silently queuing a response
        that can never be observed again."""
        self._use_strategy(mock_config, "maf_lite")

        async def fake_flow(_ask):
            while True:
                await asyncio.sleep(0.01)
                yield "working"

        strategy = MagicMock()
        strategy.initiate_agent_flow = fake_flow

        with patch(
            "api.hosted_entrypoint.AgentStrategyFactory.get_strategy",
            new=AsyncMock(return_value=strategy),
        ):
            created = client.post(
                "/responses",
                json={
                    "input": "Long-running question",
                    "background": True,
                    "store": True,
                },
            )

        assert created.status_code == 422
        assert "background" in created.json()["detail"]

    def test_responses_rejects_missing_foundry_call_id_for_toolbox_strategy(
        self,
        client,
        mock_config,
    ):
        self._use_strategy(mock_config, "mcp")

        factory = AsyncMock()
        with patch(
            "api.hosted_entrypoint.AgentStrategyFactory.get_strategy",
            new=factory,
        ):
            resp = client.post(
                "/responses",
                json={"input": "Hello", "stream": True, "store": True},
            )

        assert resp.status_code == 401
        assert "x-agent-foundry-call-id" in resp.json()["detail"]
        factory.assert_not_called()

    @pytest.mark.parametrize(
        ("method", "path"),
        [
            ("get", "/responses/{response_id}"),
            ("delete", "/responses/{response_id}"),
            ("post", "/responses/{response_id}/cancel"),
            ("get", "/responses/{response_id}/input_items"),
        ],
    )
    def test_responses_storage_routes_require_toolbox_call_id(
        self,
        client,
        mock_config,
        method,
        path,
    ):
        from azure.ai.agentserver.responses._id_generator import IdGenerator

        self._use_strategy(mock_config, "mcp")
        response_id = IdGenerator.new_response_id()

        response = getattr(client, method)(path.format(response_id=response_id))

        assert response.status_code == 401
        assert "x-agent-foundry-call-id" in response.json()["detail"]

    def test_responses_propagates_platform_call_id_for_toolbox_strategy(
        self,
        client,
        mock_config,
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
            response = client.post(
                "/responses",
                headers={"x-agent-foundry-call-id": "call-toolbox-123"},
                json={"input": "Question", "stream": True, "store": True},
            )

        assert response.status_code == 200
        assert "response.completed" in response.text
        assert strategy.foundry_call_id == "call-toolbox-123"

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

    def test_invocations_rejects_malformed_json(self, client):
        resp = client.post(
            "/invocations",
            content="{",
            headers={"content-type": "application/json"},
        )

        assert resp.status_code == 400
        assert resp.json()["detail"] == (
            "The Invocations request body must be valid JSON."
        )

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
        frames = _parse_sse_stream(resp.text)
        for frame in frames:
            _RESPONSE_EVENT_ADAPTER.validate_python(frame["data"])
        assert frames[0]["data"]["response"].get("metadata") is None
        assert frames[-1]["data"]["response"].get("metadata") is None

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
            " call-id ",
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
