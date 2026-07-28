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
from copy import deepcopy
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

    @pytest.mark.parametrize("strategy", ["multimodal", "nl2sql", "multiagent", "unknown_strategy", "mcp"])
    def test_unsupported_strategies_raise_value_error(self, strategy: str):
        with pytest.raises(ValueError, match="not supported in the hosted runtime"):
            guard_hosted_strategy(strategy)

    def test_error_message_includes_eligible_strategies(self):
        with pytest.raises(ValueError) as exc_info:
            guard_hosted_strategy("nl2sql")

        msg = str(exc_info.value)
        for eligible in HOSTED_ELIGIBLE_STRATEGIES:
            assert eligible in msg

    def test_mcp_is_not_in_eligible_strategies(self):
        assert "mcp" not in HOSTED_ELIGIBLE_STRATEGIES


# ── build_hosted_conversation SDK-boundary tests ─────────────────────────────

class TestBuildHostedConversation:
    """Tests for the build_hosted_conversation helper."""

    def test_external_conversation_id_sets_thread_id_for_server_thread_strategies(self):
        from strategies.hosted_strategies import build_hosted_conversation

        for strategy_key in ("maf_agent_service", "single_agent_rag"):
            conv = build_hosted_conversation(
                strategy_key,
                "foundry-conv-abc",
                [],
                external_conversation_id=True,
            )
            assert conv["thread_id"] == "foundry-conv-abc", (
                f"{strategy_key} should carry thread_id when external id is provided"
            )

    def test_absent_external_conversation_id_does_not_set_thread_id(self):
        """A synthesised UUID must not be forwarded as thread_id."""
        from strategies.hosted_strategies import build_hosted_conversation

        for strategy_key in ("maf_agent_service", "single_agent_rag"):
            conv = build_hosted_conversation(
                strategy_key,
                "00000000-0000-0000-0000-000000000000",
                [],
                external_conversation_id=False,
            )
            assert "thread_id" not in conv, (
                f"{strategy_key} must not receive a synthesised UUID as thread_id"
            )

    def test_absent_external_id_still_tags_agent_backend_for_server_thread_strategies(self):
        """agent_backend tag must be set so reset_legacy_thread behaves correctly."""
        from strategies.hosted_strategies import build_hosted_conversation
        from strategies.agent_provider_v2 import AGENT_BACKEND_TAG

        for strategy_key in ("maf_agent_service", "single_agent_rag"):
            conv = build_hosted_conversation(
                strategy_key,
                "any-id",
                [],
                external_conversation_id=False,
            )
            assert conv.get("agent_backend") == AGENT_BACKEND_TAG

    def test_non_server_thread_strategy_never_sets_thread_id(self):
        from strategies.hosted_strategies import build_hosted_conversation

        conv = build_hosted_conversation(
            "maf_lite",
            "foundry-conv-xyz",
            [],
            external_conversation_id=True,
        )
        assert "thread_id" not in conv

    def test_messages_are_deep_copied(self):
        from strategies.hosted_strategies import build_hosted_conversation

        original = [{"role": "user", "text": "hello"}]
        conv = build_hosted_conversation("maf_lite", "conv-1", original)
        conv["messages"][0]["role"] = "mutated"
        assert original[0]["role"] == "user"




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

    @pytest.mark.parametrize("strategy_key", ["maf_agent_service", "single_agent_rag"])
    @pytest.mark.asyncio
    async def test_absent_conversation_id_does_not_pre_set_thread_id_for_server_thread_strategies(
        self,
        strategy_key: str,
    ):
        """When Foundry omits conversation_id, server-thread strategies must not
        receive a synthesised UUID as thread_id; each strategy creates a real
        Foundry conversation on first use via get_new_thread() / ensure_conversation_id()."""
        from api.hosted_entrypoint import _hosted_stream

        received_conversations = []

        class RecordingStrategy:
            def set_context(self, _conversation_id):
                pass

            async def initiate_agent_flow(self, _ask):
                received_conversations.append(deepcopy(self.conversation))
                yield "answer"

        with patch(
            "api.hosted_entrypoint.AgentStrategyFactory.get_strategy",
            new=AsyncMock(return_value=RecordingStrategy()),
        ):
            turn = TurnRequest(ask="Hi", conversation_id=None)
            [event async for event in _hosted_stream(turn, strategy_key)]

        assert len(received_conversations) == 1
        assert "thread_id" not in received_conversations[0], (
            f"{strategy_key} must not receive a synthesised UUID as thread_id"
        )

    @pytest.mark.parametrize("strategy_key", ["maf_agent_service", "single_agent_rag"])
    @pytest.mark.asyncio
    async def test_supplied_conversation_id_is_forwarded_as_thread_id_for_server_thread_strategies(
        self,
        strategy_key: str,
    ):
        """When Foundry supplies a real conversation_id, server-thread strategies
        receive it as thread_id so they can resume the existing server-side thread."""
        from api.hosted_entrypoint import _hosted_stream

        received_conversations = []

        class RecordingStrategy:
            def set_context(self, _conversation_id):
                pass

            async def initiate_agent_flow(self, _ask):
                received_conversations.append(deepcopy(self.conversation))
                yield "answer"

        with patch(
            "api.hosted_entrypoint.AgentStrategyFactory.get_strategy",
            new=AsyncMock(return_value=RecordingStrategy()),
        ):
            turn = TurnRequest(ask="Hi", conversation_id="foundry-conv-real")
            [event async for event in _hosted_stream(turn, strategy_key)]

        assert len(received_conversations) == 1
        assert received_conversations[0]["thread_id"] == "foundry-conv-real"


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
            events = [e async for e in _hosted_stream(turn, "maf_lite")]

        assert citation in events

    @pytest.mark.parametrize("strategy_key", ["maf_lite", "single_agent_rag"])
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
            first_turn = TurnRequest(ask="first question", conversation_id="conv-1")
            [event async for event in _hosted_stream(first_turn, strategy_key)]

            prior_history = [
                {"role": "user", "text": "first question"},
                {"role": "assistant", "text": "answer to first question"},
            ]
            second_turn = TurnRequest(ask="follow-up", conversation_id="conv-1")
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
            def set_context(self, _conversation_id):
                pass

            async def initiate_agent_flow(self, _ask):
                received_conversations.append(deepcopy(self.conversation))
                yield "answer"

        factory = AsyncMock(
            side_effect=[RecordingStrategy(), RecordingStrategy()],
        )
        with patch(
            "api.hosted_entrypoint.AgentStrategyFactory.get_strategy",
            new=factory,
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

    def test_invocations_rejects_trailing_non_user_message(self, client):
        """A message list ending with an assistant turn is malformed; the Foundry
        invocation contract always places the current user ask last."""
        resp = client.post(
            "/invocations",
            json={
                "messages": [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "trailing assistant turn"},
                ],
            },
        )

        assert resp.status_code == 422
        assert "last message" in resp.json()["detail"].lower()

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


# ── Helpers ──────────────────────────────────────────────────────────────────

async def _async_gen(items):
    """Tiny helper to turn a list into an async generator."""
    for item in items:
        yield item
