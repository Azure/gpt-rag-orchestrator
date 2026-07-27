"""Compatibility tests for the runtime-neutral turn boundary."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agent_framework import (
    CitationAnnotation,
    FunctionCallContent,
    FunctionResultContent,
    TextContent,
)

from api.turn_sse import serialize_turn_event
from orchestration.agent_events import AgentEventTranslator
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
from orchestration.orchestrator import Orchestrator


def _make_turn(**overrides: Any) -> TurnRequest:
    values = {
        "ask": "What is the retrieval policy?",
        "conversation_id": "conv-abc",
        "question_id": "q-001",
        "user_context": {"principal_id": "user-1"},
        "request_access_token": None,
        "correlation_id": "req_abc123",
    }
    values.update(overrides)
    return TurnRequest(**values)


def _isolated_import_modules(import_statement: str) -> set[str]:
    repo_root = Path(__file__).resolve().parents[1]
    script = (
        "import json, sys\n"
        "before = set(sys.modules)\n"
        f"{import_statement}\n"
        "print(json.dumps(sorted(set(sys.modules) - before)))\n"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src")
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return set(json.loads(completed.stdout))


async def _collect(stream):
    return [event async for event in stream]


class TestDependencyNeutralImport:
    def test_turn_submodule_import_does_not_load_runtime_dependencies(self):
        modules = _isolated_import_modules("import orchestration.turn")

        assert "orchestration.orchestrator" not in modules
        assert "fastapi" not in modules
        assert "pydantic" not in modules
        assert not any(name == "azure" or name.startswith("azure.") for name in modules)

    def test_package_turn_export_remains_dependency_neutral(self):
        modules = _isolated_import_modules("from orchestration import TurnRequest")

        assert "orchestration.orchestrator" not in modules
        assert "fastapi" not in modules
        assert "pydantic" not in modules
        assert not any(name == "azure" or name.startswith("azure.") for name in modules)


class TestTurnContracts:
    def test_request_is_a_stdlib_dataclass_with_isolated_context(self):
        first = TurnRequest(ask="first")
        second = TurnRequest(ask="second")
        first.user_context["key"] = "value"

        assert dataclasses.is_dataclass(TurnRequest)
        assert second.user_context == {}

    def test_output_contract_covers_required_event_kinds(self):
        events = [
            TurnConversationEvent("conv-1"),
            TurnTextEvent("answer"),
            TurnCitationEvent(TurnCitation("source-1", title="Source")),
            TurnToolActivityEvent(
                TurnToolActivity("search", TurnToolStatus.STARTED, call_id="call-1")
            ),
            TurnErrorEvent(),
            TurnCancelledEvent(),
        ]

        assert all(dataclasses.is_dataclass(event) for event in events)


class TestClassicSseSerialization:
    def test_serializes_every_typed_event_kind(self):
        assert serialize_turn_event(TurnConversationEvent("conv-1")) == "conv-1 "
        assert serialize_turn_event(TurnTextEvent("answer")) == "answer"
        assert (
            serialize_turn_event(
                TurnCitationEvent(TurnCitation("source-1"), text="[source]")
            )
            == "[source]"
        )
        assert (
            serialize_turn_event(
                TurnToolActivityEvent(
                    TurnToolActivity("search", TurnToolStatus.COMPLETED),
                    text="[search complete]",
                )
            )
            == "[search complete]"
        )
        assert serialize_turn_event(TurnCitationEvent(TurnCitation("source-1"))) is None
        assert (
            serialize_turn_event(
                TurnToolActivityEvent(
                    TurnToolActivity("search", TurnToolStatus.STARTED)
                )
            )
            is None
        )
        assert (
            serialize_turn_event(TurnErrorEvent())
            == "event: error\ndata: An internal server error occurred.\n\n"
        )
        assert serialize_turn_event(TurnCancelledEvent()) is None


class TestAgentEventTranslation:
    def test_translates_tool_lifecycle_and_citations(self):
        translator = AgentEventTranslator()
        update = MagicMock(
            contents=[
                FunctionCallContent(
                    call_id="call-1",
                    name="search_knowledge_base",
                    arguments={"query": "policy"},
                ),
                FunctionResultContent(call_id="call-1", result={"count": 1}),
                TextContent(
                    "answer",
                    annotations=[
                        CitationAnnotation(
                            title="Policy",
                            url="https://example.test/policy",
                            file_id="policy.pdf",
                            snippet="Relevant excerpt",
                        )
                    ],
                ),
            ]
        )

        events = list(translator.translate(update))

        assert events == [
            TurnToolActivityEvent(
                TurnToolActivity(
                    "search_knowledge_base",
                    TurnToolStatus.STARTED,
                    call_id="call-1",
                )
            ),
            TurnToolActivityEvent(
                TurnToolActivity(
                    "search_knowledge_base",
                    TurnToolStatus.COMPLETED,
                    call_id="call-1",
                )
            ),
            TurnCitationEvent(
                TurnCitation(
                    "policy.pdf",
                    title="Policy",
                    url="https://example.test/policy",
                    snippet="Relevant excerpt",
                )
            ),
        ]

    def test_emits_one_started_event_for_streamed_argument_deltas(self):
        translator = AgentEventTranslator()

        first = MagicMock(
            contents=[
                FunctionCallContent(
                    call_id="call-1",
                    name="search_knowledge_base",
                    arguments='{"query":',
                )
            ]
        )
        continuation = MagicMock(
            contents=[
                FunctionCallContent(
                    call_id="call-1",
                    name="search_knowledge_base",
                    arguments='"policy"}',
                ),
                FunctionCallContent(
                    call_id="",
                    name="",
                    arguments='"ignored continuation"',
                ),
            ]
        )

        events = [
            *translator.translate(first),
            *translator.translate(continuation),
        ]

        assert events == [
            TurnToolActivityEvent(
                TurnToolActivity(
                    "search_knowledge_base",
                    TurnToolStatus.STARTED,
                    call_id="call-1",
                )
            )
        ]


class TestFromTurnRequest:
    @pytest.fixture(autouse=True)
    def _patch_deps(self, patch_dependencies, mock_config, mock_cosmos):
        with (
            patch("orchestration.orchestrator.get_config", return_value=mock_config),
            patch(
                "orchestration.orchestrator.get_cosmosdb_client",
                return_value=mock_cosmos,
            ),
        ):
            yield

    @pytest.mark.asyncio
    async def test_propagates_request_fields(self):
        strategy = MagicMock()
        turn = _make_turn(request_access_token="token")
        with patch(
            "orchestration.orchestrator.AgentStrategyFactory.get_strategy",
            new=AsyncMock(return_value=strategy),
        ):
            orchestrator = await Orchestrator.from_turn_request(turn)

        assert orchestrator.conversation_id == "conv-abc"
        assert orchestrator.user_context == {"principal_id": "user-1"}
        assert orchestrator.request_access_token == "token"
        assert orchestrator.correlation_id == "req_abc123"
        strategy.set_context.assert_called_once_with("conv-abc")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "strategy_key",
        [
            "single_agent_rag",
            "maf_agent_service",
            "maf_lite",
            "mcp",
            "nl2sql",
            "multimodal",
        ],
    )
    async def test_forwards_registered_selector_to_factory(
        self,
        strategy_key,
        mock_config,
    ):
        """Prove selector forwarding, not live Azure-backed construction."""
        original_get = mock_config.get.side_effect
        mock_config.get.side_effect = (
            lambda key, default=None, type=str: strategy_key
            if key == "AGENT_STRATEGY"
            else original_get(key, default, type)
        )
        strategy = MagicMock()
        get_strategy = AsyncMock(return_value=strategy)
        with patch(
            "orchestration.orchestrator.AgentStrategyFactory.get_strategy",
            new=get_strategy,
        ):
            await Orchestrator.from_turn_request(_make_turn())

        get_strategy.assert_awaited_once_with(strategy_key)


class TestStreamTurn:
    @staticmethod
    def _orchestrator(conversation_id: str = "conv-1"):
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.conversation_id = conversation_id
        return orchestrator

    @pytest.mark.asyncio
    async def test_exposes_identity_and_text_without_classic_prefix(self):
        orchestrator = self._orchestrator()

        async def classic_stream(_ask, _question_id, *, _event_sink=None):
            yield "conv-1 "
            yield "Hello "
            yield "world"

        orchestrator.stream_response = classic_stream
        events = await _collect(orchestrator.stream_turn(_make_turn()))

        assert events == [
            TurnConversationEvent("conv-1"),
            TurnTextEvent("Hello "),
            TurnTextEvent("world"),
        ]

    @pytest.mark.asyncio
    async def test_fastapi_serialization_is_byte_compatible_with_classic_stream(self):
        orchestrator = self._orchestrator()
        classic_chunks = ["conv-1 ", "Hello ", "world", " [source 1]"]

        async def classic_stream(_ask, _question_id, *, _event_sink=None):
            for chunk in classic_chunks:
                yield chunk

        orchestrator.stream_response = classic_stream
        events = await _collect(orchestrator.stream_turn(_make_turn()))
        serialized = [
            chunk
            for event in events
            if (chunk := serialize_turn_event(event)) is not None
        ]

        assert serialized == classic_chunks
        assert "".join(serialized).encode() == "".join(classic_chunks).encode()

    @pytest.mark.asyncio
    async def test_routes_ask_and_question_id(self):
        orchestrator = self._orchestrator()
        captured = {}

        async def classic_stream(ask, question_id, *, _event_sink=None):
            captured.update(ask=ask, question_id=question_id)
            yield "conv-1 "

        orchestrator.stream_response = classic_stream
        await _collect(orchestrator.stream_turn(_make_turn()))

        assert captured == {
            "ask": "What is the retrieval policy?",
            "question_id": "q-001",
        }

    @pytest.mark.asyncio
    async def test_exposes_structured_strategy_events_in_stream_order(self):
        orchestrator = self._orchestrator()
        citation = TurnCitationEvent(TurnCitation("source-1", title="Source"))
        tool = TurnToolActivityEvent(
            TurnToolActivity("search", TurnToolStatus.STARTED, call_id="call-1")
        )

        async def classic_stream(_ask, _question_id, *, _event_sink=None):
            yield "conv-1 "
            _event_sink(citation)
            _event_sink(tool)
            yield "answer"

        orchestrator.stream_response = classic_stream
        events = await _collect(orchestrator.stream_turn(_make_turn()))

        assert events == [
            TurnConversationEvent("conv-1"),
            citation,
            tool,
            TurnTextEvent("answer"),
        ]

    @pytest.mark.asyncio
    async def test_emits_typed_error_then_propagates_exception(self):
        orchestrator = self._orchestrator()

        async def classic_stream(_ask, _question_id, *, _event_sink=None):
            raise RuntimeError("strategy failure")
            yield

        orchestrator.stream_response = classic_stream
        stream = orchestrator.stream_turn(_make_turn())

        assert await anext(stream) == TurnErrorEvent()
        with pytest.raises(RuntimeError, match="strategy failure"):
            await anext(stream)

    @pytest.mark.asyncio
    async def test_emits_typed_cancellation_then_propagates_cancellation(self):
        orchestrator = self._orchestrator()

        async def classic_stream(_ask, _question_id, *, _event_sink=None):
            raise asyncio.CancelledError
            yield

        orchestrator.stream_response = classic_stream
        stream = orchestrator.stream_turn(_make_turn())

        assert await anext(stream) == TurnCancelledEvent()
        with pytest.raises(asyncio.CancelledError):
            await anext(stream)


class TestClassicApiPreservation:
    def test_lazy_package_export_preserves_orchestrator_consumer(self):
        from orchestration import Orchestrator

        assert callable(Orchestrator.create)
        assert callable(Orchestrator.stream_response)
        assert callable(Orchestrator.save_feedback)
