"""Compatibility tests for the runtime-neutral orchestration boundary.

These tests verify that:
- ``TurnRequest`` and ``TurnEvent`` are well-formed typed contracts (no
  transport imports).
- Importing ``orchestration.turn`` in a clean process does *not* pull in
  FastAPI, Pydantic, or Azure runtime modules (subprocess sys.modules check).
- ``Orchestrator.from_turn_request`` produces a correctly configured
  orchestrator instance.
- ``Orchestrator.stream_turn`` yields ``TurnEvent`` objects and delegates
  faithfully to ``stream_response``.
- SSE serialisation via ``TurnEvent.to_sse_str`` is byte-for-byte compatible
  with the classic ``stream_response`` output.
- Strategy selector forwarding: each registered key triggers the factory lookup.
- Cancellation and exceptions propagate correctly through ``stream_turn``.
- Classic FastAPI behavior is unaffected by the new boundary layer.
"""

from __future__ import annotations

import asyncio
import dataclasses
import subprocess
import sys
import textwrap
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_turn(**kwargs: Any):
    from orchestration.turn import TurnRequest
    defaults = dict(
        ask="What is the retrieval policy?",
        conversation_id="conv-abc",
        question_id="q-001",
        user_context={"principal_id": "user-1"},
        request_access_token=None,
        correlation_id="req_abc123",
    )
    defaults.update(kwargs)
    return TurnRequest(**defaults)


async def _collect_events(gen: AsyncIterator) -> list:
    """Drain an async generator into a list."""
    result = []
    async for item in gen:
        result.append(item)
    return result


# ---------------------------------------------------------------------------
# TurnRequest contract tests — no transport dependency allowed
# ---------------------------------------------------------------------------

class TestTurnRequestContract:
    """TurnRequest must be transport-neutral (no FastAPI / Pydantic imports)."""

    def test_turn_request_is_a_dataclass(self):
        from orchestration.turn import TurnRequest
        assert dataclasses.is_dataclass(TurnRequest)

    def test_turn_event_is_a_dataclass(self):
        from orchestration.turn import TurnEvent
        assert dataclasses.is_dataclass(TurnEvent)

    # ------------------------------------------------------------------
    # Subprocess sys.modules regression tests
    # ------------------------------------------------------------------

    def _run_isolation_check(self, forbidden_prefix: str) -> None:
        """Spawn a clean interpreter, import orchestration.turn, then assert
        that no module whose name starts with ``forbidden_prefix`` is loaded."""
        src_dir = str(
            __import__("pathlib").Path(__file__).resolve().parent.parent / "src"
        )
        script = textwrap.dedent(f"""\
            import sys
            sys.path.insert(0, {src_dir!r})
            import orchestration.turn
            bad = [m for m in sys.modules if m == {forbidden_prefix!r} or m.startswith({forbidden_prefix!r} + ".")]
            if bad:
                print("FAIL:" + str(bad))
                sys.exit(1)
            print("OK")
        """)
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"subprocess exited {result.returncode}: stdout={result.stdout!r} "
            f"stderr={result.stderr!r}"
        )
        assert result.stdout.strip() == "OK", (
            f"Importing orchestration.turn pulled in {forbidden_prefix!r}: "
            f"{result.stdout.strip()}"
        )

    def test_turn_module_does_not_import_fastapi(self):
        """Importing orchestration.turn must not pull in fastapi."""
        self._run_isolation_check("fastapi")

    def test_turn_module_does_not_import_pydantic(self):
        """Importing orchestration.turn must not pull in pydantic."""
        self._run_isolation_check("pydantic")

    def test_turn_module_does_not_import_azure(self):
        """Importing orchestration.turn must not pull in azure SDK modules."""
        self._run_isolation_check("azure")

    def test_turn_module_does_not_import_orchestrator(self):
        """Importing orchestration.turn must not eagerly load Orchestrator."""
        src_dir = str(
            __import__("pathlib").Path(__file__).resolve().parent.parent / "src"
        )
        script = textwrap.dedent(f"""\
            import sys
            sys.path.insert(0, {src_dir!r})
            import orchestration.turn
            loaded = "orchestration.orchestrator" in sys.modules
            print("FAIL" if loaded else "OK")
        """)
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert result.stdout.strip() == "OK", (
            "Importing orchestration.turn also loaded orchestration.orchestrator"
        )

    # ------------------------------------------------------------------
    # TurnRequest field tests
    # ------------------------------------------------------------------

    def test_required_field_ask(self):
        from orchestration.turn import TurnRequest
        t = TurnRequest(ask="hello")
        assert t.ask == "hello"

    def test_optional_fields_default_to_none_or_empty(self):
        from orchestration.turn import TurnRequest
        t = TurnRequest(ask="x")
        assert t.conversation_id is None
        assert t.question_id is None
        assert t.user_context == {}
        assert t.request_access_token is None
        assert t.correlation_id is None

    def test_all_fields_settable(self):
        from orchestration.turn import TurnRequest
        t = TurnRequest(
            ask="question",
            conversation_id="conv-1",
            question_id="q-1",
            user_context={"principal_id": "u-1"},
            request_access_token="tok",
            correlation_id="req_xyz",
        )
        assert t.ask == "question"
        assert t.conversation_id == "conv-1"
        assert t.question_id == "q-1"
        assert t.user_context == {"principal_id": "u-1"}
        assert t.request_access_token == "tok"
        assert t.correlation_id == "req_xyz"

    def test_user_context_is_independent_between_instances(self):
        """Mutable default must not be shared between instances."""
        from orchestration.turn import TurnRequest
        a = TurnRequest(ask="a")
        b = TurnRequest(ask="b")
        a.user_context["key"] = "value"
        assert "key" not in b.user_context

    # ------------------------------------------------------------------
    # TurnEvent field and serialisation tests
    # ------------------------------------------------------------------

    def test_turn_event_kind_required(self):
        from orchestration.turn import TurnEvent
        e = TurnEvent(kind="text")
        assert e.kind == "text"
        assert e.data == ""

    def test_turn_event_all_kinds_constructible(self):
        from orchestration.turn import TurnEvent
        for kind in ("conversation_id", "text", "citations", "tool_call", "error", "cancelled"):
            e = TurnEvent(kind=kind, data="payload")
            assert e.kind == kind
            assert e.data == "payload"

    def test_to_sse_str_conversation_id_appends_space(self):
        from orchestration.turn import TurnEvent
        e = TurnEvent(kind="conversation_id", data="conv-123")
        assert e.to_sse_str() == "conv-123 "

    def test_to_sse_str_text_returns_data_unchanged(self):
        from orchestration.turn import TurnEvent
        e = TurnEvent(kind="text", data="Hello world")
        assert e.to_sse_str() == "Hello world"

    def test_to_sse_str_other_kinds_return_data(self):
        from orchestration.turn import TurnEvent
        for kind in ("citations", "tool_call", "error", "cancelled"):
            e = TurnEvent(kind=kind, data="payload")
            assert e.to_sse_str() == "payload"


# ---------------------------------------------------------------------------
# Orchestrator.from_turn_request — factory boundary tests
# ---------------------------------------------------------------------------

class TestFromTurnRequest:
    """from_turn_request must propagate all TurnRequest fields to create()."""

    @pytest.fixture(autouse=True)
    def _patch_deps(self, patch_dependencies, mock_config, mock_cosmos):
        with (
            patch("orchestration.orchestrator.get_config", return_value=mock_config),
            patch("orchestration.orchestrator.get_cosmosdb_client", return_value=mock_cosmos),
        ):
            yield

    @pytest.mark.asyncio
    async def test_from_turn_request_returns_orchestrator(self):
        from orchestration.orchestrator import Orchestrator
        from orchestration.turn import TurnRequest

        turn = TurnRequest(ask="hello", conversation_id="conv-1")

        mock_strategy = MagicMock()
        mock_strategy.set_context = MagicMock()

        with patch("orchestration.orchestrator.AgentStrategyFactory.get_strategy",
                   new=AsyncMock(return_value=mock_strategy)):
            orch = await Orchestrator.from_turn_request(turn)

        assert isinstance(orch, Orchestrator)

    @pytest.mark.asyncio
    async def test_conversation_id_propagated(self):
        from orchestration.orchestrator import Orchestrator
        from orchestration.turn import TurnRequest

        turn = TurnRequest(ask="hello", conversation_id="conv-abc")

        mock_strategy = MagicMock()
        mock_strategy.set_context = MagicMock()

        with patch("orchestration.orchestrator.AgentStrategyFactory.get_strategy",
                   new=AsyncMock(return_value=mock_strategy)):
            orch = await Orchestrator.from_turn_request(turn)

        assert orch.conversation_id == "conv-abc"

    @pytest.mark.asyncio
    async def test_user_context_propagated(self):
        from orchestration.orchestrator import Orchestrator
        from orchestration.turn import TurnRequest

        ctx = {"principal_id": "user-xyz", "principal_name": "Alice"}
        turn = TurnRequest(ask="q", user_context=ctx)

        mock_strategy = MagicMock()
        mock_strategy.set_context = MagicMock()

        with patch("orchestration.orchestrator.AgentStrategyFactory.get_strategy",
                   new=AsyncMock(return_value=mock_strategy)):
            orch = await Orchestrator.from_turn_request(turn)

        assert orch.user_context == ctx

    @pytest.mark.asyncio
    async def test_correlation_id_propagated(self):
        from orchestration.orchestrator import Orchestrator
        from orchestration.turn import TurnRequest

        turn = TurnRequest(ask="q", correlation_id="req_correlate")

        mock_strategy = MagicMock()
        mock_strategy.set_context = MagicMock()

        with patch("orchestration.orchestrator.AgentStrategyFactory.get_strategy",
                   new=AsyncMock(return_value=mock_strategy)):
            orch = await Orchestrator.from_turn_request(turn)

        assert orch.correlation_id == "req_correlate"

    @pytest.mark.asyncio
    async def test_access_token_propagated(self):
        from orchestration.orchestrator import Orchestrator
        from orchestration.turn import TurnRequest

        turn = TurnRequest(ask="q", request_access_token="obo-token-123")

        mock_strategy = MagicMock()
        mock_strategy.set_context = MagicMock()

        with patch("orchestration.orchestrator.AgentStrategyFactory.get_strategy",
                   new=AsyncMock(return_value=mock_strategy)):
            orch = await Orchestrator.from_turn_request(turn)

        assert orch.request_access_token == "obo-token-123"


# ---------------------------------------------------------------------------
# Orchestrator.stream_turn — typed streaming boundary tests
# ---------------------------------------------------------------------------

class TestStreamTurn:
    """stream_turn must yield TurnEvent objects and delegate to stream_response."""

    @pytest.fixture(autouse=True)
    def _patch_deps(self, patch_dependencies, mock_config, mock_cosmos):
        with (
            patch("orchestration.orchestrator.get_config", return_value=mock_config),
            patch("orchestration.orchestrator.get_cosmosdb_client", return_value=mock_cosmos),
        ):
            yield

    def _make_orchestrator(self, mock_config, mock_cosmos):
        from orchestration.orchestrator import Orchestrator
        orch = Orchestrator.__new__(Orchestrator)
        orch.conversation_id = "conv-1"
        orch.principal_id = "user-1"
        orch.correlation_id = None
        orch.user_context = {}
        orch.request_access_token = None
        orch.database_client = mock_cosmos
        orch.database_container = "conversations"
        from orchestration.conversation_compaction import load_conversation_compaction_config
        orch.conversation_compaction_config = load_conversation_compaction_config(mock_config)
        return orch

    @pytest.mark.asyncio
    async def test_stream_turn_yields_turn_events(self, mock_config, mock_cosmos):
        """stream_turn must yield TurnEvent objects, not raw strings."""
        from orchestration.turn import TurnEvent, TurnRequest

        orch = self._make_orchestrator(mock_config, mock_cosmos)

        async def fake_stream_response(ask, question_id=None):
            yield "conv-1 "
            yield "Hello "
            yield "world"

        orch.stream_response = fake_stream_response

        turn = TurnRequest(ask="hello", conversation_id="conv-1", question_id="q-1")
        events = await _collect_events(orch.stream_turn(turn))

        assert all(isinstance(e, TurnEvent) for e in events)

    @pytest.mark.asyncio
    async def test_stream_turn_first_event_is_conversation_id(
        self, mock_config, mock_cosmos
    ):
        """First event must have kind='conversation_id' with the conversation id."""
        from orchestration.turn import TurnRequest

        orch = self._make_orchestrator(mock_config, mock_cosmos)

        async def fake_stream_response(ask, question_id=None):
            yield "conv-1 "
            yield "answer"

        orch.stream_response = fake_stream_response

        turn = TurnRequest(ask="hello", conversation_id="conv-1")
        events = await _collect_events(orch.stream_turn(turn))

        assert events[0].kind == "conversation_id"
        assert events[0].data == "conv-1"

    @pytest.mark.asyncio
    async def test_stream_turn_subsequent_events_are_text(
        self, mock_config, mock_cosmos
    ):
        """All events after the first must have kind='text'."""
        from orchestration.turn import TurnRequest

        orch = self._make_orchestrator(mock_config, mock_cosmos)

        async def fake_stream_response(ask, question_id=None):
            yield "conv-1 "
            yield "Hello "
            yield "world"

        orch.stream_response = fake_stream_response

        turn = TurnRequest(ask="hello", conversation_id="conv-1")
        events = await _collect_events(orch.stream_turn(turn))

        assert all(e.kind == "text" for e in events[1:])
        assert [e.data for e in events[1:]] == ["Hello ", "world"]

    @pytest.mark.asyncio
    async def test_stream_turn_sse_parity(self, mock_config, mock_cosmos):
        """to_sse_str() must reproduce the exact SSE bytes that stream_response yields."""
        from orchestration.turn import TurnRequest

        orch = self._make_orchestrator(mock_config, mock_cosmos)
        raw_chunks = ["conv-1 ", "Hello ", "world", "[citation JSON]"]

        async def fake_stream_response(ask, question_id=None):
            for c in raw_chunks:
                yield c

        orch.stream_response = fake_stream_response

        # Collect SSE bytes via the typed boundary
        turn = TurnRequest(ask="hello", conversation_id="conv-1", question_id="q-1")
        events = await _collect_events(orch.stream_turn(turn))
        serialised = [e.to_sse_str() for e in events]

        assert serialised == raw_chunks, (
            "TurnEvent.to_sse_str() output does not match classic stream_response output"
        )

    @pytest.mark.asyncio
    async def test_stream_turn_passes_ask_to_stream_response(
        self, mock_config, mock_cosmos
    ):
        orch = self._make_orchestrator(mock_config, mock_cosmos)

        captured: dict = {}

        async def fake_stream_response(ask, question_id=None):
            captured["ask"] = ask
            captured["question_id"] = question_id
            yield "conv-1 "
            yield "done"

        orch.stream_response = fake_stream_response

        from orchestration.turn import TurnRequest
        turn = TurnRequest(ask="my question", question_id="q-42")
        await _collect_events(orch.stream_turn(turn))

        assert captured["ask"] == "my question"
        assert captured["question_id"] == "q-42"

    @pytest.mark.asyncio
    async def test_stream_turn_propagates_cancellation(
        self, mock_config, mock_cosmos
    ):
        orch = self._make_orchestrator(mock_config, mock_cosmos)

        async def fake_stream_response(ask, question_id=None):
            raise asyncio.CancelledError()
            yield  # make it an async generator

        orch.stream_response = fake_stream_response

        from orchestration.turn import TurnRequest
        turn = TurnRequest(ask="cancel me")
        with pytest.raises(asyncio.CancelledError):
            await _collect_events(orch.stream_turn(turn))

    @pytest.mark.asyncio
    async def test_stream_turn_propagates_exceptions(
        self, mock_config, mock_cosmos
    ):
        orch = self._make_orchestrator(mock_config, mock_cosmos)

        async def fake_stream_response(ask, question_id=None):
            raise RuntimeError("strategy failure")
            yield  # make it an async generator

        orch.stream_response = fake_stream_response

        from orchestration.turn import TurnRequest
        turn = TurnRequest(ask="will fail")
        with pytest.raises(RuntimeError, match="strategy failure"):
            await _collect_events(orch.stream_turn(turn))


# ---------------------------------------------------------------------------
# Strategy selector forwarding through the typed boundary
# ---------------------------------------------------------------------------

class TestStrategySelectorForwarding:
    """Each registered strategy key must be forwarded to the factory by from_turn_request.

    These tests verify that the typed boundary correctly forwards the active
    strategy key to ``AgentStrategyFactory.get_strategy``.  The factory is
    mocked so no Azure credentials are required; the claim tested is
    *selector forwarding*, not strategy instantiation.
    """

    @pytest.fixture(autouse=True)
    def _patch_deps(self, patch_dependencies, mock_config, mock_cosmos):
        with (
            patch("orchestration.orchestrator.get_config", return_value=mock_config),
            patch("orchestration.orchestrator.get_cosmosdb_client", return_value=mock_cosmos),
        ):
            yield

    @pytest.mark.asyncio
    @pytest.mark.parametrize("strategy_key", [
        "single_agent_rag",
        "maf_agent_service",
        "maf_lite",
        "mcp",
        "nl2sql",
        "multimodal",
    ])
    async def test_strategy_key_forwarded_to_factory(
        self, strategy_key, mock_config, mock_cosmos
    ):
        """from_turn_request must forward the configured strategy key to get_strategy."""
        from orchestration.orchestrator import Orchestrator
        from orchestration.turn import TurnRequest

        mock_config.get.side_effect = lambda key, default=None, type=str: {
            "AGENT_STRATEGY": strategy_key,
            "AI_FOUNDRY_PROJECT_ENDPOINT": "https://fake.openai.azure.com",
            "AI_FOUNDRY_ACCOUNT_ENDPOINT": "https://fake-account.openai.azure.com",
            "CHAT_DEPLOYMENT_NAME": "gpt-4o",
            "OPENAI_API_VERSION": "2025-04-01-preview",
            "PROMPT_SOURCE": "file",
            "CONVERSATIONS_DATABASE_CONTAINER": "conversations",
            "CONVERSATION_MAX_TOKENS": "8000",
            "CONVERSATION_MAX_MESSAGES": "50",
            "CONVERSATION_MAX_QUESTIONS": "100",
        }.get(key, default)

        mock_strategy = MagicMock()
        mock_strategy.set_context = MagicMock()

        turn = TurnRequest(ask="test question")
        with patch(
            "orchestration.orchestrator.AgentStrategyFactory.get_strategy",
            new=AsyncMock(return_value=mock_strategy),
        ) as mock_get:
            orch = await Orchestrator.from_turn_request(turn)
            mock_get.assert_awaited_once_with(strategy_key)

        assert isinstance(orch, Orchestrator)

    def test_all_parametrized_keys_are_registered_in_factory(self):
        """The factory registry must contain every key used in parametrized tests above."""
        from strategies.agent_strategy_factory import AgentStrategyFactory

        registered = AgentStrategyFactory.registered_strategy_names()
        for key in ("single_agent_rag", "maf_agent_service", "maf_lite", "mcp", "nl2sql", "multimodal"):
            assert key in registered, (
                f"Strategy key {key!r} is missing from AgentStrategyFactory._REGISTRY"
            )


# ---------------------------------------------------------------------------
# Classic FastAPI behavior preservation
# ---------------------------------------------------------------------------

class TestClassicBehaviorPreservation:
    """Verify that the Orchestrator public API is unchanged for classic callers."""

    def test_orchestrator_still_has_create_classmethod(self):
        from orchestration.orchestrator import Orchestrator
        assert callable(getattr(Orchestrator, "create", None))

    def test_orchestrator_still_has_stream_response(self):
        from orchestration.orchestrator import Orchestrator
        assert callable(getattr(Orchestrator, "stream_response", None))

    def test_orchestrator_still_has_save_feedback(self):
        from orchestration.orchestrator import Orchestrator
        assert callable(getattr(Orchestrator, "save_feedback", None))

    def test_turn_request_exported_from_orchestration_package(self):
        from orchestration import TurnRequest  # noqa: F401
        assert TurnRequest is not None

    def test_turn_event_exported_from_orchestration_package(self):
        from orchestration import TurnEvent  # noqa: F401
        assert TurnEvent is not None

    def test_orchestrator_exported_from_orchestration_package(self):
        from orchestration import Orchestrator  # noqa: F401
        assert Orchestrator is not None

    def test_all_registered_strategy_keys_in_factory(self):
        from strategies.agent_strategy_factory import AgentStrategyFactory

        registered = AgentStrategyFactory.registered_strategy_names()
        assert "single_agent_rag" in registered
        assert "maf_agent_service" in registered
        assert "maf_lite" in registered
        assert "mcp" in registered
        assert "nl2sql" in registered
        assert "multimodal" in registered
