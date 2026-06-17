"""Regression tests for multi-turn tool-call chaining in SingleAgentRAGStrategyV2.

Covers the fix for Azure/GPT-RAG#505: every follow-up turn used to fail with
``400 No tool call found for function call output`` because the chat thread was
resumed from the previous turn's per-turn response id (``resp_``). On a follow-up
turn the tool output was chained to the resumed turn id instead of the in-loop
response that held the ``function_call``, so the service rejected it.

The fix backs the chat thread with a dedicated server-side conversation object
(``conv_``) created once and resumed on every turn. These tests pin that
contract: the conversation object is created exactly once, reused thereafter,
and the strategy resumes the thread from that stable id on every turn without
persisting per-turn response ids.
"""

import types
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

# Import a strategies module at top level so the dependencies/connectors import
# graph is initialized in the correct order before the test imports the strategy
# (mirrors the other strategy test modules and avoids a circular import).
from strategies.agent_strategies import AgentStrategies  # noqa: F401
from strategies import agent_provider_v2


class TestEnsureConversationId:
    @pytest.fixture(autouse=True)
    def _reset_module_state(self):
        # Isolate the module-scope OpenAI client cache between tests.
        saved_client = agent_provider_v2._openai_client
        saved_project = agent_provider_v2._project_client
        agent_provider_v2._openai_client = None
        try:
            yield
        finally:
            agent_provider_v2._openai_client = saved_client
            agent_provider_v2._project_client = saved_project

    @pytest.mark.asyncio
    async def test_creates_conversation_object_once_and_reuses(self):
        created = types.SimpleNamespace(id="conv_stable")
        oai = MagicMock()
        oai.conversations.create = AsyncMock(return_value=created)

        with patch.object(
            agent_provider_v2, "_get_openai_client", AsyncMock(return_value=oai)
        ):
            conv = {}
            first = await agent_provider_v2.ensure_conversation_id(conv)
            assert first == "conv_stable"
            assert conv["thread_id"] == "conv_stable"

            # Second turn reuses the stored id without creating a new object.
            second = await agent_provider_v2.ensure_conversation_id(conv)
            assert second == "conv_stable"

        oai.conversations.create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_returns_existing_thread_id_without_client(self):
        # When an id is already present, no OpenAI client is needed at all.
        with patch.object(
            agent_provider_v2, "_get_openai_client", AsyncMock(side_effect=AssertionError)
        ):
            conv = {"thread_id": "conv_existing"}
            assert await agent_provider_v2.ensure_conversation_id(conv) == "conv_existing"

    @pytest.mark.asyncio
    async def test_raises_when_provider_not_initialized(self):
        agent_provider_v2._project_client = None
        agent_provider_v2._openai_client = None
        with pytest.raises(RuntimeError):
            await agent_provider_v2._get_openai_client()


class TestStreamAgentThreadResume:
    @pytest.fixture(autouse=True)
    def _patch(self, patch_dependencies, mock_config):
        with patch(
            "strategies.single_agent_rag_strategy_v2.get_config",
            return_value=mock_config,
        ):
            yield

    def _make_strategy(self):
        from strategies.single_agent_rag_strategy_v2 import SingleAgentRAGStrategyV2

        with patch(
            "strategies.single_agent_rag_strategy_v2.get_search_client",
            return_value=MagicMock(),
        ), patch(
            "strategies.single_agent_rag_strategy_v2.get_genai_client",
            return_value=MagicMock(),
        ):
            s = SingleAgentRAGStrategyV2()
        s.search_client = MagicMock()
        s.project_endpoint = "https://example.services.ai.azure.com/api/projects/p"
        s.credential = MagicMock()
        s.model_name = "chat"
        return s

    @pytest.mark.asyncio
    async def test_resumes_same_conversation_object_across_turns(self):
        """Both turns must resume from one stable conversation object id and the
        strategy must never overwrite it with a per-turn response id."""
        s = self._make_strategy()
        conv = {}
        s.conversation = conv

        # A fake agent: async context manager whose get_new_thread records the
        # service_thread_id it was resumed from.
        resume_ids = []

        def _get_new_thread(*, service_thread_id=None):
            resume_ids.append(service_thread_id)
            return types.SimpleNamespace(service_thread_id=service_thread_id)

        agent = MagicMock()
        agent.__aenter__ = AsyncMock(return_value=agent)
        agent.__aexit__ = AsyncMock(return_value=False)
        agent.get_new_thread = _get_new_thread

        provider = MagicMock()
        provider.as_agent = MagicMock(return_value=agent)

        async def _fake_stream(*args, **kwargs):
            yield types.SimpleNamespace(text="hello")

        # One shared conversation object id handed out on first creation.
        created = types.SimpleNamespace(id="conv_stable")
        oai = MagicMock()
        oai.conversations.create = AsyncMock(return_value=created)

        with patch.object(
            agent_provider_v2, "get_provider", AsyncMock(return_value=provider)
        ), patch.object(
            agent_provider_v2, "get_or_create_agent_details", AsyncMock(return_value=MagicMock())
        ), patch.object(
            agent_provider_v2, "stream_agent_run", _fake_stream
        ), patch.object(
            agent_provider_v2, "_get_openai_client", AsyncMock(return_value=oai)
        ):
            # Turn 1
            out1 = "".join([c async for c in s._stream_agent("first question")])
            # Turn 2 (same conversation dict, as the orchestrator reuses it)
            out2 = "".join([c async for c in s._stream_agent("second question")])

        assert out1 == "hello"
        assert out2 == "hello"
        # The conversation object is created exactly once.
        oai.conversations.create.assert_awaited_once()
        # Both turns resume from the same stable conversation object id.
        assert resume_ids == ["conv_stable", "conv_stable"]
        # The stored thread id is the conversation object, never a per-turn resp id.
        assert conv["thread_id"] == "conv_stable"
