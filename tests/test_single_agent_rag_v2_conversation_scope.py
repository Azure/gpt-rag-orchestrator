"""Regression tests for conversation-scoped retrieval in SingleAgentRAGStrategyV2.

Covers the fix for Azure/GPT-RAG#478: uploaded documents (tagged with the chat
conversation id by the ingestion service) must be included in the AI Search
retrieval filter. The keystone bug was that the strategy never propagated the
conversation id into the search request context, so the filter fell back to the
shared corpus only and uploaded chunks were never retrieved.
"""

import pytest
from unittest.mock import patch, MagicMock

# Import a strategies module at top level so the dependencies/connectors import
# graph is initialized in the correct order before the test imports the strategy
# (mirrors the other strategy test modules and avoids a circular import).
from strategies.agent_strategies import AgentStrategies  # noqa: F401


class TestSingleAgentRagV2ConversationScope:
    @pytest.fixture(autouse=True)
    def _patch(self, patch_dependencies, mock_config):
        with patch(
            "strategies.single_agent_rag_strategy_v2.get_config",
            return_value=mock_config,
        ):
            yield

    def _make_strategy(self):
        from strategies.single_agent_rag_strategy_v2 import SingleAgentRAGStrategyV2

        # get_search_client()/get_genai_client() reach out to App Config/Azure at
        # construction time; patch them so __init__ stays offline. We assert on
        # the search client mock below.
        with patch(
            "strategies.single_agent_rag_strategy_v2.get_search_client",
            return_value=MagicMock(),
        ), patch(
            "strategies.single_agent_rag_strategy_v2.get_genai_client",
            return_value=MagicMock(),
        ):
            s = SingleAgentRAGStrategyV2()
        # Ensure a clean mock search client for per-request context assertions.
        s.search_client = MagicMock()
        return s

    def test_set_context_stores_conversation_id(self):
        s = self._make_strategy()
        assert s.conversation_id is None
        s.set_context("conv-123")
        assert s.conversation_id == "conv-123"

    def test_set_context_ignores_empty(self):
        s = self._make_strategy()
        s.set_context("conv-123")
        s.set_context(None)
        s.set_context("")
        # An empty/None update must not clobber a previously set id.
        assert s.conversation_id == "conv-123"

    def test_resolve_prefers_conversation_dict_id(self):
        s = self._make_strategy()
        s.set_context("from-set-context")
        s.conversation = {"id": "from-dict", "thread_id": "agent-thread"}
        # The conversation dict id wins over set_context, and thread_id is never used.
        assert s._resolve_conversation_id() == "from-dict"

    def test_resolve_falls_back_to_set_context(self):
        s = self._make_strategy()
        s.set_context("from-set-context")
        s.conversation = {}  # no id in the dict
        assert s._resolve_conversation_id() == "from-set-context"

    def test_resolve_never_uses_thread_id(self):
        s = self._make_strategy()
        s.conversation = {"thread_id": "agent-thread"}  # only thread_id present
        assert s._resolve_conversation_id() is None

    def test_apply_context_passes_conversation_id_to_search_client(self):
        s = self._make_strategy()
        s.conversation = {"id": "conv-abc"}
        s._apply_search_request_context()

        s.search_client.set_request_context.assert_called_once()
        kwargs = s.search_client.set_request_context.call_args.kwargs
        assert kwargs["conversation_id"] == "conv-abc"

    def test_apply_context_noop_without_search_client(self):
        s = self._make_strategy()
        s.search_client = None
        # Must not raise when retrieval is disabled.
        s._apply_search_request_context()

    @pytest.mark.asyncio
    async def test_bound_search_tool_delegates_to_apply_context(self):
        """Production-path regression guard for issue #478.

        Invokes the exact retrieval tool produced by `_build_search_tool()`
        (the factory used by `_stream_agent`) and asserts it calls
        `_apply_search_request_context()`. This guarantees the conversation
        scope fix continues to run on the real code path, not just on a
        helper method that could become dead code.
        """
        s = self._make_strategy()
        s.conversation = {"id": "conv-xyz"}

        async def _fake_search(query):
            return {"documents": []}

        s.search_client.search_knowledge_base = _fake_search
        s._format_search_results = lambda r: "ok"

        tool = s._build_search_tool()
        assert tool is not None
        assert tool.__name__ == "search_knowledge_base"

        result = await tool("q")
        assert result == "ok"

        s.search_client.set_request_context.assert_called_once()
        kwargs = s.search_client.set_request_context.call_args.kwargs
        assert kwargs["conversation_id"] == "conv-xyz"

    def test_build_search_tool_returns_none_without_search_client(self):
        s = self._make_strategy()
        s.search_client = None
        assert s._build_search_tool() is None
