"""Regression tests for the direct-LLM (empty index) path in SingleAgentRAGStrategyV2.

Covers the multi-turn bug where an empty search index routes to
``_stream_direct_llm`` and the persisted conversation history (stored under the
``text`` key) was re-inserted verbatim into the chat completions payload. The
OpenAI API expects ``content``, so the history messages carried ``content=None``
and the second turn failed with::

    400 - Invalid value for 'content': expected a string, got null. param: messages.[1].content

The fix normalizes ``text`` -> ``content``, drops empty entries, and keeps the
history in chronological order. An empty index must never surface an error.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from strategies.agent_strategies import AgentStrategies  # noqa: F401


class _FakeDelta:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.delta = _FakeDelta(content)


class _FakeChunk:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeStream:
    def __init__(self, contents):
        self._contents = contents

    def __aiter__(self):
        self._it = iter(self._contents)
        return self

    async def __anext__(self):
        try:
            return _FakeChunk(next(self._it))
        except StopIteration:
            raise StopAsyncIteration


class TestSingleAgentRagV2DirectLLMHistory:
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
        return s

    async def _run(self, strategy, user_message):
        captured = {}

        async def _fake_create(**kwargs):
            captured["kwargs"] = kwargs
            return _FakeStream(["Olá", " mundo"])

        strategy.llm_client = MagicMock()
        strategy.llm_client.chat_deployment = "chat"
        strategy.llm_client.openai_client.chat.completions.create = AsyncMock(
            side_effect=_fake_create
        )

        with patch.object(
            strategy, "_read_prompt", new=AsyncMock(return_value="SYSTEM")
        ):
            out = ""
            async for chunk in strategy._stream_direct_llm(user_message):
                out += chunk
        return captured["kwargs"]["messages"], out

    @pytest.mark.asyncio
    async def test_history_normalized_to_content(self):
        s = self._make_strategy()
        # History as persisted by the direct-LLM path: uses the "text" key.
        s.conversation = {
            "messages": [
                {"role": "user", "text": "olá, tudo bem?"},
                {"role": "assistant", "text": "Olá! Estou bem."},
            ]
        }

        messages, out = await self._run(s, "como funciona a injecao eletronica?")

        # No message may carry a null/None content (the cause of the 400).
        assert all(m.get("content") for m in messages)
        # System first, history in chronological order, then the current user turn.
        assert messages[0] == {"role": "system", "content": "SYSTEM"}
        assert messages[1] == {"role": "user", "content": "olá, tudo bem?"}
        assert messages[2] == {"role": "assistant", "content": "Olá! Estou bem."}
        assert messages[-1] == {
            "role": "user",
            "content": "como funciona a injecao eletronica?",
        }
        assert out == "Olá mundo"

    @pytest.mark.asyncio
    async def test_empty_and_unknown_history_entries_skipped(self):
        s = self._make_strategy()
        s.conversation = {
            "messages": [
                {"role": "assistant", "text": ""},        # empty -> skip
                {"role": "tool", "text": "ignore me"},     # unknown role -> skip
                {"role": "user", "content": "kept"},        # already content -> kept
                {"role": "assistant", "text": None},        # None -> skip
            ]
        }

        messages, _ = await self._run(s, "next question")

        assert all(m.get("content") for m in messages)
        roles_contents = [(m["role"], m["content"]) for m in messages]
        assert ("user", "kept") in roles_contents
        assert ("tool", "ignore me") not in roles_contents
        assert messages[0]["role"] == "system"
        assert messages[-1] == {"role": "user", "content": "next question"}

    @pytest.mark.asyncio
    async def test_no_history_still_builds_valid_payload(self):
        s = self._make_strategy()
        s.conversation = {}

        messages, _ = await self._run(s, "first question")

        assert messages == [
            {"role": "system", "content": "SYSTEM"},
            {"role": "user", "content": "first question"},
        ]

    @pytest.mark.asyncio
    async def test_empty_index_renders_grounded_no_context_prompt(self):
        """With a KB configured (SEARCH_RETRIEVAL_ENABLED=true) but no content, the
        empty-index path must render the strict no-context prompt and must NOT
        re-enable the tool-calling (aisearch) block, since no search tool is bound.
        """
        s = self._make_strategy()
        s.conversation = {}

        captured_ctx = {}

        async def _fake_read_prompt(name, use_jinja2=False, jinja2_context=None):
            captured_ctx.update(jinja2_context or {})
            return "SYSTEM"

        async def _fake_create(**kwargs):
            return _FakeStream(["ok"])

        s.llm_client = MagicMock()
        s.llm_client.chat_deployment = "chat"
        s.llm_client.openai_client.chat.completions.create = AsyncMock(
            side_effect=_fake_create
        )

        with patch.object(s, "_read_prompt", new=_fake_read_prompt):
            async for _ in s._stream_direct_llm("pergunta factual"):
                pass

        assert captured_ctx.get("grounded_no_context")
        assert captured_ctx.get("aisearch_enabled") is False
        assert captured_ctx.get("bing_grounding_enabled") is False
