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


class TestReusableAgentName:
    def test_uses_configured_agent_id_when_present(self):
        from strategies.single_agent_rag_strategy_v2 import _resolve_agent_name

        cfg = MagicMock()
        cfg.get.return_value = " gptrag-single-agent-rag-b622680c09 "

        assert _resolve_agent_name(cfg) == "gptrag-single-agent-rag-b622680c09"
        cfg.get.assert_called_once_with("AGENT_ID", "")

    def test_uses_stable_default_when_agent_id_is_empty(self):
        from strategies.single_agent_rag_strategy_v2 import (
            DEFAULT_REUSABLE_AGENT_NAME,
            _resolve_agent_name,
        )

        cfg = MagicMock()
        cfg.get.return_value = ""

        assert _resolve_agent_name(cfg) == DEFAULT_REUSABLE_AGENT_NAME


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


class TestStreamAgentRunFallback:
    @pytest.mark.asyncio
    async def test_removes_optional_token_limit_when_rejected(self):
        class FakeAgent:
            def __init__(self):
                self.options = []

            async def run_stream(self, _message, *, thread, options):
                self.options.append(options)
                if len(self.options) == 1:
                    raise RuntimeError("invalid_payload")
                yield types.SimpleNamespace(text="persisted")

        agent = FakeAgent()
        output = [
            chunk.text
            async for chunk in agent_provider_v2.stream_agent_run(
                agent,
                "hello",
                thread=MagicMock(),
                options={"max_tokens": 100},
            )
        ]

        assert output == ["persisted"]
        assert agent.options == [
            {"max_tokens": 100},
            {},
        ]

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
        run_options = []
        persisted_turns = []

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
            run_options.append(kwargs["options"])
            yield types.SimpleNamespace(text="hello")

        async def _persist_turn(conversation_id, user_message, assistant_message):
            persisted_turns.append(
                (conversation_id, user_message, assistant_message)
            )

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
        ), patch.object(
            agent_provider_v2, "persist_conversation_turn", _persist_turn
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
        assert run_options == [
            {"max_tokens": s.max_completion_tokens},
            {"max_tokens": s.max_completion_tokens},
        ]
        assert persisted_turns == [
            ("conv_stable", "first question", "hello"),
            ("conv_stable", "second question", "hello"),
        ]


class TestStreamAgentHostedStateless:
    """The hosted runtime must be history-blind and stateless: zero managed
    Conversations data-plane operations, no service-managed thread, the
    complete caller-supplied ordered history replayed explicitly every turn,
    and no server-side persistence of the turn."""

    @pytest.fixture(autouse=True)
    def _patch(self, patch_dependencies, mock_config):
        with patch(
            "strategies.single_agent_rag_strategy_v2.get_config",
            return_value=mock_config,
        ):
            yield

    def _make_strategy(self):
        from strategies.base_agent_strategy import hosted_runtime_construction
        from strategies.single_agent_rag_strategy_v2 import SingleAgentRAGStrategyV2

        with hosted_runtime_construction(), patch(
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
    async def test_hosted_turn_never_touches_conversations_data_plane(self):
        s = self._make_strategy()
        assert s.hosted_runtime is True
        s.conversation = {
            "id": "caller-supplied-conv-id",
            "messages": [
                {"role": "user", "text": "first question"},
                {"role": "assistant", "text": "first answer"},
            ],
        }

        captured_thread_kwargs = []
        run_calls = []

        def _get_new_thread(**kwargs):
            captured_thread_kwargs.append(kwargs)
            return types.SimpleNamespace(service_thread_id=None)

        agent = MagicMock()
        agent.__aenter__ = AsyncMock(return_value=agent)
        agent.__aexit__ = AsyncMock(return_value=False)
        agent.get_new_thread = _get_new_thread

        provider = MagicMock()
        provider.as_agent = MagicMock(return_value=agent)

        async def _fake_stream(_agent, run_input, *, thread, options):
            run_calls.append((run_input, options))
            yield types.SimpleNamespace(text="hello")

        ensure_conversation_id = AsyncMock()
        persist_conversation_turn = AsyncMock()
        get_openai_client = AsyncMock()

        with patch.object(
            agent_provider_v2, "get_provider", AsyncMock(return_value=provider)
        ), patch.object(
            agent_provider_v2, "get_or_create_agent_details", AsyncMock(return_value=MagicMock())
        ), patch.object(
            agent_provider_v2, "stream_agent_run", _fake_stream
        ), patch.object(
            agent_provider_v2, "ensure_conversation_id", ensure_conversation_id
        ), patch.object(
            agent_provider_v2, "persist_conversation_turn", persist_conversation_turn
        ), patch.object(
            agent_provider_v2, "_get_openai_client", get_openai_client
        ):
            out = "".join([c async for c in s._stream_agent("follow-up")])

        assert out == "hello"
        # No service-managed thread was requested: purely local/ephemeral.
        assert captured_thread_kwargs == [{}]
        # No Conversations data-plane call of any kind.
        ensure_conversation_id.assert_not_awaited()
        persist_conversation_turn.assert_not_awaited()
        get_openai_client.assert_not_awaited()
        # The complete ordered history plus the current ask was replayed
        # locally, and the turn is explicitly never stored server-side.
        [(run_input, options)] = run_calls
        assert [(m.role.value, m.text) for m in run_input] == [
            ("user", "first question"),
            ("assistant", "first answer"),
            ("user", "follow-up"),
        ]
        assert options == {"max_tokens": s.max_completion_tokens, "store": False}
        # No thread/backend tag leaks into the request-local conversation dict.
        assert "thread_id" not in s.conversation
        assert "agent_backend" not in s.conversation


class TestPersistConversationTurn:
    @pytest.mark.asyncio
    async def test_appends_complete_user_and_assistant_messages(self):
        oai = MagicMock()
        oai.conversations.items.create = AsyncMock()

        with patch.object(
            agent_provider_v2, "_get_openai_client", AsyncMock(return_value=oai)
        ), patch.object(agent_provider_v2.uuid, "uuid4") as uuid4:
            uuid4.return_value.hex = "abc123"
            await agent_provider_v2.persist_conversation_turn(
                "conv_stable",
                "remember JADE-7394",
                "JADE-7394",
            )

        oai.conversations.items.create.assert_awaited_once_with(
            "conv_stable",
            items=[
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "remember JADE-7394",
                        }
                    ],
                },
                {
                    "type": "message",
                    "id": "msg_abc123",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "JADE-7394",
                            "annotations": [],
                        }
                    ],
                },
            ],
        )

    @pytest.mark.asyncio
    async def test_reconciles_ambiguous_create_failure_without_duplicate(self):
        oai = MagicMock()
        oai.conversations.items.create = AsyncMock(
            side_effect=RuntimeError("response lost")
        )
        oai.conversations.items.list = AsyncMock(
            return_value=types.SimpleNamespace(
                data=[
                    types.SimpleNamespace(
                        role="assistant",
                        content=[types.SimpleNamespace(text="JADE-7394")],
                    ),
                    types.SimpleNamespace(
                        role="user",
                        content=[
                            types.SimpleNamespace(text="remember JADE-7394")
                        ],
                    ),
                ]
            )
        )

        with patch.object(
            agent_provider_v2, "_get_openai_client", AsyncMock(return_value=oai)
        ):
            await agent_provider_v2.persist_conversation_turn(
                "conv_stable",
                "remember JADE-7394",
                "JADE-7394",
            )

        oai.conversations.items.list.assert_awaited_once_with(
            "conv_stable",
            limit=2,
            order="desc",
        )

    @pytest.mark.asyncio
    async def test_raises_ambiguous_failure_when_tail_does_not_match(self):
        oai = MagicMock()
        oai.conversations.items.create = AsyncMock(
            side_effect=RuntimeError("response lost")
        )
        oai.conversations.items.list = AsyncMock(
            return_value=types.SimpleNamespace(data=[])
        )

        with patch.object(
            agent_provider_v2, "_get_openai_client", AsyncMock(return_value=oai)
        ), pytest.raises(RuntimeError, match="response lost"):
            await agent_provider_v2.persist_conversation_turn(
                "conv_stable",
                "remember JADE-7394",
                "JADE-7394",
            )

    @pytest.mark.asyncio
    async def test_persistence_failure_is_not_emitted_as_answer_text(
        self, patch_dependencies, mock_config
    ):
        from strategies.single_agent_rag_strategy_v2 import SingleAgentRAGStrategyV2

        with patch(
            "strategies.single_agent_rag_strategy_v2.get_config",
            return_value=mock_config,
        ), patch(
            "strategies.single_agent_rag_strategy_v2.get_search_client",
            return_value=MagicMock(),
        ), patch(
            "strategies.single_agent_rag_strategy_v2.get_genai_client",
            return_value=MagicMock(),
        ):
            strategy = SingleAgentRAGStrategyV2()

        strategy.search_client = MagicMock()
        strategy.project_endpoint = "https://example.services.ai.azure.com/api/projects/p"
        strategy.credential = MagicMock()
        strategy.model_name = "chat"
        strategy.conversation = {
            "thread_id": "conv_stable",
            "agent_backend": agent_provider_v2.AGENT_BACKEND_TAG,
        }

        agent = MagicMock()
        agent.__aenter__ = AsyncMock(return_value=agent)
        agent.__aexit__ = AsyncMock(return_value=False)
        agent.get_new_thread.return_value = MagicMock()
        provider = MagicMock()
        provider.as_agent.return_value = agent

        async def _fake_stream(*args, **kwargs):
            yield types.SimpleNamespace(text="completed answer")

        with patch.object(
            agent_provider_v2, "get_provider", AsyncMock(return_value=provider)
        ), patch.object(
            agent_provider_v2,
            "get_or_create_agent_details",
            AsyncMock(return_value=MagicMock()),
        ), patch.object(
            agent_provider_v2, "stream_agent_run", _fake_stream
        ), patch.object(
            agent_provider_v2,
            "persist_conversation_turn",
            AsyncMock(side_effect=RuntimeError("storage unavailable")),
        ):
            stream = strategy._stream_agent("question")
            assert await anext(stream) == "completed answer"
            with pytest.raises(RuntimeError, match="storage unavailable"):
                await anext(stream)
