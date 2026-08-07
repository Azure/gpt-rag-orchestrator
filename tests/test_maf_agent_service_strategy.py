"""Tests for MafAgentServiceStrategy (src/strategies/maf_agent_service_strategy.py)."""

import types
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from strategies.agent_strategies import AgentStrategies
from strategies import agent_provider_v2


class TestMafAgentServiceStrategy:
    @pytest.fixture(autouse=True)
    def _patch(self, patch_dependencies, mock_config):
        # Also patch the config import inside the strategy module
        with patch("strategies.maf_agent_service_strategy.get_config", return_value=mock_config):
            yield

    def test_strategy_type(self):
        from strategies.maf_agent_service_strategy import MafAgentServiceStrategy
        s = MafAgentServiceStrategy()
        assert s.strategy_type == AgentStrategies.MAF_AGENT_SERVICE

    def test_prompt_namespace_returns_maf(self):
        from strategies.maf_agent_service_strategy import MafAgentServiceStrategy
        s = MafAgentServiceStrategy()
        assert s._prompt_namespace() == "maf"

    def test_user_profile_container(self):
        from strategies.maf_agent_service_strategy import MafAgentServiceStrategy
        s = MafAgentServiceStrategy()
        assert s.user_profile_container == "conversations"

    @pytest.mark.asyncio
    async def test_hosted_mode_never_reads_or_writes_user_profiles(self):
        from strategies.base_agent_strategy import hosted_runtime_construction
        from strategies.maf_agent_service_strategy import MafAgentServiceStrategy

        with hosted_runtime_construction():
            strategy = MafAgentServiceStrategy()
        strategy._load_user_profile = AsyncMock()
        strategy._save_user_profile = AsyncMock()
        memory = MagicMock()
        memory.flush = AsyncMock()

        created = await strategy._create_user_memory("untrusted-caller")
        await strategy._persist_user_memory("untrusted-caller", memory)

        assert strategy.cosmos is None
        assert strategy.profile_memory_enabled is False
        assert created is None
        strategy._load_user_profile.assert_not_awaited()
        strategy._save_user_profile.assert_not_awaited()
        memory.flush.assert_not_awaited()


class TestMafAgentServiceHostedStateless:
    """Security regression: the hosted runtime is history-blind and
    stateless. It must never bind to a service-managed thread/conversation,
    must replay the caller-supplied ordered history explicitly, and must
    never persist the turn server-side (``store: False``)."""

    @pytest.fixture(autouse=True)
    def _patch(self, patch_dependencies, mock_config):
        with patch(
            "strategies.maf_agent_service_strategy.get_config",
            return_value=mock_config,
        ):
            yield

    def _make_strategy(self):
        from strategies.base_agent_strategy import hosted_runtime_construction
        from strategies.maf_agent_service_strategy import MafAgentServiceStrategy

        with hosted_runtime_construction():
            return MafAgentServiceStrategy()

    @pytest.mark.asyncio
    async def test_hosted_turn_never_binds_a_service_managed_thread(self):
        strategy = self._make_strategy()
        assert strategy.hosted_runtime is True
        strategy.conversation = {
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

        reset_legacy_thread = MagicMock()

        with patch.object(
            agent_provider_v2, "get_provider", AsyncMock(return_value=provider)
        ), patch.object(
            agent_provider_v2,
            "get_or_create_agent_details",
            AsyncMock(return_value=MagicMock()),
        ), patch.object(
            agent_provider_v2, "stream_agent_run", _fake_stream
        ), patch.object(
            agent_provider_v2, "reset_legacy_thread", reset_legacy_thread
        ):
            out = "".join(
                [c async for c in strategy.initiate_agent_flow("follow-up")]
            )

        assert out == "hello"
        # Purely local/ephemeral thread: no service_thread_id requested.
        assert captured_thread_kwargs == [{}]
        # The classic-only legacy-thread reconciliation never runs either.
        reset_legacy_thread.assert_not_called()
        # The complete ordered history plus the current ask was replayed
        # locally, and the turn is explicitly never stored server-side.
        [(run_input, options)] = run_calls
        assert [(m.role.value, m.text) for m in run_input] == [
            ("user", "first question"),
            ("assistant", "first answer"),
            ("user", "follow-up"),
        ]
        assert options == {
            "max_tokens": strategy.max_completion_tokens,
            "store": False,
        }
        # No thread/backend tag leaks into the request-local conversation dict.
        assert "thread_id" not in strategy.conversation
