"""Tests for MafAgentServiceStrategy (src/strategies/maf_agent_service_strategy.py)."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from strategies.agent_strategies import AgentStrategies


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
