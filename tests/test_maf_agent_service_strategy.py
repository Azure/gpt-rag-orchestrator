"""Tests for MafAgentServiceStrategy (src/strategies/maf_agent_service_strategy.py)."""

import pytest
from unittest.mock import patch, MagicMock

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
