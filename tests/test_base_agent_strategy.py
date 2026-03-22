"""Tests for BaseAgentStrategy (src/strategies/base_agent_strategy.py)."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from strategies.agent_strategies import AgentStrategies


class TestBaseAgentStrategy:
    @pytest.fixture(autouse=True)
    def _patch(self, patch_dependencies):
        yield

    def _make_concrete(self):
        """Create a minimal concrete subclass of BaseAgentStrategy for testing."""
        from strategies.base_agent_strategy import BaseAgentStrategy

        class ConcreteStrategy(BaseAgentStrategy):
            async def initiate_agent_flow(self, user_message):
                yield "test"

        s = ConcreteStrategy()
        s.strategy_type = AgentStrategies.SINGLE_AGENT_RAG
        return s

    def test_prompt_namespace_returns_strategy_value(self):
        s = self._make_concrete()
        assert s._prompt_namespace() == "single_agent_rag"

    def test_prompt_namespace_maf_agent_service(self):
        s = self._make_concrete()
        s.strategy_type = AgentStrategies.MCP
        assert s._prompt_namespace() == "mcp"

    def test_prompt_dir_raises_for_missing_dir(self):
        s = self._make_concrete()
        s.strategy_type = AgentStrategies.MULTIAGENT  # no prompts dir
        with pytest.raises(FileNotFoundError):
            s._prompt_dir()
