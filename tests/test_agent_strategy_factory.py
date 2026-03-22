"""Tests for the AgentStrategyFactory (src/strategies/agent_strategy_factory.py)."""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock


class TestAgentStrategyFactory:
    """Factory should return the correct strategy type for each key."""

    @pytest.fixture(autouse=True)
    def _patch_deps(self, patch_dependencies):
        """Ensure Azure singletons are mocked for every test in this class."""

    @pytest.mark.asyncio
    async def test_single_agent_rag_returns_v2(self):
        with patch("strategies.agent_strategy_factory.SingleAgentRAGStrategyV2") as MockV2:
            MockV2.create = AsyncMock(return_value=MagicMock())
            from strategies.agent_strategy_factory import AgentStrategyFactory
            strategy = await AgentStrategyFactory.get_strategy("single_agent_rag")
            MockV2.create.assert_awaited_once()
            assert strategy is MockV2.create.return_value

    @pytest.mark.asyncio
    async def test_maf_agent_service(self):
        with patch("strategies.agent_strategy_factory.MafAgentServiceStrategy") as MockMaf:
            MockMaf.return_value = MagicMock()
            from strategies.agent_strategy_factory import AgentStrategyFactory
            strategy = await AgentStrategyFactory.get_strategy("maf_agent_service")
            MockMaf.assert_called_once()
            assert strategy is MockMaf.return_value

    @pytest.mark.asyncio
    async def test_maf_lite(self):
        with patch("strategies.agent_strategy_factory.MafLiteStrategy") as MockLite:
            MockLite.return_value = MagicMock()
            from strategies.agent_strategy_factory import AgentStrategyFactory
            strategy = await AgentStrategyFactory.get_strategy("maf_lite")
            MockLite.assert_called_once()
            assert strategy is MockLite.return_value

    @pytest.mark.asyncio
    async def test_mcp(self):
        with patch("strategies.agent_strategy_factory.McpStrategy") as MockMcp:
            MockMcp.create = AsyncMock(return_value=MagicMock())
            from strategies.agent_strategy_factory import AgentStrategyFactory
            strategy = await AgentStrategyFactory.get_strategy("mcp")
            MockMcp.create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_nl2sql(self):
        with patch("strategies.agent_strategy_factory.NL2SQLStrategy") as MockSql:
            MockSql.return_value = MagicMock()
            from strategies.agent_strategy_factory import AgentStrategyFactory
            strategy = await AgentStrategyFactory.get_strategy("nl2sql")
            MockSql.assert_called_once()

    @pytest.mark.asyncio
    async def test_unknown_key_raises(self):
        from strategies.agent_strategy_factory import AgentStrategyFactory
        with pytest.raises(ValueError, match="Unknown strategy key"):
            await AgentStrategyFactory.get_strategy("no_such_strategy")

    @pytest.mark.asyncio
    async def test_v1_key_raises(self):
        """The removed V1 key must now raise ValueError."""
        from strategies.agent_strategy_factory import AgentStrategyFactory
        with pytest.raises(ValueError, match="Unknown strategy key"):
            await AgentStrategyFactory.get_strategy("single_agent_rag_v1")
