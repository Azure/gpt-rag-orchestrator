"""Tests for the AgentStrategies enum (src/strategies/agent_strategies.py)."""

import pytest
from strategies.agent_strategies import AgentStrategies


class TestAgentStrategiesEnum:
    """Verify the enum contains exactly the expected members after the V1 removal and MAF rename."""

    def test_single_agent_rag_exists(self):
        assert AgentStrategies.SINGLE_AGENT_RAG.value == "single_agent_rag"

    def test_maf_agent_service_exists(self):
        assert AgentStrategies.MAF_AGENT_SERVICE.value == "maf_agent_service"

    def test_maf_lite_exists(self):
        assert AgentStrategies.MAF_LITE.value == "maf_lite"

    def test_mcp_exists(self):
        assert AgentStrategies.MCP.value == "mcp"

    def test_nl2sql_exists(self):
        assert AgentStrategies.NL2SQL.value == "nl2sql"

    def test_multimodal_placeholder_exists(self):
        assert AgentStrategies.MULTIMODAL.value == "multimodal"

    def test_multiagent_placeholder_exists(self):
        assert AgentStrategies.MULTIAGENT.value == "multiagent"

    # --- Removed members ---

    def test_single_agent_rag_v1_removed(self):
        with pytest.raises(AttributeError):
            _ = AgentStrategies.SINGLE_AGENT_RAG_V1

    def test_maf_old_key_removed(self):
        """The bare ``MAF`` enum key was renamed to ``MAF_AGENT_SERVICE``."""
        with pytest.raises(AttributeError):
            _ = AgentStrategies.MAF
