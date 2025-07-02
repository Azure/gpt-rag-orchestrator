from .single_agent_rag_strategy import SingleAgentRAGStrategy
from .agent_strategies import AgentStrategies
from .mcp_strategy import McpStrategy
from .nl2sql_strategy import NL2SQLStrategy

class AgentStrategyFactory:
    @staticmethod
    def get_strategy(key: str):
        """
        Return an instance of the strategy class corresponding to the given key.
        """
        if key == AgentStrategies.SINGLE_AGENT_RAG.value:
            return SingleAgentRAGStrategy()
        if key == AgentStrategies.MCP.value:
            return McpStrategy()
        if key == AgentStrategies.NL2SQL.value:
            return NL2SQLStrategy()
        # if key == AgentStrategies.MULTIMODAL.value:
            # return ...
        # if key == AgentStrategies.MULTIAGENT.value:
            # return ...
        raise ValueError(f"Unknown strategy key: {key}")
