from .single_agent_rag_strategy_v1 import SingleAgentRAGStrategyV1
from .agent_strategies import AgentStrategies
from .mcp_strategy import McpStrategy
from .nl2sql_strategy import NL2SQLStrategy

class AgentStrategyFactory:
    @staticmethod
    async def get_strategy(key: str):
        """
        Return an instance of the strategy class corresponding to the given key.
        """
        if key in (AgentStrategies.SINGLE_AGENT_RAG.value, AgentStrategies.SINGLE_AGENT_RAG_V1.value):
            return SingleAgentRAGStrategyV1()
        if key == AgentStrategies.MCP.value:
            return await McpStrategy.create()
        if key == AgentStrategies.NL2SQL.value:
            return NL2SQLStrategy()
        # if key == AgentStrategies.MULTIMODAL.value:
            # return ...
        # if key == AgentStrategies.MULTIAGENT.value:
            # return ...
        raise ValueError(f"Unknown strategy key: {key}")
