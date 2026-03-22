from .single_agent_rag_strategy_v2 import SingleAgentRAGStrategyV2
from .maf_agent_service_strategy import MafAgentServiceStrategy
from .maf_lite_strategy import MafLiteStrategy
from .agent_strategies import AgentStrategies
from .mcp_strategy import McpStrategy
from .nl2sql_strategy import NL2SQLStrategy

class AgentStrategyFactory:
    @staticmethod
    async def get_strategy(key: str):
        """
        Return an instance of the strategy class corresponding to the given key.
        """
        if key == AgentStrategies.SINGLE_AGENT_RAG.value:
            return await SingleAgentRAGStrategyV2.create()
        if key == AgentStrategies.MAF_AGENT_SERVICE.value:
            return MafAgentServiceStrategy()
        if key == AgentStrategies.MAF_LITE.value:
            return MafLiteStrategy()
        if key == AgentStrategies.MCP.value:
            return await McpStrategy.create()
        if key == AgentStrategies.NL2SQL.value:
            return NL2SQLStrategy()
        # if key == AgentStrategies.MULTIMODAL.value:
            # return ...
        # if key == AgentStrategies.MULTIAGENT.value:
            # return ...
        raise ValueError(f"Unknown strategy key: {key}")
