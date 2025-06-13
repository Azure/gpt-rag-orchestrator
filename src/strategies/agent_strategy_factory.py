from .single_agent_rag_strategy import SingleAgentRAGStrategy
from .agent_strategies import AgentStrategies

class AgentStrategyFactory:
    @staticmethod
    def get_strategy(key: str):
        """
        Return an instance of the strategy class corresponding to the given key.
        """
        if key == AgentStrategies.SINGLE_AGENT_RAG.value:
            return SingleAgentRAGStrategy()
        # if key == AgentStrategies.MULTIAGENT.value:
            # return ...
        # if key == AgentStrategies.MCP.value:
            # return ...
        # if key == AgentStrategies.MULTIMODAL.value:
            # return ...
        # if key == AgentStrategies.NL2SQL.value:
            # return ...
        raise ValueError(f"Unknown strategy key: {key}")
