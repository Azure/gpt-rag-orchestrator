import inspect

from .single_agent_rag_strategy_v2 import SingleAgentRAGStrategyV2
from .maf_agent_service_strategy import MafAgentServiceStrategy
from .maf_lite_strategy import MafLiteStrategy
from .multimodal_strategy import MultimodalStrategy
from .agent_strategies import AgentStrategies
from .mcp_strategy import McpStrategy
from .nl2sql_strategy import NL2SQLStrategy

class AgentStrategyFactory:
    _REGISTRY = {
        AgentStrategies.SINGLE_AGENT_RAG.value: lambda: SingleAgentRAGStrategyV2.create(),
        AgentStrategies.MAF_AGENT_SERVICE.value: lambda: MafAgentServiceStrategy(),
        AgentStrategies.MAF_LITE.value: lambda: MafLiteStrategy(),
        AgentStrategies.MCP.value: lambda: McpStrategy.create(),
        AgentStrategies.NL2SQL.value: lambda: NL2SQLStrategy(),
        AgentStrategies.MULTIMODAL.value: lambda: MultimodalStrategy(),
    }

    @classmethod
    def registered_strategy_names(cls) -> frozenset[str]:
        """Return active strategy keys covered by the shared request audit seam."""
        return frozenset(cls._REGISTRY)

    @staticmethod
    async def get_strategy(key: str):
        """
        Return an instance of the strategy class corresponding to the given key.
        """
        builder = AgentStrategyFactory._REGISTRY.get(key)
        if builder is None:
            raise ValueError(f"Unknown strategy key: {key}")
        strategy = builder()
        return await strategy if inspect.isawaitable(strategy) else strategy
