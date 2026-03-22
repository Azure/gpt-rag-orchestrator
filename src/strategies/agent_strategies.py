from enum import Enum

class AgentStrategies(Enum):

    SINGLE_AGENT_RAG = "single_agent_rag"
    # Add additional strategies here
    MULTIAGENT         = "multiagent"
    MCP                = "mcp"
    MAF_AGENT_SERVICE  = "maf_agent_service"
    MAF_LITE           = "maf_lite"
    MULTIMODAL         = "multimodal"
    NL2SQL             = "nl2sql"