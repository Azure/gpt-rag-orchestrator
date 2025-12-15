from enum import Enum

class AgentStrategies(Enum):

    SINGLE_AGENT_RAG = "single_agent_rag"
    SINGLE_AGENT_RAG_V1 = "single_agent_rag_v1"
    # Add additional strategies here
    MULTIAGENT = "multiagent"
    MCP        = "mcp"
    MULTIMODAL = "multimodal"
    NL2SQL     = "nl2sql"