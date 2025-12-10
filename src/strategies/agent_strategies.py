from enum import Enum

class AgentStrategies(Enum):

    SINGLE_AGENT_RAG = "single_agent_rag"
    # Add additional strategies here
    MULTIAGENT = "multiagent"
    MCP        = "mcp"
    MULTIMODAL = "multimodal"
    NL2SQL     = "nl2sql"
    REALTIME_VOICE = "realtime"