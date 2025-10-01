from shared.progress_streamer import ProgressSteps

# Environment variables
ENV_ENVIRONMENT = "ENVIRONMENT"
ENV_O1_ENDPOINT = "O1_ENDPOINT"
ENV_O1_KEY = "O1_KEY"
ENV_MCP_FUNCTION_NAME = "MCP_FUNCTION_NAME"

# App secrets
SECRET_MCP_FUNCTION_KEY = "mcp-host--functionkey"

# MCP tool names
TOOL_AGENTIC_SEARCH = "agentic_search"
TOOL_DATA_ANALYST = "data_analyst"
TOOL_WEB_FETCH = "web_fetch"
TOOL_DOCUMENT_CHAT="document_chat"

# Mapping from tool name to progress step
TOOL_PROGRESS_STEP = {
    TOOL_AGENTIC_SEARCH: ProgressSteps.AGENTIC_SEARCH,
    TOOL_DATA_ANALYST: ProgressSteps.DATA_ANALYSIS,
    TOOL_WEB_FETCH: ProgressSteps.WEB_FETCH,
    TOOL_DOCUMENT_CHAT: ProgressSteps.DOCUMENT_CHAT
}

# Display names for tools
TOOL_DISPLAY_NAME = {
    TOOL_AGENTIC_SEARCH: "Agentic Search",
    TOOL_DATA_ANALYST: "Data Analyst",
    TOOL_WEB_FETCH: "Web Fetch",
    TOOL_DOCUMENT_CHAT: "Document Chat"
}
