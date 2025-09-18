import logging
import os
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient

from shared.util import get_secret
from shared.progress_streamer import ProgressStreamer, STEP_MESSAGES
from orc.graphs.utils import clean_chat_history_for_llm
from orc.graphs.constants import (
    ENV_ENVIRONMENT,
    ENV_MCP_FUNCTION_NAME,
    SECRET_MCP_FUNCTION_KEY,
    TOOL_AGENTIC_SEARCH,
    TOOL_DATA_ANALYST,
    TOOL_WEB_FETCH,
    TOOL_PROGRESS_STEP,
)
from shared.prompts import MCP_SYSTEM_PROMPT


logger = logging.getLogger(__name__)


class MCPExecutor:
    """Handles MCP tool discovery, tool-call planning, and execution."""

    def __init__(
        self,
        organization_id: Optional[str],
        user_id: Optional[str],
        config,
        progress_streamer: Optional[ProgressStreamer] = None,
    ) -> None:
        self.organization_id = organization_id
        self.user_id = user_id
        self.config = config
        self.progress_streamer = progress_streamer

    def _is_local_environment(self) -> bool:
        return os.getenv(ENV_ENVIRONMENT, "").lower() == "local"

    async def _init_mcp_client(self) -> MultiServerMCPClient:
        try:
            mcp_function_secret = get_secret(SECRET_MCP_FUNCTION_KEY)
            mcp_function_name = os.getenv(ENV_MCP_FUNCTION_NAME)
        except Exception as e:
            logger.error(f"Error getting MCP function variables: {str(e)}")
            raise RuntimeError(f"Error getting MCP function variables: {str(e)}")

        if self._is_local_environment():
            mcp_url = "http://localhost:7073/runtime/webhooks/mcp/sse"
        else:
            mcp_url = f"https://{mcp_function_name}.azurewebsites.net/runtime/webhooks/mcp/sse?code={mcp_function_secret}"

        client = MultiServerMCPClient(
            {
                "search": {
                    "url": mcp_url,
                    "transport": "sse",
                }
            }
        )
        return client

    def _emit_progress(self, tool_name: str, percent: int) -> None:
        if not self.progress_streamer:
            return
        step = TOOL_PROGRESS_STEP.get(tool_name)
        if not step:
            return
        self.progress_streamer.emit_progress(step, STEP_MESSAGES[step], percent)

    def _configure_agentic_search_args(
        self, tool_call: Dict[str, Any], state
    ) -> None:
        if tool_call["name"] == TOOL_AGENTIC_SEARCH:
            tool_call["args"].update(
                {
                    "organization_id": self.organization_id,
                    "rewritten_query": state.rewritten_query,
                    "reranker_threshold": self.config.reranker_threshold,
                    "historical_conversation": clean_chat_history_for_llm(state.messages),
                    "web_search_threshold": self.config.web_search_results,
                }
            )

    def _configure_data_analyst_args(self, tool_call: Dict[str, Any], state) -> None:
        if tool_call["name"] == TOOL_DATA_ANALYST:
            tool_call["args"].update(
                {
                    "organization_id": self.organization_id,
                    "code_thread_id": state.code_thread_id,
                    "user_id": self.user_id,
                }
            )

    def _configure_web_fetch_args(self, tool_call: Dict[str, Any], state) -> None:
        if tool_call["name"] == TOOL_WEB_FETCH:
            tool_call["args"].update(
                {
                    "query": state.question
                }
            )

    async def get_tool_calls(
        self,
        state,
        llm: AzureChatOpenAI,
        conversation_data: dict,
    ) -> dict:
        logger.info(f"[MCP] Initializing MCP client")
        client = await self._init_mcp_client()

        logger.info(f"[MCP] Getting tools from MCP client")
        tools = await client.get_tools()
        logger.info(f"[MCP] Found {len(tools)} tools")

        history = conversation_data.get("history", [])
        logger.info(f"[MCP] Retrieved {len(history)} conversation history messages for context")
        clean_history = clean_chat_history_for_llm(history)
        logger.info(
            f"[MCP] Cleaned conversation history: {clean_history[:200] if len(clean_history) > 200 else clean_history}"
        )

        last_mcp_tool_used = state.last_mcp_tool_used
        human_prompt = f"""
        User's question:{state.question}

        Previous tool used (if any):{last_mcp_tool_used}

        Here is the conversation history to determine if the current question is a follow up question to a previous question:
        {clean_history}
        """

        llm_with_tools = llm.bind_tools(tools, tool_choice="any")
        message = [SystemMessage(content=MCP_SYSTEM_PROMPT), HumanMessage(content=human_prompt)]
        try:
            response = await llm_with_tools.ainvoke(message)
            logger.info(
                f"[MCP] Tool calls found: {len(response.tool_calls) if hasattr(response, 'tool_calls') and response.tool_calls else 0}"
            )
            if hasattr(response, "tool_calls") and response.tool_calls:
                logger.info(f"[MCP] Tool calls: {response.tool_calls}")
            else:
                logger.warning(
                    f"[MCP] No tool calls returned by LLM. Response content: {response.content[:200] if hasattr(response, 'content') else 'No content'}..."
                )
                logger.info(f"[MCP] Available tools count: {len(tools)}")
                logger.info(
                    f"[MCP] Available tool names: {[tool.name for tool in tools] if tools else 'No tools'}"
                )
                logger.info(
                    f"[MCP] Human prompt sent to LLM: {human_prompt[:300]}..."
                )
        except Exception as e:
            logger.error(f"[MCP] Error getting tool calls from LLM: {str(e)}")
            raise RuntimeError(f"[MCP] Error getting tool calls from LLM: {str(e)}")

        return {
            "mcp_tool_used": response.tool_calls if hasattr(response, "tool_calls") else [],
        }

    async def execute_tool_calls(self, state) -> dict:
        mcp_tool_used = state.mcp_tool_used
        tool_results: List[Any] = []
        if not mcp_tool_used:
            logger.info("[MCP] No tool calls to execute")
            return {"tool_results": tool_results}

        logger.info(f"[MCP] Executing {len(mcp_tool_used)} tool(s)...")

        client = await self._init_mcp_client()
        mcp_available_tools = await client.get_tools()
        tools_by_name = {t.name: t for t in mcp_available_tools}

        for tool_call in mcp_tool_used:
            tool_name = tool_call["name"]

            self._emit_progress(tool_name, 40)

            if tool_name == TOOL_AGENTIC_SEARCH:
                self._configure_agentic_search_args(tool_call, state)
            elif tool_name == TOOL_DATA_ANALYST:
                self._configure_data_analyst_args(tool_call, state)
            elif tool_name == TOOL_WEB_FETCH:
                self._configure_web_fetch_args(tool_call, state)

            tool = tools_by_name.get(tool_name)
            if tool:
                try:
                    logger.info(f"[MCP] Running {tool_name}...")
                    tool_result = await tool.ainvoke(tool_call["args"])
                    tool_results.append(tool_result)
                    logger.info(f"[MCP] {tool_name} completed successfully")
                except Exception as e:
                    logger.error(f"[MCP] Error executing {tool_name}: {e}")
                    tool_results.append(f"Error: {e}")
            else:
                error_msg = f"[MCP] Tool '{tool_name}' not found in available tools"
                logger.error(error_msg)
                tool_results.append(error_msg)

        if tool_results:
            preview = str(tool_results[0])
            if len(preview) > 200:
                preview = preview[:200] + "..."
            logger.info(f"[MCP] Tool results: {preview}")
        else:
            logger.info("[MCP] No tool results to return")

        return {"tool_results": tool_results}

