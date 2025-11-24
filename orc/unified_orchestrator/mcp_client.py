"""
MCP Client Module

This module handles MCP Server connection and tool execution.
It manages tool discovery, context injection, and tool wrapping for LangChain integration.
"""

import os
import logging
from typing import List, Optional, Dict, Any

from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from shared.util import get_secret
from .models import ConversationState, OrchestratorConfig

logger = logging.getLogger(__name__)


class MCPClient:
    """
    Handles MCP Server connection and tool execution.

    Responsibilities:
    - Connect to MCP Server via SSE
    - Discover available tools
    - Configure tool arguments with context
    - Execute tools and return results
    """

    TOOL_AGENTIC_SEARCH = "agentic_search"
    TOOL_DATA_ANALYST = "data_analyst"
    TOOL_WEB_FETCH = "web_fetch"
    TOOL_DOCUMENT_CHAT = "document_chat"

    def __init__(self, organization_id: str, user_id: str, config: OrchestratorConfig):
        """
        Initialize MCPClient.

        Args:
            organization_id: Organization identifier
            user_id: User identifier
            config: Orchestrator configuration
        """
        self.organization_id = organization_id
        self.user_id = user_id
        self.config = config
        self.client = None
        logger.info(f"[MCPClient] Initialized for org: {organization_id}")

    def _is_local_environment(self) -> bool:
        """
        Check if running in local environment.

        Returns:
            True if local environment, False otherwise
        """
        return os.getenv("ENVIRONMENT", "").lower() == "local"

    def _get_mcp_url(self) -> str:
        """
        Build MCP Server URL based on environment.

        Returns:
            MCP Server URL

        Raises:
            ValueError: If required environment variables are missing
        """
        if self._is_local_environment():
            mcp_url = "http://localhost:7073/runtime/webhooks/mcp/sse"
            logger.info(f"[MCPClient] Using local MCP URL: {mcp_url}")
        else:
            try:
                mcp_function_name = os.getenv("MCP_FUNCTION_NAME")
                mcp_function_secret = get_secret("mcp-host--functionkey")

                if not mcp_function_name:
                    raise ValueError("MCP_FUNCTION_NAME environment variable not set")

                mcp_url = f"https://{mcp_function_name}.azurewebsites.net/runtime/webhooks/mcp/sse?code={mcp_function_secret}"  # todo: check to see if we can add the function code in a more secured way
                logger.info(
                    f"[MCPClient] Using production MCP URL for function: {mcp_function_name}"
                )

            except Exception as e:
                logger.error(f"[MCPClient] Error getting MCP configuration: {e}")
                raise ValueError(f"Failed to get MCP configuration: {e}")

        return mcp_url

    async def connect(self) -> None:
        """
        Establish SSE connection to MCP Server.

        Initializes MultiServerMCPClient with SSE transport and connects
        to the MCP Server. Handles connection errors gracefully.

        Raises:
            ConnectionError: If connection fails
        """
        logger.info("[MCPClient] Connecting to MCP Server")

        try:
            mcp_url = self._get_mcp_url()
            self.client = MultiServerMCPClient(
                {
                    "search": {  # todo: change mcp server name later
                        "url": mcp_url,
                        "transport": "sse",
                    }
                }
            )

            logger.info("[MCPClient] Successfully connected to MCP Server")

        except Exception as e:
            logger.error(f"[MCPClient] Failed to connect to MCP Server: {e}")
            raise ConnectionError(f"Failed to connect to MCP Server: {e}")

    async def get_available_tools(
        self, exclude_document_chat: bool = False
    ) -> List[Any]:
        """
        Discover tools from MCP Server.

        Retrieves available tools from the MCP Server and optionally filters
        out document_chat when no documents are present. Caches tools to avoid
        repeated discovery calls.

        Args:
            exclude_document_chat: Whether to exclude document_chat tool

        Returns:
            List of available tools

        Raises:
            RuntimeError: If client not connected or tool discovery fails
        """
        logger.info("[MCPClient] Getting available tools")

        if not self.client:
            logger.error("[MCPClient] Client not connected, call connect() first")
            raise RuntimeError("MCP Client not connected. Call connect() first.")

        try:
            tools = await self.client.get_tools()
            logger.info(f"[MCPClient] Discovered {len(tools)} tools")

            # Filter out document_chat when no documents provided
            if exclude_document_chat:
                before_count = len(tools)
                tools = [
                    t
                    for t in tools
                    if getattr(t, "name", "") != self.TOOL_DOCUMENT_CHAT
                ]
                after_count = len(tools)

                if after_count != before_count:
                    logger.info(
                        f"[MCPClient] Filtered out '{self.TOOL_DOCUMENT_CHAT}' tool "
                        f"({before_count} -> {after_count} tools)"
                    )

            logger.info(f"[MCPClient] Returning {len(tools)} available tools")
            return tools

        except Exception as e:
            logger.error(f"[MCPClient] Error getting tools: {e}")
            raise RuntimeError(f"Failed to get tools from MCP Server: {e}")

    def _create_contextual_tool(
        self, mcp_tool: Any, state: ConversationState, context: Dict[str, Any]
    ) -> StructuredTool:
        """
        Creates a LangChain StructuredTool that wraps an MCP tool with context injection.

        This allows LLM-driven tool selection while ensuring that organization-specific
        context (org_id, user_id, conversation history, etc.) is automatically injected
        into tool arguments before execution.

        Args:
            mcp_tool: Original MCP tool from MultiServerMCPClient
            state: Current conversation state with query/blob info
            context: Context dict with org_id, user_id, history, etc.

        Returns:
            StructuredTool that can be used with bind_tools
        """
        tool_name = mcp_tool.name
        logger.debug(f"[MCPClient] Creating contextual wrapper for: {tool_name}")

        def _validate_blob_names() -> bool:
            """Check if current blob names match cached file refs - document chat tool only"""
            current_blobs = set(state.blob_names)
            cached_blobs = set(
                ref.get("blob_name", "") for ref in state.uploaded_file_refs
            )
            return current_blobs == cached_blobs

        async def wrapped_func(**kwargs: Any) -> Any:
            """Execute tool with auto-injected context."""
            logger.info(f"[MCPClient] Executing {tool_name} with context injection")

            if tool_name == self.TOOL_AGENTIC_SEARCH:
                kwargs.update(
                    {
                        "organization_id": context["organization_id"],
                        "rewritten_query": state.rewritten_query,
                        "reranker_threshold": context.get("reranker_threshold", 2.0),
                        "historical_conversation": context.get(
                            "conversation_history", ""
                        ),
                        "web_search_threshold": context.get("web_search_threshold", 2),
                    }
                )
                logger.debug(
                    f"[MCPClient] Injected context for agentic_search: "
                    f"org_id={context['organization_id']}, "
                    f"rewritten_query={state.rewritten_query[:50]}..."
                )

            elif tool_name == self.TOOL_DATA_ANALYST:
                kwargs.update(
                    {
                        "organization_id": context["organization_id"],
                        "code_thread_id": state.code_thread_id,
                        "user_id": context["user_id"],
                    }
                )
                logger.debug(
                    f"[MCPClient] Injected context for data_analyst: "
                    f"org_id={context['organization_id']}, "
                    f"user_id={context['user_id']}, "
                    f"code_thread_id={state.code_thread_id}"
                )

            elif tool_name == self.TOOL_DOCUMENT_CHAT:
                doc_kwargs = {
                    "document_names": state.blob_names,
                }

                # Only send cached_file_info if blob names match exactly
                if _validate_blob_names() and state.uploaded_file_refs:
                    doc_kwargs["cached_file_info"] = state.uploaded_file_refs
                    logger.info(
                        f"[MCPClient] Reusing {len(state.uploaded_file_refs)} "
                        "cached files for document_chat"
                    )
                else:
                    logger.info(
                        f"[MCPClient] Processing {len(state.blob_names)} "
                        "fresh documents for document_chat"
                    )

                kwargs.update(doc_kwargs)
                logger.debug(
                    f"[MCPClient] Injected context for document_chat: "
                    f"{len(state.blob_names)} documents"
                )

            elif tool_name == self.TOOL_WEB_FETCH:
                logger.debug("[MCPClient] web_fetch will use query provided by Claude")

            else:
                logger.warning(
                    f"[MCPClient] Unknown tool '{tool_name}', "
                    "no context injection applied"
                )

            logger.info(f"[MCPClient] Invoking {tool_name}...")
            result = await mcp_tool.ainvoke(kwargs)

            # Log result preview
            result_preview = str(result)
            if len(result_preview) > 200:
                result_preview = result_preview[:200] + "..."
            logger.info(f"[MCPClient] {tool_name} completed: {result_preview}")

            return result

        # Create StructuredTool from the wrapped function
        return StructuredTool.from_function(
            coroutine=wrapped_func,
            name=tool_name,
            description=mcp_tool.description,
            args_schema=getattr(mcp_tool, "args_schema", None),
        )

    async def get_wrapped_tools(
        self,
        state: ConversationState,
        conversation_history: str = "",
        exclude_document_chat: bool = False,
    ) -> List[StructuredTool]:
        """
        Get MCP tools wrapped with context injection.

        This method retrieves available tools from the MCP server and wraps
        them with StructuredTool to auto-inject organization-specific
        context during execution.

        Args:
            state: Current conversation state
            conversation_history: Formatted conversation history
            exclude_document_chat: Whether to exclude document_chat tool

        Returns:
            List of StructuredTool instances ready for bind_tools

        Raises:
            RuntimeError: If client not connected or tool wrapping fails
        """
        logger.info("[MCPClient] Getting wrapped tools for bind_tools")

        available_tools = await self.get_available_tools(
            exclude_document_chat=exclude_document_chat
        )

        # Build context dict for injection
        context = {
            "organization_id": self.organization_id,
            "user_id": self.user_id,
            "conversation_history": conversation_history,
            "reranker_threshold": self.config.reranker_threshold,
            "web_search_threshold": self.config.web_search_results,
        }

        # Wrap each tool using _create_contextual_tool
        wrapped_tools = [
            self._create_contextual_tool(tool, state, context)
            for tool in available_tools
        ]

        logger.info(
            f"[MCPClient] Wrapped {len(wrapped_tools)} tools with context injection"
        )

        return wrapped_tools
