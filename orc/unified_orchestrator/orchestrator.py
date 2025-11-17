"""
Main Conversation Orchestrator

This module contains the ConversationOrchestrator class, which is the main entry point
for conversation processing. It coordinates all sub-components and manages the LangGraph
workflow execution.
"""

import os
import logging
import uuid
import time
import json
import traceback
import aiohttp
import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone

from langsmith import traceable
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_openai import AzureChatOpenAI
from langchain_anthropic import ChatAnthropic

from shared.cosmos_db import store_agent_error
from shared.util import get_organization, get_secret
from shared.prompts import MCP_SYSTEM_PROMPT

from .models import ConversationState, OrchestratorConfig
from .state_manager import StateManager
from .context_builder import ContextBuilder
from .query_planner import QueryPlanner
from .mcp_client import MCPClient
from .response_generator import ResponseGenerator
from .utils import log_info, transform_artifacts_to_images, transform_artifacts_to_blobs, get_tool_progress_message


# Configure logging
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("azure.cosmos").setLevel(logging.WARNING)
logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    format="[%(asctime)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    force=True,
)

logger = logging.getLogger(__name__)


class ConversationOrchestrator:
    """
    Unified conversation orchestrator that manages the entire conversation flow.

    This is the main entry point for conversation processing. It coordinates
    all sub-components and manages the LangGraph workflow execution.
    """

    def __init__(
        self, organization_id: str, config: Optional[OrchestratorConfig] = None
    ):
        """
        Initialize the unified orchestrator.

        Args:
            organization_id: Organization identifier
            config: Optional configuration (uses defaults if not provided)
        """
        self.organization_id = organization_id
        self.config = config or OrchestratorConfig()
        self.storage_url = os.environ.get("AZURE_STORAGE_ACCOUNT_URL")

        logger.info(
            f"[ConversationOrchestrator] Initializing for org: {organization_id}"
        )
        logger.info(
            f"[ConversationOrchestrator] Planning model: {self.config.planning_model}"
        )
        logger.info(
            f"[ConversationOrchestrator] Response model: {self.config.response_model}"
        )

        # Initialize LLM clients
        self.planning_llm = self._init_planning_llm()
        self.response_llm = self._init_response_llm()
        self.tool_calling_llm = self._init_tool_calling_llm()

        # Load organization data
        try:
            self.organization_data = get_organization(organization_id)
            logger.info(
                "[ConversationOrchestrator] Successfully loaded organization data"
            )
        except Exception as e:
            logger.error(
                f"[ConversationOrchestrator] Failed to load organization data: {e}"
            )
            self.organization_data = {
                "segmentSynonyms": "",
                "brandInformation": "",
                "industryInformation": "",
                "additionalInstructions": "",
            }

        # Sub-components will be initialized per-request
        # These are set during generate_response_with_progress
        self.state_manager = None
        self.context_builder = None
        self.query_planner = None
        self.mcp_client = None
        self.response_generator = None

        # Request-specific state
        self.current_conversation_id = None
        self.current_conversation_data = None
        self.current_user_info = None
        self.current_user_settings = None
        self.current_user_timezone = None
        self.current_response_text = ""
        self.current_blob_urls = []
        self.current_start_time = 0
        self._progress_queue = []
        self.wrapped_tools = None  # Wrapped tools for bind_tools (built at runtime)

        logger.info("[ConversationOrchestrator] Initialization complete")


    def _init_planning_llm(self) -> AzureChatOpenAI:
        """
        Initialize Azure OpenAI LLM for planning tasks.

        Returns:
            Configured AzureChatOpenAI instance
        """
        logger.info(
            "[ConversationOrchestrator] Initializing planning LLM (Azure OpenAI)"
        )

        endpoint = os.getenv("O1_ENDPOINT")
        api_key = os.getenv("O1_KEY")

        if not endpoint or not api_key:
            raise ValueError("O1_ENDPOINT and O1_KEY environment variables must be set")

        return AzureChatOpenAI(
            temperature=self.config.planning_temperature,
            openai_api_version=self.config.planning_api_version,
            azure_deployment=self.config.planning_model,
            streaming=False,
            timeout=30,
            max_retries=3,
            azure_endpoint=endpoint,
            api_key=api_key,
        )

    def _init_response_llm(self) -> ChatAnthropic:
        """
        Initialize Anthropic Claude Sonnet LLM for response generation with extended thinking.

        Returns:
            Configured ChatAnthropic instance with extended thinking enabled
        """
        logger.info(
            "[ConversationOrchestrator] Initializing response LLM (Claude Sonnet with extended thinking)"
        )

        api_key = os.getenv("ANTHROPIC_API_KEY")

        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable must be set")

        return ChatAnthropic(
            model=self.config.response_model,
            temperature=self.config.response_temperature,
            streaming=True,
            api_key=api_key,
            max_tokens=self.config.response_max_tokens,
            max_retries=3,
            thinking={"type": "enabled", "budget_tokens": self.config.thinking_budget},
        )

    def _init_tool_calling_llm(self) -> ChatAnthropic:
        """
        Initialize Anthropic Claude Haiku LLM for tool calling.

        Uses Haiku for faster and cheaper tool selection decisions.

        Returns:
            Configured ChatAnthropic instance
        """
        logger.info(
            "[ConversationOrchestrator] Initializing tool calling LLM (Claude Haiku)"
        )

        api_key = os.getenv("ANTHROPIC_API_KEY")

        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable must be set")

        return ChatAnthropic(
            model=self.config.tool_calling_model,
            temperature=self.config.tool_calling_temperature,
            streaming=False,
            api_key=api_key,
            max_tokens=self.config.tool_calling_max_tokens,
            max_retries=3,
        )


    def _store_error(
        self,
        error: Exception,
        context: str,
        question: Optional[str] = None,
    ) -> None:
        """
        Centralized error storage to Cosmos DB.

        Args:
            error: The exception that occurred
            context: Context description (e.g., "query_rewrite", "tool_execution")
            question: Optional question that caused the error
        """
        try:
            error_data = {
                "user_id": (
                    self.current_user_info.get("id") if self.current_user_info else None
                ),
                "error": f"{context}: {str(error)}",
                "ask": question or (getattr(self, "current_question", None)),
            }

            # Add conversation_id if available
            if self.current_conversation_id:
                error_data["conversation_id"] = self.current_conversation_id

            # Add organization_id if available
            if self.organization_id:
                error_data["organization_id"] = self.organization_id

            # Add error type and stack trace for detailed errors
            if context in ["query_rewrite_error", "query_augmentation_error"]:
                error_data["error_message"] = str(error)
                error_data["error_type"] = context
                error_data["stack_trace"] = traceback.format_exc()

            store_agent_error(**error_data)
            logger.debug(f"[ErrorHandler] Stored error for context: {context}")

        except Exception as store_error:
            logger.error(
                f"[ErrorHandler] Failed to store error: {store_error}",
                extra={"original_error": str(error), "context": context},
            )


    @traceable(run_type="tool", name="data_analyst_stream")
    async def _stream_data_analyst(
        self,
        query: str,  # rewritten by claude before sending to the mcp server
        organization_id: str,
        code_thread_id: Optional[str],
        user_id: Optional[str],
    ) -> Dict[str, Any]:
        """
        Call data_analyst streaming endpoint and emit thinking tokens.

        Args:
            query: User's query
            organization_id: Organization ID
            code_thread_id: Optional thread ID
            user_id: Optional user ID

        Returns:
            Dict with complete response matching tool result format
        """
        logger.info(
            f"[StreamDataAnalyst] Starting streaming call for query: {query[:100]}"
        )
        is_local = os.getenv("ENVIRONMENT", "").lower() == "local"
        if is_local:
            base_url = "http://localhost:7073"
        else:
            mcp_function_name = os.getenv("MCP_FUNCTION_NAME")
            mcp_function_secret = get_secret("mcp-host--functionkey")
            base_url = f"https://{mcp_function_name}.azurewebsites.net"

        stream_url = f"{base_url}/api/data-analyst-stream"

        payload = {
            "query": query,
            "organization_id": organization_id,
            "code_thread_id": code_thread_id,
            "user_id": user_id,
        }

        # Add function key for production
        headers = {"Content-Type": "application/json"}
        params = {}
        if not is_local:
            params["code"] = mcp_function_secret

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    stream_url,
                    json=payload,
                    headers=headers,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=300),
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(
                            f"[StreamDataAnalyst] HTTP {response.status}: {error_text}"
                        )
                        raise RuntimeError(
                            f"Streaming endpoint returned {response.status}"
                        )

                    logger.info("[StreamDataAnalyst] Connected to SSE stream")

                    complete_data = None
                    buffer = ""
                    async for chunk in response.content.iter_any():
                        if not chunk:
                            continue

                        # Decode and add to buffer
                        buffer += chunk.decode("utf-8")
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line = line.strip()

                            if not line or not line.startswith("data:"):
                                continue

                            # Extract JSON after "data: "
                            json_str = line[5:].strip()

                            if json_str == "[DONE]":
                                break

                            try:
                                chunk_data = json.loads(json_str)
                                chunk_type = chunk_data.get("type")

                                if chunk_type == "thinking":
                                    thinking_data = {
                                        "type": "thinking",
                                        "content": chunk_data.get("content", ""),
                                        "timestamp": time.time(),
                                    }
                                    self._progress_queue.append(
                                        f"__THINKING__{json.dumps(thinking_data)}__THINKING__\n"
                                    )

                                # Forward content tokens to UI (all content from data analyst treated as thinking)
                                elif chunk_type == "content":
                                    content_data = {
                                        "type": "data_analyst_content",
                                        "content": chunk_data.get("content", ""),
                                        "timestamp": time.time(),
                                    }
                                    self._progress_queue.append(
                                        f"__PROGRESS__{json.dumps(content_data)}__PROGRESS__\n"
                                    )

                                elif chunk_type == "complete":
                                    complete_data = chunk_data.get("data", {})
                                    logger.info(
                                        "[StreamDataAnalyst] Received complete event"
                                    )

                                elif chunk_type == "done":
                                    logger.info("[StreamDataAnalyst] Stream done")
                                    break

                                elif chunk_type == "error":
                                    error_msg = chunk_data.get("error", "Unknown error")
                                    logger.error(
                                        f"[StreamDataAnalyst] Stream error: {error_msg}"
                                    )
                                    raise RuntimeError(f"Streaming error: {error_msg}")

                            except json.JSONDecodeError as e:
                                logger.warning(
                                    f"[StreamDataAnalyst] Failed to parse chunk: {json_str[:100]}"
                                )
                                continue

            if not complete_data:
                raise RuntimeError("Stream ended without complete data")

            logger.info(
                f"[StreamDataAnalyst] Complete: success={complete_data.get('success')}, "
                f"artifacts={len(complete_data.get('artifacts', []))}"
            )

            return {
                "success": complete_data.get("success", False),
                "code_thread_id": complete_data.get("container_id", ""),
                "images_processed": transform_artifacts_to_images(
                    complete_data.get("artifacts", [])
                ),
                "blob_urls": transform_artifacts_to_blobs(
                    complete_data.get("artifacts", [])
                ),
                "last_agent_message": complete_data.get("response", ""),
                "error": complete_data.get("error"),
            }

        except Exception as e:
            logger.error(f"[StreamDataAnalyst] Error: {e}", exc_info=True)
            raise


    # ============================================================================
    # Workflow Node Methods
    # ============================================================================

    async def _initialize_node(self, state: ConversationState) -> Dict[str, Any]:
        """
        Initialize node: Load conversation data and extract metadata.

        Loads conversation data using StateManager, extracts existing metadata
        (thread IDs, file refs, last tool), and initializes ConversationState.
        Emits initialization progress (5%).

        Args:
            self: ConversationOrchestrator instance
            state: Current conversation state

        Returns:
            Dictionary with updated state fields
        """
        log_info(
            f"[Initialize Node] Starting - Conv: {self.current_conversation_id}, "
            f"User: {self.current_user_info.get('id')}, Org: {self.organization_id}"
        )

        try:
            progress_data = {
                "type": "progress",
                "step": "initialize",
                "message": "Loading conversation history...",
                "progress": 5,
                "timestamp": time.time(),
            }
            self._progress_queue.append(
                f"__PROGRESS__{json.dumps(progress_data)}__PROGRESS__\n"
            )

            conversation_data = self.state_manager.load_conversation(
                conversation_id=self.current_conversation_id,
                user_timezone=self.current_user_timezone,
            )

            # Extract metadata from conversation data
            code_thread_id = conversation_data.get("code_thread_id")
            last_mcp_tool_used = conversation_data.get("last_mcp_tool_used", "")
            uploaded_file_refs = conversation_data.get("uploaded_file_refs", [])

            logger.info(
                f"[Initialize Node] Loaded conversation with code_thread_id: {code_thread_id}, "
                f"last_tool: {last_mcp_tool_used}, cached_files: {len(uploaded_file_refs)}",
                extra={
                    "conversation_id": self.current_conversation_id,
                    "code_thread_id": code_thread_id,
                    "last_mcp_tool_used": last_mcp_tool_used,
                    "cached_files_count": len(uploaded_file_refs),
                },
            )

            # Store conversation data for later use
            self.current_conversation_data = conversation_data

            return {
                "code_thread_id": code_thread_id,
                "last_mcp_tool_used": last_mcp_tool_used,
                "uploaded_file_refs": uploaded_file_refs,
            }

        except Exception as e:
            logger.error(
                f"[Initialize Node] Error during initialization: {str(e)}",
                extra={
                    "conversation_id": self.current_conversation_id,
                    "user_id": self.current_user_info.get("id"),
                    "organization_id": self.organization_id,
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )

            self._store_error(e, "initialization_error")

            error_data = {
                "type": "error",
                "message": "Failed to load conversation history. Starting fresh conversation.",
                "timestamp": time.time(),
            }
            self._progress_queue.append(
                f"__PROGRESS__{json.dumps(error_data)}__PROGRESS__\n"
            )

            self.current_conversation_data = {
                "start_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "history": [],
                "memory_data": "",
                "interaction": {},
                "type": "default",
                "code_thread_id": None,
                "last_mcp_tool_used": "",
                "uploaded_file_refs": [],
            }
            return {
                "code_thread_id": None,
                "last_mcp_tool_used": "",
                "uploaded_file_refs": [],
            }



    async def _rewrite_node(self, state: ConversationState) -> Dict[str, Any]:
        """
        Rewrite node: Rewrite query with organization context.

        Calls QueryPlanner.rewrite_query() to process the user's question
        with organization-specific context and segment aliases.
        Emits query rewrite progress (15%).

        Args:
            self: ConversationOrchestrator instance
            state: Current conversation state

        Returns:
            Dictionary with rewritten_query
        """
        log_info(f"[Rewrite Node] Starting - Question: {state.question[:100]}...")

        try:
            progress_data = {
                "type": "progress",
                "step": "rewrite",
                "message": "Analyzing your question...",
                "progress": 15,
                "timestamp": time.time(),
            }
            self._progress_queue.append(
                f"__PROGRESS__{json.dumps(progress_data)}__PROGRESS__\n"
            )

            rewrite_result = await self.query_planner.rewrite_query(
                state=state,
                conversation_data=self.current_conversation_data,
                context_builder=self.context_builder,
            )

            log_info(
                f"[Rewrite Node] Complete: '{rewrite_result['rewritten_query'][:100]}...'"
            )

            return {
                "rewritten_query": rewrite_result["rewritten_query"],
            }

        except Exception as e:
            logger.error(
                f"[Rewrite Node] Error during query rewriting: {str(e)}",
                extra={
                    "conversation_id": self.current_conversation_id,
                    "user_id": self.current_user_info.get("id"),
                    "organization_id": self.organization_id,
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )

            self._store_error(e, "query_rewrite_error")

            return {
                "rewritten_query": state.question,
            }



    async def _augment_node(self, state: ConversationState) -> Dict[str, Any]:
        """
        Augment node: Augment query with conversation history.

        Calls QueryPlanner.augment_query() to enhance the query with
        context from previous conversation turns.
        Emits query augmentation progress (25%).

        Args:
            self: ConversationOrchestrator instance
            state: Current conversation state

        Returns:
            Dictionary with augmented_query
        """
        log_info(
            f"[Augment Node] Starting - Rewritten: {state.rewritten_query[:100]}..."
        )

        try:
            progress_data = {
                "type": "progress",
                "step": "augment",
                "message": "Adding conversation context...",
                "progress": 25,
                "timestamp": time.time(),
            }
            self._progress_queue.append(
                f"__PROGRESS__{json.dumps(progress_data)}__PROGRESS__\n"
            )

            augment_result = await self.query_planner.augment_query(
                state=state,
                conversation_data=self.current_conversation_data,
                context_builder=self.context_builder,
            )

            log_info(
                f"[Augment Node] Complete: '{augment_result['augmented_query'][:100]}...'"
            )

            return {
                "augmented_query": augment_result["augmented_query"],
            }

        except Exception as e:
            logger.error(
                f"[Augment Node] Error during query augmentation: {str(e)}",
                extra={
                    "conversation_id": self.current_conversation_id,
                    "user_id": self.current_user_info.get("id"),
                    "organization_id": self.organization_id,
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )

            self._store_error(e, "query_augmentation_error")

            # Fallback to rewritten query (or original if rewrite also failed)
            return {
                "augmented_query": state.rewritten_query or state.question,
            }



    async def _categorize_node(self, state: ConversationState) -> Dict[str, Any]:
        """
        Categorize node: Categorize query into marketing categories.

        Calls QueryPlanner.categorize_query() to classify the query.
        Updates state with query category. Emits categorization progress (30%).

        Args:
            self: ConversationOrchestrator instance
            state: Current conversation state

        Returns:
            Dictionary with query_category
        """
        log_info("[Categorize Node] Starting categorization")

        try:
            progress_data = {
                "type": "progress",
                "step": "categorize",
                "message": "Categorizing your request...",
                "progress": 30,
                "timestamp": time.time(),
            }
            self._progress_queue.append(
                f"__PROGRESS__{json.dumps(progress_data)}__PROGRESS__\n"
            )

            categorize_result = await self.query_planner.categorize_query(
                state=state,
                conversation_data=self.current_conversation_data,
                context_builder=self.context_builder,
            )

            logger.info(
                f"[Categorize Node] Query categorized as: {categorize_result['query_category']}",
                extra={
                    "conversation_id": self.current_conversation_id,
                    "query_category": categorize_result["query_category"],
                },
            )

            return {
                "query_category": categorize_result["query_category"],
            }

        except Exception as e:
            logger.error(
                f"[Categorize Node] Error during query categorization: {str(e)}",
                extra={
                    "conversation_id": self.current_conversation_id,
                    "user_id": self.current_user_info.get("id"),
                    "organization_id": self.organization_id,
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )

            # Store error in Cosmos DB
            self._store_error(e, "query_categorization_error")

            # Emit warning progress
            warning_data = {
                "type": "progress",
                "step": "categorize",
                "message": "Using general category (categorization failed)...",
                "progress": 30,
                "timestamp": time.time(),
            }
            self._progress_queue.append(
                f"__PROGRESS__{json.dumps(warning_data)}__PROGRESS__\n"
            )

            # Fallback to General category
            return {
                "query_category": "General",
            }



    async def _prepare_tools_node(self, state: ConversationState) -> Dict[str, Any]:
        """
        Prepare tools node: Connect to MCP and build wrapped tools.

        Connects to MCP Server, retrieves available tools, wraps them with
        ContextualToolWrapper for context injection, and stores them for use
        in plan_tools node. Excludes document_chat if no documents uploaded.

        Args:
            self: ConversationOrchestrator instance
            state: Current conversation state

        Returns:
            Empty dict (tools stored in self.wrapped_tools)
        """
        log_info("[Prepare Tools Node] Connecting to MCP and building wrapped tools")

        if state.blob_names:
            message = "Preparing document analysis tools..."
        else:
            message = "Preparing tools..."

        progress_data = {
            "type": "progress",
            "step": "tool_preparation",
            "message": message,
            "progress": 35,
            "timestamp": time.time(),
        }
        self._progress_queue.append(
            f"__PROGRESS__{json.dumps(progress_data)}__PROGRESS__\n"
        )

        try:
            await self.mcp_client.connect()
            logger.info("[Prepare Tools Node] Connected to MCP Server")
        except Exception as e:
            logger.error(f"[Prepare Tools Node] Failed to connect to MCP: {e}")
            # Set empty tools list - will skip tool execution
            self.wrapped_tools = []
            return {}

        # Get conversation history for tool context
        conversation_history = self.context_builder.format_conversation_history(
            self.current_conversation_data.get("history", [])
        )

        # Get wrapped tools
        try:
            exclude_doc_chat = len(state.blob_names) == 0
            self.wrapped_tools = await self.mcp_client.get_wrapped_tools(
                state=state,
                conversation_history=conversation_history,
                exclude_document_chat=exclude_doc_chat,
            )

            # Force document_chat if documents uploaded
            if state.blob_names:
                self.wrapped_tools = [
                    t for t in self.wrapped_tools if t.name == "document_chat"
                ]
                logger.info(
                    f"[Prepare Tools Node] Forced document_chat for {len(state.blob_names)} documents"
                )
            # Force data_analyst if data analyst mode is active
            elif state.is_data_analyst_mode:
                self.wrapped_tools = [
                    t for t in self.wrapped_tools if t.name == "data_analyst"
                ]
                logger.info(
                    "[Prepare Tools Node] Forced data_analyst tool (data analyst mode active)"
                )

            logger.info(
                f"[Prepare Tools Node] Prepared {len(self.wrapped_tools)} tools"
            )

        except Exception as e:
            logger.error(f"[Prepare Tools Node] Failed to prepare tools: {e}")
            self.wrapped_tools = []

        return {}



    async def _prepare_messages_node(self, state: ConversationState) -> Dict[str, Any]:
        """
        Prepare messages node: Build initial messages for tool calling.

        Creates system and user messages for the tool execution loop.
        Adds instructions to force document_chat if documents are uploaded.
        Includes formatted conversation history and last tool used in system prompt.

        Args:
            self: ConversationOrchestrator instance
            state: Current conversation state

        Returns:
            Dictionary with initial messages list
        """
        logger.info(
            "[Prepare Messages Node] Building initial messages for tool calling"
        )

        history = self.current_conversation_data.get("history", [])
        formatted_history = self.context_builder.format_conversation_history(history)

        last_tool_used = state.last_mcp_tool_used or ""

        system_msg = MCP_SYSTEM_PROMPT

        if formatted_history:
            system_msg += f"""

    <----------- CONVERSATION HISTORY ------------>
    Here is the conversation history to help you understand the context of the current question and frame a a relevant query for the tool use.
    {formatted_history}
    <----------- END OF CONVERSATION HISTORY ------------>
    """
            logger.info(
                f"[Prepare Messages Node] Added conversation history ({len(history)} messages) to system prompt"
            )

        if last_tool_used:
            system_msg += f"""

    <----------- PREVIOUS TOOL USED ------------>
    The last tool used in this conversation (if available) was: {last_tool_used}

    Consider this when deciding which tool to use for follow-up questions. Most of the time, user would like to continue to use the same tool throughout the session. 
    If user requests a chart after using the data_analyst tool, always trigger the data_analyst tool again to perform the visualization. Don't ask user for the chart requirements. They don't even know. Just make sure the chart looks clear, accurate, and reflect user's intention.
    <----------- END OF PREVIOUS TOOL USED ------------>
    """
            logger.info(
                f"[Prepare Messages Node] Added last tool used ({last_tool_used}) to system prompt"
            )

        messages = [
            SystemMessage(content=system_msg),
            HumanMessage(content=state.question),
        ]

        logger.info(f"[Prepare Messages Node] Created {len(messages)} initial messages")

        return {"messages": messages}



    async def _plan_tools_node(self, state: ConversationState) -> Dict[str, Any]:
        """
        Plan tools node: Claude with bind_tools decides which tools to use.

        Uses bind_tools to let Claude decide which tools to call based on
        the current messages. Returns updated messages with AIMessage containing
        tool_calls if tools should be used.

        Args:
            self: ConversationOrchestrator instance
            state: Current conversation state

        Returns:
            Dictionary with updated messages
        """
        log_info(
            f"[Plan Tools Node] Invoking Claude with {len(self.wrapped_tools or [])} tools"
        )

        progress_data = {
            "type": "progress",
            "step": "tool_planning",
            "message": "Planning tool strategy...",
            "progress": 45,
            "timestamp": time.time(),
        }
        self._progress_queue.append(
            f"__PROGRESS__{json.dumps(progress_data)}__PROGRESS__\n"
        )

        try:
            # If no tools available, skip directly to response generation
            if not self.wrapped_tools:
                logger.warning(
                    "[Plan Tools Node] No tools available, skipping tool calling"
                )
                return {"messages": state.messages}

            tool_names = [t.name for t in self.wrapped_tools]
            logger.info(f"[Plan Tools Node] Available tools: {tool_names}")

            if (
                len(self.wrapped_tools) == 1
                and self.wrapped_tools[0].name == "document_chat"
            ):
                logger.info("[Plan Tools Node] Forcing document_chat tool usage")
                model_with_tools = self.tool_calling_llm.bind_tools(
                    self.wrapped_tools,
                    tool_choice={"type": "tool", "name": "document_chat"},
                )
            elif (
                len(self.wrapped_tools) == 1
                and self.wrapped_tools[0].name == "data_analyst"
            ):
                logger.info("[Plan Tools Node] Forcing data_analyst tool usage")
                model_with_tools = self.tool_calling_llm.bind_tools(
                    self.wrapped_tools,
                    tool_choice={"type": "tool", "name": "data_analyst"},
                )
            else:
                model_with_tools = self.tool_calling_llm.bind_tools(self.wrapped_tools)

            response = await model_with_tools.ainvoke(state.messages)

            if hasattr(response, "tool_calls") and response.tool_calls:
                selected_tools = [
                    tc.get("name", "unknown") for tc in response.tool_calls
                ]
                logger.info(
                    f"[Plan Tools Node] Claude requested {len(response.tool_calls)} tool calls: {selected_tools}"
                )
                if selected_tools:
                    tool_name = selected_tools[0]  # Use first tool for progress message
                    tool_message = get_tool_progress_message(
                        tool_name, "planning"
                    )
                    planning_progress = {
                        "type": "progress",
                        "step": "tool_selected",
                        "message": tool_message,
                        "progress": 50,
                        "timestamp": time.time(),
                        "tool": tool_name,
                    }
                    self._progress_queue.append(
                        f"__PROGRESS__{json.dumps(planning_progress)}__PROGRESS__\n"
                    )
            else:
                logger.warning(
                    "[Plan Tools Node] Claude did not request any tools despite available tools. "
                    f"Response content: {getattr(response, 'content', '')[:200]}"
                )

            return {"messages": state.messages + [response]}

        except Exception as e:
            logger.error(f"[Plan Tools Node] Error invoking Claude: {e}", exc_info=True)
            return {"messages": state.messages}



    async def _execute_tools_node(self, state: ConversationState) -> Dict[str, Any]:
        """
        Execute tools node: Execute tools requested by the model.

        Uses ToolNode to execute the tools that were requested in the most
        recent AIMessage. For data_analyst, uses streaming endpoint directly.
        Returns updated messages with ToolMessage results.

        Args:
            self: ConversationOrchestrator instance
            state: Current conversation state

        Returns:
            Dictionary with updated messages including tool results
        """
        log_info("[Execute Tools Node] Executing requested tools")

        # Detect which tool is being executed from the last message
        tool_name = None
        tool_call_id = None
        tool_args = {}
        if state.messages and len(state.messages) > 0:
            last_message = state.messages[-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                tool_call = last_message.tool_calls[0]
                tool_name = tool_call.get("name")
                tool_call_id = tool_call.get("id")
                tool_args = tool_call.get("args", {})

        # Emit tool-specific execution message
        if tool_name:
            tool_message = get_tool_progress_message(tool_name, "executing")
            progress_data = {
                "type": "progress",
                "step": "tool_execution",
                "message": tool_message,
                "progress": 55,
                "timestamp": time.time(),
                "tool": tool_name,
            }
        else:
            progress_data = {
                "type": "progress",
                "step": "tool_execution",
                "message": "Executing tools...",
                "progress": 55,
                "timestamp": time.time(),
            }

        self._progress_queue.append(
            f"__PROGRESS__{json.dumps(progress_data)}__PROGRESS__\n"
        )

        try:
            if tool_name == "data_analyst":
                logger.info("[Execute Tools Node] Using streaming for data_analyst")

                query = tool_args.get("query", state.question)
                org_id = self.organization_id
                code_thread_id = state.code_thread_id
                user_id = (
                    self.current_user_info.get("id") if self.current_user_info else None
                )

                result_data = await self._stream_data_analyst(
                    query=query,
                    organization_id=org_id,
                    code_thread_id=code_thread_id,
                    user_id=user_id,
                )

                tool_message = ToolMessage(
                    content=json.dumps(result_data),
                    tool_call_id=tool_call_id,
                    name="data_analyst",
                )

                logger.info("[Execute Tools Node] data_analyst streaming complete")

                return {"messages": [tool_message]}

            else:
                tool_node = ToolNode(self.wrapped_tools)
                result = await tool_node.ainvoke(state)

                logger.info(
                    f"[Execute Tools Node] Executed tools, got {len(result.get('messages', []))} result messages"
                )

                return result

        except Exception as e:
            logger.error(f"[Execute Tools Node] Error executing tools: {e}")
            return {}



    async def _extract_context_node(self, state: ConversationState) -> Dict[str, Any]:
        """
        Extract context node: Parse tool results from messages.

        Extracts context documents, blob URLs, and file references from the
        messages that contain tool results. Also extracts metadata like
        last_mcp_tool_used and code_thread_id.

        Args:
            self: ConversationOrchestrator instance
            state: Current conversation state

        Returns:
            Dictionary with extracted context and metadata
        """
        log_info("[Extract Context Node] Extracting context from tool result messages")

        # Detect which tool was executed to show tool-specific processing message
        tool_name = None
        for msg in state.messages:
            if hasattr(msg, "name"):
                tool_name = msg.name
                break

        tool_processing_messages = {
            "agentic_search": "Processing search results...",
            "data_analyst": "Processing data analysis results...",
            "web_fetch": "Processing web content...",
            "document_chat": "Processing document content...",
        }

        message = (
            tool_processing_messages.get(tool_name, "Processing results...")
            if tool_name
            else "Processing results..."
        )

        progress_data = {
            "type": "progress",
            "step": "context_extraction",
            "message": message,
            "progress": 60,
            "timestamp": time.time(),
        }
        if tool_name:
            progress_data["tool"] = tool_name

        self._progress_queue.append(
            f"__PROGRESS__{json.dumps(progress_data)}__PROGRESS__\n"
        )

        try:
            context_docs, blob_urls, uploaded_file_refs = (
                self.context_builder.extract_context_from_messages(state.messages)
            )

            # Extract metadata
            last_mcp_tool_used = ""
            code_thread_id = state.code_thread_id

            for msg in state.messages:
                if hasattr(msg, "name"):
                    last_mcp_tool_used = msg.name

                    # Extract code_thread_id from data_analyst
                    if msg.name == "data_analyst" and hasattr(msg, "content"):
                        content = msg.content
                        if isinstance(content, str):
                            try:
                                result_dict = json.loads(content)
                                if isinstance(result_dict, dict):
                                    code_thread_id = result_dict.get(
                                        "code_thread_id", code_thread_id
                                    )
                            except Exception:
                                pass

            # Store blob URLs for metadata emission
            self.current_blob_urls = blob_urls

            logger.info(
                f"[Extract Context Node] Extracted {len(context_docs)} docs, "
                f"{len(blob_urls)} blobs, {len(uploaded_file_refs)} file refs"
            )

            return {
                "context_docs": context_docs,
                "code_thread_id": code_thread_id,
                "last_mcp_tool_used": last_mcp_tool_used,
                "uploaded_file_refs": (
                    uploaded_file_refs
                    if uploaded_file_refs
                    else state.uploaded_file_refs
                ),
            }

        except Exception as e:
            logger.error(f"[Extract Context Node] Error extracting context: {e}")
            return {
                "context_docs": [],
                "code_thread_id": state.code_thread_id,
                "last_mcp_tool_used": "",
                "uploaded_file_refs": state.uploaded_file_refs,
            }



    async def _generate_response_node(self, state: ConversationState) -> Dict[str, Any]:
        """
        Generate response node: Use Claude to generate final response.

        Builds system and user prompts with extracted context and streams
        the final response from Claude. This is the last step before saving.

        Args:
            self: ConversationOrchestrator instance
            state: Current conversation state

        Returns:
            Empty dict (response stored in self.current_response_text)
        """
        log_info("[Generate Response Node] Generating final response with Claude")

        progress_data = {
            "type": "progress",
            "step": "response_generation",
            "message": "Finalizing Thoughts & Generating Response...",
            "progress": 70,
            "timestamp": time.time(),
        }
        self._progress_queue.append(
            f"__PROGRESS__{json.dumps(progress_data)}__PROGRESS__\n"
        )

        conversation_history = self.context_builder.format_conversation_history(
            self.current_conversation_data.get("history", [])
        )

        system_prompt = self.response_generator.build_system_prompt(
            state=state,
            context_builder=self.context_builder,
            conversation_history=conversation_history,
            user_settings=self.current_user_settings,
        )

        user_prompt = self.response_generator.build_user_prompt(
            state=state, user_settings=self.current_user_settings
        )

        logger.info(
            f"[Generate Response Node] Using temperature: {self.config.response_temperature}"
        )

        response_text = ""
        try:
            async for token in self.response_generator.generate_streaming_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            ):
                response_text += token
                self._progress_queue.append(token)

            logger.info(
                f"[Generate Response Node] Generated {len(response_text)} characters"
            )

        except Exception as e:
            logger.error(f"[Generate Response Node] Error generating response: {e}")
            response_text = "I apologize, but I encountered an error while generating the response. Please try again."
            self._progress_queue.append(response_text)

        # Sanitize and store
        sanitized_response = self.response_generator.sanitize_response(response_text)
        self.current_response_text = sanitized_response

        return {}



    async def _save_node(self, state: ConversationState) -> Dict[str, Any]:
        """
        Save node: Persist conversation to Cosmos DB.

        Updates conversation history with user question and assistant response.
        Includes thoughts and metadata in assistant message. Serializes LangGraph
        memory and saves to Cosmos DB. Emits completion progress (100%).

        Args:
            self: ConversationOrchestrator instance
            state: Current conversation state

        Returns:
            Empty dictionary (no state updates)
        """
        log_info(f"[Save Node] Saving conversation: {self.current_conversation_id}")

        try:
            # Build thoughts for debugging
            thoughts = {
                "model_used": self.config.response_model,
                "query_category": state.query_category,
                "original_query": state.question,
                "rewritten_query": state.rewritten_query,
            }

            if state.last_mcp_tool_used:
                thoughts["mcp_tool_used"] = state.last_mcp_tool_used

            if state.context_docs:
                flattened_docs = []
                for item in state.context_docs:
                    if isinstance(item, list):
                        flattened_docs.extend(item)
                    else:
                        flattened_docs.append(item)
                thoughts["context_docs"] = (
                    flattened_docs if flattened_docs else state.context_docs
                )

            # Emit metadata
            metadata = {
                "conversation_id": self.current_conversation_id,
                "thoughts": thoughts,
                "images_blob_urls": self.current_blob_urls,
            }
            self._progress_queue.append(
                f"__METADATA__{json.dumps(metadata)}__METADATA__\n"
            )
            logger.debug(
                "[Save Node] Emitted metadata with thoughts",
                extra={
                    "conversation_id": self.current_conversation_id,
                    "thoughts_keys": list(thoughts.keys()),
                    "blob_urls_count": len(self.current_blob_urls),
                },
            )

            response_time = time.time() - self.current_start_time

            # Save conversation using StateManager
            self.state_manager.save_conversation(
                conversation_id=self.current_conversation_id,
                conversation_data=self.current_conversation_data,
                state=state,
                user_info=self.current_user_info,
                response_time=response_time,
                response_text=self.current_response_text,
                thoughts=thoughts,
            )

            logger.info(
                f"[Save Node] Conversation saved (response_time: {response_time:.2f}s)",
                extra={
                    "conversation_id": self.current_conversation_id,
                    "response_time": response_time,
                    "response_length": len(self.current_response_text),
                },
            )

        except Exception as e:
            logger.error(
                f"[Save Node] Error saving conversation: {str(e)}",
                extra={
                    "conversation_id": self.current_conversation_id,
                    "user_id": self.current_user_info.get("id"),
                    "organization_id": self.organization_id,
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )

            # Store error in Cosmos DB
            self._store_error(e, "conversation_save_error")

            # Don't fail the entire request if save fails
            # The response has already been generated and streamed
            logger.warning("[Save Node] Continuing despite save error")

        # Emit completion progress
        progress_data = {
            "type": "progress",
            "step": "complete",
            "message": "Complete",
            "progress": 100,
            "timestamp": time.time(),
        }
        self._progress_queue.append(
            f"__PROGRESS__{json.dumps(progress_data)}__PROGRESS__\n"
        )

        # Return empty dict (no state updates)
        return {}



    def _build_graph(self, memory: MemorySaver) -> StateGraph:
        """
        Construct the LangGraph workflow.

        Defines graph nodes (initialize, rewrite, categorize, generate, save)
        and edges between nodes. Uses ConversationState as state object.

        Args:
            memory: MemorySaver instance for checkpointing

        Returns:
            Compiled StateGraph
        """
        logger.info(
            "[ConversationOrchestrator] Building LangGraph workflow",
            extra={
                "conversation_id": self.current_conversation_id,
                "organization_id": self.organization_id,
            },
        )

        graph = StateGraph(ConversationState)

        # Add nodes - use functools.partial to bind orchestrator to async node functions
        graph.add_node("initialize", self._initialize_node)
        graph.add_node("rewrite", self._rewrite_node)
        graph.add_node("augment", self._augment_node)
        graph.add_node("categorize", self._categorize_node)
        graph.add_node("prepare_tools", self._prepare_tools_node)
        graph.add_node("prepare_messages", self._prepare_messages_node)
        graph.add_node("plan_tools", self._plan_tools_node)
        graph.add_node("execute_tools", self._execute_tools_node)
        graph.add_node("extract_context", self._extract_context_node)
        graph.add_node("generate_response", self._generate_response_node)
        graph.add_node("save", self._save_node)

        logger.debug(
            "[ConversationOrchestrator] Added 11 nodes to graph",
            extra={"conversation_id": self.current_conversation_id},
        )

        # Define routing function for tool execution
        def route_after_tool_planning(state: ConversationState) -> str:
            """Route to execute_tools if tools were planned, otherwise to extract_context"""
            # Check if the last message has tool calls
            if state.messages and len(state.messages) > 0:
                last_message = state.messages[-1]
                if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                    return "execute_tools"
            return "extract_context"

        # Define edges
        graph.add_edge(START, "initialize")
        graph.add_edge("initialize", "rewrite")
        graph.add_edge("rewrite", "augment")
        graph.add_edge("augment", "categorize")
        graph.add_edge("categorize", "prepare_tools")
        graph.add_edge("prepare_tools", "prepare_messages")
        graph.add_edge("prepare_messages", "plan_tools")

        # Conditional edge: route based on whether tools were planned
        graph.add_conditional_edges(
            "plan_tools",
            route_after_tool_planning,
            {
                "execute_tools": "execute_tools",
                "extract_context": "extract_context",
            },
        )

        # After tool execution, go directly to context extraction
        graph.add_edge("execute_tools", "extract_context")
        graph.add_edge("extract_context", "generate_response")
        graph.add_edge("generate_response", "save")
        graph.add_edge("save", END)

        logger.debug(
            "[ConversationOrchestrator] Defined graph edges with single-pass tool execution",
            extra={"conversation_id": self.current_conversation_id},
        )
        compiled_graph = graph.compile(checkpointer=memory)

        logger.info(
            "[ConversationOrchestrator] LangGraph workflow built successfully",
            extra={"conversation_id": self.current_conversation_id},
        )
        return compiled_graph


    async def _stream_graph_execution(
        self, graph: StateGraph, state: ConversationState, config: Dict[str, Any]
    ):
        """
        Execute graph with streaming progress updates.

        Uses LangGraph's astream_events for fine-grained streaming.
        Processes events and emits progress updates.

        Args:
            graph: Compiled LangGraph workflow
            state: Initial conversation state
            config: LangGraph configuration

        Yields:
            Progress updates and events from the graph execution
        """
        logger.info(
            "[ConversationOrchestrator] Starting graph execution with streaming",
            extra={
                "conversation_id": self.current_conversation_id,
                "user_id": self.current_user_info.get("id"),
                "organization_id": self.organization_id,
                "question": state.question[:100],
            },
        )

        try:
            output_queue = asyncio.Queue()
            graph_done = asyncio.Event()

            async def progress_monitor():
                """Monitor progress queue and forward items."""
                try:
                    while not graph_done.is_set():
                        while self._progress_queue:
                            item = self._progress_queue.pop(0)
                            await output_queue.put(("progress", item))

                        # Small delay to avoid busy waiting
                        await asyncio.sleep(0.05)

                    # Final flush after graph completes
                    while self._progress_queue:
                        item = self._progress_queue.pop(0)
                        await output_queue.put(("progress", item))

                except Exception as e:
                    logger.error(f"Progress monitor error: {e}", exc_info=True)

            async def graph_executor():
                """Execute graph and forward events."""
                try:
                    async for event in graph.astream_events(
                        state, config, version="v2"
                    ):
                        event_type = event.get("event", "")

                        if event_type == "on_chain_start":
                            node_name = event.get("name", "")
                            logger.info(f"[Graph Event] Node started: {node_name}")
                        elif event_type == "on_chain_end":
                            node_name = event.get("name", "")
                            logger.info(f"[Graph Event] Node completed: {node_name}")
                        elif event_type == "on_chain_error":
                            error = event.get("data", {}).get("error", "Unknown error")
                            logger.error(f"[Graph Event] Error: {error}")
                            raise RuntimeError(f"Graph execution error: {error}")

                except Exception as e:
                    await output_queue.put(("error", e))
                finally:
                    graph_done.set()
                    await output_queue.put(("done", None))

            # Start both tasks
            monitor_task = asyncio.create_task(progress_monitor())
            graph_task = asyncio.create_task(graph_executor())

            # Yield items as they become available
            while True:
                try:
                    item_type, item_data = await asyncio.wait_for(
                        output_queue.get(), timeout=0.1
                    )

                    if item_type == "progress":
                        yield item_data
                    elif item_type == "error":
                        raise item_data
                    elif item_type == "done":
                        break

                except asyncio.TimeoutError:
                    if graph_done.is_set() and output_queue.empty():
                        break
                    continue

            # Ensure both tasks complete
            await graph_task
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

            # Yield any remaining items in the progress queue
            while self._progress_queue:
                item = self._progress_queue.pop(0)
                yield item

            logger.info(
                "[ConversationOrchestrator] Graph execution completed successfully",
                extra={
                    "conversation_id": self.current_conversation_id,
                    "total_time": time.time() - self.current_start_time,
                },
            )

        except Exception as e:
            logger.error(
                f"[ConversationOrchestrator] Error during graph execution: {str(e)}",
                extra={
                    "conversation_id": self.current_conversation_id,
                    "user_id": self.current_user_info.get("id"),
                    "organization_id": self.organization_id,
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )
            raise


    @traceable(run_type="llm")
    async def generate_response_with_progress(
        self,
        conversation_id: str,
        question: str,
        user_info: Dict[str, Any],
        user_settings: Optional[Dict[str, Any]] = None,
        user_timezone: Optional[str] = None,
        blob_names: Optional[List[str]] = None,
        is_data_analyst_mode: Optional[bool] = None,
    ):
        """
        Main entry point for generating responses with progress streaming.

        This method orchestrates the entire conversation flow:
        1. Initialize sub-components per-request
        2. Build LangGraph workflow
        3. Execute graph with streaming
        4. Yield progress updates, metadata, and response tokens
        5. Handle errors gracefully

        Args:
            conversation_id: Conversation identifier (generated if None)
            question: User's question
            user_info: User information (id, name)
            user_settings: User preferences (temperature, model, detail_level)
            user_timezone: User's timezone
            blob_names: List of uploaded file names
            is_data_analyst_mode: Whether data analyst mode is active

        Yields:
            Progress updates (__PROGRESS__), metadata (__METADATA__), and response tokens
        """
        start_time = time.time()
        conversation_id = conversation_id or str(uuid.uuid4())
        blob_names = blob_names or []
        user_settings = user_settings or {}
        is_data_analyst_mode = is_data_analyst_mode or False

        log_info(f"[ConversationOrchestrator] Starting conversation: {conversation_id}")
        log_info(f"[ConversationOrchestrator] Question: {question[:100]}...")
        log_info(
            f"[ConversationOrchestrator] User: {user_info.get('id')}, Org: {self.organization_id}"
        )

        self.current_conversation_id = conversation_id
        self.current_user_info = user_info
        self.current_user_settings = user_settings
        self.current_user_timezone = user_timezone
        self.current_start_time = start_time
        self.current_response_text = ""
        self.current_blob_urls = []
        self._progress_queue = []
        self.current_question = question  # Store for error handling

        # Override config temperature with user setting if provided
        if "temperature" in user_settings:
            self.config.response_temperature = user_settings["temperature"]
            logger.info(
                f"[ConversationOrchestrator] Using user temperature: {self.config.response_temperature}"
            )

        try:
            user_id = user_info.get("id")

            self.state_manager = StateManager(
                organization_id=self.organization_id, user_id=user_id
            )

            self.context_builder = ContextBuilder(
                organization_data=self.organization_data
            )

            self.query_planner = QueryPlanner(
                llm=self.planning_llm, organization_data=self.organization_data
            )

            self.mcp_client = MCPClient(
                organization_id=self.organization_id,
                user_id=user_id,
                config=self.config,
            )

            self.response_generator = ResponseGenerator(
                claude_llm=self.response_llm,
                organization_data=self.organization_data,
                storage_url=self.storage_url,
            )

            logger.info(
                "[ConversationOrchestrator] Sub-components initialized",
                extra={
                    "conversation_id": conversation_id,
                    "user_id": user_id,
                    "organization_id": self.organization_id,
                },
            )

            initial_state = ConversationState(
                question=question,
                blob_names=blob_names,
                is_data_analyst_mode=is_data_analyst_mode,
            )

            # Create memory saver for checkpointing
            memory = MemorySaver()
            graph = self._build_graph(memory)

            # Create configuration for graph execution
            config = {"configurable": {"thread_id": conversation_id}}

            logger.info(
                "[ConversationOrchestrator] Starting graph execution",
                extra={
                    "conversation_id": conversation_id,
                    "thread_id": conversation_id,
                    "blob_names_count": len(blob_names),
                },
            )

            async for item in self._stream_graph_execution(
                graph, initial_state, config
            ):
                for handler in logging.root.handlers:
                    handler.flush()

                yield item

            logger.info(
                f"[ConversationOrchestrator] Conversation completed successfully "
                f"(total_time: {time.time() - start_time:.2f}s)"
            )

        except Exception as e:
            logger.error(
                f"[ConversationOrchestrator] Error in response generation: {str(e)}",
                extra={
                    "conversation_id": conversation_id,
                    "user_id": user_info.get("id"),
                    "organization_id": self.organization_id,
                    "question": question[:100],
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )

            # Store error for debugging
            self._store_error(e, "orchestrator_error")

            # Emit user-friendly error message
            error_data = {
                "type": "error",
                "message": "I'm sorry, I encountered an error while processing your request. Please try again.",
                "timestamp": time.time(),
            }
            yield f"__PROGRESS__{json.dumps(error_data)}__PROGRESS__\n"
