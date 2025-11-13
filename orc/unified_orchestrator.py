"""
Unified Conversation Orchestrator

This module provides a streamlined, unified orchestrator that merges the previously
separated graph and orchestrator layers into a single cohesive component. It uses
LangGraph for workflow orchestration while connecting to MCP tools as a standard client.

Architecture:
- ConversationOrchestrator: Main entry point for conversation processing
- StateManager: Manages conversation state and persistence
- QueryPlanner: Handles query rewriting, augmentation, and categorization
- MCPClient: Connects to MCP Server and executes tools
- ContextBuilder: Formats organization data and conversation history
- ResponseGenerator: Generates streaming LLM responses
"""

import os
import logging
import uuid
import re
import time
import json
import traceback
import aiohttp
import asyncio
import queue
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Annotated
from datetime import datetime, timezone
from dotenv import load_dotenv

from langsmith import traceable
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import StructuredTool
from langchain_openai import AzureChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_mcp_adapters.client import MultiServerMCPClient

from shared.cosmos_db import (
    get_conversation_data,
    update_conversation_data,
    store_agent_error,
)
from shared.util import (
    get_organization,
    get_verbosity_instruction,
    get_secret,
)
from shared.prompts import (
    MARKETING_ANSWER_PROMPT,
    CREATIVE_BRIEF_PROMPT,
    MARKETING_PLAN_PROMPT,
    BRAND_POSITION_STATEMENT_PROMPT,
    CREATIVE_COPYWRITER_PROMPT,
    QUERY_REWRITING_PROMPT,
    AUGMENTED_QUERY_PROMPT,
    FA_HELPDESK_PROMPT,
    MCP_SYSTEM_PROMPT,
)

load_dotenv()

logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("azure.cosmos").setLevel(logging.WARNING)
logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    format="[%(asctime)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    force=True,
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def log_info(message: str, **kwargs):
    """Log message with both logger and print for visibility in Azure Functions"""
    logger.info(message, **kwargs)
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    print(f"[{timestamp}] {message}")


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class ConversationState:
    """
    Core state object that flows through the LangGraph workflow.

    This dataclass maintains all conversation context, query processing results,
    tool execution data, and persistence metadata throughout the workflow.
    """

    # Input
    question: str
    blob_names: List[str] = field(default_factory=list)
    is_data_analyst_mode: bool = False

    # Query Processing
    rewritten_query: str = ""
    augmented_query: str = ""
    query_category: str = "General"

    # Conversation Context
    messages: Annotated[List[BaseMessage], add_messages] = field(default_factory=list)
    context_docs: List[Any] = field(default_factory=list)

    # Persistence
    code_thread_id: Optional[str] = (
        None  # todo: change to container_id as we switch to claude
    )
    last_mcp_tool_used: str = ""
    uploaded_file_refs: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class OrchestratorConfig:
    """
    Configuration for the unified orchestrator.

    Defines LLM parameters, retrieval settings, MCP configuration,
    and feature flags for the orchestrator.
    """

    # Planning Model Configuration (Azure OpenAI - gpt-4.1)
    # Used for: query rewriting, categorization, tool selection
    planning_model: str = "gpt-4.1"
    planning_temperature: float = 0.3
    planning_max_tokens: int = 50000  # account for conversation history
    planning_api_version: str = "2025-04-01-preview"

    # Response Model Configuration (Anthropic Claude Sonnet)
    response_model: str = "claude-sonnet-4-5-20250929"
    response_temperature: float = 0.3
    response_max_tokens: int = 64000

    # Tool Calling Model Configuration (Anthropic Claude Haiku - faster/cheaper)
    tool_calling_model: str = "claude-haiku-4-5"
    tool_calling_temperature: float = 0.0
    tool_calling_max_tokens: int = 5000

    # Retrieval Configuration
    retriever_top_k: int = 5
    reranker_threshold: float = 2.0
    web_search_results: int = 2

    # MCP Configuration
    mcp_timeout: int = 600
    mcp_max_retries: int = 3


# ============================================================================
# Component Classes (Placeholders)
# ============================================================================


class StateManager:
    """
    Manages conversation state throughout the workflow.

    Responsibilities:
    - Load/save conversation data from Cosmos DB
    - Serialize/deserialize LangGraph memory
    - Track uploaded files and code threads
    - Persist metadata and thoughts
    """

    def __init__(self, organization_id: str, user_id: str):
        """
        Initialize StateManager.

        Args:
            organization_id: Organization identifier
            user_id: User identifier
        """
        self.organization_id = organization_id
        self.user_id = user_id
        logger.info(
            f"[StateManager] Initialized for org: {organization_id}, user: {user_id}"
        )

    def load_conversation(
        self, conversation_id: str, user_timezone: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load conversation data from Cosmos DB.

        Extracts thread IDs, file refs, and last tool used from conversation history.
        Handles missing conversation data gracefully by returning empty defaults.

        Args:
            conversation_id: Conversation identifier
            user_timezone: User's timezone for timestamp formatting

        Returns:
            Dictionary containing conversation history and metadata with keys:
            - history: List of conversation messages
            - code_thread_id: Data analyst thread ID (if any)
            - last_mcp_tool_used: Last MCP tool name (if any)
            - uploaded_file_refs: Cached file metadata (if any)
        """
        logger.info(f"[StateManager] Loading conversation: {conversation_id}")

        try:
            conversation_data = get_conversation_data(
                conversation_id=conversation_id,
                user_id=self.user_id,
                user_timezone=user_timezone,
            )

            if "history" not in conversation_data:
                logger.warning(
                    f"[StateManager] No history found for conversation {conversation_id}"
                )
                conversation_data["history"] = []

            code_thread_id = None  # data analyst
            last_mcp_tool_used = ""  # all tools in general
            uploaded_file_refs = []  # chat w doc tool

            history = conversation_data.get("history", [])

            # Iterate through history in reverse to find most recent assistant message
            for message in reversed(history):
                if message.get("role") == "assistant":
                    if "code_thread_id" in message and message["code_thread_id"]:
                        code_thread_id = message["code_thread_id"]

                    if (
                        "last_mcp_tool_used" in message
                        and message["last_mcp_tool_used"]
                    ):
                        last_mcp_tool_used = message["last_mcp_tool_used"]

                    if (
                        "uploaded_file_refs" in message
                        and message["uploaded_file_refs"]
                    ):
                        uploaded_file_refs = message["uploaded_file_refs"]

                    # Break after finding the first (most recent) assistant message
                    break

            logger.info(
                f"[StateManager] Loaded conversation with {len(history)} messages, "
                f"code_thread_id: {code_thread_id}, last_tool: {last_mcp_tool_used}, "
                f"cached_files: {len(uploaded_file_refs)}"
            )

            conversation_data["history"] = history
            conversation_data["code_thread_id"] = code_thread_id
            conversation_data["last_mcp_tool_used"] = last_mcp_tool_used
            conversation_data["uploaded_file_refs"] = uploaded_file_refs

            return conversation_data

        except Exception as e:
            logger.error(
                f"[StateManager] Error loading conversation {conversation_id}: {e}"
            )
            return {
                "start_date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                "history": [],
                "memory_data": "",
                "interaction": {},
                "type": "default",
                "code_thread_id": None,
                "last_mcp_tool_used": "",
                "uploaded_file_refs": [],
            }

    def save_conversation(
        self,
        conversation_id: str,
        conversation_data: Dict[str, Any],
        state: ConversationState,
        user_info: Dict[str, Any],
        response_time: float,
        response_text: str,
        thoughts: Dict[str, Any],
    ) -> None:
        """
        Save conversation data to Cosmos DB.

        Updates conversation history with new user question and assistant response.
        Includes thoughts and metadata in assistant messages.
        Saves interaction metadata (user_id, response_time, org_id).

        Args:
            conversation_id: Conversation identifier
            conversation_data: Existing conversation data
            state: Current conversation state
            user_info: User information (id, name)
            response_time: Time taken to generate response
            response_text: Generated response text
            thoughts: Diagnostic information for debugging
        """
        logger.info(f"[StateManager] Saving conversation: {conversation_id}")

        try:
            history = conversation_data.get("history", [])

            user_message = {
                "role": "user",
                "content": state.question,
            }
            history.append(user_message)
            logger.debug(f"[StateManager] Added user message: {state.question[:50]}...")

            assistant_message = {
                "role": "assistant",
                "content": response_text,
                "thoughts": thoughts,
            }

            if state.code_thread_id:
                assistant_message["code_thread_id"] = state.code_thread_id
                logger.debug(
                    f"[StateManager] Saved code_thread_id: {state.code_thread_id}"
                )

            if state.last_mcp_tool_used:
                assistant_message["last_mcp_tool_used"] = state.last_mcp_tool_used
                logger.debug(
                    f"[StateManager] Saved last_mcp_tool_used: {state.last_mcp_tool_used}"
                )

            if state.uploaded_file_refs:
                assistant_message["uploaded_file_refs"] = state.uploaded_file_refs
                logger.debug(
                    f"[StateManager] Saved {len(state.uploaded_file_refs)} file refs"
                )

            history.append(assistant_message)
            logger.debug(
                f"[StateManager] Added assistant message with thoughts: {list(thoughts.keys())}"
            )

            conversation_data["history"] = history

            conversation_data["interaction"] = {
                "user_id": user_info.get("id"),
                "user_name": user_info.get("name"),
                "response_time": response_time,
                "organization_id": self.organization_id,
            }

            update_conversation_data(
                conversation_id=conversation_id,
                user_id=self.user_id,
                conversation_data=conversation_data,
            )

            logger.info(
                f"[StateManager] Successfully saved conversation with "
                f"{len(history)} total messages, response_time: {response_time:.2f}s"
            )

        except Exception as e:
            logger.error(
                f"[StateManager] Error saving conversation {conversation_id}: {e}"
            )


class ContextBuilder:
    """
    Builds context prompts from organization data and conversation history.

    Responsibilities:
    - Format organization context (segments, brand, industry)
    - Clean and format conversation history
    - Extract context from tool results
    - Extract metadata (blob URLs, thread IDs)
    """

    def __init__(self, organization_data: Dict[str, Any]):
        """
        Initialize ContextBuilder.

        Args:
            organization_data: Organization information from Cosmos DB
        """
        self.organization_data = organization_data or {}
        logger.info("[ContextBuilder] Initialized")

    def build_organization_context(
        self, history: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Build organization context string for prompts.

        Formats segment aliases, brand information, industry information,
        and optionally conversation history from organization data.

        Args:
            history: Optional list of conversation history messages

        Returns:
            Formatted organization context string with sections for:
            - Historical conversation context (if provided)
            - Segment alias mappings (segmentSynonyms)
            - Brand information (brandInformation)
            - Industry information (industryInformation)
        """
        logger.debug("[ContextBuilder] Building organization context")

        segment_synonyms = self.organization_data.get("segmentSynonyms", "")
        brand_info = self.organization_data.get("brandInformation", "")
        industry_info = self.organization_data.get("industryInformation", "")

        logger.debug(
            f"[ContextBuilder] Org data present - "
            f"segments: {bool(segment_synonyms)}, "
            f"brand: {bool(brand_info)}, "
            f"industry: {bool(industry_info)}, "
            f"history: {len(history) if history else 0} messages"
        )

        context = ""

        # Add conversation history if provided
        if history:
            formatted_history = self.format_conversation_history(history)
            if formatted_history:
                context += f"""
                <----------- HISTORICAL CONVERSATION CONTEXT ------------>
                {formatted_history}
                <----------- END OF HISTORICAL CONVERSATION CONTEXT ------------>
                """
                logger.debug(
                    "[ContextBuilder] Added conversation history to organization context"
                )

        context += f"""
            <----------- PROVIDED SEGMENT ALIAS (VERY CRITICAL, MUST FOLLOW) ------------>
            Alias to segment mappings typically look like this (Official Name -> Alias):
            A -> B

            This mapping is mostly used in consumer segmentation context.

            Critical Rule – Contextual Consistency with Alias Mapping:
            • Always check whether the segment reference in the historical conversation is an alias (B). For example, historical conversation may mention "B" segment, but whenever you read the context in order to rewrite the query, you must map it to the official segment name "A" using the alias mapping table.
            • ALWAYS use the official name (A) in the rewritten query.
            • DO NOT use the alias (B) in the rewritten query.

            Here is the actual alias to segment mappings:

            **Official Segment Name Mappings (Official Name -> Alias):**
            {segment_synonyms}

            For example, if the historical conversation mentions "B", and the original question also mentions "B", you must rewrite the question to use "A" instead of "B".

            Look, if a mapping in the instruction is like this:
            students -> young kids

            Though the historical conversation and the original question may mention "students", you must rewrite the question to use "young kids" instead of "students".
            <----------- END OF PROVIDED SEGMENT ALIAS ------------>

            <----------- PROVIDED Brand Information ------------>
            This is the Brand information for the organization that the user belongs to.
            When relevant, incorporate Brand information to tailor responses, ensuring that answers are highly relevant to the user's company, goals, and operational environment.        
            Here is the Brand information:
            {brand_info}
            <----------- END OF PROVIDED Brand Information ------------>

            <----------- PROVIDED INDUSTRY DEFINITION ------------>
            This is the industry definition for the organization. This helps to understand the context of the organization and tailor responses accordingly
            Here is the industry definition:
            {industry_info}
            <----------- END OF PROVIDED INDUSTRY DEFINITION ------------>
            """
        return context

    def format_conversation_history(
        self, history: List[Dict[str, Any]], max_messages: int = 6
    ) -> str:
        """
        Format conversation history for LLM consumption.

        Cleans and formats history by:
        - Truncating to most recent messages
        - Removing markdown images
        - Formatting as "Human:" and "AI Message:" exchanges

        Args:
            history: List of conversation messages (dicts with 'role' and 'content')
            max_messages: Maximum number of messages to include (default: 6)

        Returns:
            Formatted conversation history string
        """
        logger.debug(
            f"[ContextBuilder] Formatting conversation history ({len(history)} messages)"
        )

        if not history:
            logger.debug("[ContextBuilder] No history to format")
            return ""

        # Truncate to most recent messages
        if len(history) > max_messages:
            truncated_history = history[-max_messages:]
            logger.debug(f"[ContextBuilder] Truncated to last {max_messages} messages")
        else:
            truncated_history = history

        formatted_messages = []
        for message in truncated_history:
            if not isinstance(message, dict):
                continue

            role = message.get("role", "").lower()
            content = message.get("content", "")

            if not content:
                continue

            # Determine display role
            if role == "user":
                display_role = "Human"
            elif role == "assistant":
                display_role = "AI Message"
            else:
                continue

            # Clean markdown images from content
            cleaned_content = re.sub(r"!\[([^\]]*)\]\(([^\)]+)\)", "", content)

            formatted_messages.append(f"{display_role}: {cleaned_content}")

        result = "\n\n".join(formatted_messages)
        logger.debug(f"[ContextBuilder] Formatted {len(formatted_messages)} messages")

        return result

    def extract_context_from_messages(
        self, messages: List[BaseMessage]
    ) -> tuple[List[Any], List[str], List[Dict[str, str]]]:
        """
        Extract context from LangChain messages (used with bind_tools approach).

        When using bind_tools, tool calls and results are embedded in message objects.
        This method extracts the actual tool results from AIMessage tool_calls and
        ToolMessage objects.

        Args:
            messages: List of LangChain messages from tool execution

        Returns:
            Tuple of (context_docs, blob_urls, uploaded_file_refs)
        """
        logger.debug(
            f"[ContextBuilder] Extracting context from {len(messages)} messages"
        )

        context_docs = []
        blob_urls = []
        uploaded_file_refs = []

        # Find tool-related messages
        for msg in messages:
            # Check for ToolMessage (contains tool results)
            if hasattr(msg, "content") and hasattr(msg, "name"):
                tool_name = getattr(msg, "name", "")
                content = msg.content

                # Parse JSON content if it's a string
                result = content
                if isinstance(content, str):
                    try:
                        result = json.loads(content)
                    except Exception:
                        pass

                logger.debug(
                    f"[ContextBuilder] Processing tool message from {tool_name}"
                )

                # Extract based on tool type
                if tool_name == "agentic_search" and isinstance(result, dict):
                    search_results = result.get("results", result)

                    filtered_docs = []

                    if isinstance(search_results, dict):
                        for subquery_key, subquery_data in search_results.items():
                            if (
                                isinstance(subquery_data, dict)
                                and "documents" in subquery_data
                            ):
                                documents = subquery_data.get("documents", [])
                                if isinstance(documents, list):
                                    for doc in documents:
                                        if isinstance(doc, dict):
                                            filtered_docs.append(
                                                {
                                                    "content": doc.get("content"),
                                                    "source": doc.get("source"),
                                                }
                                            )

                    if filtered_docs:
                        context_docs.append(filtered_docs)
                        logger.debug(
                            f"[ContextBuilder] Added {len(filtered_docs)} filtered documents from agentic_search"
                        )
                    else:
                        # If not a list, append as-is (fallback)
                        context_docs.append(search_results)
                        logger.debug(
                            "[ContextBuilder] Added agentic_search results (non-list format)"
                        )

                elif tool_name == "data_analyst" and isinstance(result, dict):
                    last_message = result.get("last_agent_message", result)
                    context_docs.append(last_message)

                    # Extract blob URLs
                    result_blob_urls = result.get("blob_urls", [])
                    if isinstance(result_blob_urls, list) and result_blob_urls:
                        for blob_item in result_blob_urls:
                            if isinstance(blob_item, dict):
                                blob_path = blob_item.get("blob_path")
                                if blob_path:
                                    blob_urls.append(blob_path)
                                    context_docs.append(
                                        f"Here is the graph/visualization link: \n\n{blob_path}"
                                    )
                                    logger.debug(
                                        f"[ContextBuilder] Added blob URL from message: {blob_path}"
                                    )

                elif tool_name == "web_fetch" and isinstance(result, dict):
                    web_content = result.get("content")
                    if web_content:
                        context_docs.append(web_content)
                    else:
                        context_docs.append(result)
                    logger.debug(
                        "[ContextBuilder] Added web_fetch content from message"
                    )

                elif tool_name == "document_chat" and isinstance(result, dict):
                    answer = result.get("answer", result)
                    context_docs.append(answer)

                    # Extract file references - caching uploaded files (openai expired in 1 hour)
                    files = result.get("files", [])
                    if files and isinstance(files, list):
                        uploaded_file_refs = files
                        logger.debug(
                            f"[ContextBuilder] Extracted {len(files)} file references from message"
                        )

        logger.info(
            f"[ContextBuilder] Extracted from messages: {len(context_docs)} context docs, "
            f"{len(blob_urls)} blob URLs, {len(uploaded_file_refs)} file refs"
        )

        return context_docs, blob_urls, uploaded_file_refs


class QueryPlanner:
    """
    Handles query processing: rewriting, augmentation, and categorization.

    Responsibilities:
    - Rewrite queries with organization context
    - Augment queries with conversation history
    - Categorize queries into proper marketing categories
    """

    def __init__(self, llm: AzureChatOpenAI, organization_data: Dict[str, Any]):
        """
        Initialize QueryPlanner.

        Args:
            llm: Azure OpenAI LLM instance for planning tasks
            organization_data: Organization information
        """
        self.llm = llm
        self.organization_data = organization_data
        logger.info("[QueryPlanner] Initialized")

    @traceable(run_type="llm", name="query_rewrite")
    async def rewrite_query(
        self,
        state: ConversationState,
        conversation_data: Dict[str, Any],
        context_builder: ContextBuilder,
    ) -> Dict[str, str]:
        """
        Rewrite query with organization context and segment aliases.

        Uses llm planning model to rewrite the user's query with:
        - Organization context (segments, brand, industry)
        - Conversation history for context
        - Segment alias mappings to use official terminology

        Args:
            state: Current conversation state
            conversation_data: Conversation history
            context_builder: ContextBuilder instance

        Returns:
            Dictionary with rewritten_query
        """
        logger.info(f"[QueryPlanner] Rewriting query: {state.question[:50]}...")

        question = state.question
        history = conversation_data.get("history", [])
        logger.info(
            f"[QueryPlanner] Retrieved {len(history)} messages from conversation history"
        )

        # Build system prompt with organization context including history
        system_prompt = f"""
        {QUERY_REWRITING_PROMPT}
        {context_builder.build_organization_context(history)}"""

        # Build user prompt
        user_prompt = f"""Original Question: 
        <-------------------------------->
        ```
        {question}. 
        ```
        <-------------------------------->

        Please rewrite the question to be used for searching the database. Make sure to follow the alias mapping instructions at all cost.
        ALSO, THE HISTORICAL CONVERSATION CONTEXT IS VERY VERY IMPORTANT TO THE USER'S FOLLOW UP QUESTIONS, $10,000 WILL BE DEDUCTED FROM YOUR ACCOUNT IF YOU DO NOT USE THE HISTORICAL CONVERSATION CONTEXT.
        Please also consider the line of business/industry of my company when rewriting the query. Don't be too verbose. 

        if the question is a very casual/conversational one, do not rewrite, return it as it is
        """

        logger.info("[QueryPlanner] Sending query rewrite request to LLM")

        try:
            rewritten_response = await self.llm.ainvoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ]
            )
            rewritten_query = rewritten_response.content
            logger.info(
                f"[QueryPlanner] Successfully rewrote query: '{rewritten_query[:100]}...'"
            )
        except Exception as e:
            logger.error(f"[QueryPlanner] Error rewriting query: {e}")
            rewritten_query = question

        return {"rewritten_query": rewritten_query}

    @traceable(run_type="llm", name="query_augment")
    async def augment_query(
        self,
        state: ConversationState,
        conversation_data: Dict[str, Any],
        context_builder: ContextBuilder,
    ) -> Dict[str, str]:
        """
        Augment query with conversation history.

        Uses gpt-4.1 to augment the query with historical context for improved
        understanding. Handles casual/conversational queries by returning them as-is.

        Args:
            state: Current conversation state
            conversation_data: Conversation history
            context_builder: ContextBuilder instance for formatting history

        Returns:
            Dictionary with augmented_query
        """
        logger.info(f"[QueryPlanner] Augmenting query: {state.question[:50]}...")

        question = state.question
        history = conversation_data.get("history", [])
        formatted_history = context_builder.format_conversation_history(history)

        augmented_query_prompt = f""" 
        Augment the query with the historical conversation context. If the query is a very casual/conversational one, do not augment, return it as it is.
        
        Here is the historical conversation context if available:
        <context>
        {formatted_history}
        </context>

        Here is the query to augment:
        <query>
        {question}
        </query>

        Return the augmented query in text format only, no additional text, explanations, or formatting.
        """

        logger.info("[QueryPlanner] Sending augmented query request to LLM")

        try:
            augmented_response = await self.llm.ainvoke(
                [
                    SystemMessage(content=AUGMENTED_QUERY_PROMPT),
                    HumanMessage(content=augmented_query_prompt),
                ]
            )
            augmented_query = augmented_response.content
            logger.info(
                f"[QueryPlanner] Successfully augmented query: '{augmented_query[:100]}...'"
            )
        except Exception as e:
            logger.error(
                f"[QueryPlanner] Failed to augment query, using original question: {e}"
            )
            augmented_query = question

        return {"augmented_query": augmented_query}

    @traceable(run_type="llm", name="query_categorize")
    async def categorize_query(
        self,
        state: ConversationState,
        conversation_data: Dict[str, Any],
        context_builder: ContextBuilder,
    ) -> Dict[str, str]:
        """
        Categorize query into marketing categories.

        Uses Claude with conversation history for context to classify queries into:
        - Creative Brief
        - Marketing Plan
        - Brand Positioning Statement
        - Creative Copywriter
        - Help Desk
        - General

        Args:
            state: Current conversation state
            conversation_data: Conversation history
            context_builder: ContextBuilder instance for formatting history

        Returns:
            Dictionary with query_category
        """
        logger.info(f"[QueryPlanner] Categorizing query: {state.question[:50]}...")

        history = conversation_data.get("history", [])
        formatted_history = context_builder.format_conversation_history(history)

        # Build categorization prompt
        category_prompt = f"""
            You are a senior marketing strategist. Your task is to classify the user's question into one of the following categories:

            - Creative Brief
            - Marketing Plan
            - Help Desk
            - Brand Positioning Statement
            - Creative Copywriter
            - General

            Use both the current question and the historical conversation context to make an informed decision. 
            Context is crucial, as users may refer to previous topics, provide follow-ups, or respond to earlier prompts. 

            To help you make an accurate decision, consider these cues for each category:

            - **Creative Brief**: Look for project kickoffs, campaign overviews, client objectives, audience targeting, timelines, deliverables, or communication goals.
            - **Marketing Plan**: Look for references to strategy, goals, budget, channels, timelines, performance metrics, or ROI.
            - **Brand Positioning Statement**: Watch for messages about defining brand essence, values, personality, competitive differentiation, or target audience perception.
            - **Creative Copywriter**: Use this category when users ask meta-questions about your fundamental functions, purpose, or identity. This applies to inquiries such as "What can you do?", "What are your capabilities?", or "How do you work?". This category is distinct from standard user commands or requests for technical support.
            - **Help Desk**: Use this category when users ask "What can you do?", "What are your capabilities?", or similar questions about your functions.
            - **General**: If the input lacks context, doesn't relate to marketing deliverables, or is unclear or unrelated to the above.

            If the question or context is not clearly related to any of the above categories, always return "General".

            ----------------------------------------
            User's Question:
            {state.question}
            ----------------------------------------
            Conversation History:
            {formatted_history}
            ----------------------------------------

            Reply with **only** the exact category name — no additional text, explanations, or formatting.
            """

        logger.info("[QueryPlanner] Sending categorization request to LLM")

        try:
            response = await self.llm.ainvoke(
                [
                    SystemMessage(content=category_prompt),
                    HumanMessage(content=state.question),
                ]
            )
            query_category = response.content.strip()
            log_info(f"[Categorize Node] Complete: Category = '{query_category}'")
        except Exception as e:
            logger.error(f"[QueryPlanner] Error categorizing query: {e}")
            query_category = "General"

        return {"query_category": query_category}


class MCPClient:
    """
    Handles MCP Server connection and tool execution.

    Responsibilities:
    - Connect to MCP Server via SSE
    - Discover available tools
    - Configure tool arguments with context
    - Execute tools and return results
    """

    # Tool name constants
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
            # Get tools from MCP Server
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


class ResponseGenerator:
    """
    Generates streaming LLM responses with context.

    Responsibilities:
    - Build system prompts with organization context
    - Build user prompts with augmented queries
    - Stream responses from Claude
    - Sanitize responses (remove storage URLs)
    """

    def __init__(
        self,
        claude_llm: ChatAnthropic,
        organization_data: Dict[str, Any],
        storage_url: str,
    ):
        """
        Initialize ResponseGenerator.

        Args:
            claude_llm: Anthropic Claude LLM instance
            organization_data: Organization information
            storage_url: Azure Storage URL for sanitization
        """
        self.claude_llm = claude_llm
        self.organization_data = organization_data
        self.storage_url = storage_url
        logger.info("[ResponseGenerator] Initialized")

    def build_system_prompt(
        self,
        state: ConversationState,
        context_builder: ContextBuilder,
        conversation_history: str,
        user_settings: Dict[str, Any],
    ) -> str:
        """
        Build system prompt with organization context and category-specific prompts.

        Includes:
        - Organization context (segments, brand, industry)
        - Retrieved context documents with citations
        - Conversation history
        - Category-specific prompts based on query category
        - Verbosity instructions based on user settings

        Args:
            state: Current conversation state
            context_builder: ContextBuilder instance
            conversation_history: Formatted conversation history
            user_settings: User preferences

        Returns:
            Complete system prompt
        """
        logger.debug("[ResponseGenerator] Building system prompt")

        # base prompt
        system_prompt = MARKETING_ANSWER_PROMPT

        # Add organization context
        org_context = context_builder.build_organization_context()
        system_prompt += f"\n\n{org_context}"

        # Add conversation history if available
        if conversation_history:
            system_prompt += f"""
                <----------- PROVIDED CHAT HISTORY ------------>
                {conversation_history}
                <----------- END OF PROVIDED CHAT HISTORY ------------>
                """
            logger.debug(
                "[ResponseGenerator] Added conversation history to system prompt"
            )

        # Add retrieved context documents if available
        if state.context_docs:
            context_str = "\n\n".join(str(doc) for doc in state.context_docs)
            system_prompt += f"""
                <----------- PROVIDED CONTEXT ------------>
                {context_str}
                <----------- END OF PROVIDED CONTEXT ------------>
                """
            logger.debug(
                f"[ResponseGenerator] Added {len(state.context_docs)} context documents to system prompt"
            )

        # Add category-specific prompt based on query category
        category_prompts = {
            "Creative Brief": CREATIVE_BRIEF_PROMPT,
            "Marketing Plan": MARKETING_PLAN_PROMPT,
            "Brand Positioning Statement": BRAND_POSITION_STATEMENT_PROMPT,
            "Creative Copywriter": CREATIVE_COPYWRITER_PROMPT,
            "Help Desk": FA_HELPDESK_PROMPT,
        }

        if state.query_category in category_prompts:
            category_prompt = category_prompts[state.query_category]
            system_prompt += f"""
                <----------- CATEGORY-SPECIFIC INSTRUCTIONS ------------>
                {category_prompt}
                <----------- END OF CATEGORY-SPECIFIC INSTRUCTIONS ------------>
                """
            logger.debug(
                f"[ResponseGenerator] Added category-specific prompt for: {state.query_category}"
            )

        # Add verbosity instructions based on user settings
        verbosity_instruction = get_verbosity_instruction(user_settings)
        system_prompt += f"""
            <----------- VERBOSITY INSTRUCTIONS ------------>
            {verbosity_instruction}
            <----------- END OF VERBOSITY INSTRUCTIONS ------------>
            """
        logger.debug(
            "[ResponseGenerator] Added verbosity instructions to system prompt"
        )

        logger.info(
            f"[ResponseGenerator] Built system prompt with {len(system_prompt)} characters"
        )
        return system_prompt

    def build_user_prompt(
        self, state: ConversationState, user_settings: Dict[str, Any]
    ) -> str:
        """
        Build user prompt with original question and augmented query.

        Includes original question and augmented query based on detail_level setting:
        - "detailed": Include augmented query
        - "brief" or "balanced": Exclude augmented query

        Args:
            state: Current conversation state
            user_settings: User preferences

        Returns:
            Complete user prompt
        """
        logger.debug("[ResponseGenerator] Building user prompt")

        user_prompt = f"Original Question: {state.question}"

        # Check detail_level setting
        detail_level = user_settings.get("detail_level", "balanced")

        # Include augmented query only for "detailed" setting
        if detail_level == "detailed" and state.augmented_query:
            user_prompt += f"\n\nAugmented Query (with historical context): {state.augmented_query}"
            logger.debug(
                "[ResponseGenerator] Included augmented query in user prompt (detail_level: detailed)"
            )
        else:
            logger.debug(
                f"[ResponseGenerator] Excluded augmented query (detail_level: {detail_level})"
            )

        logger.info(
            f"[ResponseGenerator] Built user prompt with {len(user_prompt)} characters"
        )
        return user_prompt

    @traceable(run_type="llm", name="claude_generate_response")
    async def generate_streaming_response(
        self, system_prompt: str, user_prompt: str, temperature: Optional[float] = None
    ):
        """
        Generate streaming response from Claude with extended thinking.

        Uses Anthropic Claude (claude-sonnet-4-5-20250929) for streaming.
        Enables extended thinking and streams both thinking tokens and answer tokens.

        Thinking tokens are formatted as: __THINKING__text__THINKING__
        Answer tokens are streamed as-is.

        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            temperature: Optional temperature override from user settings

        Yields:
            Response tokens (thinking and answer)
        """
        logger.info("[ResponseGenerator] Starting streaming response generation")

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]

            logger.debug("[ResponseGenerator] Invoking Claude with streaming enabled")

            async for chunk in self.claude_llm.astream(
                messages, temperature=temperature
            ):
                if hasattr(chunk, "content") and chunk.content:
                    # Check if this is a thinking token
                    # Claude's extended thinking tokens are typically marked differently
                    # For now, we'll stream all content as answer tokens
                    # If Claude provides thinking tokens in a specific format, we can detect and wrap them
                    yield chunk.content

            logger.info("[ResponseGenerator] Completed streaming response generation")

        except Exception as e:
            logger.error(
                f"[ResponseGenerator] Error during streaming response generation: {e}"
            )
            error_message = "I apologize, but I encountered an error while generating the response. Please try again."
            yield error_message

    def sanitize_response(self, text: str) -> str:
        """
        Remove Azure Storage URLs from response text.

        Removes any URLs containing the Azure Storage account URL to prevent
        exposing internal storage paths to users.

        Args:
            text: Response text to sanitize

        Returns:
            Sanitized response text with storage URLs removed
        """
        logger.debug("[ResponseGenerator] Sanitizing response")

        if not self.storage_url or not text:
            logger.debug("[ResponseGenerator] No sanitization needed")
            return text

        # Escape special regex characters in storage URL
        escaped_storage_url = re.escape(self.storage_url)

        # Pattern to match URLs containing the storage URL
        # This will match the entire URL including any query parameters
        pattern = rf"{escaped_storage_url}[^\s\)]*"

        # Count matches before sanitization
        matches = re.findall(pattern, text)
        if matches:
            logger.info(
                f"[ResponseGenerator] Found {len(matches)} storage URLs to sanitize"
            )
            sanitized_text = re.sub(pattern, "[URL removed for security]", text)
            logger.debug("[ResponseGenerator] Sanitization complete")
            return sanitized_text
        else:
            logger.debug("[ResponseGenerator] No storage URLs found in response")
            return text


# ============================================================================
# Main Orchestrator Class
# ============================================================================


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
        query: str, # rewritten by claude before sending to the mcp server
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
                "images_processed": self._transform_artifacts_to_images(
                    complete_data.get("artifacts", [])
                ),
                "blob_urls": self._transform_artifacts_to_blobs(
                    complete_data.get("artifacts", [])
                ),
                "last_agent_message": complete_data.get("response", ""),
                "error": complete_data.get("error"),
            }

        except Exception as e:
            logger.error(f"[StreamDataAnalyst] Error: {e}", exc_info=True)
            raise

    def _transform_artifacts_to_images(self, artifacts: List[Dict]) -> List[Dict]:
        """Transform streaming artifacts to images_processed format."""
        return [
            {
                "file_id": art.get("blob_path", ""),
                "filename": art.get("filename", "unknown"),
                "size_bytes": art.get("size", 0),
                "content_type": "image/png",
            }
            for art in artifacts
        ]

    def _transform_artifacts_to_blobs(self, artifacts: List[Dict]) -> List[Dict]:
        """Transform streaming artifacts to blob_urls format."""
        return [
            {
                "filename": art.get("filename", "unknown"),
                "blob_url": art.get("blob_url", ""),
                "blob_path": art.get("blob_path", ""),
            }
            for art in artifacts
        ]

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
        Initialize Anthropic Claude Sonnet LLM for response generation.

        Returns:
            Configured ChatAnthropic instance
        """
        logger.info(
            "[ConversationOrchestrator] Initializing response LLM (Claude Sonnet)"
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

    async def _initialize_node(self, state: ConversationState) -> Dict[str, Any]:
        """
        Initialize node: Load conversation data and extract metadata.

        Loads conversation data using StateManager, extracts existing metadata
        (thread IDs, file refs, last tool), and initializes ConversationState.
        Emits initialization progress (5%).

        Args:
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

    def _get_tool_progress_message(self, tool_name: str, stage: str) -> str:
        """
        Get tool-specific progress message for UI.

        Args:
            tool_name: Name of the MCP tool
            stage: Stage of execution ('planning' or 'executing')

        Returns:
            User-friendly progress message
        """
        tool_messages = {
            "agentic_search": {
                "planning": "Planning knowledge base search...",
                "executing": "Searching your knowledge base...",
            },
            "data_analyst": {
                "planning": "Planning data analysis...",
                "executing": "Analyzing your data...",
            },
            "web_fetch": {
                "planning": "Planning web content fetch...",
                "executing": "Fetching web content...",
            },
            "document_chat": {
                "planning": "Preparing document analysis...",
                "executing": "Reading your documents...",
            },
        }

        return tool_messages.get(tool_name, {}).get(
            stage, f"{stage.capitalize()} tools..."
        )

    async def _plan_tools_node(self, state: ConversationState) -> Dict[str, Any]:
        """
        Plan tools node: Claude with bind_tools decides which tools to use.

        Uses bind_tools to let Claude decide which tools to call based on
        the current messages. Returns updated messages with AIMessage containing
        tool_calls if tools should be used.

        Args:
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
                    tool_message = self._get_tool_progress_message(
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
            tool_message = self._get_tool_progress_message(tool_name, "executing")
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
            state: Current conversation state

        Returns:
            Empty dict (response stored in self.current_response_text)
        """
        log_info("[Generate Response Node] Generating final response with Claude")

        progress_data = {
            "type": "progress",
            "step": "response_generation",
            "message": "Generating response...",
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
                temperature=self.config.response_temperature,
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

        # Add nodes
        graph.add_node("initialize", self._initialize_node)
        graph.add_node("rewrite", self._rewrite_node)
        graph.add_node("augment", self._augment_node)
        graph.add_node("categorize", self._categorize_node)
        graph.add_node("prepare_tools", self._prepare_tools_node)
        graph.add_node("prepare_messages", self._prepare_messages_node)
        graph.add_node("plan_tools", self._plan_tools_node)  # decide tools
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

        # After tool execution, go directly to context extraction - todo: figure out a chance to loop to change tool here
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
