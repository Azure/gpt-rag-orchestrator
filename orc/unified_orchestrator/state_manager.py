"""
State Manager Module

This module manages conversation state throughout the workflow, handling
persistence with Cosmos DB and tracking conversation metadata.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from shared.cosmos_db import get_conversation_data, update_conversation_data
from .models import ConversationState

logger = logging.getLogger(__name__)


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
