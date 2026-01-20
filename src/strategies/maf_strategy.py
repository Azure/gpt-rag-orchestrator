"""
Minimal Microsoft Agent Framework (MAF) strategy using Azure AI Foundry Agent Service.

This is a simplified conversational agent without RAG/tools for basic conversation capabilities.
Extension points are available for adding tools, RAG, and Bing grounding in future iterations.
"""

import logging
import time
from typing import Optional

# Suppress Azure SDK HTTP logging BEFORE importing azure packages
for _azure_logger in [
    "azure.core.pipeline.policies.http_logging_policy",
    "azure.identity",
    "azure.core",
    "azure"
]:
    logger = logging.getLogger(_azure_logger)
    logger.setLevel(logging.CRITICAL)
    logger.propagate = False
    logger.disabled = True
    logger.handlers.clear()

from azure.ai.agents.models import ListSortOrder, MessageTextContent

from .base_agent_strategy import BaseAgentStrategy
from .agent_strategies import AgentStrategies
from dependencies import get_config


class MafStrategy(BaseAgentStrategy):
    """
    Minimal conversational agent using Azure AI Foundry Agent Service.

    This strategy provides basic conversation capabilities without RAG or tools.
    It manages thread lifecycle for conversation continuity and supports both
    persistent agents (via AGENT_ID) and temporary agents created per-session.
    """

    async def create():
        """Factory method to create an instance of MafStrategy."""
        logging.debug("[Agent Flow] Creating MafStrategy instance...")
        return MafStrategy()

    def __init__(self):
        """Initialize the MAF strategy with minimal configuration."""
        super().__init__()

        logging.debug("[Init] Initializing MafStrategy...")

        cfg = get_config()
        self.strategy_type = AgentStrategies.MAF

        # Allow the user to specify an existing agent ID (optional)
        self.existing_agent_id = cfg.get("AGENT_ID", "") or None

        # No tools for minimal version - extension point for future iterations
        self.tools_list = []
        self.tool_resources = {}

        logging.debug("[Init] MafStrategy initialized (no tools)")

    async def initiate_agent_flow(self, user_message: str):
        """
        Initiate the agent flow for a conversational interaction.

        Steps:
        1. Get or create thread (conversation continuity)
        2. Get or create agent
        3. Send user message
        4. Stream response
        5. Consolidate history
        6. Cleanup temporary agent if needed
        """
        flow_start = time.time()
        logging.debug(f"[Agent Flow] initiate_agent_flow called with user_message: {user_message!r}")
        conv = self.conversation
        thread_id = conv.get("thread_id")

        async with self.project_client as project_client:
            # Step 1: Manage thread lifecycle (create or reuse)
            thread = await self._get_or_create_thread(project_client, thread_id)
            conv["thread_id"] = thread.id

            # Step 2: Create or reuse agent
            agent, created_agent = await self._get_or_create_agent(project_client)
            conv["agent_id"] = agent.id

            # Step 3: Send user message to thread
            await self._send_user_message(project_client, thread.id, user_message)

            # Step 4: Stream agent response
            async for chunk in self._stream_agent_response(
                project_client, agent.id, thread.id
            ):
                yield chunk

            # Step 5: Consolidate conversation history
            await self._consolidate_conversation_history(project_client, thread.id)

            # Step 6: Cleanup temporary agent if created
            if created_agent:
                await self._cleanup_agent(project_client, agent.id)

            logging.info(f"[Agent Flow] Total flow time: {round(time.time() - flow_start, 2)}s")

    # ============================================================
    # Agent Flow Helper Methods
    # ============================================================

    async def _get_or_create_thread(self, project_client, thread_id: Optional[str]):
        """Create a new thread or retrieve an existing one."""
        try:
            if thread_id:
                logging.debug(f"[Agent Flow] Retrieving existing thread: {thread_id}")
                thread = await project_client.agents.threads.get(thread_id)
                logging.info(f"[Agent Flow] Reused thread: {thread.id}")
            else:
                logging.debug("[Agent Flow] Creating new thread")
                thread = await project_client.agents.threads.create()
                logging.info(f"[Agent Flow] Created new thread: {thread.id}")
            return thread
        except Exception as e:
            logging.error(f"[Agent Flow] Thread operation failed: {e}", exc_info=True)
            raise Exception(f"Thread creation failed: {str(e)}") from e

    async def _get_or_create_agent(self, project_client):
        """Create a new agent or retrieve an existing one."""
        created_agent = False

        try:
            if self.existing_agent_id:
                logging.debug(f"[Agent Flow] Retrieving existing agent: {self.existing_agent_id}")
                agent = await project_client.agents.get_agent(self.existing_agent_id)
                logging.info(f"[Agent Flow] Reused agent: {agent.id}")
            else:
                logging.debug("[Agent Flow] Creating new agent")

                prompt_context = {
                    "strategy": self.strategy_type.value,
                    "user_context": self.user_context or {},
                }

                instructions = await self._read_prompt("main")

                agent = await project_client.agents.create_agent(
                    model=self.model_name,
                    name="maf-agent",
                    instructions=instructions,
                    tools=self.tools_list,
                    tool_resources=self.tool_resources
                )
                created_agent = True
                logging.info(f"[Agent Flow] Created new agent: {agent.id}")

            return agent, created_agent
        except Exception as e:
            logging.error(f"[Agent Flow] Agent operation failed: {e}", exc_info=True)
            raise Exception(f"Agent creation failed: {str(e)}") from e

    async def _send_user_message(self, project_client, thread_id: str, user_message: str):
        """Send user message to the thread."""
        try:
            logging.debug(f"[Agent Flow] Sending message to thread {thread_id}")
            await project_client.agents.messages.create(
                thread_id=thread_id,
                role="user",
                content=user_message
            )
            logging.debug("[Agent Flow] User message sent")
        except Exception as e:
            logging.error(f"[Agent Flow] Failed to send message: {e}", exc_info=True)
            raise Exception(f"Message sending failed: {str(e)}") from e

    async def _stream_agent_response(self, project_client, agent_id: str, thread_id: str):
        """Stream agent response chunks."""
        try:
            stream_start = time.time()

            async with await project_client.agents.runs.stream(
                thread_id=thread_id,
                agent_id=agent_id
            ) as stream:
                async for event_type, event_data, raw in stream:
                    # Stream message deltas
                    if event_type == "thread.message.delta" and hasattr(event_data, "text"):
                        chunk = event_data.text
                        if chunk:
                            yield chunk

                    # Handle run failure
                    if event_type == "thread.run.failed":
                        err = event_data.last_error.message
                        logging.error(f"[Stream] Run failed: {err}")
                        raise Exception(err)

            logging.info(f"[Stream] Streaming completed in {round(time.time() - stream_start, 2)}s")

        except Exception as e:
            logging.error(f"[Stream] Streaming failed: {e}", exc_info=True)
            raise Exception(f"Agent response streaming failed: {str(e)}") from e

    async def _consolidate_conversation_history(self, project_client, thread_id: str):
        """Fetch and consolidate conversation history from thread."""
        try:
            logging.debug("[Agent Flow] Consolidating conversation history")
            conv = self.conversation
            conv["messages"] = []

            messages = project_client.agents.messages.list(
                thread_id=thread_id,
                order=ListSortOrder.ASCENDING
            )

            msg_count = 0
            async for msg in messages:
                if not msg.content:
                    continue

                last_content = msg.content[-1]
                if isinstance(last_content, MessageTextContent):
                    text_val = last_content.text.value
                    msg_count += 1
                    conv["messages"].append({
                        "role": msg.role,
                        "text": text_val
                    })

            logging.info(f"[Agent Flow] Retrieved {msg_count} messages")

            if self.user_context:
                conv['user_context'] = self.user_context
        except Exception as e:
            logging.error(f"[Agent Flow] Failed to consolidate history: {e}", exc_info=True)
            # Non-critical - log and continue

    async def _cleanup_agent(self, project_client, agent_id: str):
        """Delete temporary agent after completion."""
        try:
            logging.debug(f"[Agent Flow] Deleting agent: {agent_id}")
            await project_client.agents.delete_agent(agent_id)
            logging.debug("[Agent Flow] Agent deleted")
        except Exception as e:
            logging.error(f"[Agent Flow] Failed to delete agent: {e}", exc_info=True)
            # Non-critical - log and continue
