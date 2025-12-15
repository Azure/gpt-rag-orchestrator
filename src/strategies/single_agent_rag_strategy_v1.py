
import logging
import json
import re
import time
from typing import Optional, Any
from pydantic import BaseModel

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

from azure.ai.agents.models import (
    BingGroundingTool,
    FunctionTool,
    ListSortOrder,
    MessageDeltaChunk,
    MessageDeltaTextUrlCitationAnnotation,
    MessageTextContent,
)

from .base_agent_strategy import BaseAgentStrategy
from .agent_strategies import AgentStrategies

from dependencies import get_config
from connectors import SearchClient

# ============================================================
# Citation Processing Helper
# ============================================================

# Pre-compiled regex for citation placeholders (e.g., 【7:0†source】)
# Compiled once at module load for better performance
CITATION_PLACEHOLDER_PATTERN = re.compile(r'【(\d+):(\d+)†[^】]*】')

def truncate_title(title: str, max_length: int = 30) -> str:
    """
    Truncate title to max_length, cutting at the last space before the limit.
    
    Args:
        title: The title to truncate
        max_length: Maximum length (default 30)
        
    Returns:
        Truncated title with '...' if needed
    """
    if not title or len(title) <= max_length:
        return title
    
    # Find the last space before max_length
    truncated = title[:max_length]
    last_space = truncated.rfind(' ')
    
    if last_space > 0:
        return truncated[:last_space] + '...'
    else:
        # No space found, just truncate at max_length
        return truncated + '...'


def process_bing_citations(delta: MessageDeltaChunk) -> str:
    """
    Process Bing Grounding citations from message delta.
    Replaces placeholders like 【3:0†source】 with proper [title](url) format.
    
    Note: Only works with OpenAI/Azure OpenAI models that include url_citation
    annotations. Other models (e.g., DeepSeek) don't provide annotations,
    so placeholders are simply removed.
    
    Args:
        delta: The message delta chunk containing text and potential annotations
    """
    text = delta.text
    if not text:
        return text
    
    # Collect annotation objects from the delta
    raw = getattr(delta, "delta", None)
    annotations = []
    
    if raw:
        raw_content = getattr(raw, "content", [])
        for piece in raw_content:
            txt = getattr(piece, "text", None)
            if txt:
                anns = getattr(txt, "annotations", None)
                if anns:
                    annotations.extend(anns)
    
    # Process URL citations (used by Bing Grounding with OpenAI/Azure models)
    for ann in annotations:
        placeholder = None
        url = None
        title = None
        
        # Convert annotation to dict (handles Pydantic models, etc.)
        ann_dict = None
        if hasattr(ann, 'model_dump'):
            ann_dict = ann.model_dump()
        elif hasattr(ann, 'dict'):
            ann_dict = ann.dict()
        elif hasattr(ann, '__dict__'):
            ann_dict = ann.__dict__
        elif isinstance(ann, dict):
            ann_dict = ann
        
        if ann_dict:
            # Handle nested _data structure (Azure SDK format)
            if '_data' in ann_dict:
                ann_dict = ann_dict['_data']
            
            ann_type = ann_dict.get('type', '')
            if ann_type == 'url_citation' or 'url_citation' in ann_dict:
                placeholder = ann_dict.get('text')
                url_citation = ann_dict.get('url_citation', {})
                
                if isinstance(url_citation, dict):
                    url = url_citation.get('url')
                    title = url_citation.get('title')
                else:
                    url = getattr(url_citation, 'url', None)
                    title = getattr(url_citation, 'title', None)
        
        if not title:
            title = url
        
        if url and placeholder and placeholder in text:
            display_title = truncate_title(title, 30)
            citation = f"[{display_title}]({url})"
            text = text.replace(placeholder, citation)
    
    # Clean up citation placeholders for models that don't include annotations (e.g., DeepSeek)
    # Uses pre-compiled CITATION_PLACEHOLDER_PATTERN for better performance
    text = CITATION_PLACEHOLDER_PATTERN.sub('', text)
    
    return text


# ============================================================
# Main Strategy Class
# ============================================================

class SingleAgentRAGStrategyV1(BaseAgentStrategy):
    """
    Implements a single-agent Retrieval-Augmented Generation (RAG) strategy
    using Azure AI Foundry. This class handles creating an agent, sending
    a user message, streaming the response, and cleaning up resources.
    """

    async def create():
        """
        Factory method to create an instance of SingleAgentRAGStrategyV1.
        Initializes the agent and tools.
        """
        logging.debug("[Agent Flow] Creating SingleAgentRAGStrategyV1 instance...")
        instance = SingleAgentRAGStrategyV1()

        return instance    

    def __init__(self):
        """
        Initialize base credentials and tools.
        """
        super().__init__()

        # Force all logs at DEBUG or above to appear
        logging.debug("[Init] Initializing SingleAgentRAGStrategyV1...")

        cfg = get_config()
        self.strategy_type = AgentStrategies.SINGLE_AGENT_RAG_V1
        
        # Agent Tools Initialization Section
        # =========================================================

        # Allow the user to specify an existing agent ID (optional)
        # Use a safe default to avoid raising when key is not present
        self.existing_agent_id = cfg.get("AGENT_ID", "") or None

        # Initialize tool containers
        self.tools_list = []
        self.tool_resources = {}

        # --- Initialize SearchClient for retrieval function tool ---
        aisearch_enabled = cfg.get("SEARCH_RETRIEVAL_ENABLED", True, type=bool)
        
        if not aisearch_enabled:
            logging.warning("[Init] SEARCH_RETRIEVAL_ENABLED set to false. SearchClient Function Tool will not be available.")
            self.search_client = None
        else:
            logging.info("[Init] Setting up SearchClient and custom retrieval function tool...")
            
            # Initialize SearchClient (handles embeddings internally)
            try:
                self.search_client = SearchClient()
                logging.info("[Init] ✅ SearchClient initialized with hybrid search support")
            except Exception as e:
                logging.error("[Init] ❌ Could not initialize SearchClient: %s", e)
                raise
        
            # Use SearchClient's search_knowledge_base method as the retrieval function
            retrieval_functions = {self.search_client.search_knowledge_base}
            retrieval_tool = FunctionTool(functions=retrieval_functions)
            
            # Add retrieval function definitions to tools list
            for tool_def in retrieval_tool.definitions:
                self.tools_list.append(tool_def)
                logging.debug(f"[Init] Added retrieval function tool: {tool_def}")
            
            logging.info("[Init] Custom retrieval function tool configured successfully")

        # --- Initialize BingGroundingTool (if configured) ---
        bing_enabled = cfg.get("BING_RETRIEVAL_ENABLED", False, type=bool)
        if not bing_enabled:
            logging.warning("[Init] BING_RETRIEVAL_ENABLED set to false. BingGroundingTool will not be available.")
        else:
            bing_conn = cfg.get("BING_CONNECTION_ID", "")
            if bing_conn:
                bing = BingGroundingTool(connection_id=bing_conn, count=5)
                bing_def = bing.definitions[0]
                self.tools_list.append(bing_def)
                logging.debug(f"[Init] Added BingGroundingTool to tools_list: {bing_def}")
            else:
                logging.error("[Init] BING_CONNECTION_ID not set in App Config variables. ")            
                
        logging.debug(f"[Init] Final tools_list: {self.tools_list}")
        logging.debug(f"[Init] Final tool_resources: {self.tool_resources}")


    async def initiate_agent_flow(self, user_message: str):
        """
        Initiates the agent flow using custom retrieval function.
        
        - Uses custom search_knowledge_base function as FunctionTool
        - Function is auto-executed when agent needs to search for information
        - Optionally uses BingGroundingTool if BING_CONNECTION_ID is configured
        """
        flow_start = time.time()
        logging.debug(f"[Agent Flow] invoke_stream called with user_message: {user_message!r}")
        conv = self.conversation
        thread_id = conv.get("thread_id")

        async with self.project_client as project_client:
            # Step 0: Register custom retrieval function for auto-execution
            if self.search_client:
                project_client.agents.enable_auto_function_calls({self.search_client.search_knowledge_base})
            
            # Step 1: Manage thread lifecycle (create or reuse)
            thread = await self._get_or_create_thread(project_client, thread_id)
            conv["thread_id"] = thread.id

            # Step 2: Create or reuse agent
            agent, create_agent = await self._get_or_create_agent(
                project_client
            )
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
            if create_agent:
                await self._cleanup_agent(project_client, agent.id)
            
            logging.info(f"[Agent Flow] Total flow time: {round(time.time() - flow_start, 2)}s")

    # ============================================================
    # Agent Flow Helper Methods (extracted from initiate_agent_flow)
    # ============================================================

    async def _get_or_create_thread(self, project_client, thread_id: Optional[str]):
        """
        Create a new thread or retrieve an existing one.
        
        Args:
            project_client: Azure AI Foundry project client
            thread_id: Existing thread ID or None
            
        Returns:
            Thread object (either newly created or retrieved)
        """
        try:
            if thread_id:
                logging.debug(f"[Agent Flow] thread_id exists; calling get(thread_id={thread_id})")
                thread = await project_client.agents.threads.get(thread_id)
                logging.info(f"[Agent Flow] Reused thread with ID: {thread.id}")
            else:
                logging.debug("[Agent Flow] thread_id not found; calling create()")
                thread = await project_client.agents.threads.create()
                logging.info(f"[Agent Flow] Created new thread with ID: {thread.id}")
            
            logging.debug(f"[Agent Flow] Stored thread.id = {thread.id}")
            return thread
        except Exception as e:
            logging.error(f"[Agent Flow] Failed to create/retrieve thread: {e}", exc_info=True)
            raise Exception(f"Thread creation failed: {str(e)}") from e

    async def _get_or_create_agent(self, project_client):
        """
        Create a new agent or retrieve an existing one.
        
        Args:
            project_client: Azure AI Foundry project client
            
        Returns:
            Tuple of (agent, create_agent_flag)
        """
        create_agent = False
        
        try:
            if self.existing_agent_id:
                logging.debug("[Agent Flow] agent_id exists; calling update_agent(...)")
                agent = await project_client.agents.get_agent(self.existing_agent_id)
                logging.info(f"[Agent Flow] Reused agent with ID: {agent.id}")
            else:
                logging.debug("[Agent Flow] creating agent(...)")
                cfg = get_config()
                bing_enabled = bool(cfg.get("BING_CONNECTION_ID", ""))
                aisearch_enabled = cfg.get("SEARCH_RETRIEVAL_ENABLED", True, type=bool)
                
                prompt_context = {
                    "strategy": self.strategy_type.value,
                    "user_context": self.user_context or {},
                    "bing_grounding_enabled": bing_enabled,
                    "aisearch_enabled": aisearch_enabled,
                }

                instructions = await self._read_prompt(
                    "main",
                    use_jinja2=True,
                    jinja2_context=prompt_context,
                )
                
                agent = await project_client.agents.create_agent(
                    model=self.model_name,
                    name="gpt-rag-agent",
                    instructions=instructions,
                    tools=self.tools_list,
                    tool_resources=self.tool_resources
                )
                create_agent = True
                logging.info(f"[Agent Flow] Created new agent with ID: {agent.id}")
            
            return agent, create_agent
        except Exception as e:
            logging.error(f"[Agent Flow] Failed to create/retrieve agent: {e}", exc_info=True)
            raise Exception(f"Agent creation failed: {str(e)}") from e

    async def _send_user_message(self, project_client, thread_id: str, user_message: str):
        """
        Send user message to the thread.
        
        Args:
            project_client: Azure AI Foundry project client
            thread_id: Target thread ID
            user_message: User's message text
        """
        try:
            logging.debug(f"[Agent Flow] Sending user message into thread {thread_id}: {user_message!r}")
            await project_client.agents.messages.create(
                thread_id=thread_id,
                role="user",
                content=user_message
            )
            logging.debug("[Agent Flow] User message sent.")
        except Exception as e:
            logging.error(f"[Agent Flow] Failed to send message to thread {thread_id}: {e}", exc_info=True)
            raise Exception(f"Message sending failed: {str(e)}") from e

    async def _stream_agent_response(self, project_client, agent_id: str, thread_id: str):
        """
        Stream agent response with citation processing.
        Captures Bing Grounding results to map citations for models that don't include annotations.
        """
        try:
            stream_start = time.time()
            tool_start = None
            response_start = None
            
            async with await project_client.agents.runs.stream(
                thread_id=thread_id,
                agent_id=agent_id
            ) as stream:
                async for event_type, event_data, raw in stream:
                    
                    # Track tool execution time
                    if event_type == "thread.run.step.created":
                        step_type = getattr(event_data, 'type', 'unknown')
                        if step_type == "tool_calls":
                            tool_start = time.time()
                    
                    if event_type == "thread.run.step.completed":
                        step_type = getattr(event_data, 'type', 'unknown')
                        if step_type == "tool_calls" and tool_start:
                            step_details = getattr(event_data, 'step_details', None)
                            if step_details:
                                for tc in getattr(step_details, 'tool_calls', []):
                                    tool_type = getattr(tc, 'type', 'unknown')
                                    logging.info(f"[Stream] Tool executed: {tool_type} ({round(time.time() - tool_start, 2)}s)")
                                        
                            tool_start = None
                        elif step_type == "message_creation" and response_start:
                            logging.info(f"[Stream] Response generated in {round(time.time() - response_start, 2)}s")
                    
                    # Track response generation time
                    if event_type == "thread.run.step.created":
                        step_type = getattr(event_data, 'type', 'unknown')
                        if step_type == "message_creation":
                            response_start = time.time()
                    
                    # Stream message deltas with citation processing
                    if event_type == "thread.message.delta" and hasattr(event_data, "text"):
                        chunk = event_data.text
                        
                        # Only process citations if chunk contains citation placeholder character
                        # This avoids unnecessary regex/annotation processing for most chunks
                        if chunk and '【' in chunk:
                            chunk = process_bing_citations(event_data)
                        
                        if not chunk:
                            chunk = raw or ""
                        if chunk:
                            yield chunk
                    
                    # Handle run failure
                    if event_type == "thread.run.failed":
                        err = event_data.last_error.message
                        logging.error(f"[Stream] Run failed: {err}")
                        raise Exception(err)
                
                logging.info(f"[Stream] Total streaming time: {round(time.time() - stream_start, 2)}s")
                
        except Exception as e:
            logging.error(f"[Orchestrator] Streaming failed: {e}", exc_info=True)
            raise Exception(f"Agent response streaming failed: {str(e)}") from e

    async def _consolidate_conversation_history(self, project_client, thread_id: str):
        """
        Fetch and consolidate conversation history from thread.
        
        Args:
            project_client: Azure AI Foundry project client
            thread_id: Thread ID to fetch history from
        """
        try:
            logging.debug("[Orchestrator] Fetching conversation history from thread...")
            conv = self.conversation
            conv["messages"] = []
            
            messages = project_client.agents.messages.list(
                thread_id=thread_id,
                order=ListSortOrder.ASCENDING
            )
            
            msg_count = 0
            total_chars = 0
            async for msg in messages:
                # Defensive check: skip messages with empty or missing content
                if not msg.content:
                    logging.debug(f"[Orchestrator] Skipping message {msg.id} with empty content")
                    continue
                
                # Debug: Log message structure to understand different model formats
                logging.debug(f"[Orchestrator] Message {msg.id}: role={msg.role}, content_items={len(msg.content)}")
                
                last_content = msg.content[-1]
                logging.debug(f"[Orchestrator] Last content type: {type(last_content).__name__}")
                
                if isinstance(last_content, MessageTextContent):
                    text_obj = last_content.text
                    text_val = text_obj.value
                    
                    # Debug: Check for annotations in the final message (may contain citations)
                    annotations = getattr(text_obj, 'annotations', None)
                    if annotations:
                        logging.debug(f"[Orchestrator] Found {len(annotations)} annotations in message {msg.id}")
                        for i, ann in enumerate(annotations):
                            logging.debug(f"[Orchestrator] Annotation {i}: {ann}")
                    else:
                        logging.debug(f"[Orchestrator] No annotations in message {msg.id}")
                    
                    msg_count += 1
                    total_chars += len(text_val)
                    
                    conv["messages"].append({
                        "role": msg.role,
                        "text": text_val
                    })
            
            logging.info(f"[Orchestrator] Retrieved {msg_count} messages ({total_chars:,} chars total)")

            if self.user_context:
                conv['user_context'] = self.user_context
        except Exception as e:
            logging.error(f"[Orchestrator] Failed to consolidate conversation history for thread {thread_id}: {e}", exc_info=True)
            # Don't raise here - conversation history is non-critical, log and continue

    async def _cleanup_agent(self, project_client, agent_id: str):
        """
        Delete temporary agent after completion.
        
        Args:
            project_client: Azure AI Foundry project client
            agent_id: Agent ID to delete
        """
        try:
            logging.debug(f"[Agent Flow] Deleting agent with ID: {agent_id}")
            await project_client.agents.delete_agent(agent_id)
            logging.debug("[Agent Flow] Agent deletion complete.")
        except Exception as e:
            logging.error(f"[Agent Flow] Failed to delete agent {agent_id}: {e}", exc_info=True)
            # Don't raise here - cleanup failure is non-critical, log and continue