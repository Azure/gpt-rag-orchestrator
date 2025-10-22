"""
Single Agent RAG Strategy for Azure AI Foundry

This module implements a Retrieval-Augmented Generation (RAG) strategy using Azure AI Search
and Azure AI Agents. It supports both traditional Azure AI Search and agentic retrieval.

Azure AI Search Field Mapping:
------------------------------
The AzureAISearchTool does NOT support custom field mapping configuration in the current SDK version.
The tool uses automatic field detection based on the index schema.

**Direct Index Queries for Filepath Retrieval**
This implementation uses the existing connectors.search.SearchClient to fetch real filepaths:

1. **Primary Strategy**: Direct Index Query (Async)
   - Uses document IDs from metadata.ids array
   - Queries Azure AI Search index via SearchClient.get_document()
   - Fetches the 'filepath' field directly from your index
   - Reuses existing SearchClient (lightweight, no extra connections)
   - Most reliable method - gets actual filepath value from your index

2. **Fallback Strategy**: URL Extraction (Sync)
   - Parses get_urls to extract filename from URL pattern
   - Used only if direct query fails or SearchClient unavailable
   - Pattern: /documents-<basename>-<ext>-

This ensures your 'filepath' field is used whenever possible, with a reliable fallback.

App Settings
---------------------
- ENABLE_AGENTIC_RETRIEVAL: Enable agentic retrieval (true/false)
- SEARCH_CONNECTION_ID: Azure AI Search connection ID
- SEARCH_RAG_INDEX_NAME: Azure AI Search index name
- SEARCH_TOP_K: Number of top results to retrieve (default: 5)
- SEARCH_SERVICE_QUERY_ENDPOINT: Azure AI Search endpoint (required for direct queries)
"""

import logging
import json
import re
from typing import Any, Optional
from urllib.parse import unquote

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
    AsyncAgentEventHandler,
    AzureAISearchQueryType,
    AzureAISearchTool,
    BingGroundingTool,
    FunctionTool,
    ToolSet,
    AgentsNamedToolChoice,
    AgentsNamedToolChoiceType,
    FunctionName,
    ListSortOrder,
    MessageDeltaChunk,
    MessageDeltaTextUrlCitationAnnotation,
    MessageTextContent,
    RunStep,
    ThreadMessage,
    ThreadRun,
)
from azure.search.documents.agent import KnowledgeAgentRetrievalClient
from azure.search.documents.agent.models import (
    KnowledgeAgentRetrievalRequest,
    KnowledgeAgentMessage,
    KnowledgeAgentMessageTextContent
)
from azure.identity import DefaultAzureCredential as SyncDefaultAzureCredential

from .base_agent_strategy import BaseAgentStrategy
from .agent_strategies import AgentStrategies

from dependencies import get_config

class SingleAgentRAGStrategy(BaseAgentStrategy):
    """
    Implements a single-agent Retrieval-Augmented Generation (RAG) strategy
    using Azure AI Foundry. This class handles creating an agent, sending
    a user message, streaming the response, and cleaning up resources.
    """

    async def create():
        """
        Factory method to create an instance of SingleAgentRAGStrategy.
        Initializes the agent and tools.
        """
        logging.debug("[Agent Flow] Creating SingleAgentRAGStrategy instance...")
        instance = SingleAgentRAGStrategy()

        return instance    

    def __init__(self):
        """
        Initialize base credentials and tools.
        """
        super().__init__()

        # Force all logs at DEBUG or above to appear
        logging.debug("[Init] Initializing SingleAgentRAGStrategy...")

        cfg = get_config()
        
        # Get Azure Search configuration for direct queries
        search_index = cfg.get("SEARCH_RAG_INDEX_NAME", "ragindex")
        
        # Import SearchClient from connectors
        from connectors import SearchClient
        search_client = None
        try:
            search_client = SearchClient()
            logging.info("[Init] ✅ SearchClient initialized for filepath queries")
        except Exception as e:
            logging.warning("[Init] ⚠️ Could not initialize SearchClient: %s", e)
        
        # Initialize citation/search helper with SearchClient for direct index queries
        self.search_parser = AISearchToolHelper(
            search_client=search_client,
            index_name=search_index if search_index else None
        )
        
        self.strategy_type = AgentStrategies.SINGLE_AGENT_RAG

        # Event handler for streaming responses
        self.event_handler = EventHandler(self.search_parser.tool_documents)
        
        # Log agentic retrieval configuration
        agentic_retrieval_setting = cfg.get("ENABLE_AGENTIC_RETRIEVAL", "false", str).lower()
        logging.info(f"[Init] Agentic Retrieval Setting: ENABLE_AGENTIC_RETRIEVAL={agentic_retrieval_setting}")

        self.agentic_retrieval_max_docs = cfg.get("AGENTIC_RETRIEVAL_MAX_DOCS", 5, int)
        self.agentic_retrieval_max_chars = cfg.get("AGENTIC_RETRIEVAL_MAX_CHARS", 2048, int)
        logging.info(
            "[Init] Agentic retrieval limits: max_docs=%s, max_chars=%s",
            self.agentic_retrieval_max_docs,
            self.agentic_retrieval_max_chars,
        )

        # Agent Tools Initialization Section
        # =========================================================

        # Allow the user to specify an existing agent ID (optional)
        # Use a safe default to avoid raising when key is not present
        self.existing_agent_id = cfg.get("AGENT_ID", "") or None

        # Initialize tool containers
        self.tools_list = []
        self.tool_resources = {}

        # --- Load Agentic Retrieval Configuration ---
        self.enable_agentic_retrieval = cfg.get("ENABLE_AGENTIC_RETRIEVAL", "false", str).lower() == "true"
        logging.debug(f"[Init] Agentic Retrieval Enabled: {self.enable_agentic_retrieval}")

        self.agentic_retrieval_client = None
        self._recent_thread_messages = {}
        self._last_user_message_by_thread = {}
        self.agentic_retrieval_credential = None

        if self.enable_agentic_retrieval:
            self.search_query_endpoint = cfg.get("SEARCH_SERVICE_QUERY_ENDPOINT", "")
            self.knowledge_agent_name = cfg.get("SEARCH_SERVICE_AGENT_NAME", "")
            if not self.knowledge_agent_name:
                resource_token = cfg.get("RESOURCE_TOKEN", "")
                if resource_token:
                    self.knowledge_agent_name = f"ragindex-{resource_token}-rag-agent"
                else:
                    logging.warning("[Init] RESOURCE_TOKEN not found. Cannot construct default knowledge agent name.")
            logging.debug(f"[Init] Knowledge Agent Name: {self.knowledge_agent_name}")
            logging.debug(f"[Init] Search Query Endpoint: {self.search_query_endpoint}")
            if not self.search_query_endpoint or not self.knowledge_agent_name:
                logging.error(
                    "[Init] Agentic retrieval is enabled but SEARCH_SERVICE_QUERY_ENDPOINT or agent name is not configured. Falling back to traditional search."
                )
                self.enable_agentic_retrieval = False
            else:
                try:
                    self.agentic_retrieval_credential = SyncDefaultAzureCredential(
                        exclude_interactive_browser_credential=True
                    )
                except Exception as cred_error:
                    logging.error(
                        "[Init] Unable to initialize synchronous credential for agentic retrieval: %s",
                        cred_error,
                        exc_info=True,
                    )
                    self.enable_agentic_retrieval = False
                    logging.error("[Init] Agentic retrieval disabled due to credential initialization failure")

        # --- Initialize BingGroundingTool (if configured) ---
        bing_conn = cfg.get("BING_CONNECTION_ID", "")
        if not bing_conn:
            logging.warning(
                "[Init] BING_CONNECTION_ID not set in App Config variables. "
                "BingGroundingTool will not be available."
            )
        else:
            bing = BingGroundingTool(connection_id=bing_conn, count=5)
            bing_def = bing.definitions[0]
            self.tools_list.append(bing_def)
            logging.debug(f"[Init] Added BingGroundingTool to tools_list: {bing_def}")

        # --- Initialize AzureAISearchTool (only if agentic retrieval is disabled) ---
        if not self.enable_agentic_retrieval:
            azure_ai_conn_id = cfg.get("SEARCH_CONNECTION_ID", "")
            index_name = cfg.get("SEARCH_RAG_INDEX_NAME", "ragindex")
            top_k = cfg.get("SEARCH_TOP_K", 5, int)
            
            logging.info(f"[Init] Configuring AzureAISearchTool:")
            logging.info(f"       Connection ID: {azure_ai_conn_id[:20]}..." if len(azure_ai_conn_id) > 20 else f"       Connection ID: {azure_ai_conn_id}")
            logging.info(f"       Index Name: {index_name}")
            logging.info(f"       Query Type: SIMPLE")
            logging.info(f"       Top K: {top_k}")
            logging.info(f"       Note: AzureAISearchTool does not support custom field mapping")
            logging.info(f"             Filenames will be extracted from document URLs")
            
            if not azure_ai_conn_id:
                logging.warning(
                    "[Init] SEARCH_CONNECTION_ID not configured. "
                    "AzureAISearchTool will be unavailable."
                )
            if not index_name:
                logging.warning(
                    "[Init] SEARCH_RAG_INDEX_NAME not configured. "
                    "AzureAISearchTool will be unavailable."
                )
            
            self.ai_search = AzureAISearchTool(
                index_connection_id=azure_ai_conn_id,
                index_name=index_name,
                query_type=AzureAISearchQueryType.SIMPLE,
                top_k=top_k,
                filter=""
            )
            ai_def = self.ai_search.definitions[0]
            ai_res = self.ai_search.resources
            self.tools_list.append(ai_def)
            self.tool_resources.update(ai_res)
            logging.info(f"[Init] AzureAISearchTool initialized successfully")
        else:
            logging.info("[Init] Using Agentic Retrieval - AzureAISearchTool will not be initialized")

        logging.debug(f"[Init] Final tools_list: {self.tools_list}")
        logging.debug(f"[Init] Final tool_resources: {self.tool_resources}")

    async def initiate_agent_flow(self, user_message: str):
        """
        Initiates the agent flow with dual behavior based on agentic retrieval setting.
        
        Implementation follows Microsoft's official tutorial pattern:
        https://learn.microsoft.com/en-us/azure/search/tutorial-rag-build-solution-agent-to-agent
        
        1. AGENTIC RETRIEVAL ENABLED (self.enable_agentic_retrieval = True):
           - Uses KnowledgeAgentRetrievalClient for intelligent retrieval
           - Creates agentic_retrieval function with proper docstring
           - Wraps function in FunctionTool and adds to ToolSet
           - Registers ToolSet via enable_auto_function_calls()
           - FORCES tool use via AgentsNamedToolChoice in stream params
           - Passes toolset to stream() for execution
           - Traditional AzureAISearchTool is NOT initialized
           - CITATION PROCESSING: Pre-processes citations via _inject_citation_labels()
             * Injects "citation" field into JSON payload during retrieval
             * Registers documents for EventHandler placeholder normalization
             * Agent receives pre-formatted citations from tool response
        
        2. AGENTIC RETRIEVAL DISABLED (self.enable_agentic_retrieval = False):
           - Uses traditional AzureAISearchTool
           - Tool is configured during __init__ and added to tools_list
           - No custom function tools are created
           - Agent uses built-in AzureAISearchTool capabilities
           - CITATION PROCESSING: Post-processes citations after streaming
             * Captures tool outputs via capture_tool_outputs()
             * Removes inline citations and agent-provided Sources section
             * Appends corrected Sources section with real filenames
        
        Both modes can optionally use BingGroundingTool if BING_CONNECTION_ID is configured.
        """
        logging.debug(f"[Agent Flow] invoke_stream called with user_message: {user_message!r}")
        conv = self.conversation
        thread_id = conv.get("thread_id")
        logging.debug(f"[Agent Flow] Current conversation state: thread_id={thread_id}")

        async with self.project_client as project_client:
            # Step 1: Manage thread lifecycle (create or reuse)
            thread = await self._get_or_create_thread(project_client, thread_id)
            conv["thread_id"] = thread.id

            # Step 2: Setup agentic retrieval tools if enabled
            agentic_tool_definition, agentic_function, agentic_toolset = await self._setup_agentic_retrieval_tools(
                project_client, thread.id
            )

            # Step 3: Create or reuse agent
            agent, create_agent = await self._get_or_create_agent(
                project_client, agentic_tool_definition
            )
            conv["agent_id"] = agent.id

            # Step 4: Send user message to thread
            await self._send_user_message(project_client, thread.id, user_message)

            # Step 5: Cache conversation context for agentic retrieval
            if self.enable_agentic_retrieval:
                await self._cache_conversation_context(project_client, thread.id, user_message)

            # Step 6: Stream agent response
            async for chunk in self._stream_agent_response(
                project_client, agent.id, thread.id, agentic_toolset
            ):
                yield chunk

            # Step 7: Consolidate conversation history
            await self._consolidate_conversation_history(project_client, thread.id)

            # Step 8: Cleanup temporary agent if created
            if create_agent:
                await self._cleanup_agent(project_client, agent.id)

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

    async def _setup_agentic_retrieval_tools(self, project_client, thread_id: str):
        """
        Setup agentic retrieval tools if enabled.
        
        Args:
            project_client: Azure AI Foundry project client
            thread_id: Current thread ID for retrieval context
            
        Returns:
            Tuple of (tool_definition, function, toolset) or (None, None, None)
        """
        agentic_tool_definition = None
        agentic_function = None
        agentic_toolset = None
        
        if self.enable_agentic_retrieval:
            logging.info("[Agent Flow] Setting up agentic retrieval tool...")
            result = self._create_agentic_retrieval_tool(project_client, thread_id)
            if result:
                agentic_tool_definition, agentic_function, agentic_toolset = result
                # CRITICAL: enable_auto_function_calls expects a SET or LIST of Python functions
                project_client.agents.enable_auto_function_calls({agentic_function})
                logging.info(f"[Agent Flow] Agentic retrieval function registered: {agentic_function.__name__}")
                logging.info(f"[Agent Flow]   - Tool definition will be added to agent")
            else:
                logging.warning("[Agent Flow] Failed to create agentic retrieval functions, proceeding without it")
        
        return agentic_tool_definition, agentic_function, agentic_toolset

    async def _get_or_create_agent(self, project_client, agentic_tool_definition):
        """
        Create a new agent or retrieve an existing one.
        
        Args:
            project_client: Azure AI Foundry project client
            agentic_tool_definition: Tool definition for agentic retrieval (if enabled)
            
        Returns:
            Tuple of (agent, create_agent_flag)
        """
        create_agent = False
        
        if self.existing_agent_id:
            logging.debug("[Agent Flow] agent_id exists; calling update_agent(...)")
            agent = await project_client.agents.get_agent(self.existing_agent_id)
            logging.info(f"[Agent Flow] Reused agent with ID: {agent.id}")
        else:
            logging.debug("[Agent Flow] creating agent(...)")
            # Use enhanced instructions for retrieval behavior
            prompt_context = {
                "enable_agentic_retrieval": self.enable_agentic_retrieval,
                "strategy": self.strategy_type.value,
                "user_context": self.user_context or {},
            }

            instructions = await self._read_prompt(
                "main",
                use_jinja2=True,
                jinja2_context=prompt_context,
            )
            
            # Prepare tools list - add agentic retrieval tool if enabled
            tools_list = self.tools_list.copy()
            if self.enable_agentic_retrieval and agentic_tool_definition:
                tools_list.append(agentic_tool_definition)
                logging.info(f"[Agent Flow] Added agentic_retrieval to agent tools list")
            
            agent = await project_client.agents.create_agent(
                model=self.model_name,
                name="gpt-rag-agent",
                instructions=instructions,
                tools=tools_list,
                tool_resources=self.tool_resources
            )
            create_agent = True
            logging.info(f"[Agent Flow] Created new agent with ID: {agent.id}")
        
        return agent, create_agent

    async def _send_user_message(self, project_client, thread_id: str, user_message: str):
        """
        Send user message to the thread.
        
        Args:
            project_client: Azure AI Foundry project client
            thread_id: Target thread ID
            user_message: User's message text
        """
        logging.debug(f"[Agent Flow] Sending user message into thread {thread_id}: {user_message!r}")
        await project_client.agents.messages.create(
            thread_id=thread_id,
            role="user",
            content=user_message
        )
        logging.debug("[Agent Flow] User message sent.")

    async def _cache_conversation_context(self, project_client, thread_id: str, user_message: str):
        """
        Cache conversation context for agentic retrieval.
        
        Args:
            project_client: Azure AI Foundry project client
            thread_id: Current thread ID
            user_message: User's message text
        """
        self._last_user_message_by_thread[thread_id] = user_message
        cached_messages = await AgenticRetrievalToolHelper.capture_recent_thread_messages(
            project_client, thread_id
        )
        if cached_messages:
            self._recent_thread_messages[thread_id] = cached_messages
        else:
            logging.warning(
                "[Retrieval] Agentic retrieval cache is empty for thread %s; retrieval will rely on tool arguments",
                thread_id,
            )

    async def _stream_agent_response(self, project_client, agent_id: str, thread_id: str, agentic_toolset):
        """
        Stream agent response with citation processing.
        
        Args:
            project_client: Azure AI Foundry project client
            agent_id: Agent ID
            thread_id: Thread ID
            agentic_toolset: Toolset for agentic retrieval (if enabled)
        """
        logging.info(f"[Orchestrator] Starting streaming run for agent_id={agent_id}, thread_id={thread_id}")
        
        # ONLY clear and track documents in traditional mode
        if not self.enable_agentic_retrieval:
            self.search_parser.clear_tracked_documents()
        
        # Buffer to accumulate streamed text so we can emit corrected sources at the end
        streamed_text_buffer: list[str] = []
        dropping_sources_section = False
        
        stream_params = {
            "thread_id": thread_id,
            "agent_id": agent_id,
            "event_handler": self.event_handler
        }
        
        # Add tool_choice for agentic retrieval to force it to be called first
        # Note: toolset is NOT passed here - functions are already registered via enable_auto_function_calls
        if self.enable_agentic_retrieval and agentic_toolset:
            stream_params["tool_choice"] = AgentsNamedToolChoice(
                type=AgentsNamedToolChoiceType.FUNCTION,
                function=FunctionName(name="agentic_retrieval")
            )
            logging.info("[Orchestrator] Agentic retrieval mode: Agent manages citations natively")
        else:
            logging.info("[Orchestrator] Traditional search mode: Will post-process citations")
        
        async with await project_client.agents.runs.stream(**stream_params) as stream:
            logging.debug("[Orchestrator] Streaming context opened, waiting for events...")
            event_count = 0
            message_chunks = 0
            
            async for event_type, event_data, raw in stream:
                event_count += 1
                
                # Process run step events
                await self._process_run_step_events(event_type, event_data)
                
                # Process message delta events
                chunk_result = await self._process_message_delta(
                    event_type, event_data, raw, streamed_text_buffer, dropping_sources_section
                )
                if chunk_result:
                    cleaned_chunk, dropping_sources_section = chunk_result
                    if cleaned_chunk:
                        message_chunks += 1
                        yield cleaned_chunk
                
                # Handle run failure
                if event_type == "thread.run.failed":
                    err = event_data.last_error.message
                    logging.error(f"[Stream] Run failed: {err}")
                    raise Exception(err)
            
            logging.info(f"[Orchestrator] Streaming completed: {event_count} events, {message_chunks} text chunks")

            # ONLY emit canonical sources in traditional mode
            if not self.enable_agentic_retrieval:
                sources_section = self.search_parser._build_sources_section()
                if sources_section:
                    # Clean up the final text: remove spacing artifacts left by citation removal
                    final_text = "".join(streamed_text_buffer)
                    
                    # 1. Normalize multiple spaces to single space
                    final_text = re.sub(r" {2,}", " ", final_text)
                    
                    # 2. Remove spaces before punctuation
                    final_text = re.sub(r"\s+([.,;:!?])", r"\1", final_text)
                    
                    # 3. Remove trailing spaces before newlines
                    final_text = re.sub(r" +\n", "\n", final_text)
                    
                    # Update the last item in buffer with cleaned version
                    if streamed_text_buffer:
                        streamed_text_buffer.clear()
                        streamed_text_buffer.append(final_text)
                    
                    prefix = "\n\n"
                    if final_text.endswith("\n"):
                        prefix = "\n"
                    logging.info("[Citations STREAM] Streaming canonical sources: %s", sources_section)
                    yield prefix + sources_section

    async def _process_run_step_events(self, event_type: str, event_data):
        """
        Process run step events for logging and citation capture.
        
        Args:
            event_type: Type of streaming event
            event_data: Event data object
        """
        if event_type == "thread.run.step.created":
            step_type = getattr(event_data, 'type', 'unknown')
            logging.info(f"[Stream] Step started: {step_type}")
        elif event_type == "thread.run.step.completed":
            step_type = getattr(event_data, 'type', 'unknown')
            logging.info(f"[Stream] Step completed: {step_type}")
            # Only capture tool outputs in traditional mode
            if not self.enable_agentic_retrieval:
                try:
                    import json
                    step_dict = event_data.as_dict() if hasattr(event_data, 'as_dict') else {}
                    logging.info("[Citations] RunStep dump: %s", json.dumps(step_dict, indent=2, default=str)[:2000])
                except Exception as dump_exc:
                    logging.warning("[Citations] Could not dump RunStep: %s", dump_exc)
                self.search_parser.capture_tool_outputs(event_data)
                # Fetch real filepaths from index asynchronously
                await self.search_parser.fetch_pending_filepaths()
        elif event_type in ["thread.run.created", "thread.run.queued", "thread.run.in_progress"]:
            status = getattr(event_data, 'status', 'unknown')
            logging.info(f"[Stream] Run: {event_type.split('.')[-1]} (status={status})")
        elif event_type == "thread.run.completed":
            logging.info(f"[Stream] Run completed successfully")
        elif "tool_calls" in event_type and event_type != "thread.run.step.delta":
            logging.info(f"[Stream] Tool event: {event_type}")

    async def _process_message_delta(
        self, event_type: str, event_data, raw, streamed_text_buffer: list[str], dropping_sources_section: bool
    ):
        """
        Process message delta events and handle citation normalization.
        
        Args:
            event_type: Type of streaming event
            event_data: Event data object
            raw: Raw chunk text
            streamed_text_buffer: Buffer for accumulating text
            dropping_sources_section: Flag indicating if we're dropping agent sources
            
        Returns:
            Tuple of (cleaned_chunk, dropping_sources_section) or None
        """
        if event_type != "thread.message.delta" or not hasattr(event_data, "text"):
            return None
        
        chunk = raw or "".join(event_data.text)

        # In agentic mode, pass chunks through with minimal processing
        # (EventHandler still normalizes residual placeholders)
        if self.enable_agentic_retrieval:
            if chunk:
                streamed_text_buffer.append(chunk)
                return chunk, dropping_sources_section
            return None

        # TRADITIONAL MODE ONLY: Process citations
        # Process chunk through EventHandler to normalize inline citations
        processed_chunk = await self.event_handler.on_message_delta(event_data)
        if processed_chunk is None:
            processed_chunk = chunk

        # Strip inline markdown links that correspond to citation placeholders
        cleaned_chunk = self.search_parser.remove_inline_citations(processed_chunk)
        
        if cleaned_chunk != processed_chunk:
            logging.debug("[Citations STREAM] Stripped inline markdown links from chunk")

        # Remove any agent-provided Sources section (will be replaced later)
        if dropping_sources_section:
            if cleaned_chunk.strip():
                logging.debug("[Citations STREAM] Skipping chunk belonging to agent-provided Sources section")
            return None, dropping_sources_section

        sources_idx = -1
        for marker in ("**Sources:**", "Sources:"):
            idx = cleaned_chunk.find(marker)
            if idx == -1:
                continue
            if idx == 0 or cleaned_chunk[idx - 1] in {"\n", "\r"}:
                sources_idx = idx
                break
        
        if sources_idx != -1:
            logging.info("[Citations STREAM] Dropping agent-provided Sources section from stream")
            cleaned_chunk = cleaned_chunk[:sources_idx]
            dropping_sources_section = True
            if not cleaned_chunk.strip():
                return None, dropping_sources_section

        if cleaned_chunk:
            streamed_text_buffer.append(cleaned_chunk)
            return cleaned_chunk, dropping_sources_section
        
        return None, dropping_sources_section

    async def _consolidate_conversation_history(self, project_client, thread_id: str):
        """
        Fetch and consolidate conversation history from thread.
        
        Args:
            project_client: Azure AI Foundry project client
            thread_id: Thread ID to fetch history from
        """
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
            if isinstance(msg.content[-1], MessageTextContent):
                text_val = msg.content[-1].text.value
                msg_count += 1
                total_chars += len(text_val)
                
                # ONLY fix citations in traditional mode
                if msg.role == "assistant" and not self.enable_agentic_retrieval:
                    text_val = self.search_parser.apply_tool_results_to_sources(
                        text_val, self.enable_agentic_retrieval
                    )
                
                conv["messages"].append({
                    "role": msg.role,
                    "text": text_val
                })
        
        logging.info(f"[Orchestrator] Retrieved {msg_count} messages ({total_chars:,} chars total)")

        if self.user_context:
            conv['user_context'] = self.user_context

    async def _cleanup_agent(self, project_client, agent_id: str):
        """
        Delete temporary agent after completion.
        
        Args:
            project_client: Azure AI Foundry project client
            agent_id: Agent ID to delete
        """
        logging.debug(f"[Agent Flow] Deleting agent with ID: {agent_id}")
        await project_client.agents.delete_agent(agent_id)
        logging.debug("[Agent Flow] Agent deletion complete.")

    # ============================================================
    # Agentic Retrieval Tool Creation
    # ============================================================

    def _create_agentic_retrieval_tool(self, project_client, thread_id: str):
        if not self.enable_agentic_retrieval:
            logging.warning("[Retrieval] Agentic retrieval is not enabled. Tool will not be created.")
            return None
        try:
            if not self.agentic_retrieval_credential:
                logging.error("[Retrieval] Synchronous credential not available for agentic retrieval client initialization")
                return None

            self.agentic_retrieval_client = KnowledgeAgentRetrievalClient(
                endpoint=self.search_query_endpoint,
                agent_name=self.knowledge_agent_name,
                credential=self.agentic_retrieval_credential
            )
            logging.info(f"[Retrieval] KnowledgeAgentRetrievalClient initialized for agent: {self.knowledge_agent_name}")
            
            def agentic_retrieval(query: Optional[str] = None) -> str:
                """
                Search and retrieve relevant information from the knowledge base to answer user questions.
                Use this function whenever you need to find information to answer the user's query.
                This function will return relevant documents with citations that you should use in your response.
                
                Returns:
                    str: JSON array of documents with ref_id, content, and metadata fields.
                """
                try:
                    logging.info(f"[agentic_retrieval] Function called for thread: {thread_id}")
                    
                    # Get messages from thread - following Microsoft's official example
                    # Take the last 5 messages in the conversation
                    converted_messages = self._recent_thread_messages.get(thread_id, [])
                    if not converted_messages:
                        logging.warning("[agentic_retrieval] No cached messages found for thread; falling back to last user message")
                        fallback_text = query or self._last_user_message_by_thread.get(thread_id, "")
                        if fallback_text:
                            converted_messages = [
                                KnowledgeAgentMessage(
                                    role="user",
                                    content=[KnowledgeAgentMessageTextContent(text=fallback_text)]
                                )
                            ]
                        else:
                            logging.error("[agentic_retrieval] Unable to determine request context for retrieval")
                            return json.dumps([{
                                "ref_id": 0,
                                "content": "Unable to determine the user query for retrieval.",
                                "title": "Retrieval Error"
                            }])
                    logging.debug(f"[Retrieval] Converted {len(converted_messages)} messages for retrieval request (query override provided: {bool(query)})")
                    
                    # Log the messages being sent to agentic retrieval
                    logging.info(f"[Retrieval] Calling retrieve with {len(converted_messages)} messages")
                    for idx, msg in enumerate(converted_messages):
                        logging.debug(f"[Retrieval] Message {idx}: role={msg.role}, content_preview={msg.content[0].text[:100] if msg.content else 'empty'}...")
                    
                    retrieval_result = self.agentic_retrieval_client.retrieve(
                        retrieval_request=KnowledgeAgentRetrievalRequest(
                            messages=converted_messages
                        )
                    )
                    
                    logging.info(f"[Retrieval] Retrieval completed successfully")
                    
                    if retrieval_result.response and len(retrieval_result.response) > 0:
                        response_content = retrieval_result.response[0].content
                        if response_content and len(response_content) > 0:
                            result_text = response_content[0].text
                            logging.debug(
                                "[Retrieval] Raw retrieval payload (truncated): %s",
                                result_text[:200],
                            )

                            transformed_text = AgenticRetrievalToolHelper._inject_citation_labels(
                                result_text, retrieval_result, self.search_parser
                            )
                            limited_text = AgenticRetrievalToolHelper.apply_agentic_retrieval_limits(
                                transformed_text,
                                self.agentic_retrieval_max_docs,
                                self.agentic_retrieval_max_chars,
                            )

                            logging.info(
                                "[Retrieval] Returning %d characters of transformed content",
                                len(limited_text),
                            )
                            if hasattr(retrieval_result, 'activity') and retrieval_result.activity:
                                logging.info(
                                    "[Retrieval] Activity: %d operations logged",
                                    len(retrieval_result.activity),
                                )
                                for activity in retrieval_result.activity[:3]:
                                    activity_type = getattr(activity, 'type', 'unknown')
                                    logging.debug(f"[Retrieval] Activity type: {activity_type}")
                            return limited_text
                    
                    logging.warning("[Retrieval] Empty response from retrieval service")
                    return json.dumps([
                        {
                            "ref_id": 0,
                            "content": "No information found.",
                            "title": "Empty Result",
                            "citation": "No Source",
                        }
                    ])
                except Exception as e:
                    logging.error(f"[Retrieval] Error during retrieval: {str(e)}", exc_info=True)
                    return json.dumps([
                        {
                            "ref_id": 0,
                            "content": "Unable to retrieve information at this time. Please try again.",
                            "title": "Retrieval Error",
                            "citation": "Retrieval Error",
                        }
                    ])
            
            # Create FunctionTool and ToolSet following Microsoft's pattern
            # - FunctionTool.definitions[0] for agent.tools list
            # - agentic_retrieval function for enable_auto_function_calls
            # - ToolSet for potential future use
            functions = FunctionTool({agentic_retrieval})
            toolset = ToolSet()
            toolset.add(functions)
            
            logging.info("[Retrieval] Agentic retrieval tool created successfully")
            # Return: (function definition, function, toolset)
            return (functions.definitions[0], agentic_retrieval, toolset)
        except Exception as e:
            logging.error(f"[Retrieval] Failed to create agentic retrieval tool: {str(e)}", exc_info=True)
            return None


class EventHandler(AsyncAgentEventHandler[str]):
    """
    Streams event handler for agent lifecycle events and citation processing.
    
    Responsibilities:
    - Intercepts message deltas during streaming
    - Normalizes citation annotations from Azure SDK
    - Replaces citation placeholders with human-readable links
    - Cleans up weird citation patterns (e.g., "(3:2†source)")
    
    Context Flow:
    - Initialized with reference to AISearchToolHelper.tool_documents
    - This shared dict maps placeholders → (title, filename)
    - Populated by either:
      * Traditional mode: AISearchToolHelper.capture_tool_outputs()
      * Agentic mode: AgenticRetrievalToolHelper._inject_citation_labels()
    - Used here to replace doc_N placeholders with real filenames during streaming
    
    Note: In agentic mode, most citations are handled natively by the agent.
    This handler still processes residual placeholders for consistency.
    """
    
    def __init__(self, tool_documents: dict = None):
        """
        Initialize the event handler with citation document registry.
        
        Args:
            tool_documents: Shared dict mapping citation placeholders to (title, filename).
                          This is typically AISearchToolHelper.tool_documents which is
                          actually CitationDocumentHelper.tool_documents under the hood.
        """
        super().__init__()
        self.tool_documents = tool_documents or {}

    async def on_message_delta(self, delta: MessageDeltaChunk) -> Optional[str]:
        """ 
        Called when a partial message is received.
        :param delta: Chunk of the message text.
        :return: The text chunk.
        """
        text = delta.text

        # Collect annotation objects, if any
        raw = getattr(delta, "delta", None)
        annotations = []
        if raw:
            content_pieces = getattr(raw, "content", [])
            for piece in content_pieces:
                txt = getattr(piece, "text", None)
                if not txt:
                    continue
                anns = getattr(txt, "annotations", None)
                if not anns:
                    continue
                annotations.extend(anns)
        
        if annotations:
            logging.debug(f"[Citations] Processing {len(annotations)} citation annotation(s)")
        
        for idx, ann in enumerate(annotations):
            # Check annotation type
            if hasattr(ann, 'type'):
                logging.debug(f"[Citations] Annotation {idx + 1} type: {ann.type}")
            
            # Check for file_citation (Azure AI Search uses this)
            if hasattr(ann, 'file_citation') or (hasattr(ann, '__contains__') and 'file_citation' in ann):
                try:
                    file_info = ann.get('file_citation') if hasattr(ann, 'get') else getattr(ann, 'file_citation', None)
                    placeholder = ann.get('text') if hasattr(ann, 'get') else getattr(ann, 'text', None)
                    
                    logging.debug(f"[Citations] File citation detected:")
                    logging.debug(f"[Citations]   - placeholder: '{placeholder}'")
                    logging.debug(f"[Citations]   - file_info: {file_info}")
                    
                    if file_info and placeholder:
                        # Extract file_id which should contain the filepath
                        file_id = file_info.get('file_id') if hasattr(file_info, 'get') else getattr(file_info, 'file_id', None)
                        
                        logging.debug(f"[Citations]   - file_id: '{file_id}'")
                        
                        if file_id and placeholder:
                            # Check if file_id is a doc_N placeholder and resolve to real filename
                            # If not yet in tool_documents, keep the placeholder (will be resolved later)
                            real_filename = file_id
                            real_title = file_id
                            
                            if file_id in self.tool_documents:
                                real_title, real_filename = self.tool_documents[file_id]
                                logging.debug(f"[Citations]   - Resolved '{file_id}' → title='{real_title}', file='{real_filename}'")
                            else:
                                logging.debug(f"[Citations]   - Keeping placeholder '{file_id}' (not yet in tool_documents)")
                            
                            # Clean up the title: remove underscores and any file extension
                            display_title = real_title
                            if display_title:
                                # Replace underscores with spaces
                                display_title = display_title.replace('_', ' ')
                                # Remove any file extension (everything after the last dot)
                                display_title = re.sub(r'\.[^.]+$', '', display_title)
                            
                            citation = f"[{display_title}]({real_filename})"
                            text = text.replace(placeholder, citation)
                            logging.debug(f"[Citations] File citation replaced: '{placeholder}' → {citation}")
                            continue
                        else:
                            logging.warning(f"[Citations] Missing file_id or placeholder in file_citation")
                except Exception as e:
                    logging.error(f"[Citations] Error processing file_citation: {e}", exc_info=True)
            
            # Check for url_citation (alternative format)
            if isinstance(ann, MessageDeltaTextUrlCitationAnnotation) or (hasattr(ann, '__contains__') and 'url_citation' in ann):
                try:
                    info = ann.get('url_citation') if hasattr(ann, 'get') else getattr(ann, 'url_citation', None)
                    placeholder = ann.get('text') if hasattr(ann, 'get') else getattr(ann, 'text', None)
                    
                    logging.debug(f"[Citations] URL citation detected:")
                    logging.debug(f"[Citations]   - placeholder: '{placeholder}'")
                    logging.debug(f"[Citations]   - info: {info}")
                    
                    if info and placeholder:
                        url = info.get('url') if hasattr(info, 'get') else getattr(info, 'url', None)
                        title = info.get('title', url) if hasattr(info, 'get') else getattr(info, 'title', url)
                        
                        logging.debug(f"[Citations]   - url: '{url}', title: '{title}'")
                        
                        if url and placeholder:
                            # Check if url is a doc_N placeholder and resolve to real filename
                            # If not yet in tool_documents, keep the placeholder (will be resolved later)
                            real_filename = url
                            real_title = title
                            
                            if url in self.tool_documents:
                                real_title, real_filename = self.tool_documents[url]
                                logging.debug(f"[Citations]   - Resolved '{url}' → title='{real_title}', file='{real_filename}'")
                            else:
                                logging.debug(f"[Citations]   - Keeping placeholder '{url}' (not yet in tool_documents)")
                            
                            # Clean up the title: remove underscores and any file extension
                            display_title = real_title
                            if display_title:
                                # Replace underscores with spaces
                                display_title = display_title.replace('_', ' ')
                                # Remove any file extension (everything after the last dot)
                                display_title = re.sub(r'\.[^.]+$', '', display_title)
                            
                            citation = f"[{display_title}]({real_filename})"
                            text = text.replace(placeholder, citation)
                            logging.debug(f"[Citations] URL citation replaced: '{placeholder}' → {citation}")
                            continue
                        else:
                            logging.warning(f"[Citations] Missing url or placeholder in url_citation")
                except Exception as e:
                    logging.error(f"[Citations] Error processing url_citation: {e}", exc_info=True)
            
            # If we got here, annotation wasn't processed
            logging.warning(f"[Citations] ⚠️ Annotation {idx + 1} not processed - unknown format")

        # After processing annotations, replace any remaining doc_N or weird citation placeholders
        if self.tool_documents and text:
            original_text = text
            
            # Replace inline doc_N references in markdown links
            # Pattern: [Title](doc_N) → [Title](Real_Filename.pdf)
            for doc_id, (title, filename) in self.tool_documents.items():
                doc_link_pattern = re.compile(r'\[([^\]]+)\]\(' + re.escape(doc_id) + r'\)')
                text = doc_link_pattern.sub(rf'[\1]({filename})', text)
            
            # Replace weird placeholders like (3:2†source) with proper citations
            # Pattern: (3:2†source) where 3 is the index and 2 is the doc number
            weird_citation_pattern = re.compile(r'\((\d+):(\d+)†[^)]*\)')
            
            def replace_weird_citation(match):
                doc_index = int(match.group(2))
                doc_key = f"doc_{doc_index}"
                if doc_key in self.tool_documents:
                    title, filename = self.tool_documents[doc_key]
                    logging.info("[Citations STREAM] Replaced weird citation in stream: %s → [%s](%s)", match.group(0), title, filename)
                    return f"[{title}]({filename})"
                return match.group(0)
            
            text = weird_citation_pattern.sub(replace_weird_citation, text)
            
            if text != original_text:
                logging.info("[Citations STREAM] Text modified during streaming: '%s...' → '%s...'", original_text[:50], text[:50])

        return text

    async def on_thread_message(self, message: ThreadMessage) -> Optional[str]:
        """
        Called when a new thread message object is created.
        :param message: The ThreadMessage instance.
        :return: Summary including message ID and status.
        """
        # Only log at DEBUG level - these are captured by stream loop
        return None

    async def on_thread_run(self, run: ThreadRun) -> Optional[str]:
        """
        Called when a new thread run event occurs.
        :param run: The ThreadRun instance.
        :return: Summary of the run status.
        """
        # Only log at DEBUG level - these are captured by stream loop
        return None

    async def on_run_step(self, step: RunStep) -> Optional[str]:
        """
        Called at each step of the run pipeline.
        :param step: The RunStep instance.
        :return: Type and status of the step.
        """
        # Only log at DEBUG level - these are captured by stream loop
        return None

    async def on_error(self, data: str) -> Optional[str]:
        """
        Called when an error occurs during the stream.
        :param data: Error information.
        :return: Formatted error message.
        """
        logging.debug(f"EventHandler.on_error called with data={data!r}")
        return f"Error in stream: {data}"

    async def on_done(self) -> Optional[str]:
        """
        Called when the streaming completes successfully.
        :return: Completion message.
        """
        logging.debug("EventHandler.on_done called")
        return "Streaming completed"

    async def on_unhandled_event(self, event_type: str, event_data: Any) -> Optional[str]:
        """
        Catches any events not handled by other methods.
        :param event_type: The type identifier of the event.
        :param event_data: The raw event payload.
        :return: Description of the unhandled event.
        """
        logging.debug(f"EventHandler.on_unhandled_event called: type={event_type}, data={event_data!r}")
        return f"Unhandled event: type={event_type}, data={event_data}"


class CitationDocumentHelper:
    """
    Core utility for citation document management across RAG retrieval modes.
    
    Central Responsibilities:
    - Extract and normalize filenames from URLs, paths, and metadata
    - Select appropriate human-readable titles for citations
    - Track document references via placeholder → (title, filename) mapping
    
    Architecture Context:
    - Used by AISearchToolHelper (traditional Azure AI Search mode)
    - Used by AgenticRetrievalToolHelper (agentic retrieval mode)
    - Shared document registry (tool_documents dict) is passed to EventHandler
    - This creates a data flow: Helper populates → EventHandler consumes
    
    Why Centralized:
    - Both retrieval modes need identical filename/title normalization logic
    - Ensures consistent citation format regardless of retrieval source
    - Single source of truth for document reference tracking
    
    Used by both traditional Azure AI Search and agentic retrieval modes
    to ensure consistent citation handling across the RAG pipeline.
    """

    def __init__(self):
        """Initialize document tracking dictionary for citation management."""
        self.tool_documents: dict[str, tuple[str, str]] = {} 

    @staticmethod
    def select_filename(candidates: list[Any]) -> Optional[str]:
        """
        Pick the best filename from candidate list using heuristics.
        
        Scoring criteria:
        - Earlier candidates in list score higher
        - Filenames with spaces score higher (likely human-readable)
        - Filenames with dashes but no spaces score lower
        - Generic patterns like "Document_N.pdf" score lower
        
        Args:
            candidates: List of potential filename strings
            
        Returns:
            Best filename or None if no valid candidates found
        """
        best_filename: Optional[str] = None
        best_score = -1

        for idx, value in enumerate(candidates):
            if not isinstance(value, str):
                continue
            candidate = value.strip()
            if not candidate:
                continue

            filename = CitationDocumentHelper._extract_filename(candidate)
            if not filename:
                continue

            # Calculate score based on position and characteristics
            score = 0
            score += max(len(candidates) - idx, 0)  # Earlier = better
            if " " in filename:
                score += 5  # Human-readable names preferred
            if "-" in filename and " " not in filename:
                score -= 1  # URL slugs less preferred
            if re.match(r"Document_?\d+\.pdf", filename, re.IGNORECASE):
                score -= 5  # Generic fallback names

            if score > best_score:
                best_filename = filename
                best_score = score

        return best_filename

    @staticmethod
    def choose_title(candidates: list[Any], fallback_filename: Optional[str]) -> Optional[str]:
        """
        Select display title, falling back to prettified filename.
        
        Args:
            candidates: List of potential title strings
            fallback_filename: Filename to prettify if no good title found
            
        Returns:
            Best title string or None
        """
        cleaned = []
        for value in candidates:
            if isinstance(value, str):
                candidate = value.strip()
                if candidate:
                    cleaned.append(candidate)

        # Return first non-generic candidate
        for candidate in cleaned:
            if candidate.lower() not in {"unknown", ""}:
                return candidate

        # Fall back to prettified filename
        if fallback_filename:
            return CitationDocumentHelper.prettify_filename(fallback_filename)

        return cleaned[0] if cleaned else None

    @staticmethod
    def derive_filename_from_title(title: Optional[str], fallback: str) -> str:
        """
        Generate sanitized filename from title.
        Ensures .pdf extension is present.
        
        Args:
            title: Title to convert to filename
            fallback: Fallback string if title is empty
            
        Returns:
            Sanitized filename with .pdf extension
        """
        base = title or fallback or "document"
        sanitized = re.sub(r"[^A-Za-z0-9]+", "_", base).strip("_") or fallback or "document"
        if not sanitized.lower().endswith(".pdf"):
            sanitized += ".pdf"
        return sanitized

    @staticmethod
    def prettify_filename(filename: str) -> str:
        """
        Convert filename to human-readable display format.
        Removes extension and replaces underscores with spaces.
        
        Args:
            filename: Raw filename
            
        Returns:
            Human-readable version of filename
        """
        if not filename:
            return filename
        segment = filename.split("/")[-1].split("\\")[-1]
        name = segment.rsplit(".", 1)[0]
        return name.replace("_", " ").strip()
    
    def register_document(self, ref_id: Any, title: str, filename: str) -> None:
        """
        Track document references for citation cleanup.
        Creates multiple placeholder variations for robust matching.
        
        Args:
            ref_id: Reference identifier (int, string, or None)
            title: Display title for the document
            filename: Actual filename/path
        """
        if not filename:
            return

        display_title = title or self.prettify_filename(filename)
        placeholders: set[str] = set()

        ref_str = str(ref_id) if ref_id is not None else None
        if ref_str:
            placeholders.add(ref_str)
            placeholders.add(f"ref_{ref_str}")
            if ref_str.isdigit():
                idx = int(ref_str)
                placeholders.update({
                    f"doc_{idx}",
                    f"Document {idx}",
                    f"Document_{idx}",
                    f"Document_{idx}.pdf",
                })

        placeholders.add(filename)

        # Register all placeholder variations
        for key in placeholders:
            cleaned_key = key.strip()
            if cleaned_key:
                self.tool_documents[cleaned_key] = (display_title, filename)

    @staticmethod
    def _extract_filename(candidate: str) -> Optional[str]:
        """
        Normalize filepath/url patterns into a filename.
        Handles Azure Search URL patterns and various path formats.
        
        Args:
            candidate: Raw URL or filepath string
            
        Returns:
            Extracted filename or None
        """
        cleaned = unquote(candidate).split("?")[0].split("#")[0].rstrip("/\\")
        if not cleaned:
            return None

        is_url = "://" in cleaned or cleaned.startswith(("http://", "https://"))
        segment = re.split(r"[/\\]", cleaned)[-1] if is_url else cleaned

        # Check for Azure Search URL pattern with embedded filename
        # Pattern: documents-<basename>-<ext>-  (generic, accepts any extension)
        documents_match = re.search(
            r"documents-([^/]+?)-([a-z0-9]+)-",
            segment,
            re.IGNORECASE,
        )
        if documents_match:
            basename = documents_match.group(1)
            ext = documents_match.group(2).lower()
            return f"{basename}.{ext}"

        # For non-URLs, preserve full path with extension
        if not is_url and ("/" in cleaned or "\\" in cleaned):
            # Accept any file with a valid extension (generic pattern)
            if re.search(r'\.[a-z0-9]+$', cleaned, re.IGNORECASE):
                return cleaned

        # Check for filename with extension in segment (generic pattern)
        explicit_match = re.search(
            r'([A-Za-z0-9_.\s-]+\.[a-z0-9]+)$',
            segment,
            re.IGNORECASE,
        )
        if explicit_match:
            return explicit_match.group(1)

        if "." in segment:
            return segment

        return None

class AgenticRetrievalToolHelper:
    """
    Agentic retrieval-specific helper for citation payload transformation.
    
    Primary Purpose:
    - Process KnowledgeAgentRetrievalClient responses
    - Inject human-readable citation labels into retrieval JSON payload
    - Apply token limits (max docs, max chars) to control costs
    
    Key Methods:
    - _build_reference_lookup(): Maps ref_id → (title, filename) from retrieval references
    - _inject_citation_labels(): Augments each document in payload with "citation" field
    - apply_agentic_retrieval_limits(): Truncates payload to prevent token explosion
    
    Data Flow (Agentic Mode):
    1. KnowledgeAgentRetrievalClient.retrieve() returns raw JSON
    2. _inject_citation_labels() processes it:
       - Extracts metadata from retrieval_result.references
       - Uses CitationDocumentHelper to normalize filenames/titles
       - Adds item["citation"] = "[Title](filename.pdf)"
       - Calls search_parser.citation_helper.register_document()
    3. Registered docs populate shared dict for EventHandler
    
    Why Separate from Traditional Search:
    - Different data structures (JSON payload vs Azure Search metadata)
    - Different citation injection points (function return vs post-streaming)
    - Agentic mode has native citation handling, this augments it
    """


    @staticmethod
    def apply_agentic_retrieval_limits(payload: str, max_docs: int, max_chars: int) -> str:
        """Clamp agentic retrieval payload by doc and character limits to control token usage.

        Args:
            payload: JSON string (list of document dicts) returned by agentic retrieval.
            max_docs: Maximum number of document objects to keep.
            max_chars: Maximum number of characters per document field (content/text).
        Returns:
            Potentially truncated JSON string with enforced limits.
        """
        if not payload:
            return payload

        doc_limit = max(max_docs, 0)
        char_limit = max(max_chars, 0)
        if doc_limit == 0 and char_limit == 0:
            return payload

        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            logging.warning("[Retrieval] Could not parse agentic payload for limiting; returning original")
            return payload

        if not isinstance(data, list):
            return payload

        truncated = False

        if doc_limit and len(data) > doc_limit:
            logging.info(
                "[Retrieval] Truncating agentic payload from %d to %d documents",
                len(data),
                doc_limit,
            )
            data = data[:doc_limit]
            truncated = True

        if char_limit:
            for item in data:
                if not isinstance(item, dict):
                    continue
                truncated |= AgenticRetrievalToolHelper._trim_field(item, "content", char_limit)
                truncated |= AgenticRetrievalToolHelper._trim_field(item, "text", char_limit)
                chunks = item.get("chunks")
                if isinstance(chunks, list):
                    for chunk in chunks:
                        if not isinstance(chunk, dict):
                            continue
                        truncated |= AgenticRetrievalToolHelper._trim_field(chunk, "content", char_limit)
                        truncated |= AgenticRetrievalToolHelper._trim_field(chunk, "text", char_limit)

        return json.dumps(data) if truncated else payload

    @staticmethod
    async def capture_recent_thread_messages(project_client, thread_id: str, limit: int = 5):
        """Collect the most recent conversational messages for retrieval context.

        Returns list[KnowledgeAgentMessage] suitable for KnowledgeAgentRetrievalRequest.
        """
        try:
            paged = project_client.agents.messages.list(
                thread_id=thread_id,
                limit=limit,
                order=ListSortOrder.DESCENDING
            )
            collected = []
            async for item in paged:
                collected.append(item)
            collected.reverse()

            converted_messages = []
            for msg in collected:
                if msg.role == "system":
                    continue
                content_text = ""
                if hasattr(msg, "content") and msg.content:
                    for content_item in msg.content:
                        if isinstance(content_item, MessageTextContent):
                            content_text = content_item.text.value
                            break
                        text_attr = getattr(content_item, "text", None)
                        if text_attr:
                            content_text = text_attr
                            break
                if content_text:
                    converted_messages.append(
                        KnowledgeAgentMessage(
                            role=msg.role,
                            content=[KnowledgeAgentMessageTextContent(text=content_text)]
                        )
                    )
            logging.debug(
                "[Retrieval] Prepared %d messages for agentic retrieval (thread_id=%s)",
                len(converted_messages),
                thread_id,
            )
            return converted_messages
        except Exception as exc:
            logging.error(
                "[Retrieval] Failed to capture recent messages for thread %s: %s",
                thread_id,
                str(exc),
                exc_info=True,
            )
            return []    
    
    @staticmethod
    def _build_reference_lookup(retrieval_result) -> dict[str, tuple[str, str]]:
        """Create a mapping from reference id to (display title, filename)."""
        lookup: dict[str, tuple[str, str]] = {}
        references = getattr(retrieval_result, "references", None) or []
        for ref in references:
            ref_id = None
            source_data = {}
            try:
                ref_dict = ref.as_dict()
                ref_id = ref_dict.get("id")
                source_data = ref_dict.get("source_data") or {}
            except AttributeError:
                ref_id = getattr(ref, "id", None)
                source_data = getattr(ref, "source_data", {}) or {}

            if ref_id is None:
                continue

            # Use CitationDocumentHelper for filename selection
            filename = CitationDocumentHelper.select_filename([
                source_data.get("filepath"),
                source_data.get("file_path"),
                source_data.get("filePath"),
                source_data.get("parent_id"),
                source_data.get("parentId"),
                source_data.get("metadata_storage_path"),
                source_data.get("metadataStoragePath"),
                source_data.get("metadata_storage_name"),
                source_data.get("metadataStorageName"),
                source_data.get("file_name"),
                source_data.get("fileName"),
                source_data.get("url"),
                source_data.get("id"),
                source_data.get("uri"),
            ])

            # Use CitationDocumentHelper for title selection
            title = CitationDocumentHelper.choose_title([
                source_data.get("displayName"),
                source_data.get("display_name"),
                source_data.get("title"),
                source_data.get("name"),
                source_data.get("citation"),
            ], filename)

            if not filename:
                # Fallback: derive from title or use ref id
                filename = CitationDocumentHelper.derive_filename_from_title(title, str(ref_id))

            if not title:
                title = CitationDocumentHelper.prettify_filename(filename)

            lookup[str(ref_id)] = (title, filename)

        return lookup

    @staticmethod
    def _inject_citation_labels(result_text: str, retrieval_result, search_parser) -> str:
        """
        Attach human-readable citation labels to retrieval payload.
        
        Args:
            result_text: JSON string containing retrieval results
            retrieval_result: Retrieval result object with references
            search_parser: AISearchToolHelper instance for document registration
            
        Returns:
            Modified JSON string with citation labels
        """
        try:
            payload = json.loads(result_text)
        except json.JSONDecodeError:
            logging.debug("[Citations] Retrieval response is not JSON; skipping citation augmentation")
            return result_text

        if not isinstance(payload, list):
            logging.debug("[Citations] Retrieval JSON is not a list; skipping citation augmentation")
            return result_text

        reference_lookup = AgenticRetrievalToolHelper._build_reference_lookup(retrieval_result)

        for idx, item in enumerate(payload):
            if not isinstance(item, dict):
                continue

            metadata = item.get("metadata") or {}
            ref_identifier = item.get("ref_id")
            ref_key = str(ref_identifier) if ref_identifier is not None else None
            ref_title = None
            ref_filename = None
            if ref_key and ref_key in reference_lookup:
                ref_title, ref_filename = reference_lookup[ref_key]

            filepath_candidates = [
                item.get("filepath"),
                metadata.get("filepath"),
                item.get("file_path"),
                metadata.get("file_path"),
                metadata.get("metadata_storage_path"),
                metadata.get("metadataStoragePath"),
                metadata.get("metadata_storage_name"),
                metadata.get("metadataStorageName"),
                metadata.get("file_name"),
                metadata.get("fileName"),
                item.get("metadata_storage_path"),
                item.get("metadata_storage_name"),
                item.get("file_name"),
                item.get("fileName"),
                item.get("url"),
                ref_filename,
            ]

            # Use CitationDocumentHelper for filename selection
            filepath = CitationDocumentHelper.select_filename(filepath_candidates)

            title_candidates = [
                metadata.get("title"),
                item.get("title"),
                metadata.get("displayName"),
                metadata.get("display_name"),
                metadata.get("name"),
                ref_title,
            ]

            # Use CitationDocumentHelper for title selection
            title = CitationDocumentHelper.choose_title(title_candidates, filepath or ref_filename)

            if not title and filepath:
                if "." in filepath:
                    title = filepath.rsplit(".", 1)[0].replace("_", " ")
                else:
                    title = filepath

            if not title:
                title = f"Document {idx}"

            if not filepath:
                if isinstance(ref_identifier, str) and ref_identifier.strip():
                    filepath = ref_identifier.strip()
                elif isinstance(ref_identifier, int):
                    filepath = f"Document_{ref_identifier}.pdf"
                else:
                    filepath = f"Document_{idx}.pdf"

            item["citation"] = f"[{title}]({filepath})"

            # Register document in shared tracking dict (used by EventHandler for citation cleanup)
            search_parser.citation_helper.register_document(
                ref_identifier if ref_identifier is not None else idx,
                title,
                filepath,
            )

        return json.dumps(payload)

    @staticmethod
    def _trim_field(container: dict[Any, Any], field: str, char_limit: int) -> bool:
        """Trim a text field in-place and append ellipsis when data exceeds the limit."""
        value = container.get(field)
        if not isinstance(value, str):
            return False
        if len(value) <= char_limit:
            return False

        trimmed = value[:char_limit].rstrip()
        if not trimmed.endswith("..."):
            trimmed += " ..."
        container[field] = trimmed
        logging.info(
            "[Retrieval] Trimmed field '%s' from %d to %d characters",
            field,
            len(value),
            len(trimmed),
        )
        return True

class AISearchToolHelper:
    """
    Helper for Azure AI Search citation processing.
    
    Handles:
    - Search output parsing
    - Sources section building
    - Inline citation removal
    - Direct Azure AI Search queries to fetch document metadata
    
    Uses CitationDocumentHelper for document tracking and filename normalization.
    """
    
    def __init__(self, search_client: Optional[Any] = None, index_name: Optional[str] = None):
        """
        Initialize with CitationDocumentHelper for document tracking.
        
        Args:
            search_client: Instance of connectors.search.SearchClient for direct queries
            index_name: Azure AI Search index name
        """
        self.citation_helper = CitationDocumentHelper()
        # Expose tool_documents for backward compatibility
        self.tool_documents = self.citation_helper.tool_documents
        
        # Azure AI Search connection for direct queries
        self._search_client = search_client
        self.index_name = index_name
        
        # Cache for pending filepath fetches (document_id -> placeholder mapping)
        self._pending_filepath_fetches = {}
        
        if search_client and index_name:
            logging.info("[Citations] ✅ Using existing SearchClient for direct document queries")
            logging.info("[Citations]    Index: %s", index_name)
        else:
            logging.info("[Citations] ⚠️ No SearchClient provided - will use URL extraction fallback")
    
    def remove_inline_citations(self, text: str) -> str:
        """Remove inline markdown citation links completely from text."""
        if not text:
            return text

        citation_targets: set[str] = set()
        for placeholder, (_, filename) in self.tool_documents.items():
            citation_targets.add(placeholder)
            citation_targets.add(filename)

        def _replace(match: re.Match[str]) -> str:
            label, target = match.groups()
            normalized_target = target.strip()
            lower_target = normalized_target.lower()

            is_citation_link = (
                normalized_target in citation_targets
                or lower_target.startswith("doc_")
            )

            if is_citation_link:
                # Replace citation with empty string - cleanup will handle spacing
                return ""

            return match.group(0)

        inline_link_pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
        cleaned = inline_link_pattern.sub(_replace, text)
        # Clean up multiple spaces left by removed citations
        cleaned = re.sub(r" {2,}", " ", cleaned)
        # Remove spaces before punctuation (handles cases like "word ." or "word ,")
        cleaned = re.sub(r"\s+([.,;:!?])", r"\1", cleaned)
        cleaned = re.sub(r"\s+\n", "\n", cleaned)
        return cleaned
    
    def clear_tracked_documents(self):
        """Clear all tracked documents."""
        self.tool_documents.clear()
    
    async def _fetch_filepath_from_index(self, document_id: str) -> Optional[str]:
        """
        Fetch filepath directly from Azure AI Search index using document ID.
        
        This method queries the Azure AI Search index to retrieve the 'filepath' field
        for a given document ID. This is more reliable than extracting from URLs.
        
        Args:
            document_id: Document ID from Azure Search (e.g., "documents-filename-pdf-c00012")
            
        Returns:
            Filepath string from the index, or None if not found
        """
        if not self._search_client or not self.index_name:
            logging.debug("[Citations] No search client available for direct query")
            return None
        
        try:
            logging.info("[Citations] 🔍 Fetching filepath from index for document_id: %s", document_id)
            
            # Use the existing SearchClient.get_document method
            document = await self._search_client.get_document(
                index_name=self.index_name,
                document_id=document_id,
                select_fields=['filepath', 'title']  # Only fetch what we need
            )
            
            if document:
                filepath = document.get('filepath')
                if filepath:
                    logging.info("[Citations] ✅ Found filepath in index: %s", filepath)
                    return filepath
                else:
                    logging.warning("[Citations] ⚠️ Document found but 'filepath' field is empty")
            else:
                logging.warning("[Citations] ⚠️ Document not found with ID: %s", document_id)
                
        except Exception as e:
            logging.error("[Citations] ❌ Error fetching document from index: %s", e, exc_info=True)
        
        return None
    
    async def fetch_pending_filepaths(self):
        """
        Fetch all pending filepaths from Azure AI Search index asynchronously.
        
        This method processes all document IDs stored during _process_document_entry
        and updates the citation mappings with real filepaths from the index.
        """
        if not self._pending_filepath_fetches:
            logging.debug("[Citations] No pending filepath fetches")
            return
        
        logging.info("[Citations] 🚀 Fetching %d filepaths from index...", len(self._pending_filepath_fetches))
        
        for document_id, info in self._pending_filepath_fetches.items():
            placeholder = info['placeholder']
            title = info['title']
            fallback_info = info.get('fallback')
            
            # Try to fetch real filepath from index
            filepath = await self._fetch_filepath_from_index(document_id)
            
            if filepath:
                # Success! Register with real filepath from index
                display_title = title if title else CitationDocumentHelper.prettify_filename(filepath)
                self.citation_helper.register_document(placeholder, display_title, filepath)
                self.citation_helper.register_document(filepath, display_title, filepath)
                logging.info("[Citations] ✅ Updated '%s' with real filepath: %s", placeholder, filepath)
            elif fallback_info:
                # Use URL-based fallback
                fallback_filename, fallback_title = fallback_info
                self.citation_helper.register_document(placeholder, fallback_title, fallback_filename)
                self.citation_helper.register_document(fallback_filename, fallback_title, fallback_filename)
                logging.warning("[Citations] ⚠️ Using URL fallback for '%s': %s", placeholder, fallback_filename)
            else:
                logging.error("[Citations] ❌ No filepath found and no fallback available for '%s'", placeholder)
        
        # Clear pending fetches
        self._pending_filepath_fetches.clear()
        logging.info("[Citations] ✅ Finished fetching filepaths from index")
    
    def apply_tool_results_to_sources(self, text: str, enable_agentic_retrieval: bool = False) -> str:
        """Replace **Sources:** section with correct filenames from Azure AI Search."""
        if enable_agentic_retrieval:
            logging.debug("[Citations] Skipping Sources rewrite (agentic retrieval enabled)")
            return text

        # First, clean up any spacing issues before punctuation (left by removed citations)
        text = re.sub(r"\s+([.,;:!?])", r"\1", text)

        sources_section = self._build_sources_section()
        if not sources_section:
            logging.warning("[Citations] ⚠️ Unable to build sources section - leaving text unchanged")
            return text

        sources_pattern = re.compile(r'(\*\*Sources:\*\*|Sources:).*$', re.DOTALL | re.MULTILINE)

        if sources_pattern.search(text):
            updated_text = sources_pattern.sub(sources_section, text)
            logging.info(
                "[Citations] ✅ REPLACED agent's Sources (likely fake) with REAL documents from Azure Search"
            )
            logging.info("[Citations] Real sources: %s", sources_section)
            return updated_text

        logging.info("[Citations] No Sources section found, appending REAL sources: %s", sources_section)
        return text.rstrip() + "\n\n" + sources_section
    
    # ============================================================
    # Azure Search Output Parsing
    # ============================================================
    
    def capture_tool_outputs(self, run_step: RunStep) -> None:
        """Capture filenames from AzureAISearchTool for citation fixing."""
        try:
            logging.info("[Citations DEBUG] capture_tool_outputs called")
            
            step_details = getattr(run_step, "step_details", None)
            logging.info("[Citations DEBUG] step_details exists: %s", step_details is not None)
            
            if not step_details:
                logging.info("[Citations DEBUG] No step_details, returning")
                return
            
            tool_calls = getattr(step_details, "tool_calls", [])
            logging.info("[Citations DEBUG] Found %d tool_calls", len(tool_calls))
            
            for i, tool_call in enumerate(tool_calls):
                tool_type = getattr(tool_call, "type", "")
                logging.info("[Citations DEBUG] tool_call %d type: %s", i, tool_type)
                
                if tool_type != "azure_ai_search":
                    continue
                
                search_data = getattr(tool_call, "azure_ai_search", None)
                logging.info("[Citations DEBUG] search_data exists: %s", search_data is not None)
                logging.info("[Citations DEBUG] search_data type: %s", type(search_data).__name__)
                
                if not search_data:
                    continue
                
                # Try both attribute and dictionary access
                output_str = None
                if hasattr(search_data, "output"):
                    output_str = search_data.output
                    logging.info("[Citations DEBUG] Got output via attribute access")
                elif isinstance(search_data, dict) and "output" in search_data:
                    output_str = search_data["output"]
                    logging.info("[Citations DEBUG] Got output via dict access")
                
                logging.info("[Citations DEBUG] output_str type: %s", type(output_str).__name__ if output_str else "None")
                logging.info("[Citations DEBUG] output_str length: %s", len(str(output_str)) if output_str else 0)
                
                if not output_str:
                    logging.warning("[Citations DEBUG] output_str is empty or None, skipping")
                    continue
                
                output_str = str(output_str)
                logging.info("[Citations] Azure AI Search output (trimmed): %s", output_str[:800])
                self._parse_search_output(output_str)
        except Exception as exc:
            logging.error("[Citations] Failed to capture tool outputs: %s", exc, exc_info=True)
    
    def _parse_search_output(self, output_str: str) -> None:
        """
        Parse Azure AI Search output to extract filename mappings.
        
        This method orchestrates the parsing process:
        1. Parse JSON output string
        2. Extract metadata arrays
        3. Process each document entry
        4. Register document mappings
        """
        logging.info("[Citations DEBUG] parse_search_output called")
        
        # Step 1: Parse JSON output
        data = self._parse_json_output(output_str)
        if not data:
            return
        
        # Step 2: Extract metadata arrays
        metadata_arrays = self._extract_metadata_arrays(data)
        if not metadata_arrays:
            return
        
        urls, titles, ids, get_urls = metadata_arrays
        
        # Step 3: Process each document entry
        # Pattern accepts any file extension (not hardcoded to specific types)
        filename_pattern = re.compile(r'/documents-([^?/]+?)-([a-z0-9]+)-', re.IGNORECASE)
        for idx, placeholder in enumerate(urls):
            self._process_document_entry(
                idx, placeholder, titles, ids, get_urls, filename_pattern
            )

    def _parse_json_output(self, output_str: str) -> Optional[dict]:
        """
        Parse JSON output string from Azure AI Search.
        
        Args:
            output_str: Raw output string from search tool
            
        Returns:
            Parsed dictionary or None if parsing fails
        """
        try:
            import ast
            logging.info("[Citations DEBUG] Attempting to parse with ast.literal_eval")
            # Safe to use ast.literal_eval here because output_str comes from Azure SDK,
            # not from user input. The SDK generates this output string internally.
            data = ast.literal_eval(output_str)
            logging.info("[Citations DEBUG] Successfully parsed, type: %s", type(data))
            
            if not isinstance(data, dict):
                logging.info("[Citations DEBUG] data is not dict: %s", type(data))
                return None
            
            # ==== DETAILED INDEX FIELDS DUMP ====
            logging.info("[INDEX FIELDS] ========== START FULL INDEX DUMP ==========")
            logging.info("[INDEX FIELDS] Top-level keys: %s", list(data.keys()))
            
            # Dump metadata section
            metadata = data.get("metadata", {})
            if metadata:
                logging.info("[INDEX FIELDS] metadata keys: %s", list(metadata.keys()))
                for key, value in metadata.items():
                    if isinstance(value, list):
                        logging.info("[INDEX FIELDS]   metadata.%s: [list with %d items]", key, len(value))
                        if len(value) > 0:
                            logging.info("[INDEX FIELDS]     First item: %s", str(value[0])[:200])
                    else:
                        logging.info("[INDEX FIELDS]   metadata.%s: %s", key, str(value)[:200])
            
            # Dump documents section (this contains actual index fields!)
            documents = data.get("documents", [])
            if documents:
                logging.info("[INDEX FIELDS] documents: [list with %d items]", len(documents))
                if len(documents) > 0:
                    first_doc = documents[0]
                    logging.info("[INDEX FIELDS] First document type: %s", type(first_doc))
                    if isinstance(first_doc, dict):
                        logging.info("[INDEX FIELDS] First document keys (ALL INDEX FIELDS): %s", list(first_doc.keys()))
                        for key, value in first_doc.items():
                            value_str = str(value)[:200] if value else "None"
                            logging.info("[INDEX FIELDS]   doc[0].%s = %s", key, value_str)
            
            # Dump any other top-level keys
            for key in data.keys():
                if key not in ["metadata", "documents"]:
                    value = data[key]
                    logging.info("[INDEX FIELDS] %s: %s", key, str(value)[:200])
            
            logging.info("[INDEX FIELDS] ========== END FULL INDEX DUMP ==========")
            
            return data
        except Exception as parse_exc:
            logging.warning("[Citations] Could not parse search output: %s", parse_exc)
            return None

    def _extract_metadata_arrays(self, data: dict) -> Optional[tuple]:
        """
        Extract metadata arrays from parsed search output.
        
        Args:
            data: Parsed search output dictionary
            
        Returns:
            Tuple of (urls, titles, ids, get_urls) or None if missing
        """
        metadata = data.get("metadata", {})
        logging.info("[Citations DEBUG] metadata keys: %s", list(metadata.keys()) if metadata else None)
        
        urls = metadata.get("urls", [])
        titles = metadata.get("titles", [])
        ids = metadata.get("ids", [])  # IDs dos documentos para busca direta
        get_urls = metadata.get("get_urls", [])

        logging.info(
            "[Citations DEBUG] counts - urls:%d, titles:%d, ids:%d, get_urls:%d", 
            len(urls), len(titles), len(ids), len(get_urls)
        )

        if not urls or not get_urls:
            logging.debug("[Citations] No urls or get_urls in metadata")
            return None
        
        return urls, titles, ids, get_urls

        if not urls or not get_urls:
            logging.debug("[Citations] No urls or get_urls in metadata")
            return None
        
        return urls, titles, filepaths, get_urls

    def _process_document_entry(
        self, idx: int, placeholder: str, titles: list, ids: list, 
        get_urls: list, filename_pattern: re.Pattern
    ):
        """
        Process a single document entry to extract filename and title.
        
        Strategy:
        1. If SearchClient available, store document ID for async fetch later
        2. Otherwise, extract from URL as fallback
        
        Args:
            idx: Index in metadata arrays
            placeholder: Citation placeholder (e.g., doc_0)
            titles: List of document titles
            ids: List of document IDs from Azure Search
            get_urls: List of document URLs
            filename_pattern: Regex pattern for extracting filename info
        """
        get_url = get_urls[idx] if idx < len(get_urls) else None
        if not get_url:
            return
        
        title = titles[idx] if idx < len(titles) else None
        document_id = ids[idx] if idx < len(ids) else None
        
        # STRATEGY 1: If SearchClient available, store for async fetch
        if document_id and self._search_client and self.index_name:
            logging.info("[Citations] 📋 Queueing document for async filepath fetch: %s", document_id)
            # Store for async fetch - include URL for fallback if needed
            decoded_url = unquote(get_url)
            match = filename_pattern.search(decoded_url)
            fallback_info = None
            if match:
                ext = match.group(2).lower()
                basename = match.group(1)
                fallback_filename = f"{basename}.{ext}"
                fallback_title = title if title else basename.replace('_', ' ')
                fallback_info = (fallback_filename, fallback_title)
            
            self._pending_filepath_fetches[document_id] = {
                'placeholder': placeholder,
                'title': title,
                'get_url': get_url,
                'fallback': fallback_info  # Store fallback but don't register yet
            }
            return
        
        # STRATEGY 2: Fallback to URL extraction
        logging.info("[Citations] ⚠️ Using URL extraction fallback for: %s", placeholder)
        decoded_url = unquote(get_url)
        match = filename_pattern.search(decoded_url)
        
        if not match:
            logging.debug("[Citations] Could not extract from URL: %s", decoded_url[:200])
            return
        
        ext = match.group(2).lower()
        basename = match.group(1)
        filename = f"{basename}.{ext}"
        display_title = title if title else basename.replace('_', ' ')
        
        # Register both placeholder and filename mappings
        self.citation_helper.register_document(placeholder, display_title, filename)
        self.citation_helper.register_document(filename, display_title, filename)
        logging.info("[Citations] 📝 Mapped '%s' → title='%s', file='%s'", placeholder, display_title, filename)

    def _build_sources_section(self) -> Optional[str]:
        """Build canonical **Sources:** line from tracked documents."""
        if not self.tool_documents:
            return None

        unique_sources: list[str] = []
        seen_files: set[str] = set()

        for _, (title, filename) in self.tool_documents.items():
            if filename not in seen_files:
                seen_files.add(filename)
                unique_sources.append(f"[{title}]({filename})")
                logging.debug("[Citations] Adding REAL source: [%s](%s)", title, filename)

        if not unique_sources:
            return None

        return "**Sources:** " + ", ".join(unique_sources)

