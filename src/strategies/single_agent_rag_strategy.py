import logging
import json
from typing import Any, Optional

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

from connectors.appconfig import AppConfigClient
from dependencies import get_config

class SingleAgentRAGStrategy(BaseAgentStrategy):
    """
    Implements a single-agent Retrieval-Augmented Generation (RAG) strategy
    using Azure AI Foundry. This class handles creating an agent, sending
    a user message, streaming the response, and cleaning up resources.
    """

    def __init__(self):
        """
        Initialize base credentials and tools.
        """
        super().__init__()

        # Force all logs at DEBUG or above to appear
        logging.debug("Initializing SingleAgentRAGStrategy...")

        # Event handler for streaming responses
        self.strategy_type = AgentStrategies.SINGLE_AGENT_RAG
        self.event_handler = EventHandler()

        cfg = get_config()

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
        logging.debug(f"Agentic Retrieval Enabled: {self.enable_agentic_retrieval}")

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
                    logging.warning("RESOURCE_TOKEN not found. Cannot construct default knowledge agent name.")
            logging.debug(f"Knowledge Agent Name: {self.knowledge_agent_name}")
            logging.debug(f"Search Query Endpoint: {self.search_query_endpoint}")
            if not self.search_query_endpoint or not self.knowledge_agent_name:
                logging.error(
                    "Agentic retrieval is enabled but SEARCH_SERVICE_QUERY_ENDPOINT or agent name is not configured. Falling back to traditional search."
                )
                self.enable_agentic_retrieval = False
            else:
                try:
                    self.agentic_retrieval_credential = SyncDefaultAzureCredential(
                        exclude_interactive_browser_credential=True
                    )
                except Exception as cred_error:
                    logging.error(
                        "Unable to initialize synchronous credential for agentic retrieval: %s",
                        cred_error,
                        exc_info=True,
                    )
                    self.enable_agentic_retrieval = False
                    logging.error("Agentic retrieval disabled due to credential initialization failure")

        # --- Initialize BingGroundingTool (if configured) ---
        bing_conn = cfg.get("BING_CONNECTION_ID", "")
        if not bing_conn:
            logging.warning(
                "BING_CONNECTION_ID not set in App Config variables. "
                "BingGroundingTool will not be available."
            )
        else:
            bing = BingGroundingTool(connection_id=bing_conn, count=5)
            bing_def = bing.definitions[0]
            self.tools_list.append(bing_def)
            logging.debug(f"Added BingGroundingTool to tools_list: {bing_def}")

        # --- Initialize AzureAISearchTool (only if agentic retrieval is disabled) ---
        if not self.enable_agentic_retrieval:
            azure_ai_conn_id = cfg.get("SEARCH_CONNECTION_ID", "")
            index_name = cfg.get("SEARCH_RAG_INDEX_NAME", "ragindex")
            logging.debug(f"seachConnectionId (cfg)  = {azure_ai_conn_id}")
            logging.debug(f"SEARCH_RAG_INDEX_NAME (cfg) = {index_name}")
            if not azure_ai_conn_id:
                logging.warning(
                    "seachConnectionId undefined (cfg). "
                    "AzureAISearchTool will be unavailable."
                )
            if not index_name:
                logging.warning(
                    "SEARCH_RAG_INDEX_NAME undefined (cfg). "
                    "AzureAISearchTool will be unavailable."
                )
            self.ai_search = AzureAISearchTool(
                index_connection_id=azure_ai_conn_id,
                index_name=index_name,
                query_type=AzureAISearchQueryType.SIMPLE,
                top_k=cfg.get("SEARCH_TOP_K", 5, int),
                filter="",
            )
            ai_def = self.ai_search.definitions[0]
            ai_res = self.ai_search.resources
            logging.debug(f"Created AzureAISearchTool definition: {ai_def}")
            logging.debug(f"AzureAISearchTool resources metadata: {ai_res}")
            self.tools_list.append(ai_def)
            self.tool_resources.update(ai_res)
        else:
            logging.info("Using Agentic Retrieval - traditional AzureAISearchTool will not be initialized")

        logging.debug(f"Final tools_list: {self.tools_list}")
        logging.debug(f"Final tool_resources: {self.tool_resources}")

    def _create_agentic_retrieval_tool(self, project_client, thread_id: str):
        if not self.enable_agentic_retrieval:
            logging.warning("Agentic retrieval is not enabled. Tool will not be created.")
            return None
        try:
            if not self.agentic_retrieval_credential:
                logging.error("Synchronous credential not available for agentic retrieval client initialization")
                return None

            self.agentic_retrieval_client = KnowledgeAgentRetrievalClient(
                endpoint=self.search_query_endpoint,
                agent_name=self.knowledge_agent_name,
                credential=self.agentic_retrieval_credential
            )
            logging.info(f"KnowledgeAgentRetrievalClient initialized for agent: {self.knowledge_agent_name}")
            
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
                    logging.debug(f"Converted {len(converted_messages)} messages for retrieval request (query override provided: {bool(query)})")
                    
                    # Log the messages being sent to agentic retrieval
                    logging.info(f"[agentic_retrieval] Calling retrieve with {len(converted_messages)} messages")
                    for idx, msg in enumerate(converted_messages):
                        logging.debug(f"[agentic_retrieval] Message {idx}: role={msg.role}, content_preview={msg.content[0].text[:100] if msg.content else 'empty'}...")
                    
                    retrieval_result = self.agentic_retrieval_client.retrieve(
                        retrieval_request=KnowledgeAgentRetrievalRequest(
                            messages=converted_messages
                        )
                    )
                    
                    logging.info(f"[agentic_retrieval] Retrieval completed successfully")
                    
                    if retrieval_result.response and len(retrieval_result.response) > 0:
                        response_content = retrieval_result.response[0].content
                        if response_content and len(response_content) > 0:
                            result_text = response_content[0].text
                            logging.debug(
                                "[agentic_retrieval] Raw retrieval payload (truncated): %s",
                                result_text[:200],
                            )

                            transformed_text = self._inject_citation_labels(result_text, retrieval_result)

                            logging.info(
                                "[agentic_retrieval] Returning %d characters of transformed content",
                                len(transformed_text),
                            )
                            if hasattr(retrieval_result, 'activity') and retrieval_result.activity:
                                logging.info(
                                    "[agentic_retrieval] Activity: %d operations logged",
                                    len(retrieval_result.activity),
                                )
                                for activity in retrieval_result.activity[:3]:
                                    activity_type = getattr(activity, 'type', 'unknown')
                                    logging.debug(f"Activity type: {activity_type}")
                            return transformed_text
                    
                    logging.warning("[agentic_retrieval] Empty response from retrieval service")
                    return json.dumps([
                        {
                            "ref_id": 0,
                            "content": "No information found.",
                            "title": "Empty Result",
                            "citation": "No Source",
                        }
                    ])
                except Exception as e:
                    logging.error(f"[agentic_retrieval] Error during retrieval: {str(e)}", exc_info=True)
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
            
            logging.info("Agentic retrieval tool created successfully")
            # Return: (function definition, function, toolset)
            return (functions.definitions[0], agentic_retrieval, toolset)
        except Exception as e:
            logging.error(f"Failed to create agentic retrieval tool: {str(e)}", exc_info=True)
            return None

    def _build_reference_lookup(self, retrieval_result) -> dict[str, str]:
        """Create a mapping from reference id to human-readable source label."""
        lookup: dict[str, str] = {}
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

            candidates = [
                source_data.get("filepath"),
                source_data.get("file_path"),
                source_data.get("metadata_storage_path"),
                source_data.get("metadata_storage_name"),
                source_data.get("file_name"),
                source_data.get("fileName"),
                source_data.get("citation"),
                source_data.get("displayName"),
                source_data.get("display_name"),
                source_data.get("title"),
                source_data.get("name"),
                source_data.get("id"),
                source_data.get("uri"),
            ]
            label = self._choose_label(candidates)
            if not label:
                label = str(ref_id)

            lookup[str(ref_id)] = label

        return lookup

    @staticmethod
    def _choose_label(candidates: list[Any]) -> Optional[str]:
        """Select the best citation label, preferring values that include a file extension."""
        cleaned = []
        for value in candidates:
            if isinstance(value, str):
                candidate = value.strip()
                if candidate:
                    cleaned.append(candidate)

        for candidate in cleaned:
            if "." in candidate:
                return candidate

        return cleaned[0] if cleaned else None

    def _inject_citation_labels(self, result_text: str, retrieval_result) -> str:
        """Attach human-readable citation labels to retrieval payload."""
        try:
            payload = json.loads(result_text)
        except json.JSONDecodeError:
            logging.debug("[agentic_retrieval] Retrieval response is not JSON; skipping citation augmentation")
            return result_text

        if not isinstance(payload, list):
            logging.debug("[agentic_retrieval] Retrieval JSON is not a list; skipping citation augmentation")
            return result_text

        reference_lookup = self._build_reference_lookup(retrieval_result)

        for idx, item in enumerate(payload):
            if not isinstance(item, dict):
                continue

            metadata = item.get("metadata") or {}
            
            # Get filepath candidates
            filepath_candidates = [
                item.get("filepath"),
                item.get("file_path"),
                item.get("file_name"),
                item.get("fileName"),
                metadata.get("filepath"),
                metadata.get("file_path"),
                metadata.get("metadata_storage_name"),
                metadata.get("metadata_storage_path"),
                metadata.get("file_name"),
                metadata.get("fileName"),
            ]
            
            # Get title candidates
            title_candidates = [
                item.get("title"),
                item.get("citation"),
                metadata.get("displayName"),
                metadata.get("display_name"),
                metadata.get("title"),
                metadata.get("name"),
            ]

            ref_identifier = item.get("ref_id")
            if ref_identifier is not None:
                ref_label = reference_lookup.get(str(ref_identifier))
                if ref_label:
                    # Check if ref_label looks like a filepath (has extension)
                    if "." in ref_label:
                        filepath_candidates.insert(0, ref_label)
                    else:
                        title_candidates.insert(0, ref_label)

            filepath = self._choose_label(filepath_candidates)
            title = self._choose_label(title_candidates)

            if not filepath:
                if isinstance(ref_identifier, str) and ref_identifier.strip():
                    filepath = ref_identifier.strip()
                elif isinstance(ref_identifier, int):
                    filepath = f"Document_{ref_identifier}.pdf"
                else:
                    filepath = f"Document_{idx}.pdf"

            if not title:
                if filepath and "." in filepath:
                    # Extract title from filepath by removing extension and replacing underscores
                    title = filepath.rsplit(".", 1)[0].replace("_", " ")
                else:
                    title = f"Document {idx}"

            # Format as [title](filepath)
            item["citation"] = f"[{title}]({filepath})"

        return json.dumps(payload)

    async def _capture_recent_thread_messages(self, project_client, thread_id: str, limit: int = 5):
        """Collect the most recent conversational messages for retrieval context."""
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
                "Prepared %d messages for agentic retrieval (thread_id=%s)",
                len(converted_messages),
                thread_id,
            )
            return converted_messages
        except Exception as exc:
            logging.error(
                "Failed to capture recent messages for thread %s: %s",
                thread_id,
                str(exc),
                exc_info=True,
            )
            return []

    async def create():
        """
        Factory method to create an instance of SingleAgentRAGStrategy.
        Initializes the agent and tools.
        """
        logging.debug("Creating SingleAgentRAGStrategy instance...")
        instance = SingleAgentRAGStrategy()

        return instance


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
        
        2. AGENTIC RETRIEVAL DISABLED (self.enable_agentic_retrieval = False):
           - Uses traditional AzureAISearchTool
           - Tool is configured during __init__ and added to tools_list
           - No custom function tools are created
           - Agent uses built-in AzureAISearchTool capabilities
        
        Both modes can optionally use BingGroundingTool if BING_CONNECTION_ID is configured.
        """
        logging.debug(f"invoke_stream called with user_message: {user_message!r}")
        conv = self.conversation
        thread_id = conv.get("thread_id")
        logging.debug(f"Current conversation state: thread_id={thread_id}")

        async with self.project_client as project_client:
            # Thread management - CREATE OR GET THREAD FIRST
            if thread_id:
                logging.debug(f"thread_id exists; calling get(thread_id={thread_id})")
                thread = await project_client.agents.threads.get(thread_id)
                logging.info(f"Reused thread with ID: {thread.id}")
            else:
                logging.debug("thread_id not found; calling create()")
                thread = await project_client.agents.threads.create()
                logging.info(f"Created new thread with ID: {thread.id}")

            conv["thread_id"] = thread.id
            logging.debug(f"Stored conv['thread_id'] = {thread.id}")

            # Setup agentic retrieval tool if enabled (BEFORE creating agent)
            agentic_tool_definition = None
            agentic_function = None
            agentic_toolset = None
            if self.enable_agentic_retrieval:
                logging.info("Setting up agentic retrieval tool...")
                result = self._create_agentic_retrieval_tool(project_client, thread.id)
                if result:
                    agentic_tool_definition, agentic_function, agentic_toolset = result
                    # CRITICAL: enable_auto_function_calls expects a SET or LIST of Python functions
                    project_client.agents.enable_auto_function_calls({agentic_function})
                    logging.info(f"Agentic retrieval function registered: {agentic_function.__name__}")
                    logging.info(f"  - Tool definition will be added to agent")
                else:
                    logging.warning("Failed to create agentic retrieval functions, proceeding without it")

            # Agent management
            create_agent = False
            if self.existing_agent_id:
                logging.debug("agent_id exists; calling update_agent(...)")
                agent = await project_client.agents.get_agent(self.existing_agent_id)
                logging.info(f"Reused agent with ID: {agent.id}")
            else:
                logging.debug("creating agent(...)")
                # Use enhanced instructions for retrieval behavior
                instructions = await self._read_prompt("main")
                instructions += """

UNIVERSAL CITATION REQUIREMENTS:

1. Every fact derived from retrieved content MUST include a citation using the format [title](filepath).
2. The title should be descriptive (e.g., document title or display name) and filepath should be the actual filename with extension.
3. Place the citation immediately after the sentence that uses the sourced information.
4. Repeat citations when the same source backs multiple statements.
5. If no relevant document is found, respond with "I don't have information about that in my knowledge base."
"""

                if self.enable_agentic_retrieval:
                    instructions += """

CRITICAL INSTRUCTIONS FOR AGENTIC RETRIEVAL:

1. You MUST call the 'agentic_retrieval' function to search for information BEFORE answering ANY question.
2. Do NOT attempt to answer from your internal knowledge without calling the function first.
3. The function returns a JSON array containing 'content', 'metadata', and a 'citation' field formatted as [title](filepath) for each source.
4. You MUST cite ALL sources using the exact format [title](filepath) as provided in the citation field (for example, [Northwind Health Plus Benefits Details](Northwind_Health_Plus_Benefits_Details.pdf)).
5. Place citations immediately after each statement that uses information from the sources.
6. If multiple statements use the same source, repeat the citation after each relevant statement.
7. If the function returns no relevant documents or empty results, respond with "I don't have information about that in my knowledge base."

Example of proper response:
Question: "What is the emergency room copay?"
After calling agentic_retrieval and receiving a document with citation "[Benefits Summary](benefits-summary.pdf)":
"The emergency room copay for in-network services is $100 [Benefits Summary](benefits-summary.pdf). For out-of-network services, the copay is $150 [Benefits Summary](benefits-summary.pdf)."

REMEMBER: Call agentic_retrieval FIRST, ALWAYS cite your sources using the provided citation labels, and NEVER answer without using the function."""
                else:
                    instructions += """

GUIDANCE FOR AZURE AI SEARCH TOOL:

1. Invoke the Azure AI Search tool to gather grounding data before answering.
2. The search results will include citation annotations that you must use in the format [title](filepath).
3. Use the exact citation format provided by the tool's annotations.
4. If the metadata lacks a descriptive name, construct a concise label and use it consistently in citations.
5. Never rely solely on internal knowledge when the search results contain relevant information.

Example: "The emergency room copay for in-network services is $100 [Benefits Summary](benefits-summary.pdf)."""
                
                # Prepare tools list - add agentic retrieval tool if enabled
                tools_list = self.tools_list.copy()
                if self.enable_agentic_retrieval and agentic_tool_definition:
                    tools_list.append(agentic_tool_definition)
                    logging.info(f"Added agentic_retrieval to agent tools list")
                
                agent = await project_client.agents.create_agent(
                    model=self.model_name,
                    name="gpt-rag-agent",
                    instructions=instructions,
                    tools=tools_list,
                    tool_resources=self.tool_resources
                )
                create_agent = True
                logging.info(f"Created new agent with ID: {agent.id}")

            conv["agent_id"] = agent.id

            # Send user message
            logging.debug(f"Sending user message into thread {thread.id}: {user_message!r}")
            await project_client.agents.messages.create(
                thread_id=thread.id,
                role="user",
                content=user_message
            )
            logging.debug("User message sent.")

            if self.enable_agentic_retrieval:
                self._last_user_message_by_thread[thread.id] = user_message
                cached_messages = await self._capture_recent_thread_messages(project_client, thread.id)
                if cached_messages:
                    self._recent_thread_messages[thread.id] = cached_messages
                else:
                    logging.warning(
                        "Agentic retrieval cache is empty for thread %s; retrieval will rely on tool arguments",
                        thread.id,
                    )

            # Stream back the agent answer
            logging.debug(f"About to call project_client.agents.runs.stream(...) "
                          f"for agent_id={agent.id}, thread_id={thread.id}")
            stream_params = {
                "thread_id": thread.id,
                "agent_id": agent.id,
                "event_handler": self.event_handler
            }
            
            # Add tool_choice for agentic retrieval to force it to be called first
            # Note: toolset is NOT passed here - functions are already registered via enable_auto_function_calls
            if self.enable_agentic_retrieval and agentic_toolset:
                stream_params["tool_choice"] = AgentsNamedToolChoice(
                    type=AgentsNamedToolChoiceType.FUNCTION,
                    function=FunctionName(name="agentic_retrieval")
                )
                logging.info("🎯 Agentic retrieval configured with FORCED tool_choice")
                logging.debug(f"   tool_choice: agentic_retrieval (function is auto-registered)")
            else:
                logging.debug("Using traditional tools (no agentic retrieval)")
            
            async with await project_client.agents.runs.stream(**stream_params) as stream:
                logging.debug("Entered streaming context; beginning to iterate over events...")
                async for event_type, event_data, raw in stream:
                    # Log important events
                    if event_type.startswith("thread.run.step"):
                        logging.info(f"Stream event: {event_type}")
                    elif event_type.startswith("thread.run"):
                        logging.info(f"Stream event: {event_type}")
                    else:
                        logging.debug(f"Stream event: type={event_type}")
                    
                    # Log tool calls
                    if "tool_calls" in event_type or (hasattr(event_data, 'type') and 'tool' in str(event_data.type)):
                        logging.info(f"🔧 Tool call event: {event_type}, data={event_data}")
                    
                    if event_type == "thread.message.delta" and hasattr(event_data, "text"):
                        chunk = raw or "".join(event_data.text)
                        yield chunk
                    elif event_type == "thread.run.failed":
                        err = event_data.last_error.message
                        logging.error(f"Stream encountered failure: {err}")
                        raise Exception(err)
                logging.debug("Streaming context closed (the run is complete).")

            # After streaming, list all messages in the thread
            logging.debug("Fetching all messages from thread in ascending order...")
            conv["messages"] = []
            messages = project_client.agents.messages.list(
                thread_id=thread.id,
                order=ListSortOrder.ASCENDING
            )
            async for msg in messages:
                if isinstance(msg.content[-1], MessageTextContent):
                    text_val = msg.content[-1].text.value
                    logging.debug(f"Retrieved message in thread: role={msg.role}, text={text_val!r}")
                    conv["messages"].append({
                        "role": msg.role,
                        "text": text_val
                    })
            logging.debug(f"Final conversation messages: {conv['messages']}")

            if self.user_context:
                conv['user_context'] = self.user_context

            if create_agent:
                logging.debug(f"Deleting agent with ID: {agent.id}")
                await project_client.agents.delete_agent(agent.id)
                logging.debug("Agent deletion complete.")


class EventHandler(AsyncAgentEventHandler[str]):
    """
    Handles events emitted during the agent run lifecycle,
    converting each into a human-readable string.
    """

    async def on_message_delta(self, delta: MessageDeltaChunk) -> Optional[str]:
        """
        Called when a partial message is received.
        :param delta: Chunk of the message text.
        :return: The text chunk.
        """
        logging.debug(f"EventHandler.on_message_delta called with delta={delta!r}")
        text = delta.text

        # Collect annotation objects, if any
        raw = getattr(delta, "delta", None)
        annotations = []
        if raw:
            for piece in getattr(raw, "content", []):
                txt = getattr(piece, "text", None)
                if not txt:
                    continue
                anns = getattr(txt, "annotations", None)
                if not anns:
                    continue
                annotations.extend(anns)

        for ann in annotations:
            if isinstance(ann, MessageDeltaTextUrlCitationAnnotation) and "url_citation" in ann:
                info = ann["url_citation"]
                placeholder = ann["text"]
            else:
                continue
            url = info.get("url")
            title = info.get("title", url)
            if url and placeholder:
                # Extract filepath from URL or use title as fallback
                filepath = None
                if url:
                    # Try to extract filename from URL
                    if "/" in url:
                        filepath = url.split("/")[-1]
                    elif "." in url:
                        filepath = url
                
                if not filepath and title:
                    # If no filepath found, check if title contains a file extension
                    if "." in title and not title.startswith("http"):
                        filepath = title
                    else:
                        # Try to construct filepath from title
                        filepath = title.replace(" ", "_") + ".pdf"
                
                # Format as [title](filepath)
                final_title = title if title else "Document"
                final_filepath = filepath if filepath else "unknown.pdf"
                text = text.replace(placeholder, f"[{final_title}]({final_filepath})")

        # logging.trace(f"on_message_delta returning text={text!r}")
        return text

    async def on_thread_message(self, message: ThreadMessage) -> Optional[str]:
        """
        Called when a new thread message object is created.
        :param message: The ThreadMessage instance.
        :return: Summary including message ID and status.
        """
        logging.debug(f"EventHandler.on_thread_message called: ID={message.id}, status={message.status}")
        return f"Thread message created: ID={message.id}, status={message.status}"

    async def on_thread_run(self, run: ThreadRun) -> Optional[str]:
        """
        Called when a new thread run event occurs.
        :param run: The ThreadRun instance.
        :return: Summary of the run status.
        """
        logging.debug(f"EventHandler.on_thread_run called: status={run.status}")
        return f"Thread run status: {run.status}"

    async def on_run_step(self, step: RunStep) -> Optional[str]:
        """
        Called at each step of the run pipeline.
        :param step: The RunStep instance.
        :return: Type and status of the step.
        """
        logging.debug(f"EventHandler.on_run_step called: type={step.type}, status={step.status}")
        return f"Run step: type={step.type}, status={step.status}"

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
