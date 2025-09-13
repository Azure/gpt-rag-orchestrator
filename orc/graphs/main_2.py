import os
import sys
import json
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
load_dotenv()
from shared.progress_streamer import ProgressStreamer, ProgressSteps, STEP_MESSAGES
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain.schema import Document
from langgraph.graph import StateGraph, END, START
from langchain_openai import AzureChatOpenAI

from shared.cosmos_db import get_conversation_data
from shared.util import get_organization
from orc.graphs.utils import (
    clean_chat_history_for_llm,
)
from orc.graphs.context_builder import ContextBuilder
from orc.graphs.mcp_executor import MCPExecutor
from orc.graphs.query_planner import QueryPlanner
from orc.graphs.constants import ENV_O1_ENDPOINT, ENV_O1_KEY

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
    force=True,  
)

logger = logging.getLogger(__name__)

azure_search_logger = logging.getLogger("azure.search")
azure_search_logger.setLevel(logging.INFO)

# Set logging level for Azure Identity libraries
azure_identity_logger = logging.getLogger("azure.identity")
azure_identity_logger.setLevel(logging.WARNING)  # Less verbose for auth

# Set logging level for all Azure libraries (fallback)
azure_logger = logging.getLogger("azure")
azure_logger.setLevel(logging.WARNING)

# Suppress noisy Azure Functions worker logs
azure_functions_worker_logger = logging.getLogger("azure_functions_worker")
azure_functions_worker_logger.setLevel(logging.WARNING)

# Set logging level for LangChain libraries
langchain_logger = logging.getLogger("langchain")
langchain_logger.setLevel(logging.WARNING)

# Set logging level for OpenAI libraries
openai_logger = logging.getLogger("openai")
openai_logger.setLevel(logging.WARNING)

# Ensure propagation is enabled for Azure Functions
logger.propagate = True
azure_search_logger.propagate = True
azure_identity_logger.propagate = True
azure_logger.propagate = True
langchain_logger.propagate = True
openai_logger.propagate = True


@dataclass
class ConversationState:
    """State container for conversation flow management.

    Attributes:
        question: Current user query
        messages: Conversation history as a list of messages
    """

    question: str
    messages: List[AIMessage | HumanMessage] = field(default_factory=list)
    context_docs: List[Document] = field(default_factory=list)
    requires_retrieval: bool = field(default=False)
    rewritten_query: str = field(default_factory=str)
    query_category: str = field(default_factory=str)
    augmented_query: str = field(default_factory=str)
    mcp_tool_used: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: List[Any] = field(default_factory=list)
    code_thread_id: Optional[str] = field(default=None)
    last_mcp_tool_used: str = field(default="")


@dataclass
class GraphConfig:
    """
    Configuration parameters for the graph builder.

    Attributes:
        azure_api_version (str): Azure OpenAI API version to use. Default is "2025-01-01-preview".
        azure_deployment (str): Name of the Azure OpenAI deployment. Default is "gpt-4.1".
        retriever_top_k (int): Number of top documents to retrieve. Default is 5.
        reranker_threshold (float): Threshold for reranking retrieved documents. Default is 2.
        web_search_results (int): Number of web search results to include. Default is 2.
        temperature (float): Sampling temperature for the language model. Default is 0.4.
        max_tokens (int): Maximum number of tokens for the model output. Default is 200000.
    """

    azure_api_version: str = (
        "2025-04-01-preview"  
    )   
    azure_deployment: str = "gpt-4.1"   
    support_model_deployment: str = "gpt-5-nano"  
    support_model_reasoning_effort: str = "low"  

    retriever_top_k: int = 5
    reranker_threshold: float = 2
    web_search_results: int = 2
    temperature: float = 0.4
    max_tokens: int = 20000


class GraphBuilder:
    """Builds and manages the conversation flow graph."""

    def __init__(
        self,
        organization_id: str = None,
        config: GraphConfig = GraphConfig(),
        conversation_id: str = None,
        user_id: str = None,
        progress_streamer: Optional[ProgressStreamer] = None,
    ):
        """Initialize with with configuration"""
        logger.info(
            f"[GraphBuilder Init] Initializing GraphBuilder for conversation: {conversation_id}"
        )
        logger.info(
            f"[GraphBuilder Init] Config - model temperature: {config.temperature}, max_tokens: {config.max_tokens}"
        )

        self.organization_id = organization_id
        self.config = config
        self.conversation_id = conversation_id
        self.user_id = user_id
        self.progress_streamer = progress_streamer

        # Initialize LLM and retriever
        self.llm = self._init_llm()
        self.support_llm = self._init_support_model()

        # Initialize organization data with error handling
        try:
            self.organization_data = get_organization(organization_id)
            logger.info(
                f"[GraphBuilder Init] Successfully retrieved organization data for ID: {organization_id}"
            )
        except Exception as e:
            logger.error(
                f"[GraphBuilder Init] Failed to retrieve organization data: {str(e)}"
            )
            logger.warning(
                "[GraphBuilder Init] Using empty organization data as fallback"
            )
            self.organization_data = {
                "segmentSynonyms": "",
                "brandInformation": "",
                "industryInformation": "",
            }

        try:
            self.conversation_data = get_conversation_data(conversation_id)
            logger.info(
                f"[GraphBuilder Init] Successfully retrieved conversation data for ID: {conversation_id}"
            )
        except Exception:
            logger.exception(
                "[GraphBuilder Init] Failed to retrieve conversation data"
            )
            logger.warning(
                "[GraphBuilder Init] Using empty conversation data as fallback"
            )
            self.conversation_data = {"history": []}

        logger.info("[GraphBuilder Init] Successfully initialized GraphBuilder")

        self.context_builder = ContextBuilder(self.organization_data)
        self.mcp_executor = MCPExecutor(
            organization_id=self.organization_id,
            user_id=self.user_id,
            config=self.config,
            progress_streamer=self.progress_streamer,
        )
        self.query_planner = QueryPlanner(self.llm)

    def _init_llm(self) -> AzureChatOpenAI:
        """Configure Azure OpenAI instance."""
        logger.info("[GraphBuilder LLM Init] Initializing Azure OpenAI client")
        config = self.config

        endpoint = os.getenv(ENV_O1_ENDPOINT)
        api_key = os.getenv(ENV_O1_KEY)

        try:
            llm = AzureChatOpenAI(
                temperature=config.temperature,
                openai_api_version=config.azure_api_version,
                azure_deployment=config.azure_deployment,
                streaming=False,
                timeout=30,
                max_retries=3,
                azure_endpoint=endpoint,
                api_key=api_key,
            )
            logger.info(
                f"[GraphBuilder LLM Init] Successfully initialized Azure OpenAI with deployment: {config.azure_deployment}"
            )
            return llm
        except Exception as e:
            logger.error(
                f"[GraphBuilder LLM Init] Failed to initialize Azure OpenAI: {str(e)}"
            )
            raise RuntimeError(f"Failed to initialize Azure OpenAI: {str(e)}")

    def _init_support_model(self) -> AzureChatOpenAI:
        """Configure Azure OpenAI instance."""
        logger.info(
            "[GraphBuilder Support Model Init] Initializing Azure OpenAI client"
        )
        config = self.config
        return AzureChatOpenAI(
            openai_api_version=config.azure_api_version,
            azure_deployment=config.support_model_deployment,
            reasoning_effort=config.support_model_reasoning_effort,
            streaming=False,
            timeout=30,
            max_retries=3,
            azure_endpoint=os.getenv(ENV_O1_ENDPOINT),
            api_key=os.getenv(ENV_O1_KEY),
        )

    def _get_conversation_data(self) -> dict:
        """
        Retrieve cached conversation data to avoid multiple DB calls.

        Returns:
            Dictionary containing conversation data with history
        """
        logger.info(
            f"[Conversation] Using cached conversation data for ID: {self.conversation_id}"
        )
        return self.conversation_data

    async def _llm_invoke(
        self, messages: List[SystemMessage | HumanMessage | AIMessage], **kwargs
    ):
        """
        LLM invocation.

        Args:
            messages: List of messages to send to LLM
            **kwargs: Additional arguments for LLM call

        Returns:
            LLM response
        """
        return await self.llm.ainvoke(messages, **kwargs)

    async def _support_llm_invoke(
        self, messages: List[SystemMessage | HumanMessage | AIMessage], **kwargs
    ):
        """
        Support LLM invocation.

        Args:
            messages: List of messages to send to LLM
            **kwargs: Additional arguments for LLM call

        Returns:
            LLM response
        """
        return await self.support_llm.ainvoke(messages, **kwargs)

    def _build_organization_context_prompt(self, history: List[dict]) -> str:
        """
        Build the organization context prompt with conversation history and organization data.

        Args:
            history: List of conversation history messages

        Returns:
            Formatted organization context prompt
        """
        return f"""
        <-------------------------------->
        
        Historical Conversation Context:
        <-------------------------------->
        ```
        {clean_chat_history_for_llm(history)}
        ```
        <-------------------------------->

        **Alias segment mappings:**
        <-------------------------------->
        alias to segment mappings typically look like this (Official Name -> Alias):
        A -> B
        
        This mapping is mostly used in consumer segmentation context. 
        
        Critical Rule – Contextual Consistency with Alias Mapping:
    •	Always check whether the segment reference in the historical conversation is an alias (B). For example, historical conversation may mention "B" segment, but whenever you read the context in order to rewrite the query, you must map it to the official segment name "A" using the alias mapping table.
    •	ALWAYS use the official name (A) in the rewritten query.
    •	DO NOT use the alias (B) in the rewritten query. 

        Here is the actual alias to segment mappings:
        
        **Official Segment Name Mappings (Official Name -> Alias):**
        ```
        {self._init_segment_alias()}
        ```

        For example, if the historical conversation mentions "B", and the original question also mentions "B", you must rewrite the question to use "A" instead of "B".

        Look, if a mapping in the instruction is like this:
        students -> young kids 

        Though the historical conversation and the original question may mention "students", you must rewrite the question to use "young kids" instead of "students".

        <-------------------------------->
        Brand Information:
        <-------------------------------->
        ```
        {self._init_brand_information()}
        ```
        <-------------------------------->

        Industry Information:
        <-------------------------------->
        ```
        {self._init_industry_information()}
        ```
        <-------------------------------->

        """

    def _get_code_thread_id(self, state: ConversationState) -> Optional[str]:
        """Extract thread id from the code interpreter tool result"""
        if state.mcp_tool_used and state.tool_results:
            for tool_call, tool_result in zip(state.mcp_tool_used, state.tool_results):
                if tool_call["name"] == "data_analyst":
                    if isinstance(tool_result, str):
                        tool_result = json.loads(tool_result)
                    if isinstance(tool_result, dict):
                        thread_id = tool_result.get("thread_id")
                        if thread_id:
                            logger.info(f"Existing thread id found: {thread_id}")
                            return thread_id
        return None

    def _get_last_mcp_tool_used(self, state: ConversationState) -> str:
        """Extract the name of the last MCP tool used"""
        if state.mcp_tool_used:
            last_tool_name = state.mcp_tool_used[-1]["name"]  # [0] works just fine
            logger.info(f"Last MCP tool used: {last_tool_name}")
            return last_tool_name
        return state.last_mcp_tool_used

    def _return_state(self, state: ConversationState) -> dict:
        # Get updated thread ID from tool results, fallback to existing thread ID
        updated_thread_id = self._get_code_thread_id(state) or state.code_thread_id

        # Get the the latest mcp tool used
        last_mcp_tool_used = self._get_last_mcp_tool_used(state)

        context_docs = self.context_builder.to_context_docs(state)
        return {
            "messages": state.messages,
            "context_docs": context_docs,
            "rewritten_query": state.rewritten_query,
            "query_category": state.query_category,
            "mcp_tool_used": state.mcp_tool_used,
            "tool_results": state.tool_results,
            "code_thread_id": updated_thread_id,
            "last_mcp_tool_used": last_mcp_tool_used,
        }

    def build(self, memory) -> StateGraph:
        """Construct the conversation processing graph."""
        logger.info("[GraphBuilder Build] Starting graph construction")

        graph = StateGraph(ConversationState)

        graph.add_node("rewrite", self._rewrite_query)
        graph.add_node("route", self._route_query)
        graph.add_node("tool_choice", self._categorize_query)
        graph.add_node("get_mcp_tool_calls", self._get_tool_calls)
        graph.add_node("execute_mcp_tool_calls", self._execute_mcp_tool_calls)
        graph.add_node("return", self._return_state)

        # Define graph flow
        graph.add_edge(START, "rewrite")
        graph.add_edge("rewrite", "route")
        graph.add_conditional_edges(
            "route",
            self._route_decision,
            {"tool_choice": "tool_choice", "return": "return"},
        )
        graph.add_edge("tool_choice", "get_mcp_tool_calls")
        graph.add_edge("get_mcp_tool_calls", "execute_mcp_tool_calls")
        graph.add_edge("execute_mcp_tool_calls", "return")
        graph.add_edge("return", END)

        compiled_graph = graph.compile(checkpointer=memory)
        logger.info(
            "[GraphBuilder Build] Successfully constructed conversation processing graph"
        )
        return compiled_graph

    async def _rewrite_query(self, state: ConversationState) -> dict:
        logger.info(
            f"[Query Rewrite] Starting async query rewrite for: '{state.question[:100]}...'"
        )

        if self.progress_streamer:
            self.progress_streamer.emit_progress(
                ProgressSteps.QUERY_REWRITE,
                STEP_MESSAGES[ProgressSteps.QUERY_REWRITE],
                20,
            )
        conversation_data = self._get_conversation_data()
        return await self.query_planner.rewrite(
            state, conversation_data, self.context_builder
        )

    async def _categorize_query(self, state: ConversationState) -> dict:
        """Categorize the query."""
        logger.info(
            f"[Query Categorization] Starting async query categorization for: '{state.question[:100]}...'"
        )

        conversation_data = self._get_conversation_data()
        return await self.query_planner.categorize(state, conversation_data)

    async def _route_query(self, state: ConversationState) -> dict:
        """Determine if external knowledge is needed."""
        logger.info(
            f"[Query Routing] Determining routing decision for query: '{state.rewritten_query[:100]}...'"
        )

        return await self.query_planner.route(state)

    def _route_decision(self, state: ConversationState) -> str:
        """Route query based on knowledge requirement."""
        decision = "tool_choice" if state.requires_retrieval else "return"
        logger.info(
            f"[Route Decision] Routing to: '{decision}' (requires_retrieval: {state.requires_retrieval})"
        )
        return decision

    async def _get_tool_calls(self, state: ConversationState) -> dict:
        """Get tool calls via MCPExecutor."""
        conversation_data = self._get_conversation_data()
        return await self.mcp_executor.get_tool_calls(
            state, self.llm, conversation_data
        )

    async def _execute_mcp_tool_calls(self, state: ConversationState) -> dict:
        """Execute tool calls via MCPExecutor."""
        return await self.mcp_executor.execute_tool_calls(state)


def create_conversation_graph(
    memory,
    organization_id=None,
    conversation_id=None,
    user_id=None,
    progress_streamer=None,
) -> StateGraph:
    """Create and return a configured conversation graph.
    Returns:
        Compiled StateGraph for conversation processing
    """
    logger.info(
        f"[Conversation Graph Creation] Creating conversation graph for conversation: {conversation_id}"
    )
    builder = GraphBuilder(
        organization_id=organization_id,
        conversation_id=conversation_id,
        user_id=user_id,
        progress_streamer=progress_streamer,
    )
    return builder.build(memory)
