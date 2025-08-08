import os
import sys
import json
import logging
import asyncio
from dataclasses import dataclass, field
from typing import List, Optional
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from typing import Dict, Any
from shared.util import get_secret

# Load environment variables FIRST, before importing modules that read them
load_dotenv()

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain.schema import Document
from langgraph.graph import StateGraph, END, START
from langchain_openai import AzureChatOpenAI

from shared.prompts import (
    MARKETING_ORC_PROMPT,
    QUERY_REWRITING_PROMPT,
    AUGMENTED_QUERY_PROMPT,
    MCP_SYSTEM_PROMPT,
)
from shared.cosmos_db import get_conversation_data
from shared.util import get_organization
from orc.graphs.utils import clean_chat_history_for_llm, extract_thread_id_from_history
from langgraph.checkpoint.memory import MemorySaver

# Set up logging for Azure Functions - this needs to be done before creating loggers
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
    force=True,  # Override any existing logging configuration
)

# Configure the main module logger
logger = logging.getLogger(__name__)

# Set logging level for Azure Search libraries
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
    messages: List[AIMessage | HumanMessage] = field(
        default_factory=list
    )  
    context_docs: List[Document] = field(default_factory=list)
    requires_retrieval: bool = field(default=False)
    rewritten_query: str = field(
        default_factory=str
    )  
    query_category: str = field(default_factory=str)
    augmented_query: str = field(default_factory=str)
    mcp_tool_used: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: List[Any] = field(default_factory=list)
    code_thread_id: Optional[str] = field(default=None)


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
    azure_api_version: str = "2025-01-01-preview"
    azure_deployment: str = "gpt-4.1"
    retriever_top_k: int = 5
    reranker_threshold: float = 2
    web_search_results: int = 2
    temperature: float = 0.4
    max_tokens: int = 200000

class GraphBuilder:
    """Builds and manages the conversation flow graph."""

    def __init__(
        self,
        organization_id: str = None,
        config: GraphConfig = GraphConfig(),
        conversation_id: str = None,
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
        
        # Initialize LLM and retriever
        self.llm = self._init_llm()
        
        # Initialize organization data with error handling
        try:
            self.organization_data = get_organization(organization_id)
            logger.info(f"[GraphBuilder Init] Successfully retrieved organization data for ID: {organization_id}")
        except Exception as e:
            logger.error(f"[GraphBuilder Init] Failed to retrieve organization data: {str(e)}")
            logger.warning("[GraphBuilder Init] Using empty organization data as fallback")
            self.organization_data = {
                "segmentSynonyms": "",
                "brandInformation": "",
                "industryInformation": "",
            }

        logger.info("[GraphBuilder Init] Successfully initialized GraphBuilder")

    def _init_llm(self) -> AzureChatOpenAI:
        """Configure Azure OpenAI instance."""
        logger.info("[GraphBuilder LLM Init] Initializing Azure OpenAI client")
        config = self.config
        
        endpoint = os.getenv("O1_ENDPOINT")
        api_key = os.getenv("O1_KEY")
        
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

    def _get_organization_data(self, data_key: str, data_name: str) -> str:
        """
        Retrieve organization data by key with consistent logging.

        Args:
            data_key: Key in organization_data dictionary
            data_name: Human-readable name for logging

        Returns:
            Organization data value or empty string if not found
        """
        data_value = self.organization_data.get(data_key, "")
        logger.info(
            f"[GraphBuilder {data_name} Init] Retrieved {data_name.lower()} for organization {self.organization_id}"
        )
        return data_value

    def _init_segment_alias(self) -> str:
        """Retrieve segment alias."""
        return self._get_organization_data("segmentSynonyms", "")

    def _init_brand_information(self) -> str:
        """Retrieve brand information."""
        return self._get_organization_data("brandInformation", "")

    def _init_industry_information(self) -> str:
        """Retrieve industry information."""
        return self._get_organization_data("industryInformation", "")

    def _get_conversation_data(self) -> dict:
        """
        Retrieve conversation data.

        Returns:
            Dictionary containing conversation data with history
        """
        logger.info(f"[Conversation] Fetching conversation data for ID: {self.conversation_id}")
        try:
            return get_conversation_data(self.conversation_id)
        except Exception as e:
            logger.error(f"[Conversation] Failed to retrieve conversation data: {str(e)}")
            logger.warning("[Conversation] Using empty conversation history as fallback")
            return {
                "history": []
            }

    async def _llm_invoke(self, messages, **kwargs):
        """
        LLM invocation.

        Args:
            messages: List of messages to send to LLM
            **kwargs: Additional arguments for LLM call

        Returns:
            LLM response
        """
        return await self.llm.ainvoke(messages, **kwargs)
    
    def _find_tool_by_name(self, tools: List[Any], tool_name: str):
        """
        Find a tool in the tools list by its name.

        Args:
            tools: List of available tools
            tool_name: Name of the tool to find

        Returns:
            The tool object if found, None otherwise
        """
        for tool in tools:
            if tool.name == tool_name:
                return tool
        return None

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

        Though the historical conversation and the orignal question may mention "students", you must rewrite the question to use "young kids" instead of "students".

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

    def _get_context_docs_from_tool_results(self, state: ConversationState) -> List[Any]:
        """Get context docs from the tool results"""
        context_docs = []

        if state.mcp_tool_used and state.tool_results:
            for i, tool_call in enumerate(state.mcp_tool_used):
                if i < len(state.tool_results):
                    tool_result = state.tool_results[i]
                    
                    if isinstance(tool_result, str):
                        tool_result = json.loads(tool_result)

                    if tool_call["name"] == "agentic_search" and isinstance(tool_result, dict):
                        context_docs.append(tool_result.get("results", tool_result))
                    elif tool_call["name"] == "data_analyst" and isinstance(tool_result, dict):
                        context_docs.append(tool_result.get("last_agent_message", tool_result))
                    else:
                        context_docs.append(tool_result)
        return context_docs
    
    def _return_state(self, state: ConversationState) -> dict:
        # Get updated thread ID from tool results, fallback to existing thread ID
        updated_thread_id = self._get_code_thread_id(state) or state.code_thread_id
        
        return {
            "messages": state.messages,
            "context_docs": self._get_context_docs_from_tool_results(state),
            "rewritten_query": state.rewritten_query,
            "query_category": state.query_category,
            "mcp_tool_used": state.mcp_tool_used,
            "tool_results": state.tool_results,
            "code_thread_id": updated_thread_id
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
        question = state.question

        system_prompt = QUERY_REWRITING_PROMPT

        conversation_data = self._get_conversation_data()
        history = conversation_data.get("history", [])
        logger.info(
            f"[Query Rewrite] Retrieved {len(history)} messages from conversation history"
        )

        # combine the system prompt with the additional system prompt
        system_prompt = (
            f"{system_prompt}\n\n{self._build_organization_context_prompt(history)}"
        )

        prompt = f"""Original Question: 
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

        logger.info("[Query Rewrite] Sending async query rewrite request to LLM")
        rewritte_query = await self._llm_invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=prompt)]
        )
        logger.info(
            f"[Query Rewrite] Successfully rewrote query: '{rewritte_query.content[:100]}...'"
        )

        if state.messages is None:
            state.messages = []

        # augment the query with the historical conversation context
        augmented_query_prompt = f""" 
        Augment the query with the historical conversation context. If the query is a very casual/conversational one, do not augment, return it as it is.
        
        Here is the historical conversation context if available:
        <context>
        {clean_chat_history_for_llm(history)}
        </context>

        Here is the query to augment:
        <query>
        {question}
        </query>

        Return the augmented query in text format only, no additional text, explanations, or formatting.
        
        """
        logger.info(
            f"[Query Augment] Sending async augmented query request to LLM {augmented_query_prompt[:100]}..."
        )
        try:
            augmented_query = await self._llm_invoke(
                [
                    SystemMessage(content=AUGMENTED_QUERY_PROMPT),
                    HumanMessage(content=augmented_query_prompt),
                ]
            )
            logger.info(
                f"[Query Augment] Successfully augmented query: '{augmented_query.content[:100]}...'"
            )
        except Exception as e:
            logger.error(
                f"[Query Augment] Failed to augment query, using original question: {e}"
            )
            augmented_query = question

        # Initialize code thread ID from conversation history (this is for the code interpreter tool)
        existing_thread_id = extract_thread_id_from_history(history)
        logger.info(f"[Query Rewrite] Initialized thread_id from history: {existing_thread_id}")

        return {
            "rewritten_query": rewritte_query.content,
            "augmented_query": (
                augmented_query.content
                if hasattr(augmented_query, "content")
                else augmented_query
            ),
            "messages": state.messages + [HumanMessage(content=question)],
            "code_thread_id": existing_thread_id,
        }

    async def _categorize_query(self, state: ConversationState) -> dict:
        """Categorize the query."""
        logger.info(
            f"[Query Categorization] Starting async query categorization for: '{state.question[:100]}...'"
        )

        conversation_data = self._get_conversation_data()
        history = conversation_data.get("history", [])
        logger.info(
            f"[Query Categorization] Using {len(history)} conversation history messages for context"
        )

        category_prompt = f"""
        You are a senior marketing strategist. Your task is to classify the user's question into one of the following categories:

        - Creative Brief
        - Marketing Plan
        - Brand Positioning Statement
        - Creative Copywriter
        - General

        Use both the current question and the historical conversation context to make an informed decision. 
        Context is crucial, as users may refer to previous topics, provide follow-ups, or respond to earlier prompts. 

        To help you make an accurate decision, consider these cues for each category:

        - **Creative Brief**: Look for project kickoffs, campaign overviews, client objectives, audience targeting, timelines, deliverables, or communication goals.
        - **Marketing Plan**: Look for references to strategy, goals, budget, channels, timelines, performance metrics, or ROI.
        - **Brand Positioning Statement**: Watch for messages about defining brand essence, values, personality, competitive differentiation, or target audience perception.
        - **Creative Copywriter**: Use this category when the user asks for help creating or refining marketing text. This includes taglines, headlines, ad copy, email subject lines, social captions, website copy, or product descriptions. Trigger this if the user is brainstorming, writing, or editing text with a creative, promotional purpose.
        - **General**: If the input lacks context, doesn't relate to marketing deliverables, or is unclear or unrelated to the above.

        If the question or context is not clearly related to any of the above categories, always return "General".

        ----------------------------------------
        User's Question:
        {state.question}
        ----------------------------------------
        Conversation History:
        {clean_chat_history_for_llm(history)}
        ----------------------------------------

        Reply with **only** the exact category name — no additional text, explanations, or formatting.
        """

        logger.info(
            "[Query Categorization] Sending async categorization request to LLM"
        )
        response = await self._llm_invoke(
            [
                SystemMessage(content=category_prompt),
                HumanMessage(content=state.question),
            ],
            temperature=0,
        )
        logger.info(
            f"[Query Categorization] Categorized query as: '{response.content}'"
        )

        return {"query_category": response.content}

    async def _route_query(self, state: ConversationState) -> dict:
        """Determine if external knowledge is needed."""
        logger.info(
            f"[Query Routing] Determining routing decision for query: '{state.rewritten_query[:100]}...'"
        )

        system_prompt = MARKETING_ORC_PROMPT

        logger.info("[Query Routing] Sending routing decision request to LLM")
        response = await self._llm_invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(
                    content=f"How should I categorize this question: \n\n{state.rewritten_query}\n\nAnswer yes/no."
                ),
            ]
        )

        llm_suggests_retrieval = response.content.lower().startswith("y")
        logger.info(
            f"[Query Routing] LLM assessment - Not a casual/conversational question, proceed to retrieve documents: {llm_suggests_retrieval}"
        )

        return {
            "requires_retrieval": llm_suggests_retrieval,
            "query_category": "General",
        }

    def _route_decision(self, state: ConversationState) -> str:
        """Route query based on knowledge requirement."""
        decision = "tool_choice" if state.requires_retrieval else "return"
        logger.info(
            f"[Route Decision] Routing to: '{decision}' (requires_retrieval: {state.requires_retrieval})"
        )
        return decision
    
    def _configure_agentic_search_args(
        self,
        tool_call: Dict[str, Any],
        state: ConversationState
    ) -> Dict[str, Any]:
        """
        Configure additional arguments for agentic_search tool calls.
        """
        if tool_call["name"] == "agentic_search":
            tool_call["args"].update(
                {
                    "organization_id": self.organization_id,
                    "rewritten_query": state.rewritten_query,
                    "reranker_threshold": self.config.reranker_threshold,
                    "historical_conversation": clean_chat_history_for_llm(state.messages), 
                    "web_search_threshold": self.config.web_search_results,
                }
            )
    
    def _configure_data_analyst_args(
        self,
        tool_call: Dict[str, Any],
        state: ConversationState
    ) -> Dict[str, Any]:
        """Configure additional arguments for data_analyst tool calls."""
        if tool_call["name"] == "data_analyst":
            tool_call["args"].update(
                {
                    "organization_id": self.organization_id,
                    "code_thread_id": state.code_thread_id
                }
            )
    
    def _is_local_environment(self) -> bool:
        """Check if the current environment is local development."""
        return os.getenv("ENVIRONMENT", "").lower() == "local"
    
    async def _init_mcp_client(self) -> MultiServerMCPClient:
        """Initialize the MCP client"""
        try: 
            mcp_function_secret = get_secret("mcp-host--functionkey")
            mcp_function_name = os.getenv("MCP_FUNCTION_NAME")
        except Exception as e:
            logger.error(f"Error getting MCP function variables: {str(e)}")
            raise RuntimeError(f"Error getting MCP function variables: {str(e)}")
        
        # Use different endpoint for local environment
        if self._is_local_environment():
            mcp_url = f"http://localhost:7073/runtime/webhooks/mcp/sse"
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
    
    async def _get_tool_calls(self, state: ConversationState) -> dict:
        """Get tool calls from the MCP server"""
        client = await self._init_mcp_client()

        tools = await client.get_tools()
        logger.info(f"Found {len(tools)} tools")

        # equip the llm with the tools
        llm_with_tools = self.llm.bind_tools(tools, tool_choice="any") # switch to auto in case we want to use no tool 

        message = [SystemMessage(content=MCP_SYSTEM_PROMPT), HumanMessage(content=state.question)]
        try:
            response = await llm_with_tools.ainvoke(message)
        except Exception as e:
            logger.error(f"Error getting tool calls from LLM: {str(e)}")
            raise RuntimeError(f"Error getting tool calls from LLM: {str(e)}")
        return {
            "mcp_tool_used": response.tool_calls,
        }
        
    async def _execute_mcp_tool_calls(self, state: ConversationState) -> List[Any]:
        """Execute tool calls."""
        mcp_tool_used = state.mcp_tool_used
        tool_results = []
        if not mcp_tool_used:
            logger.info("No tool calls to execute")
            return tool_results
        
        logger.info(f"Executing {len(mcp_tool_used)} tool(s)...")
        
        # gotta call it here again since langchain gives me too much trouble to store tool calls results in the state
        client = await self._init_mcp_client()
        mcp_available_tools = await client.get_tools()
        
        for tool_call in mcp_tool_used:
            tool_name = tool_call["name"]
            if tool_name == "agentic_search":
                self._configure_agentic_search_args(tool_call, state)
            elif tool_name == "data_analyst":
                self._configure_data_analyst_args(tool_call, state)

            tool = self._find_tool_by_name(mcp_available_tools, tool_name)
            if tool:
                try:
                    logger.info(f"Running {tool_name}...")
                    tool_result = await tool.ainvoke(tool_call["args"])
                    tool_results.append(tool_result)
                    logger.info(f"{tool_name} completed successfully")
                except Exception as e:
                    logger.error(f"Error executing {tool_name}: {e}")
                    tool_results.append(f"Error: {e}")
            else:
                error_msg = f"Tool '{tool_name}' not found in available tools"
                logger.error(error_msg)
                tool_results.append(error_msg)
        
        if tool_results:
            preview = str(tool_results[0])
            if len(preview) > 200:
                preview = preview[:200] + "..."
            logger.info(f"Tool results: {preview}")
            return {
                "tool_results": tool_results,
            }
            
        else:
            logger.info("No tool results to return")
            return {
                "tool_results": tool_results,
            }

def create_conversation_graph(
    memory, organization_id=None, conversation_id=None
) -> StateGraph:
    """Create and return a configured conversation graph.
    Returns:
        Compiled StateGraph for conversation processing
    """
    logger.info(
        f"[Conversation Graph Creation] Creating conversation graph for conversation: {conversation_id}"
    )
    builder = GraphBuilder(
        organization_id=organization_id, conversation_id=conversation_id
    )
    return builder.build(memory)


# if __name__ == "__main__":

#     config = GraphConfig()
#     # Initialize memory saver
#     memory = MemorySaver()

#     graph_builder = GraphBuilder(
#         organization_id="6c33b530-22f6-49ca-831b-25d587056237",
#         config=config,
#         conversation_id="123",
#     )

#     graph = graph_builder.build(memory=memory)
    
#     print(f"\nInvoking Graph with Sample Question...")
    
#     question = "How has total POS $ and POS Units evolved month-over-month from Jan 2024 through the latest month in 2025?"

#     async def run_graph():
#         try:
#             config_dict = {"configurable": {"thread_id": "test-thread-123"}}
#             result = await graph.ainvoke({"question": question}, config=config_dict)
            
#             print(result)
                
#         except Exception as e:
#             print(f"Error invoking graph: {str(e)}")
    
#     asyncio.run(run_graph())
