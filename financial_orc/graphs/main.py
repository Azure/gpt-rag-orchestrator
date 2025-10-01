import os
import json
import requests
from typing import List, Annotated, Sequence, TypedDict, Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import (
    AIMessage,
    ToolMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langchain_openai import AzureChatOpenAI
from shared.cosmos_db import (
    get_conversation_data,
    update_conversation_data,
)
from pydantic import BaseModel

# tools
from .tools.tavily_tool import (
    conduct_tavily_search_news,
    conduct_tavily_search_general,
    format_tavily_results,
)
from .tools.database_retriever import CustomRetriever, format_retrieved_content
from datetime import datetime


########################################
# Define agent graph
########################################


class AgentState(TypedDict):
    """The state of the agent."""
    
    question: str
    messages: Annotated[Sequence[BaseMessage], add_messages]
    report: str
    chat_summary: str = ""


class GraphConfig:
    "Config for the graph builder"

    azure_api_version: str = "2024-05-01-preview"
    azure_deployment: str = "gpt-4o-orchestrator"
    index_name: str = "financial-index"
    retriever_top_k: int = 1
    reranker_threshold: float = 1.2
    web_search_results: int = 2
    temperature: float = 0.3
    max_tokens: int = 5000
    verbose: bool = True


class GraphBuilder:
    """Builds and manages the conversation flow graph."""

    def __init__(
        self,
        organization_id: str = None,
        config: GraphConfig = GraphConfig(),
        conversation_id: str = None,
        document_id: str = None,
        document_type: str = "report",
    ):
        """Initialize with configuration and validate environment variables"""
        # Validate required environment variables
        required_env_vars = [
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_API_VERSION",
            "AZURE_AI_SEARCH_API_KEY",
            "AZURE_SEARCH_API_VERSION",
            "AZURE_SEARCH_SERVICE",
        ]

        missing_vars = [var for var in required_env_vars if not os.environ.get(var)]
        if missing_vars:
            raise EnvironmentError(
                f"Missing required environment variables: {', '.join(missing_vars)}"
            )

        # Initialize instance variables
        self.organization_id = organization_id
        self.document_id = document_id
        self.document_type = document_type
        self.config = config
        self.retriever = self._init_retriever()
        self.conversation_id = conversation_id

    def _init_llm(self) -> AzureChatOpenAI:
        return AzureChatOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            azure_deployment="Agent",
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            temperature=0.3,
        )
        
    def _should_continue(self) -> str:
        """Route query based on knowledge requirement."""
        # not sure of how to make the comparison here
        return "return_state"

    def _init_retriever(self) -> CustomRetriever:
        try:
            index_name = self.config.index_name
            return CustomRetriever(
                indexes=[index_name],
                topK=self.config.retriever_top_k,
                reranker_threshold=self.config.reranker_threshold,
                organization_id=self.organization_id,
                document_id=self.document_id,
                verbose=self.config.verbose
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize Azure AI Search Retriever: {str(e)}"
            )

    def build(self, memory) -> StateGraph:
        """Construct the conversation processing graph."""
        # set up graph
        graph = StateGraph(AgentState)

        # Add processing nodes
        graph.add_node("report_preload", self._report_retriever)
        graph.add_node("agent", self._agent)
        graph.add_node("tools", self._tool_node)
        graph.add_node("return_state", self._return_state)

        # Define graph flow
        graph.add_edge(START, "report_preload")
        graph.add_edge("report_preload", "agent")
        graph.add_conditional_edges(
            "agent",
            self._orchestrator,
            {"continue": "tools", "return_state": "return_state"},
        )
        graph.add_edge("tools", "return_state")
        graph.add_edge("return_state", END)

        # Compile the graph
        return graph.compile(checkpointer=memory)

    def _report_retriever(self, state: AgentState) -> dict:
        """Retrieve the initial report."""
        try:
            # Get documents from retriever
            documents = self.retriever.invoke(self.document_type)
            # format the retrieved content
            formatted_content = format_retrieved_content(documents)

            if self.config.verbose:
                print(
                    f"[financial-orchestrator-agent] RETRIEVED DOCUMENTS: {len(documents)}"
                )

            if not documents or len(documents) == 0:
                if self.config.verbose:
                    print(
                        "[financial-orchestrator-agent] No documents retrieved, using fallback content"
                    )
                documents = [
                    Document(page_content="No information found about the report")
                ]

            return {"report": formatted_content}
        except Exception as e:
            if self.config.verbose:
                print(
                    f"[financial-orchestrator-agent] Error in report_retriever: {str(e)}"
                )
            # Return a fallback document to prevent crashes
            return {
                "report": [
                    Document(page_content="Error retrieving report information.")
                ]
            }

    class ToolDecision(BaseModel):
        orc_decision: Literal["tools", "return_state"]

    def _agent(self, state: AgentState) -> dict:
        """Currently doing nothing, it just exists as a dummy node"""
        pass

    def _orchestrator(self, state: AgentState) -> dict:
        """Decide if the question should be answered using a tool or not."""

        system_prompt = """ 
        You're a helpful assistant. Please decide if the question should be answered using a tool or not.
        """
        user_query = state.get("question")
        #user_query = state.get("messages", ["no messages"])[-1].content
        prompt = f"""
        Question: {user_query}
        """

        # structured_llm = self._init_llm().with_structured_output(self.ToolDecision)
        # response = structured_llm.invoke(prompt)

        # comment from dev: tools are generating errors, so I am commenting this out for now
        # it does not use to have tools, no idea what are these used for

        # if response.orc_decision == "tools":
        #     return "tools"
        # else:
        return "return_state"

    @tool
    def web_search(query: str) -> str:
        """Conduct web search for user query that is not included in the report
        This tool should be used when the question is about recent news or events."""

        # Step 2. Executing a context search query
        result = conduct_tavily_search_news(query)
        # format the results
        formatted_results = format_tavily_results(result)

        return formatted_results

    @tool
    def general_search(query: str) -> str:
        """Conduct general web search for user query that is not included in the report
        This tool should be used when the question is more general, and the requested information does not have to be up to date.
        """
        result = conduct_tavily_search_general(query)
        formatted_results = format_tavily_results(result)
        return formatted_results

    tools = [web_search, general_search]

    # create a dictionary of tools by name
    tools_by_name = {tool.name: tool for tool in tools}

    def _tool_node(self, state: AgentState):
        outputs = []
        try:
            for tool_call in state["messages"][-1].tool_calls:
                tool_result = self.tools_by_name[tool_call["name"]].invoke(
                    tool_call["args"]
                )
                outputs.append(
                    ToolMessage(
                        content=json.dumps(tool_result),
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                    )
                )
        except KeyError as e:
            raise ValueError(f"Tool not found: {e}")
        except Exception as e:
            raise Exception(f"Error processing tool calls: {e}")
        return {"messages": outputs}

    def _return_state(self, state: AgentState) -> dict:
        return {
            "messages": state["messages"],
            "report": state["report"],
            "chat_summary": state["chat_summary"],
        }


def create_conversation_graph(
    memory, organization_id=None, conversation_id=None, document_id=None, document_type="report"
) -> StateGraph:
    """Create and return a configured conversation graph.
    Returns:
        Compiled StateGraph for conversation processing
    """
    print(f"Creating conversation graph for organization: {organization_id}")
    builder = GraphBuilder(
        organization_id=organization_id, 
        conversation_id=conversation_id, 
        document_id=document_id,
        document_type=document_type
    )
    return builder.build(memory)
