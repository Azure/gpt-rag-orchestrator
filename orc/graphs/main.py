import os
from dataclasses import dataclass, field
from typing import List
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.schema import Document
from langgraph.graph import StateGraph, END, START
from langchain_openai import AzureChatOpenAI
from orc.graphs.tools import CustomRetriever, GoogleSearch
from shared.prompts import MARKETING_ORC_PROMPT, MARKETING_ANSWER_PROMPT, QUERY_REWRITING_PROMPT


@dataclass
class ConversationState:
    """State container for conversation flow management.

    Attributes:
        question: Current user query
        messages: Conversation history as a list of messages
        context_docs: Retrieved documents from various sources
        requires_web_search: Flag indicating if web search is needed
    """

    question: str
    messages: List[AIMessage | HumanMessage] = field(default_factory=list)
    context_docs: List[Document] = field(default_factory=list)
    requires_web_search: bool = False
    rewritten_query: str = field(default_factory=str) # rewritten query for better search 
    # conversation_history: List[AIMessage | HumanMessage] = field(default_factory=list) # TODO: Add conversation history, blank for now
    # conversation_summary: str = field(default_factory=str) # TODO: Add conversation summary, blank for now

@dataclass
class GraphConfig: 
    "Config for the graph builder"
    azure_api_version: str = "2024-05-01-preview"
    azure_deployment: str = "gpt-4o-orchestrator"
    retriever_top_k: int = 5
    reranker_threshold: float = 2.5
    web_search_results: int = 2
    temperature: float = 0

class GraphBuilder:
    """Builds and manages the conversation flow graph."""

    def __init__(self, config: GraphConfig = GraphConfig()):
        """Initialize with with configuration"""
        self.config = config
        self.llm = self._init_llm()
        self.retriever = self._init_retriever()
        self.web_search = self._init_web_search()

    def _init_llm(self, config: GraphConfig = GraphConfig()) -> AzureChatOpenAI:
        """Configure Azure OpenAI instance."""
        try:
            return AzureChatOpenAI(
                temperature=config.temperature,
                openai_api_version=config.azure_api_version,
                azure_deployment=config.azure_deployment,
                streaming=True,
                timeout=30,
                max_retries=3
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Azure OpenAI: {str(e)}")
    
    def _init_retriever(self, config: GraphConfig = GraphConfig()) -> CustomRetriever:
        try:    
            return CustomRetriever(
                indexes = [os.environ["AZURE_AI_SEARCH_INDEX_NAME"]],
                topK = config.retriever_top_k,
                reranker_threshold = config.reranker_threshold
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Azure AI Search Retriever: {str(e)}")
        
    def _init_web_search(self, config: GraphConfig = GraphConfig()): 
        try:
            return GoogleSearch(k=config.web_search_results)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Google Search: {str(e)}")

    def build(self) -> StateGraph:
        """Construct the conversation processing graph."""
        graph = StateGraph(ConversationState)

        # Add processing nodes
        graph.add_node("rewrite", self._rewrite_query)
        graph.add_node("route", self._route_query)
        graph.add_node("retrieve", self._retrieve_context)
        graph.add_node("search", self._web_search)
        graph.add_node("generate", self._generate_response)

        # Define graph flow
        graph.add_edge(START, "rewrite")
        graph.add_edge("rewrite", "route")
        graph.add_conditional_edges(
            "route",
            self._route_decision,
            {"retrieve": "retrieve", "generate": "generate"},
        )
        graph.add_conditional_edges(
            "retrieve",
            self._needs_web_search,
            {"search": "search", "generate": "generate"},
        )
        graph.add_edge("search", "generate")
        graph.add_edge("generate", END)

        return graph.compile()


    def _rewrite_query(self, state: ConversationState) -> dict: 
        question = state.question 

        system_prompt = QUERY_REWRITING_PROMPT

        prompt = f" Original Question: \n\n{question}. Please rewrite the question to be used for searching the database." #TODO: take into account the historical context of the conversation

        rewritte_query = self.llm.invoke([SystemMessage(content = system_prompt), HumanMessage(content = prompt)])

        return {
            "rewritten_query": rewritte_query.content,
        }


    def _route_query(self, state: ConversationState) -> dict:
        """Determine if external knowledge is needed."""

    
        system_prompt = MARKETING_ORC_PROMPT

        response = self.llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(
                    content=f"How should I categorize this question: \n\n{state.rewritten_query}\n\nAnswer yes/no."
                )
            ]
        )
        return {
            "requires_web_search": response.content.lower().startswith("y"),
        }

    def _route_decision(self, state: ConversationState) -> str:
        """Route query based on knowledge requirement."""
        return "retrieve" if state.requires_web_search else "generate"

    def _retrieve_context(self, state: ConversationState) -> dict:
        """Get relevant documents from vector store."""
        docs = self.retriever.invoke(state.rewritten_query)
        return {
            "context_docs": docs,
            "requires_web_search": len(docs) < 3,
        }

    def _needs_web_search(self, state: ConversationState) -> str:
        """Check if web search is needed based on retrieval results."""
        return "search" if state.requires_web_search else "generate"

    def _web_search(self, state: ConversationState) -> dict:
        """Perform web search and combine with existing context."""
        web_docs = self.web_search.search(state.rewritten_query)
        return {
            "context_docs": state.context_docs + web_docs,
            "requires_web_search": state.requires_web_search,
        }

    def _generate_response(self, state: ConversationState) -> dict:
        """Generate final response using context and query."""
        # Combine all document content
        context = ""
        if state.context_docs:
            context_parts = []
            for doc in state.context_docs:
                content = f"Content: \n\n{doc.page_content}"
                if doc.metadata.get("source"):
                    content += f"\n\nSource: {doc.metadata['source']}"
                context_parts.append(content)
            context = "\n\n==============================================\n\n".join(context_parts)

        system_prompt = MARKETING_ANSWER_PROMPT
        prompt = f"Context: {context}\nQuestion: {state.rewritten_query}\nProvide a detailed answer."
        # Generate response and update message history
        response = self.llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=prompt)])
        current_messages = state.messages if state.messages is not None else []

        return {
            "messages": [
                *current_messages,
                HumanMessage(content=state.rewritten_query),
                AIMessage(content=response.content),
            ],
            "context_docs": state.context_docs,
            "requires_web_search": state.requires_web_search,
        }


def create_conversation_graph(api_version: str = "2024-05-01-preview") -> StateGraph:
    """Create and return a configured conversation graph.

    Args:
        api_version: Azure API version to use

    Returns:
        Compiled StateGraph for conversation processing
    """
    builder = GraphBuilder(api_version)
    return builder.build()
