import os
from dataclasses import dataclass, field
from typing import List
from langchain_core.messages import AIMessage, HumanMessage
from langchain.schema import Document
from langgraph.graph import StateGraph, END, START
from langchain_openai import AzureChatOpenAI
from orc.graphs.tools import CustomRetriever, GoogleSearch


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


class GraphBuilder:
    """Builds and manages the conversation flow graph."""

    def __init__(self, azure_api_version: str = "2024-05-01-preview"):
        """Initialize with Azure configuration and tools."""
        self.llm = self._init_llm(azure_api_version)
        self.retriever = CustomRetriever(
            indexes=[os.environ["AZURE_AI_SEARCH_INDEX_NAME"]],
            topK=5,
            reranker_threshold=1.2,
        )
        self.web_search = GoogleSearch(k=3)

    def _init_llm(self, api_version: str) -> AzureChatOpenAI:
        """Configure Azure OpenAI instance."""
        return AzureChatOpenAI(
            temperature=0,
            openai_api_version=api_version,
            azure_deployment="gpt-4o-orchestrator",
            streaming=True,
        )

    def build(self) -> StateGraph:
        """Construct the conversation processing graph."""
        graph = StateGraph(ConversationState)

        # Add processing nodes
        graph.add_node("route", self._route_query)
        graph.add_node("retrieve", self._retrieve_context)
        graph.add_node("search", self._web_search)
        graph.add_node("generate", self._generate_response)

        # Define graph flow
        graph.add_edge(START, "route")
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

    def _route_query(self, state: ConversationState) -> dict:
        """Determine if external knowledge is needed."""
        response = self.llm.invoke(
            [
                HumanMessage(
                    content=f"Does this question require external knowledge: {state.question}? Answer yes/no."
                )
            ]
        )
        return {
            "question": state.question,
            "messages": state.messages,
            "requires_web_search": response.content.lower().startswith("y"),
        }

    def _route_decision(self, state: ConversationState) -> str:
        """Route query based on knowledge requirement."""
        return "retrieve" if state.requires_web_search else "generate"

    def _retrieve_context(self, state: ConversationState) -> dict:
        """Get relevant documents from vector store."""
        docs = self.retriever.get_relevant_documents(state.question)
        return {
            "question": state.question,
            "messages": state.messages,
            "context_docs": docs,
            "requires_web_search": len(docs) < 3,
        }

    def _needs_web_search(self, state: ConversationState) -> str:
        """Check if web search is needed based on retrieval results."""
        return "search" if state.requires_web_search else "generate"

    def _web_search(self, state: ConversationState) -> dict:
        """Perform web search and combine with existing context."""
        web_docs = self.web_search.search(state.question)
        return {
            "question": state.question,
            "messages": state.messages,
            "context_docs": state.context_docs + web_docs,
            "requires_web_search": state.requires_web_search,
        }

    def _generate_response(self, state: ConversationState) -> dict:
        """Generate final response using context and query."""
        # Combine all document content
        context = (
            "\n".join(doc.page_content for doc in state.context_docs)
            if state.context_docs
            else ""
        )
        prompt = f"Context: {context}\nQuestion: {state.question}\nProvide a detailed answer."

        # Generate response and update message history
        response = self.llm.invoke([HumanMessage(content=prompt)])
        current_messages = state.messages if state.messages is not None else []

        return {
            "question": state.question,
            "messages": [
                *current_messages,
                HumanMessage(content=state.question),
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
