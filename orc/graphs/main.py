import os
from dataclasses import dataclass, field
from typing import List
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    RemoveMessage,
)
from langchain.schema import Document
from langgraph.graph import StateGraph, END, START
from langchain_openai import AzureChatOpenAI
from orc.graphs.tools import CustomRetriever, GoogleSearch
from langgraph.checkpoint.memory import MemorySaver
from shared.tools import num_tokens_from_string, messages_to_string
from shared.prompts import (
    MARKETING_ORC_PROMPT,
    MARKETING_ANSWER_PROMPT,
    QUERY_REWRITING_PROMPT,
)
from langchain_core.runnables import RunnableParallel


# initialize memory saver
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
    messages: List[AIMessage | HumanMessage] = field(
        default_factory=list
    )  # track all messages in the conversation
    context_docs: List[Document] = field(default_factory=list)
    requires_web_search: bool = field(default=False)
    rewritten_query: str = field(
        default_factory=str
    )  # rewritten query for better search
    chat_summary: str = field(default_factory=str)
    token_count: int = field(default_factory=int)


def _summarize_chat(
    token_count: int,
    messages: List[AIMessage | HumanMessage],
    chat_summary: str,
    max_tokens: int,
    llm: AzureChatOpenAI,
) -> dict:
    """Summarize chat history if it exceeds token limit.


    Args:
        state: Current conversation state

    Returns:
        dict: Contains updated chat_summary and token_count
    """

    if token_count > max_tokens:
        try:
            if chat_summary:
                messages = [
                    SystemMessage(
                        content="You are a helpful assistant that summarizes conversations."
                    ),
                    HumanMessage(
                        content=f"Previous summary:\n{chat_summary}\n\nNew messages to incorporate:\n{messages}\n\nPlease extend the summary. Return only the summary text."
                    ),
                ]

            else:
                messages = [
                    SystemMessage(
                        content="You are a helpful assistant that summarizes conversations."
                    ),
                    HumanMessage(
                        content=f"Summarize this conversation history. Return only the summary text:\n{messages}"
                    ),
                ]

            new_summary = llm.invoke(messages)

            return new_summary.content

        except Exception as e:
            # Log the error but continue with empty summary
            print(f"Error summarizing chat: {str(e)}")
            return chat_summary or ""
    else:
        return chat_summary or ""


@dataclass
class GraphConfig:
    "Config for the graph builder"

    azure_api_version: str = "2024-05-01-preview"
    azure_deployment: str = "gpt-4o-orchestrator"
    retriever_top_k: int = 5
    reranker_threshold: float = 2.5
    web_search_results: int = 2
    temperature: float = 0.3
    max_tokens: int = 5000


class GraphBuilder:
    """Builds and manages the conversation flow graph."""

    def __init__(self, organization_id: str = None, config: GraphConfig = GraphConfig()):
        """Initialize with with configuration"""
        self.organization_id = organization_id
        self.config = config
        self.llm = self._init_llm()
        self.retriever = self._init_retriever()
        self.web_search = self._init_web_search()

    def _init_llm(self) -> AzureChatOpenAI:
        """Configure Azure OpenAI instance."""
        config = self.config
        try:
            return AzureChatOpenAI(
                temperature=config.temperature,
                openai_api_version=config.azure_api_version,
                azure_deployment=config.azure_deployment,
                streaming=True,
                timeout=30,
                max_retries=3,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Azure OpenAI: {str(e)}")

    def _init_retriever(self) -> CustomRetriever:
        try:
            config = self.config
            index_name = os.getenv("AZURE_AI_SEARCH_INDEX_NAME")
            if not index_name:
                raise ValueError(
                    "AZURE_AI_SEARCH_INDEX_NAME is not set in the environment variables"
                )
            return CustomRetriever(
                indexes = [index_name],
                topK = config.retriever_top_k,
                reranker_threshold = config.reranker_threshold,
                organization_id = self.organization_id
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize Azure AI Search Retriever: {str(e)}"
            )

    def _init_web_search(self):
        try:
            config = self.config
            return GoogleSearch(k=config.web_search_results)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Google Search: {str(e)}")

    def _return_state(self, state: ConversationState) -> dict:
        return {
            "messages": state.messages,
            "context_docs": state.context_docs,
            "chat_summary": state.chat_summary,
            "token_count": state.token_count,
            "requires_web_search": state.requires_web_search,
            "rewritten_query": state.rewritten_query,
        }

    def build(self, memory) -> StateGraph:
        """Construct the conversation processing graph."""
        # set up graph
        graph = StateGraph(ConversationState)
        # Add processing nodes

        graph.add_node("rewrite", self._rewrite_query)
        graph.add_node("route", self._route_query)
        graph.add_node("retrieve", self._retrieve_context)
        graph.add_node("search", self._web_search)
        graph.add_node("return", self._return_state)
        # graph.add_node("generate", self._generate_response)

        # Define graph flow
        graph.add_edge(START, "rewrite")
        graph.add_edge("rewrite", "route")
        graph.add_conditional_edges(
            "route",
            self._route_decision,
            {"retrieve": "retrieve", "return": "return"},
        )

        graph.add_conditional_edges(
            "retrieve",
            self._needs_web_search,
            {"search": "search", "return": "return"},
        )
        graph.add_edge("search", "return")
        graph.add_edge("return", END)
        # graph.add_edge("generate", END)

        return graph.compile(checkpointer=memory)

    def _rewrite_query(self, state: ConversationState) -> dict:
        question = state.question

        system_prompt = QUERY_REWRITING_PROMPT

        prompt = f"""Original Question:
        ```
        {question}. 
        ```
        
        Historical Conversation Context:
        
        ```
        {state.messages}
        ```

        Chat Summary:

        ```
        {state.chat_summary}
        ```

        Please rewrite the question to be used for searching the database.
        """

        rewritte_query = self.llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=prompt)]
        )

        if state.messages is None:
            state.messages = []

        return {
            "rewritten_query": rewritte_query.content,
            # save the original question to state
            "messages": state.messages + [HumanMessage(content=question)],
        }

    def _route_query(self, state: ConversationState) -> dict:
        """Determine if external knowledge is needed."""

        system_prompt = MARKETING_ORC_PROMPT

        response = self.llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(
                    content=f"How should I categorize this question: \n\n{state.rewritten_query}\n\nAnswer yes/no."
                ),
            ]
        )
        return {
            "requires_web_search": response.content.lower().startswith("y"),
        }

    def _route_decision(self, state: ConversationState) -> str:
        """Route query based on knowledge requirement."""
        return "retrieve" if state.requires_web_search else "return"

    def _retrieve_context(self, state: ConversationState) -> dict:
        """Get relevant documents from vector store."""
        docs = self.retriever.invoke(state.rewritten_query)
        return {
            "context_docs": docs,
            "requires_web_search": len(docs) < 3,
        }

    def _needs_web_search(self, state: ConversationState) -> str:
        """Check if web search is needed based on retrieval results."""
        return "search" if state.requires_web_search else "return"

    def _web_search(self, state: ConversationState) -> dict:
        """Perform web search and combine with existing context."""
        web_docs = self.web_search.search(state.rewritten_query)
        return {
            "context_docs": state.context_docs + web_docs,
            "requires_web_search": state.requires_web_search,
        }

    def _generate_response(self, state: ConversationState) -> dict:
        """Generate final response using context and query."""
        context = ""
        if state.context_docs:
            context = "\n\n==============================================\n\n".join(
                [
                    f"\nContent: \n\n{doc.page_content}"
                    + (
                        f"\n\nSource: {doc.metadata['source']}"
                        if doc.metadata.get("source")
                        else ""
                    )
                    for doc in state.context_docs
                ]
            )

        system_prompt = MARKETING_ANSWER_PROMPT
        prompt = f"""
        
        Question: 
        
        <----------- USER QUESTION ------------>
        REWRITTEN QUESTION: {state.rewritten_query}

        ORIGINAL QUESTION: {state.question}
        <----------- END OF USER QUESTION ------------>
        
        
        Context: (MUST PROVIDE CITATIONS FOR ALL SOURCES USED IN THE ANSWER)
        
        <----------- PROVIDED CONTEXT ------------>
        {context}
        <----------- END OF PROVIDED CONTEXT ------------>

        Chat History:

        <----------- PROVIDED CHAT HISTORY ------------>
        {state.messages}
        <----------- END OF PROVIDED CHAT HISTORY ------------>

        Chat Summary:

        <----------- PROVIDED CHAT SUMMARY ------------>
        {state.chat_summary}
        <----------- END OF PROVIDED CHAT SUMMARY ------------>

        Provide a detailed answer.
        """

        # Generate response and update message history
        response = self.llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=prompt)]
        )

        #####################################################################################
        # Summary and chat history work
        #####################################################################################
        current_messages = state.messages if state.messages is not None else []

        try:
            # Try to count tokens, fallback to conservative estimate if it fails
            pre_token_count = (
                num_tokens_from_string(messages_to_string(current_messages))
                if current_messages
                else 0
            )
            print(f"Pre token count: {pre_token_count}")

            # Summarize chat history if it exceeds token limit
            if pre_token_count > self.config.max_tokens:
                chat_summary = _summarize_chat(
                    pre_token_count,
                    current_messages,
                    state.chat_summary,
                    self.config.max_tokens,
                    self.llm,
                )
            else:
                chat_summary = state.chat_summary

            # Prepare new messages
            new_messages = [
                HumanMessage(content=state.rewritten_query),
                AIMessage(content=response.content),
            ]
            total_messages = (
                (current_messages + new_messages)
                if pre_token_count <= self.config.max_tokens
                else new_messages
            )
            post_token_count = num_tokens_from_string(
                messages_to_string(total_messages)
            )
            print(f"Post token count: {post_token_count}")

        except Exception as e:
            print(f"Warning: Token counting failed: {str(e)}")
            # Fallback to simple length-based estimate
            pre_token_count = sum(len(str(m.content)) // 4 for m in current_messages)

            total_messages = current_messages + [
                HumanMessage(content=state.rewritten_query),
                AIMessage(content=response.content),
            ]

            post_token_count = sum(len(str(m.content)) // 4 for m in total_messages)

        return {
            "messages": total_messages,
            "context_docs": state.context_docs,
            "token_count": post_token_count,
            "chat_summary": chat_summary,
        }


def create_conversation_graph(memory, organization_id = None) -> StateGraph:
    """Create and return a configured conversation graph.
    Returns:
        Compiled StateGraph for conversation processing
    """
    print(f"Creating conversation graph for organization: {organization_id}")
    builder = GraphBuilder(organization_id=organization_id)
    return builder.build(memory)
