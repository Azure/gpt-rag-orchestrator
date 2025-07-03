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
from orc.graphs.tools import CustomRetriever, TavilySearch, retrieve_and_convert_to_document_format
from langgraph.checkpoint.memory import MemorySaver
from shared.prompts import (
    MARKETING_ORC_PROMPT,
    QUERY_REWRITING_PROMPT,
)
from langchain_core.runnables import RunnableParallel
from shared.cosmos_db import get_conversation_data
from shared.util import get_organization


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
    query_category: str = field(default_factory=str)
    agentic_search_mode: bool = field(default=True)

def clean_chat_history(chat_history: List[dict], agentic_search_mode: bool = False) -> str:
    """
    Clean the chat history and format it as a string for LLM consumption.

    Args:
        chat_history (list): List of chat message dictionaries

    Returns:
        str: Formatted chat history string in the format:
                Human: {message}
                AI Message: {message}
    """
    formatted_history = []
    if len(chat_history) > 4:
        chat_history = chat_history[-4:]
    else:
        chat_history = chat_history

    if agentic_search_mode:
        chat_history = [{"role": message.get("role", "").lower(), "content": message.get("content", "")} for message in chat_history]
        return chat_history

    for message in chat_history:
        if not message.get("content"):
            continue

        role = message.get("role", "").lower()
        content = message.get("content", "")

        if role and content:
            display_role = "Human" if role == "user" else "AI Message"
            formatted_history.append(f"{display_role}: {content}")

    return "\n\n".join(formatted_history)

@dataclass
class GraphConfig:
    "Config for the graph builder"

    azure_api_version: str = "2025-01-01-preview"
    azure_deployment: str = "gpt-4.1"
    retriever_top_k: int = 5
    reranker_threshold: float = 2.1
    web_search_results: int = 2
    temperature: float = 0.4
    max_tokens: int = 5000
    agentic_search_mode: bool = True


class GraphBuilder:
    """Builds and manages the conversation flow graph."""

    def __init__(
        self,
        organization_id: str = None,
        config: GraphConfig = GraphConfig(),
        conversation_id: str = None,
    ):
        """Initialize with with configuration"""
        self.organization_id = organization_id
        self.config = config
        self.llm = self._init_llm()
        self.retriever = self._init_retriever()
        self.web_search = self._init_web_search()
        self.conversation_id = conversation_id
        self.organization_data = get_organization(organization_id)

    def _init_llm(self) -> AzureChatOpenAI:
        """Configure Azure OpenAI instance."""
        config = self.config
        try:
            return AzureChatOpenAI(
                temperature=config.temperature,
                openai_api_version=config.azure_api_version,
                azure_deployment=config.azure_deployment,
                streaming=False,
                timeout=30,
                max_retries=3,
                azure_endpoint=os.getenv("O1_ENDPOINT"),
                api_key=os.getenv("O1_KEY")
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Azure OpenAI: {str(e)}")
    
    def _init_segment_alias(self) -> str:
        """Retrieve segment alias."""
        return self.organization_data.get('segmentSynonyms','')
    
    def _init_brand_information(self) -> str:
        """Retrieve brand information."""
        return self.organization_data.get('brandInformation','')
    
    def _init_industry_information(self) -> str:
        """Retrieve industry information."""
        return self.organization_data.get('industryInformation','')

    def _init_retriever(self) -> CustomRetriever:
        try:
            config = self.config
            # index_name = os.getenv("AZURE_AI_SEARCH_INDEX_NAME")
            index_name = "ragindex-test"
            if not index_name:
                raise ValueError(
                    "AZURE_AI_SEARCH_INDEX_NAME is not set in the environment variables"
                )
            return CustomRetriever(
                indexes=[index_name],
                topK=config.retriever_top_k,
                reranker_threshold=config.reranker_threshold,
                organization_id=self.organization_id,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize Azure AI Search Retriever: {str(e)}"
            )
    
    def _run_agentic_retriever(self, conversation_history: List[dict]):
        return retrieve_and_convert_to_document_format(conversation_history, self.organization_id)

    def _init_web_search(self):
        try:
            config = self.config
            # return GoogleSearch(k=config.web_search_results)
            return TavilySearch(max_results=config.web_search_results)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Web Search: {str(e)}")

    def _return_state(self, state: ConversationState) -> dict:
        return {
            "messages": state.messages,
            "context_docs": state.context_docs,
            "chat_summary": state.chat_summary,
            "token_count": state.token_count,
            "requires_web_search": state.requires_web_search,
            "rewritten_query": state.rewritten_query,
            "query_category": state.query_category,
        }

    def build(self, memory) -> StateGraph:
        """Construct the conversation processing graph."""
        # set up graph
        graph = StateGraph(ConversationState)
        # Add processing nodes

        graph.add_node("rewrite", self._rewrite_query)
        graph.add_node("tool_choice", self._categorize_query)
        graph.add_node("route", self._route_query)
        graph.add_node("retrieve", self._retrieve_context)
        graph.add_node("search", self._web_search)
        graph.add_node("return", self._return_state)

        # Define graph flow
        graph.add_edge(START, "rewrite")
        graph.add_edge("rewrite", "tool_choice")
        graph.add_edge("tool_choice", "route")
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

        return graph.compile(checkpointer=memory)

    def _rewrite_query(self, state: ConversationState) -> dict:
        question = state.question

        system_prompt = QUERY_REWRITING_PROMPT

        conversation_data = get_conversation_data(self.conversation_id)
        history = conversation_data.get("history", [])

        # additional system prompt
        additional_system_prompt = f"""
        <-------------------------------->
        
        Historical Conversation Context:
        <-------------------------------->
        ```
        {clean_chat_history(history)}
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
        # combine the system prompt with the additional system prompt
        system_prompt = f"{system_prompt}\n\n{additional_system_prompt}"

        prompt = f"""Original Question: 
        <-------------------------------->
        ```
        {question}. 
        ```
        <-------------------------------->

        Please rewrite the question to be used for searching the database. Make sure to follow the alias mapping instructions at all cost.
        ALSO, THE HISTORICAL CONVERSATION CONTEXT IS VERY VERY IMPORTANT TO THE USER'S FOLLOW UP QUESTIONS, $10,000 WILL BE DEDUCTED FROM YOUR ACCOUNT IF YOU DO NOT USE THE HISTORICAL CONVERSATION CONTEXT.
        Please also consider the line of business/industry of my company when rewriting the query. Don't be too verbose. 
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

    def _categorize_query(self, state: ConversationState) -> dict:
        """Categorize the query."""

        conversation_data = get_conversation_data(self.conversation_id)
        history = conversation_data.get("history", [])

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
        - **General**: If the input lacks context, doesn’t relate to marketing deliverables, or is unclear or unrelated to the above.

        If the question or context is not clearly related to any of the above categories, always return "General".

        ----------------------------------------
        User's Question:
        {state.question}
        ----------------------------------------
        Conversation History:
        {clean_chat_history(history)}
        ----------------------------------------

        Reply with **only** the exact category name — no additional text, explanations, or formatting.
        """
        response = self.llm.invoke(
            [
                SystemMessage(content=category_prompt),
                HumanMessage(content=state.question),
            ],
            temperature=0
        )

        return {
            "query_category": response.content
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
        if self.config.agentic_search_mode:
            print("[Retrieve Context] Agentic search mode is enabled")

            # get the conversation history
            conversation_history = get_conversation_data(self.conversation_id).get("history", [])

            # clean the conversation history
            conversation_history = clean_chat_history(conversation_history, agentic_search_mode=True)

            # append the rewritten query to the conversation history
            conversation_history.append({"role": "user", "content": state.rewritten_query})

            docs = self._run_agentic_retriever(conversation_history)

        else:
            docs = self.retriever.invoke(state.rewritten_query)

            print(f"total number of docs: {len(docs)}")

            if docs:
                # print id of the first document in the list
                print(f'Document ID of the top ranked doc: {docs[0].id}')
                # get adjacent chunks
                adjacent_chunks = self.retriever._search_adjacent_pages(docs[0].id)
                # reformat adjacent chunks
                if adjacent_chunks:
                    adjacent_chunks = [Document(page_content=chunk['content'], metadata={'source': chunk['filepath'], 'score': "adjacent chunk"}) for chunk in adjacent_chunks]
                    print(f"total number of docs before adding adjacent chunks: {len(docs)}")
                    docs.extend(adjacent_chunks)
                    print(f"total number of docs after adding adjacent chunks: {len(docs)}")
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


def create_conversation_graph(
    memory, organization_id=None, conversation_id=None
) -> StateGraph:
    """Create and return a configured conversation graph.
    Returns:
        Compiled StateGraph for conversation processing
    """
    print(f"Creating conversation graph for organization: {organization_id}")
    builder = GraphBuilder(
        organization_id=organization_id, conversation_id=conversation_id
    )
    return builder.build(memory)
