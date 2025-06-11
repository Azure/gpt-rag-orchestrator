import os
import json
import requests
import os
from typing import List, Annotated, Literal
from collections import OrderedDict
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain.schema import Document
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.language_models import LanguageModelLike
from typing import TypedDict, List, Dict
from langgraph.graph.graph import CompiledGraph
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.retrievers import TavilySearchAPIRetriever
from langgraph.graph.message import add_messages, AnyMessage
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START
from shared.prompts import DOCSEARCH_PROMPT, RETRIEVAL_REWRITER_PROMPT
from shared.util import get_secret

# import tools
from orc.graphs.tools import CustomRetriever, GoogleSearch
from concurrent.futures import ThreadPoolExecutor

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Document are relevant to the question, 'yes' or 'no' "
    )


class RetrievalState(TypedDict):
    """
    Represent the state of our graph

    Attributes:
    question: question
    generation: llm generation
    web_search: whether to add search
    documents: list of retrieved documents (websearch and database retrieval)
    summary: summary of the entire conversation"""

    question: str
    generation: str
    web_search: str
    summary: str
    retrieval_messages: Annotated[list[AnyMessage], add_messages]
    combined_messages: Annotated[list[AnyMessage], add_messages]
    documents: List[str]


def create_retrieval_graph(
    model: LanguageModelLike,
    model_two: LanguageModelLike,
    verbose: bool = True,
) -> CompiledGraph:
    
    web_search_tool = GoogleSearch(k=3)
    rag_chain = DOCSEARCH_PROMPT | model | StrOutputParser()
    # index_name = os.environ["AZURE_AI_SEARCH_INDEX_NAME"]
    index_name = "ragindex"
    indexes = [index_name]

    retriever = CustomRetriever(
        indexes=indexes,
        topK=5,
        reranker_threshold=1.2,
    )
    retrieval_question_rewriter = (
        RETRIEVAL_REWRITER_PROMPT | model_two | StrOutputParser()
    )

    def retrieval_transform_query(state: RetrievalState):
        """
        Transform the query to optimize retrieval process.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates question key with a re-phrased query
        """

        if verbose:
            print("---TRANSFORM QUERY FOR RETRIEVAL OPTIMIZATION---")

        question = state["question"]
        messages = state["retrieval_messages"]

        # Re-write question
        better_user_query = retrieval_question_rewriter.invoke(
            {"question": question, "previous_conversation": messages}
        )

        # add to messages schema
        messages = [HumanMessage(content=better_user_query)]

        return {
            "question": better_user_query,
            "retrieval_messages": messages,
            "combined_messages": messages,
        }

    def retrieve(state: RetrievalState):
        """
        Retrieve documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        if verbose:
            print("---RETRIEVE---")
        question = state["question"]

        # Retrieval
        documents = retriever.get_search_results(query = question)

        # set web search to no if there are more than 3 documents
        if len(documents) > 3:
            web_search = "No"
        else:
            web_search = "Yes"

        return {"documents": documents, "web_search": web_search}

    def generate(state: RetrievalState) -> Literal["conversation_summary", "__end__"]:
        """
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """

        if verbose:
            print("---GENERATE---")

        question = state["question"]
        documents = state["documents"]
        summary = state.get("summary", "")
        messages = state["retrieval_messages"]

        if summary:
            system_message = f"Summary of the conversation earlier: \n\n{summary}"
            previous_conversation = [SystemMessage(content=system_message)] + messages[
                :-1
            ]
        else:
            previous_conversation = messages

        # RAG generation
        generation = rag_chain.invoke(
            {
                "context": documents,
                "previous_conversation": previous_conversation,
                "question": question,
            }
        )

        # Add this generation to messages schema
        messages = [AIMessage(content=generation)]

        return {
            "generation": generation,
            "retrieval_messages": messages,
            "combined_messages": messages,
        }

    def decide_to_generate(state):
        """
        Determines whether to generate an answer, or re-generate a question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Binary decision for next node to call
        """

        if verbose:
            print("---ASSESS DOCUMENTS---")

        web_search = state["web_search"]

        if web_search == "Yes":
            # insufficient relevant documents -> conducting websearch
            if verbose:
                print("---DECISION: PROCEED TO CONDUCT WEB SEARCH---")
            return "web_search_node"
        else:
            # We have relevant documents, so generate answer
            if verbose:
                print("---DECISION: GENERATE---")
            return "generate"

    def web_search(state: RetrievalState):
        """
        Conduct Web search to add more relevant context

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with appended web results
        """
        if verbose:
            print("---WEB SEARCH---")

        question = state["question"]
        documents = state["documents"]

        # Web search - note the updated method call
        docs = web_search_tool.search(question)

        # append to the existing document
        documents.extend(docs)

        return {"documents": documents}

    # parent graph
    retrieval_stategraph = StateGraph(RetrievalState)

    # Define the nodes
    retrieval_stategraph.add_node(
        "transform_query", retrieval_transform_query
    )  # rewrite user query
    retrieval_stategraph.add_node("retrieve", retrieve)  # retrieve
    retrieval_stategraph.add_node("generate", generate)  # generatae
    retrieval_stategraph.add_node("web_search_node", web_search)  # web search

    # Build graph
    retrieval_stategraph.add_edge(START, "transform_query")
    retrieval_stategraph.add_edge("transform_query", "retrieve")
    retrieval_stategraph.add_conditional_edges(
        "retrieve",
        decide_to_generate,
        {
            "web_search_node": "web_search_node",
            "generate": "generate",
        },
    )
    retrieval_stategraph.add_edge("web_search_node", "generate")
    retrieval_stategraph.add_edge("generate", END)
    retrieval_stategraph.add_edge("conversation_summary", END)

    # Compile
    rag = retrieval_stategraph.compile()

    return rag
