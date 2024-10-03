import os
from typing import Annotated
from langgraph.graph.graph import CompiledGraph
from langchain_core.language_models import LanguageModelLike
from typing import TypedDict, List
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.messages import RemoveMessage
from langgraph.graph.message import add_messages, AnyMessage
from typing_extensions import TypedDict
from langchain.schema import Document
from langchain_community.utilities import BingSearchAPIWrapper
from langgraph.graph import END, StateGraph, START
from shared.prompts import GENERAL_PROMPT, ANSWER_GRADER_PROMPT, IMPROVED_GENERAL_PROMPT


BING_SEARCH_API_KEY = os.environ.get("BING_SEARCH_API_KEY")
BING_SEARCH_URL = os.environ.get("BING_SEARCH_URL")


class GradeAnswer(BaseModel):
    """Binary score for completeness of the answer."""

    binary_score: str = Field(
        description="Answer is relevant and satisfies the question, 'yes' or 'no'"
    )


class GeneralModState(TypedDict):
    """
    Represent the state our the General LLM subgraph

    Attributes:
    question: question provided by user:
    generation: llm generation
    messages: keep track of the conversation
    """

    question: str
    general_model_messages: Annotated[list[AnyMessage], add_messages]
    combined_messages: Annotated[list[AnyMessage], add_messages]
    relevance: str = "no"
    bing_documents: List[Document]


def create_general_graph(
    model: LanguageModelLike,
    verbose: bool = False,
) -> CompiledGraph:
    general_chain_model = GENERAL_PROMPT | model
    structured_llm_answer_grader = model.with_structured_output(GradeAnswer)
    answer_grader = ANSWER_GRADER_PROMPT | structured_llm_answer_grader
    final_general_chain_model = IMPROVED_GENERAL_PROMPT | model

    def general_llm_node(state: GeneralModState):
        """
        General LLM node in the graph
        """

        question = state["question"]

        if verbose:
            print("---ROUTE DECISION: GENERAL LLM")

        response = general_chain_model.invoke({"question": question})

        return {
            "general_model_messages": [question, response],
            "combined_messages": [question, response],
        }

    def completeness_checker(state: GeneralModState) -> dict:
        """
        Check whether the answer has fully addressed the user's question.

        Args:
            state (GeneralModState): The current state of the general model.

        Returns:
            dict: A dictionary containing the binary score for the answer's completeness.
        """
        question = state["question"]
        answer = state["general_model_messages"][-1].content

        completeness_score = answer_grader.invoke(
            {"question": question, "answer": answer}
        )
        return {"relevance": completeness_score.binary_score}

    def completeness_decision(state: GeneralModState) -> str:
        """
        Decide whether the answer is already good enough or needs additional context from Bing search.

        Args:
            state (GeneralModState): The current state of the general model.

        Returns:
            str: The next action to take, either "bing_web_search" or "__end__".
        """
        if state["relevance"] == "no":
            if verbose:
                print("---DECISION: CONDUCT WEB SEARCH TO ADD CONTEXT TO ANSWER---")
            return "bing_web_search"
        else:
            if verbose:
                print("---DECISION: GENERATED ANSWER IS GOOD ENOUGH---")
            return "__end__"

    def bing_search(query: str, count: int = 2) -> List[Document]:
        """
        Perform a Bing web search and return the results as a list of Document objects.

        Args:
            query (str): The search query string.
            count (int, optional): The number of search results to return. Defaults to 2.

        Returns:
            List[Document]: A list of Document objects containing the search results.
        """

        search = BingSearchAPIWrapper()
        search_results = search._bing_search_results(query, count=count)

        documents = []

        for page in search_results:
            doc = Document(
                page_content=f"Title: {page['name']}\nSnippet: {page['snippet']}",
                metadata={"source": page["url"]},
            )
            documents.append(doc)

        return documents

    def bing_search_node(state: GeneralModState) -> GeneralModState:
        """
        Generate additional context from Bing search and update the state.

        Args:
            state (GeneralModState): The current state of the general model.

        Returns:
            GeneralModState: Updated state with new Bing search results and modified messages.
        """
        bing_documents = state.get("bing_documents", [])
        general_model_messages = state["general_model_messages"]
        question = state["question"]

        if verbose:
            print("---BING SEARCH---")

        # Perform web search
        docs = bing_search(question)

        # Append new documents to the existing ones
        # bing_documents.extend(docs)

        # Create a message to remove the last conversation message
        messages_to_delete = RemoveMessage(id=general_model_messages[-1].id)

        # Return updated state
        return {
            "bing_documents": docs,
            "general_model_messages": [messages_to_delete],
            "combined_messages": [messages_to_delete],
        }

    def final_general_generation(state: GeneralModState):
        """
        Enhance the previous answer using additional context from Bing search results.

        This function takes the current state, extracts relevant information, and uses
        the final_general_chain_model to generate an improved response based on the
        original question, previous answer, and additional context from Bing.

        Args:
            state (GeneralModState): The current state containing question, previous answer,
                                    and Bing search results.

        Returns:
            dict: A dictionary containing the improved response in both 'general_model_messages'
                and 'combined_messages' keys.
        """
        # Extract relevant information from the state
        bing_documents = state["bing_documents"]
        previous_answer = state["general_model_messages"][-1]
        question = state["question"]

        # Generate improved response using the final_general_chain_model
        improved_response = final_general_chain_model.invoke(
            {
                "question": question,
                "previous_answer": previous_answer,
                "bing_documents": bing_documents,
            }
        )

        # Return the improved response
        return {
            "general_model_messages": [improved_response],
            "combined_messages": [improved_response],
        }

    # general llm graph
    # build graph
    general_stategraph = StateGraph(GeneralModState)

    ## nodes in graph
    general_stategraph.add_node("general_llm_node", general_llm_node)
    general_stategraph.add_node("completeness_checker", completeness_checker)
    general_stategraph.add_node("bing_web_search", bing_search_node)
    general_stategraph.add_node("Answer_Regeneration", final_general_generation)

    ## edges in graph
    general_stategraph.add_edge(START, "general_llm_node")
    general_stategraph.add_edge("general_llm_node", "completeness_checker")
    general_stategraph.add_conditional_edges(
        "completeness_checker",
        completeness_decision,
        {"bing_web_search": "bing_web_search", "__end__": END},
    )

    general_stategraph.add_edge("bing_web_search", "Answer_Regeneration")
    general_stategraph.add_edge("Answer_Regeneration", END)

    ## compile the general llm subgraph
    general_llm_subgraph = general_stategraph.compile()

    return general_llm_subgraph
