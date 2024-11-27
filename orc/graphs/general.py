import os
from typing import Annotated
from langgraph.graph.graph import CompiledGraph
from langchain_core.language_models import LanguageModelLike
from typing import TypedDict, List, Dict
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.messages import RemoveMessage
from langgraph.graph.message import add_messages, AnyMessage
from typing_extensions import TypedDict
from langchain.schema import Document
from langchain_community.utilities import GoogleSerperAPIWrapper
from langgraph.graph import END, StateGraph, START
from shared.prompts import GENERAL_PROMPT, ANSWER_GRADER_PROMPT, IMPROVED_GENERAL_PROMPT
from shared.util import get_secret


# obtain google search api key
GOOGLE_SEARCH_API_KEY = os.environ.get("SERPER_API_KEY") or get_secret("GoogleSearchKey")




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
    google_documents: documents retrieved from Google search
    """
    question: str
    general_model_messages: Annotated[list[AnyMessage], add_messages]
    combined_messages: Annotated[list[AnyMessage], add_messages]
    relevance: str = "no"
    google_documents: List[Document]

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
        Decide whether the answer is already good enough or needs additional context from google search.

        Args:
            state (GeneralModState): The current state of the general model.

        Returns:
            str: The next action to take, either "google_web_search" or "__end__".
        """
        if state["relevance"] == "no":
            if verbose:
                print("---DECISION: CONDUCT WEB SEARCH TO ADD CONTEXT TO ANSWER---")
            return "google_web_search"
        else:
            if verbose:
                print("---DECISION: GENERATED ANSWER IS GOOD ENOUGH---")
            return "__end__"
        
    
    google_search = GoogleSerperAPIWrapper(k=3, serper_api_key= GOOGLE_SEARCH_API_KEY)


    def ggsearch_reformat(result: Dict) -> List[Document]:
        """
        Reformats Google search results into a list of Document objects.

        Args:
            result (Dict): The raw search results from Google.

        Returns:
            List[Document]: A list of Document objects containing the search results.
        """
        documents = []
        try:
            # Process Knowledge Graph results if present
            if 'knowledgeGraph' in result:
                kg = result['knowledgeGraph']
                doc = Document(
                    page_content=kg.get('description', ''),
                    metadata={'source': kg.get('descriptionLink', ''), 'title': kg.get('title', '')}
                )
                documents.append(doc)
            
            # Process organic search results
            if 'organic' in result:
                for item in result['organic']:
                    doc = Document(
                        page_content=item.get('snippet', ''),
                        metadata={'source': item.get('link', ''), 'title': item.get('title', '')}
                    )
                    documents.append(doc)
            
            if not documents:
                raise ValueError("No search results found")
            
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            documents.append(Document(
                page_content="No search results found or an error occurred.",
                metadata={'source': 'Error', 'title': 'Search Error'}
            ))
        
        return documents
    
    def google_search_node(state: GeneralModState) -> GeneralModState:
        """ 
        Retrieve documents from Google search.
        
        Args:
            state (InterviewState): The current state of the interview.
        
        Returns:
            dict: A dictionary containing the retrieved documents as context.
        """
        general_model_messages = state["general_model_messages"]
        question = state["question"]
        if verbose:
            print("---GOOGLE SEARCH---")

        # Perform Google search using the generated search query
        docs = ggsearch_reformat(google_search.results(question))
        
        # Create a message to remove the last conversation message
        messages_to_delete = RemoveMessage(id=general_model_messages[-1].id)

        # Return updated state
        return {
            "google_documents": docs,
            "general_model_messages": [messages_to_delete],
            "combined_messages": [messages_to_delete],
        }


    def final_general_generation(state: GeneralModState):
        """
        Enhance the previous answer using additional context from google search results.

        This function takes the current state, extracts relevant information, and uses
        the final_general_chain_model to generate an improved response based on the
        original question, previous answer, and additional context from google.

        Args:
            state (GeneralModState): The current state containing question, previous answer,
                                    and google search results.

        Returns:
            dict: A dictionary containing the improved response in both 'general_model_messages'
                and 'combined_messages' keys.
        """
        # Extract relevant information from the state
        google_documents = state["google_documents"]
        previous_answer = state["general_model_messages"][-1]
        question = state["question"]

        # Generate improved response using the final_general_chain_model
        improved_response = final_general_chain_model.invoke(
            {
                "question": question,
                "previous_answer": previous_answer,
                "google_documents": google_documents,
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
    general_stategraph.add_node("google_web_search", google_search_node)
    general_stategraph.add_node("Answer_Regeneration", final_general_generation)

    ## edges in graph
    general_stategraph.add_edge(START, "general_llm_node")
    general_stategraph.add_edge("general_llm_node", "completeness_checker")
    general_stategraph.add_conditional_edges(
        "completeness_checker",
        completeness_decision,
        {"google_web_search": "google_web_search", "__end__": END},
    )
    general_stategraph.add_edge("google_web_search", "Answer_Regeneration")
    general_stategraph.add_edge("Answer_Regeneration", END)

    ## compile the general llm subgraph
    general_llm_subgraph = general_stategraph.compile()

    return general_llm_subgraph
