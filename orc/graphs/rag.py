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
from shared.prompts import DOCSEARCH_PROMPT, RETRIEVAL_REWRITER_PROMPT, GRADE_PROMPT
from shared.util import get_secret

# obtain google search api key
GOOGLE_SEARCH_API_KEY = os.environ.get("SERPER_API_KEY") or get_secret("GoogleSearchKey")


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


class CustomRetriever(BaseRetriever):
    """
    Custom Retriever class that extends the BaseRetriever class to perform multi-index hybrid search
    and return ordered dictionary with the combined results
    """

    topK: int
    reranker_threshold: int
    indexes: List

    def get_search_results(
        self,
        query: str,
        indexes: list,
        k: int = 5,
        reranker_threshold: float = 1.2,  # range between 0 and 4 (high to low)
    ) -> List[dict]:
        """Performs multi-index hybrid search and returns ordered dictionary with the combined results"""

        headers = {
            "Content-Type": "application/json",
            "api-key": os.environ["AZURE_AI_SEARCH_API_KEY"],
        }
        params = {"api-version": os.environ["AZURE_SEARCH_API_VERSION"]}

        agg_search_results = dict()

        for index in indexes:
            search_payload = {
                "search": query,
                "select": "id, title, content, filepath",
                "queryType": "semantic",
                "vectorQueries": [
                    {
                        "text": query,
                        "fields": "vector",
                        "kind": "text",
                        "k": k,
                        "threshold": {
                            "kind": "vectorSimilarity",
                            "value": 0.5,  # 0.333 - 1.00 (Cosine), 0 to 1 for Euclidean and DotProduct.
                        },
                    }
                ],
                "semanticConfiguration": "my-semantic-config",  # change the name depends on your config name
                "captions": "extractive",
                "answers": "extractive",
                "count": "true",
                "top": k,
            }

            AZURE_SEARCH_SERVICE = os.environ.get("AZURE_SEARCH_SERVICE")
            AZURE_SEARCH_ENDPOINT_SF = (
                f"https://{AZURE_SEARCH_SERVICE}.search.windows.net"
            )

            resp = requests.post(
                AZURE_SEARCH_ENDPOINT_SF + "/indexes/" + index + "/docs/search",
                data=json.dumps(search_payload),
                headers=headers,
                params=params,
            )

            search_results = resp.json()
            agg_search_results[index] = search_results

        content = dict()
        ordered_content = OrderedDict()

        for index, search_results in agg_search_results.items():
            for result in search_results["value"]:
                if (
                    result["@search.rerankerScore"] > reranker_threshold
                ):  # Range between 0 and 4
                    content[result["id"]] = {
                        "title": result["title"],
                        "name": (result["name"] if "name" in result else ""),
                        "chunk": (result["content"] if "content" in result else ""),
                        "location": (
                            result["filepath"] if "filepath" in result else ""
                        ),
                        "caption": result["@search.captions"][0]["text"],
                        "score": result["@search.rerankerScore"],
                        "index": index,
                    }

        topk = k

        count = 0  # To keep track of the number of results added
        for id in sorted(content, key=lambda x: content[x]["score"], reverse=True):
            ordered_content[id] = content[id]
            count += 1
            if count >= topk:  # Stop after adding topK results
                break

        return ordered_content

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        Modify the _get_relevant_documents methods in BaseRetriever so that it aligns with our previous settings
        Retrieved Documents are sorted based on reranker score (semantic score)
        """
        ordered_results = self.get_search_results(
            query,
            self.indexes,
            k=self.topK,
            reranker_threshold=self.reranker_threshold,
        )

        top_docs = []

        for key, value in ordered_results.items():
            location = value["location"] if value["location"] is not None else ""
            top_docs.append(
                Document(
                    page_content=value["chunk"],
                    metadata={"source": location, "score": value["score"]},
                )
            )

        return top_docs


def create_retrieval_graph(
    model: LanguageModelLike,
    model_two: LanguageModelLike,
    verbose: bool = False,
) -> CompiledGraph:
    # web_search_tool = TavilySearchAPIRetriever(k=3, 
    #                                            search_depth= 'advanced',
    #                                            include_raw_content = True,
    #                                            )
    web_search_tool = GoogleSerperAPIWrapper(k=3, serper_api_key= GOOGLE_SEARCH_API_KEY)
    rag_chain = DOCSEARCH_PROMPT | model | StrOutputParser()
    index_name = os.environ["AZURE_AI_SEARCH_INDEX_NAME"]
    indexes = [index_name]

    retriever = CustomRetriever(
        indexes=indexes,
        topK=5,
        reranker_threshold=1.2,
    )
    retrieval_question_rewriter = (
        RETRIEVAL_REWRITER_PROMPT | model_two | StrOutputParser()
    )
    structured_llm_grader = model_two.with_structured_output(GradeDocuments)
    retrieval_grader = GRADE_PROMPT | structured_llm_grader

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
        documents = retriever.invoke(question)

        return {"documents": documents}

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

    def grade_documents(state: RetrievalState):
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with only filtered relevant documents and web search decision
        """

        if verbose:
            print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")

        question = state["question"]
        documents = state["documents"]
        previous_conversation = state["retrieval_messages"] + [state.get("summary", "")]

        # Score each doc
        filtered_docs = []
        web_search = "No"
        if not documents:
            if verbose:
                print("---NO RELEVANT DOCUMENTS RETRIEVED FROM THE DATABASE---")
            relevant_doc_count = 0
            web_search = "Yes"
        else:
            if verbose:
                print("---EVALUATING RETRIEVED DOCUMENTS---")
            for d in documents:
                score = retrieval_grader.invoke(
                    {
                        "question": question,
                        "previous_conversation": previous_conversation,
                        "document": d.page_content,
                    }
                )
                grade = score.binary_score
                if grade == "yes":
                    if verbose:
                        print("---GRADE: DOCUMENT RELEVANT---")
                    filtered_docs.append(d)
                else:
                    if verbose:
                        print("---GRADE: DOCUMENT NOT RELEVANT---")

        # count the number of retrieved documents
        relevant_doc_count = len(filtered_docs)

        if verbose:
            print(f"---NUMBER OF DATABASE RETRIEVED DOCUMENTS---: {relevant_doc_count}")

        if relevant_doc_count >= 3:
            web_search = "No"
        else:
            web_search = "Yes"

        return {"documents": filtered_docs, "web_search": web_search}

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

        # Web search
        docs = ggsearch_reformat(web_search_tool.results(question))

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
    retrieval_stategraph.add_node("grade_documents", grade_documents)  # grade documents
    retrieval_stategraph.add_node("generate", generate)  # generatae
    retrieval_stategraph.add_node("web_search_node", web_search)  # web search

    # Build graph
    retrieval_stategraph.add_edge(START, "transform_query")
    retrieval_stategraph.add_edge("transform_query", "retrieve")
    retrieval_stategraph.add_edge("retrieve", "grade_documents")
    retrieval_stategraph.add_conditional_edges(
        "grade_documents",
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
