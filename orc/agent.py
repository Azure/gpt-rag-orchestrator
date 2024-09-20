import os
import json
import requests
import os
from typing import List, Annotated, Literal
from collections import OrderedDict
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.language_models import LanguageModelLike
from typing import Optional, TypedDict, List
from langgraph.checkpoint import BaseCheckpointSaver
from langgraph.graph.graph import CompiledGraph
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph.message import add_messages, AnyMessage
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    RemoveMessage,
    SystemMessage,
)
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START
from shared.prompts import DOCSEARCH_PROMPT


def get_search_results(
    query: str,
    indexes: list,
    k: int = 3,
    reranker_threshold: int = 2,  # range between 0 and 4 (high to low)
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
            "select": "id, title, chunk, name, location",
            "queryType": "semantic",
            "vectorQueries": [
                {
                    "text": query,
                    "fields": "chunkVector",
                    "kind": "text",
                    "k": k,
                    "threshold": {
                        "kind": "vectorSimilarity",
                        "value": 0.6,  # 0.333 - 1.00 (Cosine), 0 to 1 for Euclidean and DotProduct.
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
        AZURE_SEARCH_ENDPOINT_SF = f"https://{AZURE_SEARCH_SERVICE}.search.windows.net"

        resp = requests.post(
            AZURE_SEARCH_ENDPOINT_SF
            + "/indexes/"
            + index
            + "/docs/search",
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
                    "name": result["name"],
                    "chunk": result["chunk"],
                    "location": (
                        result["location"]
                        if result["location"]
                        else ""
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


class CustomRetriever(BaseRetriever):
    topK: int
    reranker_threshold: int
    indexes: List

    """Modify the _get_relevant_documents methods in BaseRetriever so that it aligns with our previous settings
       Retrieved Documents are sorted based on reranker score (semantic score)
    """

    def _get_relevant_documents(self, query: str) -> List[Document]:

        ordered_results = get_search_results(
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


class GraphState(TypedDict):
    """
    Represent the state of our graph

    Attributes:
    question: question
    generation: llm generation
    web_search: whether to add search
    documents: list of documents"""

    question: str
    generation: str
    web_search: str
    summary: str
    messages: Annotated[list[AnyMessage], add_messages]
    documents: List[str]


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Document are relevant to the question, 'yes' or 'no' "
    )


def retrieval_grader_chain(model):
    structured_llm_grader = model.with_structured_output(GradeDocuments)

    # Prompt
    system = """You are a grader responsible for evaluating the relevance of a retrieved document to a user question.
        Thoroughly examine the entire document, focusing on both keywords and overall semantic meaning related to the question.
        Consider the context, implicit meanings, and nuances that might connect the document to the question.
        Provide a binary score of 'yes' for relevant or partially relevant, and 'no' for irrelevant, based on a comprehensive analysis.
        If the question ask about information from  prior conversations or last questions, return 'yes'. """

    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Retrieved document: \n\n {document} \n\n Previous Conversation {previous_conversation} \n\n User question: {question}",
            ),
        ]
    )

    # Chain the grader and llm together
    retrieval_grader = grade_prompt | structured_llm_grader

    return retrieval_grader


# User Query Re-writer 1: Optimize retrieval result
def retrieval_question_rewriter_chain(model):
    # prompt
    system = """
        You are a query rewriter that improves input questions for information retrieval in a vector database.\n
        Don't mention the specific year unless it is mentioned in the original query.\n
        Identify the core intent and optimize the query by:
        1. Correcting spelling and grammar errors.
        2. Use synonyms to broaden search results if necessary.
        3. Ensure the rewritten query is concise and directly addresses the intended information.
    """

    retrieval_rewrite_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Here is the initial question: \n\n {question} \n Formulate an improved question ",
            ),
        ]
    )

    retrieval_question_rewriter = retrieval_rewrite_prompt | model | StrOutputParser()

    return retrieval_question_rewriter


# User Query Re-writer 2 (for web search): Make sure that we are getting the most optimal web search results
def retrieval_question_rewriter_web_chain(model):
    system = """ You are a question re-writer that converts an input question to a better version that is optimized
    for web search. Look at the input and try to reason about the underlying semantic intent/meaning. Don't mention the specific year unless it is mentioned in the original query.
    """

    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Here is the intitial question: \n\n {question} \n Formulate an improved question ",
            ),
        ]
    )

    question_rewriter = re_write_prompt | model | StrOutputParser()

    return question_rewriter


def create_agent(
    model: LanguageModelLike,
    mini_model: LanguageModelLike,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    verbose: bool = False,
) -> CompiledGraph:
    retrieval_question_rewriter = retrieval_question_rewriter_chain(mini_model)
    question_rewriter = retrieval_question_rewriter_web_chain(mini_model)
    retrieval_grader = retrieval_grader_chain(mini_model)
    web_search_tool = TavilySearchResults(max_results=2)
    rag_chain = DOCSEARCH_PROMPT | model | StrOutputParser()
    index_name = os.environ["AZURE_AI_SEARCH_INDEX_NAME"]
    indexes = [index_name]

    retriever = CustomRetriever(
        indexes=indexes,
        topK=3,
        reranker_threshold=2,
    )

    def retrieval_transform_query(state):
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

        # Re-write question
        better_user_query = retrieval_question_rewriter.invoke({"question": question})

        # add to messages schema
        messages = [HumanMessage(content=better_user_query)]
        return {"question": better_user_query, "messages": messages}

    def retrieve(state):
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

    def generate(state) -> Literal["conversation_summary", "__end__"]:
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
        messages = state["messages"]

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
                "question": messages[-1],
            }
        )

        # Add this generation to messages schema
        messages = [AIMessage(content=generation)]
        return {"generation": generation, "messages": messages}

    def grade_documents(state):
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with only filtered relevant documents
        """
        if verbose:
            print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]
        previous_conversation = state["messages"] + [state.get("summary", "")]

        # Score each doc
        filtered_docs = []
        web_search = "No"
        if not documents:
            print("---NO RELEVANT DOCUMENTS RETRIEVED DURING THE RETRIEVAL PROCESS---")
            web_search = "Yes"
        else:
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
                    print("---GRADE: DOCUMENT RELEVANT---")
                    filtered_docs.append(d)
                else:
                    print("---GRADE: DOCUMENT NOT RELEVANT---")
                    web_search = "Yes"
        return {"documents": filtered_docs, "web_search": web_search}

    def web_search(state):
        """
        Web search based on the re-phrased question.

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
        docs = web_search_tool.invoke({"query": question})
        for doc in docs:
            # pass metadata as a dict with source as key, fornat here is consistent with retrieved document format
            web_result = Document(
                metadata={"source": doc["url"]}, page_content=doc["content"]
            )
            documents.append(web_result)

        return {"documents": documents}

    def transform_query(state):
        """
        Transform the query to produce a better question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates question key with a re-phrased question
        """
        if verbose:
            print("---TRANSFORM QUERY FOR WEBSEARCH---")
        question = state["question"]

        # Re-write question
        better_question = question_rewriter.invoke({"question": question})
        return {"question": better_question}

    ### Edges

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
            # All documents have been filtered check_relevance
            # We will re-generate a new query
            if verbose:
                print(
                    "---DECISION: SOME/ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, OR NO DOCUMENT RETRIEVED---"
                )
            return "transform_query"
        else:
            # We have relevant documents, so generate answer
            if verbose:
                print("---DECISION: GENERATE---")
            return "generate"

    def conversation_summary(state):
        """Summary the entire conversation to date"""
        summary = state.get("summary", "")
        messages = state["messages"]

        if summary:
            summary_prompt = f"Here is the conversation summary so far: {summary}\n\n Please take into account the above information to the summary and summarize all:"
        else:
            summary_prompt = "Create a summary of the entire conversation so far:"

        new_messages = messages + [HumanMessage(content = summary_prompt)]

        # remove "RemoveMessage" from the list
        new_messages = [m for m in new_messages if not isinstance(m, RemoveMessage)]

        conversation_summary = mini_model.invoke(new_messages)

        # delete all but the 3 most recent messages
        retained_messages = [RemoveMessage(id=m.id) for m in messages[:-6]]
        return {"summary": conversation_summary.content, "messages": retained_messages}

    def summary_decide(state):
        """Decide whether it's necessary to summarize the conversation
        If the conversation has been more than 6 (3 humans 3 AI responses) then we should summarize it
        """
        if len(state["messages"]) > 6:
            return "conversation_summary"
        return "__end__"

    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node(
        "retrieval_transform_query", retrieval_transform_query
    )  # rewrite user query

    workflow.add_node("retrieve", retrieve)  # retrieve
    workflow.add_node("grade_documents", grade_documents)  # grade documents
    workflow.add_node("generate", generate)  # generatae
    workflow.add_node("transform_query", transform_query)  # transform_query_web
    workflow.add_node("web_search_node", web_search)  # web search
    workflow.add_node(
        "conversation_summary", conversation_summary
    )  # summary conversation to date

    # Build graph
    workflow.add_edge(START, "retrieval_transform_query")
    workflow.add_edge("retrieval_transform_query", "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )
    workflow.add_edge("transform_query", "web_search_node")
    workflow.add_edge("web_search_node", "generate")
    workflow.add_conditional_edges(
        "generate",
        summary_decide,
        {"conversation_summary": "conversation_summary", "__end__": "__end__"},
    )
    workflow.add_edge("conversation_summary", END)

    # Compile
    app = workflow.compile(checkpointer=checkpointer)

    return app
