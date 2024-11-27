import os
import json
import requests
from collections import OrderedDict
from typing import List, Annotated, Sequence, TypedDict
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import (
    AIMessage,
    ToolMessage,
    BaseMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.retrievers import TavilySearchAPIRetriever
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_openai import AzureChatOpenAI


LOGLEVEL = os.environ.get("LOGLEVEL", "DEBUG").upper()
logging.basicConfig(level=LOGLEVEL)


# Define agent graph
class AgentState(TypedDict):
    """The state of the agent."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    report: str


# Update CustomRetriever with error handling
class CustomRetriever(BaseRetriever):
    """
    Custom retriever class that extends BaseRetriever to work with Azure AI Search.

    Attributes:
        topK (int): Number of top results to retrieve.
        reranker_threshold (float): Threshold for reranker score.
        indexes (List): List of index names to search.
        sas_token (str): SAS token for authentication.
    """

    topK: int
    reranker_threshold: float
    indexes: List
    sas_token: str = None

    def get_search_results(
        self,
        query: str,
        indexes: list,
        k: int = 3,
        reranker_threshold: float = 1.2,  # range between 0 and 4 (high to low)
        sas_token: str = "",
    ) -> List[dict]:
        """
        Performs multi-index hybrid search and returns ordered dictionary with the combined results.

        Args:
            query (str): The search query.
            indexes (list): List of index names to search.
            k (int): Number of top results to retrieve. Default is 5.
            reranker_threshold (float): Threshold for reranker score. Default is 1.2.
            sas_token (str): SAS token for authentication. Default is empty string.

        Returns:
            OrderedDict: Ordered dictionary of search results.
        """

        headers = {
            "Content-Type": "application/json",
            "api-key": os.environ["AZURE_AI_SEARCH_API_KEY"],
        }
        params = {"api-version": os.environ["AZURE_SEARCH_API_VERSION"]}

        agg_search_results = dict()

        for index in indexes:
            search_payload = {
                "search": query,
                "select": "chunk_id, file_name, chunk, url, date_last_modified",
                "queryType": "semantic",
                "vectorQueries": [
                    {
                        "text": query,
                        "fields": "text_vector",
                        "kind": "text",
                        "k": k,
                        "threshold": {
                            "kind": "vectorSimilarity",
                            "value": 0.5,  # 0.333 - 1.00 (Cosine), 0 to 1 for Euclidean and DotProduct.
                        },
                    }
                ],
                "semanticConfiguration": "financial-index-semantic-configuration",  # change the name depends on your config name
                "captions": "extractive",
                "answers": "extractive",
                "count": "true",
                "top": k,
            }

            try:
                resp = requests.post(
                    os.environ["AZURE_SEARCH_ENDPOINT"]
                    + "/indexes/"
                    + index
                    + "/docs/search",
                    data=json.dumps(search_payload),
                    headers=headers,
                    params=params,
                )
                # resp.raise_for_status()
                search_results = resp.json()
                agg_search_results[index] = search_results
            except Exception as e:
                logging.info(f"[financial-orchestrator-agent] Error in get_search_results: {str(e)}")
                return []

        content = dict()
        ordered_content = OrderedDict()

        for index, search_results in agg_search_results.items():
            for result in search_results["value"]:
                if (
                    result["@search.rerankerScore"] > reranker_threshold
                ):  # Range between 0 and 4
                    content[result["chunk_id"]] = {
                        "filename": result["file_name"],
                        # "title": result['title'],
                        "chunk": (result["chunk"] if "chunk" in result else ""),
                        "location": (result["url"] if "url" in result else ""),
                        "caption": result["@search.captions"][0]["text"],
                        "score": result["@search.rerankerScore"],
                        "index": index,
                    }

        for index, search_results in agg_search_results.items():
            for result in search_results["value"]:
                if (
                    result["@search.rerankerScore"] > reranker_threshold
                ):  # Range between 0 and 4
                    content[result["chunk_id"]] = {
                        "filename": result["file_name"],
                        "chunk": result["chunk"],
                        "location": (
                            result["url"] + f"?{sas_token}" if result["url"] else ""
                        ),
                        "date_last_modified": result["date_last_modified"],
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
        Retrieves relevant documents based on the given query.

        Args:
            query (str): The search query.

        Returns:
            List[Document]: List of relevant documents.
        """

        ordered_results = self.get_search_results(
            query,
            self.indexes,
            k=self.topK,
            reranker_threshold=self.reranker_threshold,
            sas_token=self.sas_token,
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


def create_main_agent(checkpointer, verbose=True):
    # Define model
    llm = AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_deployment="Agent",
        openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        temperature=0.3,
    )
    ###################################################
    # Define retriever tool
    ###################################################
    # Define the index name for Azure AI Search
    index_name = "financial-index"
    indexes = [index_name]  # we can add more indexes here if needed

    # Initialize custom retriever on Azure AI Search
    k = 1
    retriever = CustomRetriever(
        indexes=indexes,
        topK=k,
        reranker_threshold=1.2,
        sas_token=os.environ["BLOB_SAS_TOKEN"],
    )

    def report_retriever(state: AgentState):
        """Retrieve the initial report for consumer segmentation."""
        try:
            # Get documents from retriever
            documents = retriever.invoke("consumer segmentation")

            logging.info(f"[financial-orchestrator-agent] RETRIEVED DOCUMENTS: {len(documents)}")

            if not documents or len(documents) == 0:
                logging.info("[financial-orchestrator-agent] No documents retrieved, using fallback content")
                documents = [
                    Document(
                        page_content="No information found about consumer segmentation."
                    )
                ]

            return {"report": documents}
        except Exception as e:
            logging.info(f"[financial-orchestrator-agent] Error in report_retriever: {str(e)}")
            # Return a fallback document to prevent crashes
            return {
                "report": [
                    Document(
                        page_content="Error retrieving consumer segmentation information."
                    )
                ]
            }

    @tool
    def web_search(query: str) -> str:
        """Conduct web search for user query that is not included in the report"""

        search = TavilySearchAPIRetriever(
            k=2,
            search_depth="advanced",
            include_generated_answer=True,
            include_raw_content=True,
        )

        # Convert the search results to a string representation
        results = search.invoke(query)

        # Format the results into a readable string
        formatted_results = []
        for result in results:
            metadata = result.metadata
            page_content = result.page_content
            formatted_results.append(
                f"URL: {metadata['source']}\n" f"Content: {page_content}\n"
            )

        logging.info(f"[financial-orchestrator-agent] WEBSEARCH RESULTS: {len(results)}")

        return "\n\n".join(formatted_results)

    tools = [web_search]

    ###################################################
    # define nodes and edges
    ###################################################
    # Tool dictionary definition remains the same
    tools_by_name = {tool.name: tool for tool in tools}

    def tool_node(state: AgentState):
        outputs = []
        try:
            for tool_call in state["messages"][-1].tool_calls:
                tool_result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
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

    def call_model(state: AgentState, config: RunnableConfig):

        report = state["report"][0].page_content

        system_prompt = """
        You are a helpful assistant. Use the available tool to help answer user queries if the provided report are deemed irrelevant

        Here is the report:
        {report}
        """.format(
            report=report
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )
        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        response = agent_executor.invoke(
            {
                "input": state["messages"][-1].content,
                "chat_history": state["messages"][:-1],
            }
        )

        return {"messages": [AIMessage(content=response["output"])]}

    # define the conditional edge that determines whether to continue or not
    def should_continue(state: AgentState):
        messages = state["messages"]
        last_message = messages[-1]
        # if there is not function call, then we finish
        if not last_message.tool_calls:
            return "end"
        # otherwise if there is, we continue
        else:
            return "continue"

    ###################################################
    # define graph
    ###################################################
    # define a new graph
    workflow = StateGraph(AgentState)

    # define the preload document node
    workflow.add_node("report_preload", report_retriever)
    # define the two nodes we will cycle between
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)

    # set the entry point as agent
    workflow.set_entry_point("report_preload")

    # we now add a conditional edge
    workflow.add_edge("report_preload", "agent")

    workflow.add_conditional_edges(
        "agent", should_continue, {"continue": "tools", "end": END}
    )

    # add a normal edge from tools to agent
    workflow.add_edge("tools", "agent")

    # compile the graph
    graph = workflow.compile(checkpointer=checkpointer)

    return graph
