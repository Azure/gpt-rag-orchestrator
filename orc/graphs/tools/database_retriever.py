import os
import json
import requests
from typing import List, Annotated
from collections import OrderedDict
from langchain_core.retrievers import BaseRetriever
from langchain.schema import Document
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages, AnyMessage



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