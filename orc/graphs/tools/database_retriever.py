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
    reranker_threshold: float
    indexes: List
    organization_id: str

    def get_search_results(
        self,
        query: str,
        indexes: str = ["ragindex"],
        k: int = 5,
        reranker_threshold: float = 2.5,  # range between 0 and 4 (high to low)
    ) -> List[dict]:
        """Performs multi-index hybrid search and returns ordered dictionary with the combined results"""

        print(f"[Database Retriever]: Starting search with query: {query}")
        print(f"[Database Retriever]: Using indexes: {indexes}")
        print(f"[Database Retriever]: Organization ID filter: {self.organization_id}")

        headers = {
            "Content-Type": "application/json",
            "api-key": os.getenv("AZURE_AI_SEARCH_API_KEY"),
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
                "filter": f"organization_id eq '{self.organization_id}' or organization_id eq null",
            }

            print(
                f"[Database Retriever]: Search payload for index {index}:",
                json.dumps(search_payload, indent=2),
            )

            AZURE_SEARCH_SERVICE = os.getenv("AZURE_SEARCH_SERVICE")
            AZURE_SEARCH_ENDPOINT_SF = (
                f"https://{AZURE_SEARCH_SERVICE}.search.windows.net"
            )

            search_url = AZURE_SEARCH_ENDPOINT_SF + "/indexes/" + index + "/docs/search"
            print(f"[Database Retriever]: Making request to: {search_url}")

            try:
                resp = requests.post(
                    search_url,
                    data=json.dumps(search_payload),
                    headers=headers,
                    params=params,
                )

                print(f"Debug: Response status code: {resp.status_code}")
                if resp.status_code != 200:
                    print(f"Debug: Error response: {resp.text}")
                    continue

                search_results = resp.json()
                print(
                    f"Debug: Got {len(search_results.get('value', []))} results from index {index}"
                )
                agg_search_results[index] = search_results

            except Exception as e:
                print(f"Debug: Exception during search request: {str(e)}")
                continue

        content = dict()
        ordered_content = OrderedDict()

        print("Debug: Processing search results...")
        for index, search_results in agg_search_results.items():
            filtered_results = [
                result
                for result in search_results.get("value", [])
                if result["@search.rerankerScore"] > reranker_threshold
            ]
            print(
                f"Debug: After filtering by reranker score > {reranker_threshold}: {len(filtered_results)} results"
            )

            for result in filtered_results:
                content[result["id"]] = {
                    "title": result.get("title", ""),
                    "name": result.get("name", ""),
                    "chunk": result.get("content", ""),
                    "location": result.get("filepath", ""),
                    "caption": (
                        result["@search.captions"][0]["text"]
                        if "@search.captions" in result
                        else ""
                    ),
                    "score": result["@search.rerankerScore"],
                    "index": index,
                }

        print(f"Debug: Total unique documents found: {len(content)}")
        topk = k

        count = 0  # To keep track of the number of results added
        for id in sorted(content, key=lambda x: content[x]["score"], reverse=True):
            ordered_content[id] = content[id]
            count += 1
            if count >= topk:  # Stop after adding topK results
                break

        return list(ordered_content.values())

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        Modify the _get_relevant_documents methods in BaseRetriever so that it aligns with our previous settings
        Retrieved Documents are sorted based on reranker score (semantic score).
        Filters out duplicate results with identical scores.
        """
        ordered_results = self.get_search_results(
            query,
            self.indexes,
            k=self.topK,
            reranker_threshold=self.reranker_threshold,
        )

        top_docs = []
        seen_scores = set()

        for key, value in (
            ordered_results.items()
            if isinstance(ordered_results, dict)
            else enumerate(ordered_results)
        ):
            score = value["score"]
            # Skip documents with duplicate scores, which are likely duplicates
            if score in seen_scores:
                continue
            seen_scores.add(score)

            location = value["location"] if value["location"] is not None else ""
            top_docs.append(
                Document(
                    page_content=value["chunk"],
                    metadata={"source": location, "score": score},
                )
            )

        return top_docs
