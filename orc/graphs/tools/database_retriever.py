import os
import json
import requests
from typing import List, Annotated
from collections import OrderedDict
from langchain_core.retrievers import BaseRetriever
from langchain.schema import Document
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages, AnyMessage
import re


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
                    "id": result.get("id", ""),
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
            # print(value)
            # Skip documents with duplicate scores, which are likely duplicates
            if score in seen_scores:
                continue
            seen_scores.add(score)

            location = value["location"] if value["location"] is not None else ""
            top_docs.append(
                Document(
                    page_content=value["chunk"],
                    metadata={"source": location, "score": score},
                    id=value["id"]
                )
            )

        return top_docs

    def _search_adjacent_pages(self, doc_id: str) -> List[dict]:
        """
        Searches for documents on pages immediately preceding and succeeding the page number
        extracted from the given doc_id in the specified Azure AI Search index.

        Args:
            doc_id: The document ID string, expected to end with '_pages_<number>'.

        Returns:
            A list of dictionaries, each representing a found adjacent document result from Azure Search,
            or an empty list if the ID format is invalid or no adjacent documents are found.
        """
        print(f"[Adjacent Search]: Received request for ID: {doc_id}")

        # 1. Parse base ID and page number using regex
        match = re.search(r"^(.*_pages_)(\d+)$", doc_id)
        # match = re.search(r"^(.*_chunks_)(\d+)$", doc_id) #USE THIS FOR THE TEST INDEX
        if not match:
            # print(f"[Adjacent Search]: Invalid ID format: {doc_id}. Does not match '.*_pages_\\d+'. Skipping.")
            print(f"[Adjacent Search]: Invalid ID format: {doc_id}. Does not match '.*_chunks_\\d+'. Skipping.")
            return []

        base_id_part = match.group(1)
        try:
            current_page = int(match.group(2))
        except ValueError:
             # This case should ideally not happen if regex matches \d+, but good practice
             print(f"[Adjacent Search]: Could not parse page number from {doc_id}. Skipping.")
             return []

        print(f"[Adjacent Search]: Parsed base ID part: '{base_id_part}', Current Page: {current_page}")

        # 2. Calculate adjacent page numbers (page-1 and page+1)
        adjacent_pages_to_search = []
        if current_page > 0:
            adjacent_pages_to_search.append(current_page - 1)
        adjacent_pages_to_search.append(current_page + 1)

        print(f"[Adjacent Search]: Target adjacent pages: {adjacent_pages_to_search}")

        # 3. Construct target IDs for adjacent pages
        target_ids = [f"{base_id_part}{page}" for page in adjacent_pages_to_search]
        print(f"[Adjacent Search]: Target full IDs: {target_ids}")

        # 4. Prepare Azure Search request details
        api_key = os.getenv("AZURE_AI_SEARCH_API_KEY")
        search_service = os.getenv("AZURE_SEARCH_SERVICE")
        api_version = os.getenv("AZURE_SEARCH_API_VERSION")

        if not all([api_key, search_service, api_version]):
             print("[Adjacent Search]: Error: Missing required Azure Search environment variables (API Key, Service Name, API Version).")
             return [] 

        headers = {
            "Content-Type": "application/json",
            "api-key": api_key,
        }
        params = {"api-version": api_version}
        azure_search_endpoint = f"https://{search_service}.search.windows.net"

        if not self.indexes:
             print("[Adjacent Search]: Error: No indexes configured for this retriever instance.")
             return []
        index_name = self.indexes[0]
        search_url = f"{azure_search_endpoint}/indexes/{index_name}/docs/search"
        print(f"[Adjacent Search]: Using search URL: {search_url}")


        adjacent_docs_results = []

        # 5. Iterate through target IDs and perform search
        for target_id in target_ids:
            print(f"[Adjacent Search]: Searching for ID: {target_id} in index {index_name}")

            # Construct the filter: must match target_id AND organization_id (or null org_id)
            id_filter = f"id eq '{target_id}'"
            # Ensure organization_id filter is correctly formatted, handle potential single quotes in org_id if necessary
            org_filter = f"(organization_id eq '{self.organization_id}' or organization_id eq null)"
            combined_filter = f"{id_filter} and {org_filter}" # Use 'and', not parentheses unless needed for precedence

            search_payload = {
                "search": "*", 
                "count": True,
                "vectorQueries": [
                    {
                      "kind": "text",
                      "text": "*", 
                      "fields": "vector" 
                    }
                ],
                "select": "id, title, content, filepath",
                "queryType": "semantic", 
                "semanticConfiguration": "my-semantic-config",
                "captions": "extractive",
                "answers": "extractive|count-3",
                "queryLanguage": "en-us",
                "filter": combined_filter,
                "top": 1 
            }

            print(f"[Adjacent Search]: Sending payload for {target_id}: {json.dumps(search_payload, indent=2)}")

            try:
                resp = requests.post(
                    search_url,
                    data=json.dumps(search_payload),
                    headers=headers,
                    params=params,
                    timeout=20 
                )
                resp.raise_for_status()

                result_data = resp.json()
                found_docs = result_data.get("value", [])

                if found_docs:
                    # Assuming the first result is the one matching the ID
                    print(f"[Adjacent Search]: Successfully found document for ID {target_id}")
                    adjacent_docs_results.append(found_docs[0])
                else:
                    print(f"[Adjacent Search]: No document found matching ID {target_id} and org filter.")

            except requests.exceptions.HTTPError as http_err:
                print(f"[Adjacent Search]: HTTP error occurred for {target_id}: {http_err} - Response: {resp.text}")
            except requests.exceptions.RequestException as req_err:
                print(f"[Adjacent Search]: Request error occurred for {target_id}: {req_err}")
            except Exception as e:
                print(f"[Adjacent Search]: An unexpected error occurred while searching for {target_id}: {str(e)}")

        print(f"[Adjacent Search]: Completed search. Found {len(adjacent_docs_results)} adjacent documents.")
        return adjacent_docs_results


if __name__ == "__main__":
    retriever = CustomRetriever(
        topK=5,
        reranker_threshold=2.5,
        indexes=["ragindex"],
        organization_id="0d4ddea52add"
    )
    results = retriever.search_adjacent_pages("0d4ddea52add_aHR0cHM6Ly9zdHJhZzB2bTJiMmh0dnV1Y2xtLmJsb2IuY29yZS53aW5kb3dzLm5ldC9kb2N1bWVudHMvU2VnbWVudGF0aW9uL0NvbnN1bWVyJTIwUHVsc2UlMjBTZWdtZW50YXRpb24lMjBTdW1tYXJ5LmRvY3g1_docintContent_pages_2")
    print(results)