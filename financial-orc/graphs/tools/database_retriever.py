from langchain_core.retrievers import BaseRetriever
from typing import List
import os
import requests
import json
from collections import OrderedDict
from langchain_core.documents import Document
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
import logging
LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
logging.basicConfig(level=LOGLEVEL)



def get_secret(secretName):
    keyVaultName = os.getenv("AZURE_KEY_VAULT_NAME")
    KVUri = f"https://{keyVaultName}.vault.azure.net"
    credential = DefaultAzureCredential()
    client = SecretClient(vault_url=KVUri, credential=credential)
    logging.info(f"[webbackend] retrieving {secretName} secret from {keyVaultName}.")
    retrieved_secret = client.get_secret(secretName)
    return retrieved_secret.value



class CustomRetriever(BaseRetriever):
    """
    Custom retriever class that extends BaseRetriever to work with Azure AI Search.

    Attributes:
        topK (int): Number of top results to retrieve.
        reranker_threshold (float): Threshold for reranker score.
        indexes (List): List of index names to search.
    """

    topK = 1
    reranker_threshold = 0.5
    vector_similarity_threshold = 0.1
    semantic_config = "financial-index-semantic-configuration"
    index_name = "financial-index"
    indexes: List
    verbose: bool

    def get_search_results(
        self,
        query: str,
        indexes: list,
        k: int = topK,
        semantic_config: str = semantic_config,
        reranker_threshold: float = reranker_threshold,
        vector_similarity_threshold: float = vector_similarity_threshold,
        ) -> List[dict]:
        """
        Performs multi-index hybrid search and returns ordered dictionary with the combined results.

        Args:
            query (str): The search query.
            indexes (list): List of index names to search.
            k (int): Number of top results to retrieve. Default is 5.
            reranker_threshold (float): Threshold for reranker score. Default is 1.2.

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
            if self.verbose:
                print(f"[CustomRetriever] Searching index: {index}")
            
            search_payload = {
                "search": query,
                "select": "chunk_id, file_name, chunk, url, date_last_modified",
                "queryType": "semantic",
                "semanticConfiguration": semantic_config,
                "captions": "extractive",
                "answers": "extractive",
                "count": "true",
                "top": k,
                "vectorQueries": [
                    {
                    "text": query,
                    "fields": "text_vector",
                    "kind": "text",
                    "k": k,
                    "threshold": {
                        "kind": "vectorSimilarity",
                        "value": vector_similarity_threshold,
                    },
                }
            ],
        }

        AZURE_SEARCH_SERVICE = os.environ.get("AZURE_SEARCH_SERVICE")
        search_endpoint = f"https://{AZURE_SEARCH_SERVICE}.search.windows.net/indexes/{index}/docs/search"
        
        if self.verbose:
            print(f"[CustomRetriever] Using endpoint: {search_endpoint}")

        try:
            resp = requests.post(
                search_endpoint,
                data=json.dumps(search_payload),
                headers=headers,
                params=params,
            )
            resp.raise_for_status() # Raises an HTTPError for bad responses
        except requests.exceptions.HTTPError as e:
            if self.verbose:
                print(f"[CustomRetriever] HTTP Request failedd: {str(e)}")
            return []

        except Exception as e:
            if self.verbose:
                print(f"[CustomRetriever] Error in get_search_results: {e}")
            return []
        if not resp.ok:
            if self.verbose:
                print(f"[CustomRetriever] Error response: {resp.status_code} - {resp.text}")
            return []
        search_results = resp.json()
        if self.verbose:
            print(f"[CustomRetriever] Results found: {len(search_results.get('value', []))}")
            
        agg_search_results[index] = search_results


        content = dict()
        ordered_content = OrderedDict()

        for index, search_results in agg_search_results.items():
            for result in search_results["value"]:
                if (
                    result["@search.rerankerScore"] > reranker_threshold
                ):  # Range between 0 and 4
                    content[result["chunk_id"]] = {
                        "filename": result["file_name"],
                        "chunk": (result["chunk"] if "chunk" in result else ""),
                        "location": (result["url"] if "url" in result else ""),
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


# format database retrieved content 
def format_retrieved_content(docs):
    """
    Format retrieved documents into a readable report with metadata.
    
    Args:
        docs (List[Document]): List of retrieved documents
        
    Returns:
        List[Document]: List of formatted documents with citations
    """
    if not docs:
        return [Document(page_content="No documents were retrieved.")]
    
    try:
        formatted_docs = []
        for i, doc in enumerate(docs, 1):            
            formatted_doc = f"""
            Document {i}:
            ********************************************************************************

            Report Content: 

            {doc.page_content.strip()}
            """

            if doc.metadata.get('source'):
                # sas_token = get_secret('blobSasToken')
                sas_token = os.getenv("BLOB_SAS_TOKEN")
                citation = f"{doc.metadata['source']}?{sas_token}"
            else:
                citation = ""
            
            formatted_docs.append(
                Document(
                    page_content=formatted_doc,
                    metadata={"citation": citation}
                )
            )
        
        return formatted_docs 
        
    except Exception as e:
        return [Document(page_content=f"Error formatting documents: {str(e)}")]

