from __future__ import annotations

import re
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
import logging

import json
from typing import Dict, List, Optional

import aiohttp
import requests
from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import Extra, root_validator
from langchain_core.retrievers import BaseRetriever
from langchain_core.utils import get_from_dict_or_env, get_from_env
from openai import AzureOpenAI

class LineListOutputParser(StrOutputParser):
    def parse(self, text: str):
        # Split by newlines
        lines = text.strip().split("\n")
        # Remove special characters
        lines = [(re.sub("\W+", " ", k)).strip() for k in lines]
        # Remove the number at the beginning of the line (if exists)
        lines = [re.sub("^\d+\s*", "", k) for k in lines]
        return lines


def retrieval_transform(docs):
    sources = [x.metadata.get("filepath", "") for x in docs]
    docs = [f"Source {i}: {x.metadata.get('filepath', '')} \n{x.page_content}" for i, x in enumerate(docs, start=1)]
    source_knowledge = "\n---\n".join(docs)
    # logging.info(f"SOURCES {sources}")
    return source_knowledge, sources


TERM_SEARCH_APPROACH='term'
VECTOR_SEARCH_APPROACH='vector'
HYBRID_SEARCH_APPROACH='hybrid'
DEFAULT_URL_SUFFIX="search.windows.net"
AZURE_SEARCH_APPROACH=HYBRID_SEARCH_APPROACH
AZURE_SEARCH_USE_SEMANTIC="true"
AZURE_SEARCH_SEMANTIC_SEARCH_CONFIG ="semantic-config"

"""Default URL Suffix for endpoint connection - commercial cloud"""

class AzureAISearchRetriever(BaseRetriever):
    """`Azure AI Search` service retriever."""

    service_name: str = ""
    """Name of Azure AI Search service"""
    index_name: str = ""
    """Name of Index inside Azure AI Search service"""
    api_key: str = ""
    """API Key. Both Admin and Query keys work, but for reading data it's
    recommended to use a Query key."""
    api_version: str = "2023-11-01"
    """API version"""
    aiosession: Optional[aiohttp.ClientSession] = None
    """ClientSession, in case we want to reuse connection for better performance."""
    content_key: str = "content"
    """Key in a retrieved result to set as the Document page_content."""
    top_k: Optional[int] = None
    """Number of results to retrieve. Set to None to retrieve all results."""
    endpoint: Optional[str] = None
    """Endpoint URL. If not set, it will be built based on service_name."""
    deployment: Optional[str] = None
    """Deployment name for the OpenAI service."""
    azure_api_key: Optional[str] = None
    """API Key for the OpenAI service."""

    class Config:
        extra = Extra.forbid
        arbitrary_types_allowed = True

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that service name, index name and api key exists in environment."""
        values["service_name"] = get_from_dict_or_env(
            values, "service_name", "AZURE_AI_SEARCH_SERVICE_NAME"
        )
        values["index_name"] = get_from_dict_or_env(
            values, "index_name", "AZURE_AI_SEARCH_INDEX_NAME"
        )
        values["api_key"] = get_from_dict_or_env(
            values, "api_key", "AZURE_AI_SEARCH_API_KEY"
        )
        return values

    def _build_search_url(self, query: str) -> str:
        url_suffix = get_from_env("", "AZURE_AI_SEARCH_URL_SUFFIX", DEFAULT_URL_SUFFIX)
        if url_suffix in self.service_name and "https://" in self.service_name:
            base_url = f"{self.service_name}/"
        elif url_suffix in self.service_name and "https://" not in self.service_name:
            base_url = f"https://{self.service_name}/"
        elif url_suffix not in self.service_name and "https://" in self.service_name:
            base_url = f"{self.service_name}.{url_suffix}/"
        elif (
            url_suffix not in self.service_name and "https://" not in self.service_name
        ):
            base_url = f"https://{self.service_name}.{url_suffix}/"
        else:
            base_url = self.service_name
        endpoint_path = f"indexes/{self.index_name}/docs/search?api-version={self.api_version}"
        return base_url + endpoint_path

    @property
    def _headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "api-key": self.api_key,
        }
    
    def generate_embeddings(self, text):
        client = AzureOpenAI(
            api_version = self.api_version,
            azure_endpoint = self.endpoint,
            api_key = self.azure_api_key,
        )
    
        embeddings =  client.embeddings.create(input = [text], model=self.deployment).data[0].embedding

        return embeddings

    def _search(self, query: str) -> List[dict]:
        search_url = self._build_search_url(query)

        embeddings_query = self.generate_embeddings(query)

        body = {
            "select": "title, chunk, chunk_id, filepath",
            "top": self.top_k
        } 

        if AZURE_SEARCH_APPROACH == TERM_SEARCH_APPROACH:
            body["search"] = query
        elif AZURE_SEARCH_APPROACH == VECTOR_SEARCH_APPROACH:
            body["vectorQueries"] = [{
                "kind": "vector",
                "vector": embeddings_query,
                "fields": "vector",
                "k": int(self.top_k)
            }]
        elif AZURE_SEARCH_APPROACH == HYBRID_SEARCH_APPROACH:
            body["search"] = query
            body["vectorQueries"] = [{
                "kind": "vector",
                "vector": embeddings_query,
                "fields": "vector",
                "k": int(self.top_k)
            }]

        if AZURE_SEARCH_USE_SEMANTIC == "true" and AZURE_SEARCH_APPROACH != VECTOR_SEARCH_APPROACH:
            body["queryType"] = "semantic"
            body["semanticConfiguration"] = AZURE_SEARCH_SEMANTIC_SEARCH_CONFIG

        response = requests.post(search_url, headers=self._headers, json=body)

        if response.status_code != 200:
            raise Exception(f"Error in search request: {response.text}")

        return json.loads(response.text)["value"]

    async def _asearch(self, query: str) -> List[dict]:
        search_url = self._build_search_url(query)
        if not self.aiosession:
            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, headers=self._headers) as response:
                    response_json = await response.json()
        else:
            async with self.aiosession.get(
                search_url, headers=self._headers
            ) as response:
                response_json = await response.json()

        return response_json["value"]

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        search_results = self._search(query)

        return [
            Document(page_content="Source: " + result.pop("filepath") + "\n" + result.pop(self.content_key), metadata=result)
            for result in search_results
        ]

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        search_results = await self._asearch(query)

        return [
            Document(page_content="Source: " + result.pop("filepath") + "\n" + result.pop(self.content_key), metadata=result)
            for result in search_results
        ]

# For backwards compatibility
class AzureCognitiveSearchRetriever(AzureAISearchRetriever):
    """`Azure Cognitive Search` service retriever.
    This version of the retriever will soon be
    depreciated. Please switch to AzureAISearchRetriever
    """
