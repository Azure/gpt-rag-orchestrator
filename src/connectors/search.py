import aiohttp
import logging
import json
import time
from typing import Optional, Any, Dict, List
from pydantic import BaseModel

from dependencies import get_config


class SearchResult(BaseModel):
    """Represents a single search result from AI Search."""
    title: str
    link: str
    content: str


class SearchClient:
    """
    Azure Cognitive Search client with hybrid search support.
    
    Handles:
    - Basic search operations (term, vector, hybrid)
    - Document retrieval by ID
    - Token acquisition and authentication
    - Embeddings generation for vector search
    """
    def __init__(self):
        """
        Initialize SearchClient with configuration.
        """
        # ==== Load all config parameters in one place ====
        self.cfg = get_config()
        self.endpoint = self.cfg.get("SEARCH_SERVICE_QUERY_ENDPOINT")
        self.api_version = self.cfg.get("AZURE_SEARCH_API_VERSION", "2024-07-01")
        self.credential = self.cfg.aiocredential
        
        # Hybrid search configuration
        self.search_top_k = int(self.cfg.get('SEARCH_RAGINDEX_TOP_K', 3))
        self.search_approach = self.cfg.get('SEARCH_APPROACH', 'hybrid')
        self.semantic_search_config = self.cfg.get('SEARCH_SEMANTIC_SEARCH_CONFIG', 'my-semantic-config')
        self.search_service = self.cfg.get('SEARCH_SERVICE_NAME')
        self.use_semantic = self.cfg.get('SEARCH_USE_SEMANTIC', 'false').lower() == 'true'
        self.index_name = self.cfg.get("SEARCH_RAG_INDEX_NAME", "ragindex")
        
        # Initialize GenAIModelClient for embeddings (only if needed for vector/hybrid search)
        self.aoai_client = None
        if self.search_approach in ["vector", "hybrid"]:
            try:
                from connectors.aifoundry import GenAIModelClient
                self.aoai_client = GenAIModelClient()
                logging.info("[SearchClient] ✅ GenAIModelClient initialized for embeddings")
            except Exception as e:
                logging.warning("[SearchClient] ⚠️ Could not initialize GenAIModelClient for embeddings: %s", e)
                logging.warning("[SearchClient] ⚠️ Falling back to term search only")
                self.search_approach = "term"
        # ==== End config block ====

        if not self.endpoint:
            raise ValueError("SEARCH_SERVICE_QUERY_ENDPOINT not set in config")
        
        logging.info("[SearchClient] ✅ Initialized with hybrid search support")
        logging.info("[SearchClient]    Index: %s", self.index_name)
        logging.info("[SearchClient]    Approach: %s", self.search_approach)
        logging.info("[SearchClient]    Top K: %s", self.search_top_k)

    async def search(self, index_name: str, body: dict) -> dict:
        """
        Executes a search POST against /indexes/{index_name}/docs/search.
        """
        url = (
            f"{self.endpoint}"
            f"/indexes/{index_name}/docs/search"
            f"?api-version={self.api_version}"
        )

        # get bearer token
        try:
            token = (await self.credential.get_token("https://search.azure.com/.default")).token
        except Exception:
            logging.exception("[search] failed to acquire token")
            raise

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=body) as resp:
                text = await resp.text()
                if resp.status >= 400:
                    logging.error(f"[search] {resp.status} {text}")
                    raise RuntimeError(f"Search failed: {resp.status} {text}")
                return await resp.json()

    async def get_document(self, index_name: str, document_id: str, select_fields: list = None) -> dict:
        """
        Retrieves a single document by ID from the index.
        GET /indexes/{index_name}/docs/{document_id}
        
        Args:
            index_name: Name of the search index
            document_id: Document key/ID
            select_fields: Optional list of fields to retrieve (e.g., ['filepath', 'title'])
            
        Returns:
            Document dictionary with requested fields
        """
        # Build URL with optional $select parameter
        url = (
            f"{self.endpoint}"
            f"/indexes/{index_name}/docs('{document_id}')"
            f"?api-version={self.api_version}"
        )
        
        if select_fields:
            fields_str = ",".join(select_fields)
            url += f"&$select={fields_str}"
        
        # Get bearer token
        try:
            token = (await self.credential.get_token("https://search.azure.com/.default")).token
        except Exception:
            logging.exception("[search] failed to acquire token for get_document")
            raise

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as resp:
                text = await resp.text()
                if resp.status == 404:
                    logging.warning(f"[search] Document not found: {document_id}")
                    return None
                if resp.status >= 400:
                    logging.error(f"[search] {resp.status} {text}")
                    raise RuntimeError(f"Get document failed: {resp.status} {text}")
                return await resp.json()

    async def search_knowledge_base(self, query: str) -> str:
        """
        Searches the knowledge base for relevant documents using hybrid search.
        
        :param query: The search query to find relevant documents.
        :return: Search results as a JSON string containing a list of documents with title, link and content.
        """
        
        logging.info(f"[Retrieval] AI Search index: {self.index_name}")
        logging.info(f"[Retrieval] Search approach: {self.search_approach}")
        logging.info(f"[Retrieval] Executing search for query: {query}")

        try:
            logging.info("[Retrieval] Using Azure AI Search for document retrieval")
            
            # Build search body according to search approach
            search_body: Dict[str, Any] = {
                "select": "title,content,url,filepath,chunk_id",
                "top": self.search_top_k
            }
            
            # Generate embeddings for vector/hybrid search
            if self.search_approach in ["vector", "hybrid"] and self.aoai_client:
                start_time = time.time()
                logging.info(f"[Retrieval] Generating embeddings for query")
                embeddings_query = await self.aoai_client.get_embeddings(query)
                logging.info(f"[Retrieval] Embeddings generated in {round(time.time() - start_time, 2)} seconds")
                
                if self.search_approach == "vector":
                    search_body["vectorQueries"] = [{
                        "kind": "vector",
                        "vector": embeddings_query,
                        "fields": "contentVector",
                        "k": self.search_top_k
                    }]
                elif self.search_approach == "hybrid":
                    search_body["search"] = query
                    search_body["vectorQueries"] = [{
                        "kind": "vector",
                        "vector": embeddings_query,
                        "fields": "contentVector",
                        "k": self.search_top_k
                    }]
            else:
                # Term search only
                search_body["search"] = query
            
            # Execute search
            search_results = await self.search(
                index_name=self.index_name,
                body=search_body
            )
            
            # Process search results
            results_list = []
            for result in search_results.get('value', []):
                title = result.get('title', 'reference') or 'reference'
                link = result.get('filepath') or result.get('url', '') or ''
                content = result.get('content', '')
                
                # Debug log each document with formatted output (remove line breaks)
                content_preview = content[:200] if len(content) > 200 else content
                content_preview = ' '.join(content_preview.split())  # Replace all whitespace/newlines with single space
                logging.debug(f"[Retrieval] Document: [{title}]({link}): {content_preview}")
                
                search_result = SearchResult(
                    title=title,
                    link=link,
                    content=content
                )
                results_list.append(search_result.model_dump())
            
            logging.info(f"[Retrieval] Found {len(results_list)} results from Azure AI Search")
            return json.dumps({"results": results_list, "query": query})
            
        except Exception as e:
            logging.error(f"[Retrieval] Azure AI Search failed: {e}", exc_info=True)
            logging.warning("[Retrieval] Falling back to mock results")


    async def fetch_filepath_from_index(self, document_id: str) -> Optional[str]:
        """
        Fetch filepath directly from Azure AI Search index using document ID.
        
        Args:
            document_id: Document ID from Azure Search
            
        Returns:
            Filepath string from the index, or None if not found
        """
        try:
            logging.info("[Citations] 🔍 Fetching filepath from index for document_id: %s", document_id)
            
            document = await self.get_document(
                index_name=self.index_name,
                document_id=document_id,
                select_fields=['filepath', 'title']
            )
            
            if document:
                filepath = document.get('filepath')
                if filepath:
                    logging.info("[Citations] ✅ Found filepath in index: %s", filepath)
                    return filepath
                else:
                    logging.warning("[Citations] ⚠️ Document found but 'filepath' field is empty")
            else:
                logging.warning("[Citations] ⚠️ Document not found with ID: %s", document_id)
                
        except Exception as e:
            logging.error("[Citations] ❌ Error fetching document from index: %s", e, exc_info=True)
        
        return None
