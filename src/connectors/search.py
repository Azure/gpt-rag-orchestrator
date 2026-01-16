import aiohttp
import logging
import json
import time
import hashlib
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

        # Per-request context (kept in memory only)
        self._request_api_access_token: Optional[str] = None
        self._allow_anonymous: bool = True

        # Cached delegated Search token (OBO) for the current request
        self._cached_search_user_token: Optional[str] = None
        self._cached_search_user_token_expires_at: float = 0.0

        # Last OBO error summary (for clear logs / strict-mode failures)
        self._last_obo_error: Optional[str] = None
        
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

    def set_request_context(self, *, api_access_token: Optional[str], allow_anonymous: bool) -> None:
        """Sets per-request context used for permission trimming.

        api_access_token is the incoming user token sent to the orchestrator (audience: this API).
        It is used as the OBO assertion to acquire a Search-audience user token.
        """
        api_access_token = (api_access_token or "").strip() or None
        if api_access_token != self._request_api_access_token:
            # Token changed (new request); drop cached OBO token.
            self._cached_search_user_token = None
            self._cached_search_user_token_expires_at = 0.0

        self._request_api_access_token = api_access_token
        self._allow_anonymous = bool(allow_anonymous)

    def _token_fingerprint(self, token: Optional[str]) -> str:
        if not token:
            return "<none>"
        try:
            return hashlib.sha256(token.encode("utf-8")).hexdigest()[:12]
        except Exception:
            return "<unknown>"

    async def _acquire_search_user_token_via_obo(self, api_access_token: str) -> Optional[str]:
        """Acquire a delegated Azure AI Search token using the OBO flow.

        This exchanges the incoming API token (user assertion) for a Search-audience token.
        """
        tenant_id = None
        client_id = None
        client_secret = None
        try:
            tenant_id = (self.cfg.get_value("OAUTH_AZURE_AD_TENANT_ID", default=None, allow_none=True) or "").strip() or None
        except Exception:
            tenant_id = None

        try:
            client_id = (self.cfg.get_value("OAUTH_AZURE_AD_CLIENT_ID", default=None, allow_none=True) or "").strip() or None
        except Exception:
            client_id = None

        try:
            client_secret = (self.cfg.get_value("OAUTH_AZURE_AD_CLIENT_SECRET", default=None, allow_none=True) or "").strip() or None
        except Exception:
            client_secret = None

        if not tenant_id or not client_id or not client_secret:
            logging.warning(
                "[Retrieval][OBO] Missing Entra configuration for OBO. tenant_id=%s client_id=%s client_secret=%s",
                "set" if tenant_id else "missing",
                "set" if client_id else "missing",
                "set" if client_secret else "missing",
            )
            return None

        token_url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"

        form = {
            "client_id": client_id,
            "client_secret": client_secret,
            "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
            "requested_token_use": "on_behalf_of",
            "scope": "https://search.azure.com/user_impersonation",
            "assertion": api_access_token,
        }

        # Important: never log the assertion.
        fp = self._token_fingerprint(api_access_token)
        logging.debug("[Retrieval][OBO] Requesting Search delegated token via OBO (assertion_fp=%s)", fp)

        async with aiohttp.ClientSession() as session:
            async with session.post(token_url, data=form) as resp:
                raw = await resp.text()
                if resp.status >= 400:
                    # Token endpoint errors often include trace_id/correlation_id.
                    try:
                        err = json.loads(raw)
                        error = err.get("error")
                        desc = err.get("error_description")
                        trace_id = err.get("trace_id")
                        correlation_id = err.get("correlation_id")
                        error_codes = err.get("error_codes")
                        self._last_obo_error = (
                            f"status={resp.status} error={error} codes={error_codes} trace_id={trace_id} correlation_id={correlation_id}"
                        )
                        logging.error(
                            "[Retrieval][OBO] OBO failed (status=%d error=%s codes=%s trace_id=%s correlation_id=%s desc=%s)",
                            resp.status,
                            error,
                            error_codes,
                            trace_id,
                            correlation_id,
                            (str(desc)[:240] + "…") if desc and len(str(desc)) > 240 else desc,
                        )
                    except Exception:
                        self._last_obo_error = f"status={resp.status} body={raw[:200]}"
                        logging.error("[Retrieval][OBO] OBO failed (status=%d body=%s)", resp.status, raw[:400])
                    return None

                data = {}
                try:
                    data = json.loads(raw)
                except Exception:
                    logging.error("[Retrieval][OBO] Token endpoint returned non-JSON response")
                    return None

                token = data.get("access_token")
                expires_in = data.get("expires_in")
                if not token:
                    self._last_obo_error = "token endpoint response missing access_token"
                    logging.error("[Retrieval][OBO] Token endpoint response missing access_token")
                    return None

                # Cache for the remainder of the request.
                try:
                    ttl = int(expires_in) if expires_in is not None else 0
                except Exception:
                    ttl = 0
                self._cached_search_user_token = token
                self._cached_search_user_token_expires_at = time.time() + max(0, ttl - 30)

                self._last_obo_error = None

                logging.info("[Retrieval][OBO] ✅ Acquired Search delegated token via OBO")
                return token

    async def _get_search_user_token_for_trimming(self) -> Optional[str]:
        # Use cached token if still valid.
        if self._cached_search_user_token and time.time() < self._cached_search_user_token_expires_at:
            return self._cached_search_user_token

        # No incoming user token -> cannot do OBO.
        if not self._request_api_access_token:
            if self._allow_anonymous:
                logging.info(
                    "[Retrieval][Trimming] No incoming user token; running without x-ms-query-source-authorization because ALLOW_ANONYMOUS=true"
                )
                return None

            logging.error(
                "[Retrieval][Trimming] Missing incoming user token. Permission trimming is required but ALLOW_ANONYMOUS=false. "
                "Refusing to call Search without x-ms-query-source-authorization."
            )
            raise RuntimeError(
                "Permission trimming requires an incoming user access token. "
                "No Authorization header was available and ALLOW_ANONYMOUS=false."
            )

        token = await self._acquire_search_user_token_via_obo(self._request_api_access_token)
        if token:
            return token

        if self._allow_anonymous:
            logging.warning(
                "[Retrieval][Trimming] OBO failed; running without x-ms-query-source-authorization because ALLOW_ANONYMOUS=true (details=%s)",
                self._last_obo_error or "<no-details>",
            )
            return None

        logging.error(
            "[Retrieval][Trimming] OBO failed and ALLOW_ANONYMOUS=false. Refusing to call Search without x-ms-query-source-authorization. Details: %s",
            self._last_obo_error or "<no-details>",
        )
        raise RuntimeError(
            "Failed to acquire Azure AI Search delegated token via OBO. "
            "Ensure API permissions include Azure Cognitive Search delegated user_impersonation and admin consent is granted."
        )

    async def search(self, index_name: str, body: dict, *, search_user_token: Optional[str] = None) -> dict:
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

        # Optional: user context token for permission trimming.
        if search_user_token:
            headers["x-ms-query-source-authorization"] = f"Bearer {search_user_token}"

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
            search_user_token = await self._get_search_user_token_for_trimming()
            if search_user_token:
                logging.info("[Retrieval][Trimming] Using x-ms-query-source-authorization (OBO token acquired)")
            else:
                logging.info("[Retrieval][Trimming] Not sending x-ms-query-source-authorization")

            search_results = await self.search(
                index_name=self.index_name,
                body=search_body,
                search_user_token=search_user_token,
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

            # In strict mode (ALLOW_ANONYMOUS=false), do not silently degrade.
            # Raise so the request fails early and the logs make the root cause obvious.
            if not self._allow_anonymous:
                raise

            logging.warning("[Retrieval] Falling back to empty results (ALLOW_ANONYMOUS=true)")
            return json.dumps({"results": [], "query": query, "error": "search_failed"})


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
