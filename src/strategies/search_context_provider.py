"""Search Context Provider with citation metadata.

Wraps Azure AI Search to return document content along with title and
filepath so the model can format proper ``[title](filepath)`` citations.
The UI resolves relative filepaths into time-limited SAS URLs.

Supports per-request OBO (On-Behalf-Of) permission trimming via the
``x_ms_query_source_authorization`` parameter natively provided by
the Azure Search SDK.
"""

import logging
import time
from collections.abc import Awaitable, Callable, MutableSequence, Sequence
from typing import Any, Optional

from agent_framework import ChatMessage, Context, ContextProvider, Role
from azure.core.credentials_async import AsyncTokenCredential
from azure.search.documents.aio import SearchClient
from azure.search.documents.models import VectorizedQuery, QueryType, QueryCaptionType

logger = logging.getLogger(__name__)

_CONTEXT_PROMPT = (
    "## Retrieved Documents\n\n"
    "The following documents were retrieved from the knowledge base. "
    "Each document starts with a header line in the format: ### [Document Title](filepath). "
    "Base your answer on these documents.\n\n"
    "**Citation rules:**\n"
    "- ONLY cite using the document title and filepath from the ### header lines above.\n"
    "- Format: [Document Title](filepath) — use the EXACT title and filepath from the header.\n"
    "- Do NOT treat any text inside the document content as a citation source. "
    "Internal references, chapter names, or bracketed text within the content are NOT valid sources.\n"
    "- Cite each source ONLY ONCE. Do NOT repeat the same citation on every bullet point or paragraph.\n"
    "- Example: According to [Product Guide](product-guide.pdf), the system supports...\n\n"
    "If the user's message is a greeting or small talk, ignore these documents and respond naturally."
)


class SearchContextProvider(ContextProvider):
    """Azure AI Search context provider that includes citation metadata."""

    def __init__(
        self,
        *,
        endpoint: str,
        index_name: str,
        credential: AsyncTokenCredential,
        top_k: int = 3,
        semantic_configuration_name: str | None = None,
        embed_fn: Callable[[str], Awaitable[list[float]]] | None = None,
        vector_field: str = "contentVector",
        max_content_chars: int = 1500,
        get_obo_token: Callable[[], Awaitable[Optional[str]]] | None = None,
    ) -> None:
        self._endpoint = endpoint
        self._index_name = index_name
        self._credential = credential
        self._top_k = top_k
        self._semantic_config = semantic_configuration_name
        self._embed_fn = embed_fn
        self._vector_field = vector_field
        self._max_content_chars = max_content_chars
        self._get_obo_token = get_obo_token

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def invoking(
        self,
        messages: ChatMessage | MutableSequence[ChatMessage],
        **kwargs: Any,
    ) -> Context:
        msgs = [messages] if isinstance(messages, ChatMessage) else list(messages)
        user_texts = [
            m.text for m in msgs
            if m and m.text and m.text.strip() and m.role == Role.USER
        ]
        if not user_texts:
            return Context()

        query = user_texts[-1]

        search_start = time.time()
        logger.info("[SearchContextProvider] Query: %r (top_k=%d, hybrid=%s)", query[:120], self._top_k, bool(self._embed_fn))

        search_params: dict[str, Any] = {
            "search_text": query,
            "top": self._top_k,
            "select": ["id", "content", "title", "filepath", "url"],
        }

        # Hybrid search: add vector query when embedding function is available
        if self._embed_fn:
            try:
                embed_start = time.time()
                vector = await self._embed_fn(query)
                search_params["vector_queries"] = [
                    VectorizedQuery(
                        vector=vector,
                        k=self._top_k,
                        fields=self._vector_field,
                    )
                ]
                logger.info("[SearchContextProvider] Embedding generated in %.2fs (dims=%d)", time.time() - embed_start, len(vector))
            except Exception as e:
                logger.warning("[SearchContextProvider] Embedding failed, falling back to keyword search: %s", e)

        if self._semantic_config:
            search_params["query_type"] = QueryType.SEMANTIC
            search_params["semantic_configuration_name"] = self._semantic_config
            search_params["query_caption"] = QueryCaptionType.EXTRACTIVE

        try:
            # Acquire OBO token for permission trimming if configured
            obo_token: Optional[str] = None
            if self._get_obo_token:
                try:
                    obo_token = await self._get_obo_token()
                except Exception as e:
                    logger.warning("[SearchContextProvider] OBO token acquisition failed: %s", e)

            if obo_token:
                search_params["x_ms_query_source_authorization"] = f"Bearer {obo_token}"
                logger.info("[SearchContextProvider] Using x-ms-query-source-authorization (OBO)")
            else:
                logger.info("[SearchContextProvider] Not sending x-ms-query-source-authorization")

            async with SearchClient(
                endpoint=self._endpoint,
                index_name=self._index_name,
                credential=self._credential,
            ) as client:
                results = await client.search(**search_params)

                parts: list[str] = []
                async for doc in results:
                    title = doc.get("title") or doc.get("filepath") or doc.get("id") or "Unknown"
                    link = doc.get("filepath") or doc.get("url") or ""
                    content = doc.get("content") or ""
                    if not content:
                        continue
                    if len(content) > self._max_content_chars:
                        content = content[:self._max_content_chars] + "..."
                    header = f"### [{title}]({link})" if link else f"### {title}"
                    parts.append(f"{header}\n{content}")
        except Exception as e:
            logger.error("[SearchContextProvider] Search failed in %.2fs: %s", time.time() - search_start, e)
            return Context()

        logger.info("[SearchContextProvider] Search returned %d documents in %.2fs", len(parts), time.time() - search_start)

        if not parts:
            return Context()

        context_text = _CONTEXT_PROMPT + "\n\n" + "\n\n---\n\n".join(parts)
        return Context(messages=[ChatMessage(role=Role.SYSTEM, text=context_text)])
