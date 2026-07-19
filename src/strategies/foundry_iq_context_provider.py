"""Foundry IQ context provider with citation metadata.

The Foundry IQ counterpart to :class:`SearchContextProvider`. It retrieves
documents from a Foundry IQ knowledge base (via :class:`FoundryIQClient`) and
emits them into the prompt using the shared context-shaping helpers so the
context is byte-identical to the Azure AI Search path. Downstream citation
behavior therefore does not change with the selected ``RETRIEVAL_BACKEND``.

Per-request OBO permission trimming is forwarded by :class:`FoundryIQClient` in
the ``x-ms-query-source-authorization`` header. That OBO header is a distinct
mechanism from the Pattern B ``filterAddOn`` security filter, which narrows a
registered GPT-RAG Azure AI Search index using its custom security fields.

Introduced for Azure/GPT-RAG#526.
"""

import logging
import time
from collections.abc import Awaitable, Callable, MutableSequence
from typing import Any, Mapping, Optional

from agent_framework import ChatMessage, Context, ContextProvider, Role

from connectors.foundry_iq import McpSourceError, get_foundry_iq_client
from connectors.foundry_iq_mcp import McpConfigurationError, McpCredentialError
from connectors.search import _classify_retrieval_error
from .context_shaping import build_context_text, format_context_part

logger = logging.getLogger(__name__)


class FoundryIQContextProvider(ContextProvider):
    """Foundry IQ knowledge base context provider with citation metadata."""

    def __init__(
        self,
        *,
        conversation_id: Optional[str] = None,
        top_k: int = 3,
        max_content_chars: int = 1500,
        get_obo_token: Callable[[], Awaitable[Optional[str]]] | None = None,
        request_access_token: Optional[str] = None,
        allow_anonymous: bool = True,
        mcp_enabled: bool = False,
        user_context: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self._conversation_id = (conversation_id or "").strip() or None
        self._top_k = top_k
        self._max_content_chars = max_content_chars
        self._get_obo_token = get_obo_token
        self._request_access_token = request_access_token
        self._allow_anonymous = allow_anonymous
        self._mcp_enabled = mcp_enabled
        self._user_context = dict(user_context or {})

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
        logger.info("[FoundryIQContextProvider] Query: %r (top_k=%d)", query[:120], self._top_k)

        obo_token: Optional[str] = None
        mcp_enabled = self._mcp_enabled
        try:
            client = get_foundry_iq_client()
            client_mcp_config = getattr(client, "mcp_config", None)
            if client_mcp_config is not None:
                mcp_enabled = bool(
                    getattr(client_mcp_config, "enabled", self._mcp_enabled)
                )

            # Acquire OBO token for per-user document-level security if configured.
            if self._get_obo_token:
                try:
                    obo_token = await self._get_obo_token()
                except Exception as e:
                    logger.warning(
                        "[FoundryIQContextProvider] OBO token acquisition failed: %s",
                        e,
                    )
                    if mcp_enabled:
                        raise McpCredentialError(
                            "Failed to acquire the required Search OBO token"
                        ) from None
                if mcp_enabled and not obo_token and not self._allow_anonymous:
                    raise McpCredentialError(
                        "Search OBO token is required when ALLOW_ANONYMOUS=false"
                    )

            retrieve_kwargs: dict[str, Any] = {
                "obo_token": obo_token,
                "conversation_id": self._conversation_id,
                "user_context": self._user_context,
            }
            if mcp_enabled:
                retrieve_kwargs["incoming_token"] = self._request_access_token
            records = await client.retrieve(query, **retrieve_kwargs)

            parts: list[str] = []
            for record in records[: self._top_k]:
                title = record.get("title") or "reference"
                link = record.get("link") or ""
                content = record.get("content") or ""
                if not content:
                    continue
                if len(content) > self._max_content_chars:
                    content = content[: self._max_content_chars] + "..."
                parts.append(format_context_part(title, link, content))
        except Exception as e:
            level, marker = _classify_retrieval_error(e)
            logger.log(
                level,
                "%s Foundry IQ retrieval failed in %.2fs: %s",
                marker,
                time.time() - search_start,
                e,
                exc_info=True,
                extra={
                    "retrieval_credential_type": "obo" if obo_token else "managed_identity",
                },
            )
            if isinstance(
                e, (McpConfigurationError, McpCredentialError, McpSourceError)
            ) and mcp_enabled:
                raise
            return Context()

        logger.info(
            "[FoundryIQContextProvider] Retrieval returned %d documents in %.2fs",
            len(parts),
            time.time() - search_start,
        )

        if not parts:
            return Context()

        context_text = build_context_text(parts)
        return Context(messages=[ChatMessage(role=Role.SYSTEM, text=context_text)])
