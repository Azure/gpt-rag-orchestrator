"""Foundry IQ knowledge base retrieval client.

Targets the Azure AI Search *knowledge base* retrieve action (Foundry IQ), the
agentic-retrieval surface that runs query planning and parallel retrieval across
the knowledge sources bound to a knowledge base. The client mirrors the
``aiohttp`` style of :class:`connectors.search.SearchClient`: it acquires a
service bearer token for ``https://search.azure.com/.default`` and, when an OBO
user token is supplied, forwards it in the ``x-ms-query-source-authorization``
header for per-user document-level security (query-time ACL/RBAC enforcement).

Pattern A vs Pattern B is a *server-side* knowledge base configuration concern,
not a client concern: the client always targets the configured knowledge base
name and the retrieve call shape is identical either way. The Pattern B security
field trimming (a ``filterAddOn`` OData filter under ``knowledgeSourceParams``)
is a separate mechanism from the OBO header and is intentionally NOT wired here —
it is deferred to a later PR.

Introduced for Azure/GPT-RAG#526. Per-user security on the retrieve action is a
preview capability, so the API version is pinned to a single configurable
constant (see :data:`DEFAULT_FOUNDRY_IQ_API_VERSION`).
"""

import logging
import time
from typing import Any, Dict, List, Optional

import aiohttp

from dependencies import get_config

# Per-user security on the knowledge base retrieve action (both the native OBO
# path and the Pattern B filterAddOn path) requires this preview API version.
# Core retrieval is GA at 2026-04-01, but we pin the preview so the security
# features are always available. Override via the FOUNDRY_IQ_API_VERSION key.
DEFAULT_FOUNDRY_IQ_API_VERSION = "2026-05-01-preview"

# Config key names (kept here so the connector layer doesn't import from api/).
KNOWLEDGE_BASE_NAME_KEY = "KNOWLEDGE_BASE_NAME"
KNOWLEDGE_BASE_ENDPOINT_KEY = "KNOWLEDGE_BASE_ENDPOINT"
FOUNDRY_IQ_API_VERSION_KEY = "FOUNDRY_IQ_API_VERSION"

# Search-audience scope used for the service token.
_SEARCH_SCOPE = "https://search.azure.com/.default"


class FoundryIQClient:
    """Client for the Foundry IQ knowledge base retrieve action."""

    def __init__(self) -> None:
        self.cfg = get_config()
        # The knowledge base lives on the search service, so the endpoint
        # defaults to the search query endpoint when KNOWLEDGE_BASE_ENDPOINT is
        # not explicitly stamped.
        self.endpoint = (
            self.cfg.get("KNOWLEDGE_BASE_ENDPOINT")
            or self.cfg.get("SEARCH_SERVICE_QUERY_ENDPOINT")
        )
        self.knowledge_base_name = self.cfg.get(KNOWLEDGE_BASE_NAME_KEY)
        self.api_version = self.cfg.get(
            FOUNDRY_IQ_API_VERSION_KEY, DEFAULT_FOUNDRY_IQ_API_VERSION
        )
        self.credential = self.cfg.aiocredential

        # Shared aiohttp session — reuses TCP connections across calls.
        self._session: Optional[aiohttp.ClientSession] = None

        if not self.endpoint:
            raise ValueError(
                "Neither KNOWLEDGE_BASE_ENDPOINT nor SEARCH_SERVICE_QUERY_ENDPOINT "
                "is set in config; cannot reach the Foundry IQ knowledge base."
            )
        if not self.knowledge_base_name:
            raise ValueError(
                "KNOWLEDGE_BASE_NAME is not set in config; cannot target a "
                "Foundry IQ knowledge base."
            )

        logging.info(
            "[FoundryIQClient] ✅ Initialized (knowledge_base=%s, api_version=%s)",
            self.knowledge_base_name,
            self.api_version,
        )

    async def _get_session(self) -> aiohttp.ClientSession:
        """Return a shared aiohttp session, creating one lazily if needed."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    def _retrieve_url(self) -> str:
        return (
            f"{self.endpoint}"
            f"/knowledgebases/{self.knowledge_base_name}/retrieve"
            f"?api-version={self.api_version}"
        )

    @staticmethod
    def _normalize_references(payload: Dict[str, Any]) -> List[Dict[str, str]]:
        """Map a retrieve response into ``[{title, link, content}]`` records.

        The knowledge base retrieve response carries a ``references`` array; each
        reference exposes a ``sourceData`` object with the configured source
        fields. We map defensively into the same ``title``/``link``/``content``
        contract the Azure AI Search path emits so downstream citation handling
        is identical across backends.
        """
        records: List[Dict[str, str]] = []
        for ref in payload.get("references", []) or []:
            source = ref.get("sourceData") or {}
            title = source.get("title") or ref.get("docKey") or source.get("id") or "reference"
            link = source.get("filepath") or source.get("url") or ""
            content = source.get("content") or ""
            if not content:
                continue
            records.append({"title": title, "link": link, "content": content})
        return records

    async def retrieve(
        self,
        query: str,
        *,
        obo_token: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Run the knowledge base retrieve action and return normalized records.

        :param query: Natural-language user query.
        :param obo_token: Optional delegated Azure AI Search user token. When
            present it is forwarded in ``x-ms-query-source-authorization`` for
            per-user document-level security. This is the OBO mechanism, kept
            distinct from the Pattern B ``filterAddOn`` security filter.
        :param conversation_id: Conversation scope, plumbed for parity and
            logging. Conversation/security-field scoping via ``filterAddOn`` is
            a server-side knowledge source concern deferred to a later PR.
        :return: A list of ``{title, link, content}`` records.
        """
        url = self._retrieve_url()
        start = time.time()

        # Service bearer token (the same search audience used by SearchClient).
        try:
            token = (await self.credential.get_token(_SEARCH_SCOPE)).token
        except Exception:
            logging.exception("[FoundryIQClient] failed to acquire service token")
            raise

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }
        if obo_token:
            # Per-user document-level security (query-time ACL/RBAC enforcement).
            headers["x-ms-query-source-authorization"] = obo_token
            logging.info("[FoundryIQClient][Trimming] Using x-ms-query-source-authorization (OBO)")
        else:
            logging.info("[FoundryIQClient][Trimming] Not sending x-ms-query-source-authorization")

        if conversation_id:
            logging.debug("[FoundryIQClient] conversation_id=%s", conversation_id)

        # Minimal retrieve request: a single user message. Answer synthesis is
        # left to the orchestrator's own model, so the default output mode (which
        # returns references) is used.
        body: Dict[str, Any] = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": query}],
                }
            ]
        }

        session = await self._get_session()
        async with session.post(url, headers=headers, json=body) as resp:
            text = await resp.text()
            if resp.status >= 400:
                logging.error("[FoundryIQClient] %s %s", resp.status, text)
                raise RuntimeError(f"Foundry IQ retrieve failed: {resp.status} {text}")
            payload = await resp.json()

        records = self._normalize_references(payload)
        logging.info(
            "[FoundryIQClient] Retrieved %d references in %.2fs (knowledge_base=%s)",
            len(records),
            time.time() - start,
            self.knowledge_base_name,
        )
        return records


_foundry_iq_client_instance: Optional[FoundryIQClient] = None


def get_foundry_iq_client() -> FoundryIQClient:
    """Return a singleton :class:`FoundryIQClient` to reuse connections/config."""
    global _foundry_iq_client_instance
    if _foundry_iq_client_instance is None:
        _foundry_iq_client_instance = FoundryIQClient()
    return _foundry_iq_client_instance
