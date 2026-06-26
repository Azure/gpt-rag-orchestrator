"""Foundry IQ knowledge base retrieval client.

Targets the Azure AI Search *knowledge base* retrieve action (Foundry IQ), the
agentic-retrieval surface that runs query planning and parallel retrieval across
the knowledge sources bound to a knowledge base. The client mirrors the
``aiohttp`` style of :class:`connectors.search.SearchClient`: it acquires a
service bearer token for ``https://search.azure.com/.default`` and, when an OBO
user token is supplied, forwards it in the ``x-ms-query-source-authorization``
header for per-user document-level security (query-time ACL/RBAC enforcement).

Pattern A vs Pattern B is mostly a *server-side* knowledge base configuration
concern: the client always targets the configured knowledge base name. The one
query-time difference is Pattern B security-field trimming. When enabled,
GPT-RAG injects a ``filterAddOn`` OData filter under ``knowledgeSourceParams``.
That filter is deliberately separate from the native OBO header. The OBO header
drives Foundry IQ permission-aware sources such as ADLS Gen2 ACLs, SharePoint,
Purview, OneLake, and Fabric. The Pattern B filter narrows the registered GPT-RAG
Azure AI Search index that carries custom security fields.

Introduced for Azure/GPT-RAG#526. Per-user security on the retrieve action is a
preview capability, so the API version is pinned to a single configurable
constant (see :data:`DEFAULT_FOUNDRY_IQ_API_VERSION`).
"""

import logging
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional

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
FOUNDRY_IQ_KNOWLEDGE_SOURCE_NAME_KEY = "FOUNDRY_IQ_KNOWLEDGE_SOURCE_NAME"
FOUNDRY_IQ_KNOWLEDGE_SOURCE_KIND_KEY = "FOUNDRY_IQ_KNOWLEDGE_SOURCE_KIND"
FOUNDRY_IQ_PATTERN_KEY = "FOUNDRY_IQ_PATTERN"
FOUNDRY_IQ_FILTER_ADD_ON_ENABLED_KEY = "FOUNDRY_IQ_FILTER_ADD_ON_ENABLED"
FOUNDRY_IQ_SECURITY_FIELD_NAME_KEY = "FOUNDRY_IQ_SECURITY_FIELD_NAME"
FOUNDRY_IQ_MAX_OUTPUT_DOCUMENTS_KEY = "FOUNDRY_IQ_MAX_OUTPUT_DOCUMENTS"

# Search-audience scope used for the service token.
_SEARCH_SCOPE = "https://search.azure.com/.default"


def _odata_escape_string(value: Optional[str]) -> str:
    """Escape a string for embedding in a single-quoted OData literal."""
    return (value or "").replace("'", "''")


def _normalize_security_ids(values: Iterable[Any]) -> List[str]:
    """Return stable, non-empty security IDs without broadening the filter."""
    normalized: List[str] = []
    seen = set()
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if not text or text == "anonymous":
            continue
        if text not in seen:
            seen.add(text)
            normalized.append(text)
    return normalized


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _as_optional_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def build_pattern_b_filter_add_on(
    *,
    conversation_id: Optional[str],
    user_context: Optional[Mapping[str, Any]],
    security_field_name: str = "metadata_security_id",
) -> str:
    """Build the Pattern B ``filterAddOn`` for GPT-RAG security fields.

    Pattern B registers GPT-RAG's existing Azure AI Search index as a Foundry IQ
    ``searchIndex`` knowledge source. The custom security fields on that index
    are not enforced by ``x-ms-query-source-authorization``. They must be
    expressed as an OData ``filterAddOn`` on the retrieve request. The filter
    preserves the direct AI Search behavior: include documents that match one of
    the caller security IDs, plus public documents where the security collection
    is empty.
    """
    safe_field = (security_field_name or "").strip() or "metadata_security_id"
    context = dict(user_context or {})

    candidate_ids: List[Any] = [
        context.get("principal_id"),
        context.get("oid"),
        context.get("user_id"),
    ]
    principal_name = context.get("principal_name") or context.get("user_name")
    if principal_name:
        candidate_ids.append(principal_name)

    for key in ("security_ids", "groups", "group_ids", "client_group_names"):
        value = context.get(key)
        if isinstance(value, (list, tuple, set)):
            candidate_ids.extend(value)
        elif value:
            candidate_ids.append(value)

    security_ids = _normalize_security_ids(candidate_ids)
    public_clause = f"not {safe_field}/any()"
    if not security_ids:
        security_clause = public_clause
    else:
        escaped_ids = ",".join(_odata_escape_string(value) for value in security_ids)
        security_clause = f"({safe_field}/any(g:search.in(g, '{escaped_ids}')) or {public_clause})"

    # Keep runtime uploads conversation-scoped while allowing the shared corpus.
    conversation_clause = None
    cid = (conversation_id or "").strip() or None
    if cid:
        conversation_clause = (
            f"(conversationId eq '{_odata_escape_string(cid)}' "
            "or (conversationId eq 'NaN' or conversationId eq null))"
        )

    if conversation_clause:
        return f"({security_clause}) and {conversation_clause}"
    return security_clause


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
        self.knowledge_source_name = (
            self.cfg.get(FOUNDRY_IQ_KNOWLEDGE_SOURCE_NAME_KEY, "") or ""
        ).strip()
        self.knowledge_source_kind = (
            self.cfg.get(FOUNDRY_IQ_KNOWLEDGE_SOURCE_KIND_KEY, "")
            or self.cfg.get(FOUNDRY_IQ_PATTERN_KEY, "")
            or ""
        ).strip()
        self.filter_add_on_enabled = _as_bool(
            self.cfg.get(FOUNDRY_IQ_FILTER_ADD_ON_ENABLED_KEY, False, type=bool)
        )
        if not self.knowledge_source_kind:
            self.knowledge_source_kind = "searchIndex" if self.filter_add_on_enabled else "azureBlob"
        if self.knowledge_source_kind == "managed":
            self.knowledge_source_kind = "azureBlob"
        self.security_field_name = self.cfg.get(
            FOUNDRY_IQ_SECURITY_FIELD_NAME_KEY, "metadata_security_id", type=str
        )
        self.max_output_documents = _as_optional_int(
            self.cfg.get(FOUNDRY_IQ_MAX_OUTPUT_DOCUMENTS_KEY, None)
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

        The knowledge base retrieve response carries a ``references`` array;
        each reference exposes a ``sourceData`` object with the configured
        source fields. Field names vary by knowledge source kind and content
        extraction mode:

        - ``azureBlob`` native sources (both ``minimal`` and ``standard``
          content extraction) return ``sourceData.snippet`` and
          ``sourceData.blob_url`` and do not include ``title``.
        - ``searchIndex`` Pattern B sources return whatever the index
          projects (typically ``content``, ``filepath``/``url``, ``title``).

        We accept the union with explicit priority so the downstream
        ``{title, link, content}`` contract is identical across backends,
        and so a future small rename does not silently drop references.
        Title falls back to the link's filename when the source does not
        carry one (true for ``azureBlob`` references).
        """
        records: List[Dict[str, str]] = []
        for ref in payload.get("references", []) or []:
            source = ref.get("sourceData") or {}

            content = (
                source.get("snippet")
                or source.get("content")
                or source.get("text")
                or ""
            )
            if not content:
                continue

            link = (
                source.get("blob_url")
                or source.get("url")
                or source.get("filepath")
                or source.get("path")
                or ref.get("blobUrl")
                or ""
            )

            title = source.get("title") or ref.get("docKey") or source.get("id")
            if not title and link:
                # Derive a human-friendly title from the blob/file name so the
                # citation surface is not an opaque uid.
                tail = link.rsplit("/", 1)[-1]
                title = tail.split("?", 1)[0] or None
            if not title:
                title = "reference"

            records.append({"title": title, "link": link, "content": content})
        return records

    async def retrieve(
        self,
        query: str,
        *,
        obo_token: Optional[str] = None,
        conversation_id: Optional[str] = None,
        user_context: Optional[Mapping[str, Any]] = None,
    ) -> List[Dict[str, str]]:
        """Run the knowledge base retrieve action and return normalized records.

        :param query: Natural-language user query.
        :param obo_token: Optional delegated Azure AI Search user token. When
            present it is forwarded in ``x-ms-query-source-authorization`` for
            per-user document-level security. This is the OBO mechanism, kept
            distinct from the Pattern B ``filterAddOn`` security filter.
        :param conversation_id: Optional conversation scope. When Pattern B
            ``filterAddOn`` is enabled, this narrows runtime-upload documents to
            the current conversation plus shared/global chunks.
        :param user_context: Optional user/security context used only for Pattern
            B security-field filters. Native permission enforcement continues to
            use ``obo_token`` and the ``x-ms-query-source-authorization`` header.
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

        # GPT-RAG configures the knowledge base with minimal reasoning, which
        # requires explicit intents rather than chat messages.
        body: Dict[str, Any] = {
            "intents": [
                {
                    "search": query,
                    "type": "semantic",
                }
            ]
        }
        if self.max_output_documents:
            body["maxOutputDocuments"] = self.max_output_documents

        knowledge_source_params: List[Dict[str, Any]] = []
        if self.knowledge_source_name:
            source_params: Dict[str, Any] = {
                "knowledgeSourceName": self.knowledge_source_name,
                "kind": self.knowledge_source_kind,
                "includeReferences": True,
                "includeReferenceSourceData": True,
            }
            if self.filter_add_on_enabled:
                if self.knowledge_source_kind != "searchIndex":
                    raise ValueError(
                        "Foundry IQ filterAddOn is only valid for searchIndex knowledge sources; "
                        f"current kind={self.knowledge_source_kind}."
                    )
                if self.api_version != DEFAULT_FOUNDRY_IQ_API_VERSION:
                    raise ValueError(
                        "Foundry IQ Pattern B filterAddOn requires "
                        f"{DEFAULT_FOUNDRY_IQ_API_VERSION}; current "
                        f"FOUNDRY_IQ_API_VERSION={self.api_version}."
                    )
                source_params["filterAddOn"] = build_pattern_b_filter_add_on(
                    conversation_id=conversation_id,
                    user_context=user_context,
                    security_field_name=self.security_field_name,
                )
                logging.info(
                    "[FoundryIQClient][PatternB] Applying filterAddOn to knowledge source %s",
                    self.knowledge_source_name,
                )
            knowledge_source_params.append(source_params)

        if knowledge_source_params:
            body["knowledgeSourceParams"] = knowledge_source_params

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
