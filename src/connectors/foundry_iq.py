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
FOUNDRY_IQ_FORWARD_SOURCE_AUTH_KEY = "FOUNDRY_IQ_FORWARD_SOURCE_AUTH"
# Hybrid file-upload sidecar (Pattern A + UI upload). When enabled, and the
# primary knowledge source is the native azureBlob corpus, the retrieve action
# also queries a second searchIndex knowledge source built over the existing
# GPT-RAG index (SEARCH_RAG_INDEX_NAME). That second source carries the runtime
# uploads and is trimmed by a conversationId filterAddOn so uploaded files are
# only visible inside the conversation that created them. Off by default.
FOUNDRY_IQ_CONVERSATION_UPLOAD_ENABLED_KEY = "FOUNDRY_IQ_CONVERSATION_UPLOAD_ENABLED"
FOUNDRY_IQ_CONVERSATION_KNOWLEDGE_SOURCE_NAME_KEY = (
    "FOUNDRY_IQ_CONVERSATION_KNOWLEDGE_SOURCE_NAME"
)
# Upper bound on the retrieve action runtime (maxRuntimeInSeconds on the
# request body). Remote knowledge source kinds such as workIQ / fabricOntology
# can take 40-60s to fan out; the default leaves headroom without penalising
# the fast local kinds. Only emitted when a remote kind is enabled so today's
# Pattern A / Pattern B request bodies remain byte-identical.
FOUNDRY_IQ_MAX_RUNTIME_SECONDS_KEY = "FOUNDRY_IQ_MAX_RUNTIME_SECONDS"
DEFAULT_FOUNDRY_IQ_MAX_RUNTIME_SECONDS = 120

# Work IQ (Microsoft 365 remote knowledge source). Opt-in; default off. When
# enabled, the retrieve request appends a workIQ knowledge source in
# knowledgeSourceParams. ACL is enforced natively by M365, so no filterAddOn
# is applied. OBO is required; MI fallback is never used for remote kinds.
WORK_IQ_ENABLED_KEY = "WORK_IQ_ENABLED"
WORK_IQ_KNOWLEDGE_SOURCE_NAME_KEY = "WORK_IQ_KNOWLEDGE_SOURCE_NAME"

# Fabric IQ (Microsoft Fabric ontology remote knowledge source). Opt-in;
# default off. When enabled, the retrieve request appends a fabricOntology
# knowledge source in knowledgeSourceParams. ACL is enforced natively by
# Fabric via the forwarded per-user OBO token (same
# x-ms-query-source-authorization header used by Work IQ), so no filterAddOn
# is applied. OBO is required; MI fallback is never used for remote kinds.
FABRIC_IQ_ENABLED_KEY = "FABRIC_IQ_ENABLED"
FABRIC_IQ_KNOWLEDGE_SOURCE_NAME_KEY = "FABRIC_IQ_KNOWLEDGE_SOURCE_NAME"

# Fabric Data Agent (Microsoft Fabric Data Agent remote knowledge source).
# Opt-in; default off. When enabled, the retrieve request appends a
# fabricDataAgent knowledge source in knowledgeSourceParams. A Fabric Data
# Agent acts as a virtual analyst that runs queries over Fabric data and
# returns answers, tables, and charts. ACL is enforced natively by Fabric
# via the forwarded per-user OBO token (same x-ms-query-source-authorization
# header used by Work IQ / Fabric ontology), so no filterAddOn is applied.
# OBO is required; MI fallback is never used for remote kinds.
FABRIC_DATA_AGENT_ENABLED_KEY = "FABRIC_DATA_AGENT_ENABLED"
FABRIC_DATA_AGENT_KNOWLEDGE_SOURCE_NAME_KEY = "FABRIC_DATA_AGENT_KNOWLEDGE_SOURCE_NAME"

# SharePoint remote (Microsoft 365 Copilot Retrieval API remote knowledge
# source). Opt-in; default off. When enabled, the retrieve request appends a
# remoteSharePoint knowledge source in knowledgeSourceParams. SharePoint
# item-level permissions are enforced natively by M365 via the forwarded
# per-user OBO token (same x-ms-query-source-authorization header used by
# Work IQ / Fabric ontology), so no filterAddOn is applied. OBO is required;
# MI fallback is never used for remote kinds. An optional retrieval-time KQL
# scope can be forwarded via filterExpressionAddOn.
SHAREPOINT_REMOTE_ENABLED_KEY = "SHAREPOINT_REMOTE_ENABLED"
SHAREPOINT_REMOTE_KNOWLEDGE_SOURCE_NAME_KEY = "SHAREPOINT_REMOTE_KNOWLEDGE_SOURCE_NAME"
SHAREPOINT_REMOTE_FILTER_EXPRESSION_ADD_ON_KEY = (
    "SHAREPOINT_REMOTE_FILTER_EXPRESSION_ADD_ON"
)

# Knowledge source kinds that must never fall back to the service managed
# identity for x-ms-query-source-authorization. These sources run against
# external systems (M365, Fabric) where impersonation, not app identity, is
# the security boundary.
_REMOTE_KNOWLEDGE_SOURCE_KINDS = frozenset(
    {"workIQ", "fabricOntology", "fabricDataAgent", "remoteSharePoint"}
)

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


def build_conversation_upload_filter_add_on(conversation_id: str) -> str:
    """Build the Foundry IQ sidecar filter for runtime conversation uploads."""
    return f"conversationId eq '{_odata_escape_string(conversation_id)}'"


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
        # When true (default), and no per-user OBO token is available, the
        # service managed-identity Search-audience token is forwarded as
        # ``x-ms-query-source-authorization`` so Foundry IQ can evaluate
        # RBAC-scoped permission filters on the bound knowledge source. This
        # is required by knowledge bases whose index has
        # ``permissionFilterOption=enabled`` and whose knowledge source uses
        # ``ingestionPermissionOptions=["rbacScope"]`` - without it the
        # retrieve action returns 502 ("Failed to query search index").
        self.forward_source_auth = _as_bool(
            self.cfg.get(FOUNDRY_IQ_FORWARD_SOURCE_AUTH_KEY, True, type=bool)
        )
        # Hybrid file-upload sidecar. Only meaningful for Pattern A (azureBlob
        # primary); Pattern B already carries a conversationId filterAddOn on its
        # single searchIndex source, so it needs no second source.
        self.conversation_upload_enabled = _as_bool(
            self.cfg.get(FOUNDRY_IQ_CONVERSATION_UPLOAD_ENABLED_KEY, False, type=bool)
        )
        self.conversation_knowledge_source_name = (
            self.cfg.get(FOUNDRY_IQ_CONVERSATION_KNOWLEDGE_SOURCE_NAME_KEY, "") or ""
        ).strip()
        # Retrieve-runtime ceiling for remote knowledge source kinds.
        self.max_runtime_seconds = _as_optional_int(
            self.cfg.get(
                FOUNDRY_IQ_MAX_RUNTIME_SECONDS_KEY,
                DEFAULT_FOUNDRY_IQ_MAX_RUNTIME_SECONDS,
            )
        ) or DEFAULT_FOUNDRY_IQ_MAX_RUNTIME_SECONDS
        # Work IQ (Microsoft 365) remote knowledge source. Opt-in; requires
        # a per-user OBO token - anonymous / MI fallback is never used.
        self.work_iq_enabled = _as_bool(
            self.cfg.get(WORK_IQ_ENABLED_KEY, False, type=bool)
        )
        self.work_iq_knowledge_source_name = (
            self.cfg.get(WORK_IQ_KNOWLEDGE_SOURCE_NAME_KEY, "") or ""
        ).strip()
        # Fabric IQ (Microsoft Fabric ontology) remote knowledge source.
        # Opt-in; requires a per-user OBO token (same
        # x-ms-query-source-authorization header used by Work IQ), so
        # anonymous / MI fallback is never used.
        self.fabric_iq_enabled = _as_bool(
            self.cfg.get(FABRIC_IQ_ENABLED_KEY, False, type=bool)
        )
        self.fabric_iq_knowledge_source_name = (
            self.cfg.get(FABRIC_IQ_KNOWLEDGE_SOURCE_NAME_KEY, "") or ""
        ).strip()
        # Fabric Data Agent remote knowledge source. Opt-in; requires a
        # per-user OBO token (same x-ms-query-source-authorization header
        # used by Work IQ / Fabric ontology), so anonymous / MI fallback is
        # never used.
        self.fabric_data_agent_enabled = _as_bool(
            self.cfg.get(FABRIC_DATA_AGENT_ENABLED_KEY, False, type=bool)
        )
        self.fabric_data_agent_knowledge_source_name = (
            self.cfg.get(FABRIC_DATA_AGENT_KNOWLEDGE_SOURCE_NAME_KEY, "") or ""
        ).strip()
        # SharePoint remote knowledge source (Copilot Retrieval API). Opt-in;
        # requires a per-user OBO token (same x-ms-query-source-authorization
        # header used by Work IQ / Fabric ontology), so anonymous / MI
        # fallback is never used.
        self.sharepoint_remote_enabled = _as_bool(
            self.cfg.get(SHAREPOINT_REMOTE_ENABLED_KEY, False, type=bool)
        )
        self.sharepoint_remote_knowledge_source_name = (
            self.cfg.get(SHAREPOINT_REMOTE_KNOWLEDGE_SOURCE_NAME_KEY, "") or ""
        ).strip()
        self.sharepoint_remote_filter_expression_add_on = (
            self.cfg.get(SHAREPOINT_REMOTE_FILTER_EXPRESSION_ADD_ON_KEY, "") or ""
        ).strip()
        self.credential = self.cfg.aiocredential

        # Shared aiohttp session - reuses TCP connections across calls.
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
    def _normalize_work_iq_reference(source: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """Map a Work IQ ``sourceData`` payload into ``{title, link, content}``.

        Work IQ references do not use the flat ``snippet`` / ``blob_url`` shape.
        Instead they carry:

        - ``attributions[]`` - one or more entries per reference. Each carries
          a ``seeMoreWebUrl`` (Outlook / SharePoint / Teams deep link) and,
          when available, a subject / filename we can use as the citation
          title.
        - ``extracts[]`` - extracted chunks with a ``text`` field. Concatenating
          the extracts gives the grounding content the LLM sees.

        Returns ``None`` when the payload has neither extracts nor attributions
        we can use, so the caller can skip empty Work IQ hits without falling
        through to the generic azureBlob / searchIndex path (which would
        otherwise emit a ``reference`` title and no link).
        """
        attributions = source.get("attributions") or []
        extracts = source.get("extracts") or []
        if not attributions and not extracts:
            return None

        extract_texts: List[str] = []
        for extract in extracts:
            if isinstance(extract, Mapping):
                text = extract.get("text")
                if text:
                    extract_texts.append(str(text))
        content = "\n\n".join(extract_texts).strip()
        if not content:
            return None

        link = ""
        title: Optional[str] = None
        for attribution in attributions:
            if not isinstance(attribution, Mapping):
                continue
            if not link:
                link = str(attribution.get("seeMoreWebUrl") or "")
            if not title:
                # Prefer human-friendly labels, fall back to filename.
                for key in ("subject", "name", "title", "filename", "fileName"):
                    value = attribution.get(key)
                    if value:
                        title = str(value)
                        break
            if link and title:
                break

        if not title and link:
            tail = link.rsplit("/", 1)[-1]
            title = tail.split("?", 1)[0] or None
        if not title:
            title = "reference"

        return {"title": title, "link": link, "content": content}

    @staticmethod
    def _normalize_fabric_iq_reference(source: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """Map a Fabric IQ ``sourceData`` payload into ``{title, link, content}``.

        Fabric IQ references carry a different shape from Work IQ:

        - ``fabricAnswer`` (optional) is a natural-language summary produced by
          the Fabric ontology retriever.
        - ``fabricRawData`` (optional) is the raw grounding evidence, typically
          CSV rows the LLM should be able to quote from verbatim.
        - ``workspaceId`` / ``ontologyId`` identify the Fabric artifact used.

        Both ``fabricAnswer`` and ``fabricRawData`` may be present; when both
        exist we concatenate them so the LLM sees the summary plus the raw
        rows. Returns ``None`` when neither field is present so the caller can
        skip empty hits rather than emit a bare ``reference`` title.

        Fabric IQ does not currently return a per-reference deep link, so
        ``link`` is left empty and the citation title falls back to the
        workspace / ontology identifiers.
        """
        answer = source.get("fabricAnswer")
        raw = source.get("fabricRawData")
        parts: List[str] = []
        if isinstance(answer, str) and answer.strip():
            parts.append(answer.strip())
        if isinstance(raw, str) and raw.strip():
            parts.append(raw.strip())
        if not parts:
            return None
        content = "\n\n".join(parts)

        workspace_id = str(source.get("workspaceId") or "").strip()
        ontology_id = str(source.get("ontologyId") or "").strip()
        if workspace_id and ontology_id:
            title = f"Fabric ontology {ontology_id} (workspace {workspace_id})"
        elif ontology_id:
            title = f"Fabric ontology {ontology_id}"
        elif workspace_id:
            title = f"Fabric workspace {workspace_id}"
        else:
            title = "Fabric ontology"

        return {"title": title, "link": "", "content": content}

    @staticmethod
    def _normalize_fabric_data_agent_reference(
        source: Dict[str, Any],
    ) -> Optional[Dict[str, str]]:
        """Map a Fabric Data Agent ``sourceData`` payload into ``{title, link, content}``.

        A Fabric Data Agent acts as a virtual analyst: it runs queries over
        Fabric data and returns answers, and (optionally) tabular / chart
        artifacts. The retrieve response typically exposes:

        - ``dataAgentAnswer`` (optional) - natural-language answer.
        - ``dataAgentRawData`` (optional) - raw grounding evidence, usually
          CSV / JSON rows the LLM should be able to quote from verbatim.
        - ``workspaceId`` / ``dataAgentId`` - identify the Fabric artifacts
          the agent ran against.

        Both ``dataAgentAnswer`` and ``dataAgentRawData`` may be present;
        when both exist we concatenate them so the LLM sees the summary plus
        the raw rows. Returns ``None`` when neither field is present so the
        caller can skip empty hits rather than emit a bare ``reference``
        title. Fabric Data Agent does not currently return a per-reference
        deep link, so ``link`` is left empty and the citation title falls
        back to the workspace / data agent identifiers.
        """
        answer = source.get("dataAgentAnswer")
        raw = source.get("dataAgentRawData")
        parts: List[str] = []
        if isinstance(answer, str) and answer.strip():
            parts.append(answer.strip())
        if isinstance(raw, str) and raw.strip():
            parts.append(raw.strip())
        if not parts:
            return None
        content = "\n\n".join(parts)

        workspace_id = str(source.get("workspaceId") or "").strip()
        data_agent_id = str(source.get("dataAgentId") or "").strip()
        if workspace_id and data_agent_id:
            title = f"Fabric data agent {data_agent_id} (workspace {workspace_id})"
        elif data_agent_id:
            title = f"Fabric data agent {data_agent_id}"
        elif workspace_id:
            title = f"Fabric workspace {workspace_id}"
        else:
            title = "Fabric data agent"

        return {"title": title, "link": "", "content": content}

    @staticmethod
    def _normalize_sharepoint_remote_reference(
        source: Dict[str, Any],
    ) -> Optional[Dict[str, str]]:
        """Map a remoteSharePoint ``sourceData`` payload into ``{title, link, content}``.

        SharePoint remote references come from the Copilot Retrieval API. A
        reference typically carries:

        - ``webUrl`` - deep link to the SharePoint item (file, page, list
          item), the natural value for the citation ``link``.
        - ``resourceMetadata`` - object with SharePoint properties such as
          ``Title`` / ``Name`` / ``FileName`` / ``LastModifiedTime``.
        - ``extracts`` - list of ``{text}`` snippets (same shape used by
          Work IQ) and/or a flat ``text`` / ``content`` / ``snippet`` field
          when the API returns pre-joined text.

        Returns ``None`` when neither snippet content nor extracts can be
        recovered so the caller can skip empty hits rather than emit a bare
        ``reference`` title.
        """
        parts: List[str] = []
        for extract in source.get("extracts") or []:
            if not isinstance(extract, Mapping):
                continue
            text = extract.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        if not parts:
            for key in ("text", "content", "snippet"):
                value = source.get(key)
                if isinstance(value, str) and value.strip():
                    parts.append(value.strip())
                    break
        if not parts:
            return None
        content = "\n\n".join(parts)

        link = ""
        web_url = source.get("webUrl")
        if isinstance(web_url, str):
            link = web_url.strip()

        title = ""
        metadata = source.get("resourceMetadata")
        if isinstance(metadata, Mapping):
            for key in ("Title", "title", "Name", "name", "FileName", "fileName"):
                value = metadata.get(key)
                if isinstance(value, str) and value.strip():
                    title = value.strip()
                    break
        if not title and link:
            tail = link.rsplit("/", 1)[-1]
            title = tail.split("?", 1)[0]
        if not title:
            title = "SharePoint item"

        return {"title": title, "link": link, "content": content}

    @classmethod
    def _normalize_references(cls, payload: Dict[str, Any]) -> List[Dict[str, str]]:
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
        - ``workIQ`` (Microsoft 365) sources return ``sourceData.attributions``
          and ``sourceData.extracts`` instead of a flat snippet field. See
          :meth:`_normalize_work_iq_reference`.
        - ``fabricOntology`` (Microsoft Fabric) sources return
          ``sourceData.fabricAnswer`` and/or ``sourceData.fabricRawData``
          alongside ``workspaceId`` / ``ontologyId``. See
          :meth:`_normalize_fabric_iq_reference`.
        - ``fabricDataAgent`` (Microsoft Fabric Data Agent) sources return
          ``sourceData.dataAgentAnswer`` and/or ``sourceData.dataAgentRawData``
          alongside ``workspaceId`` / ``dataAgentId``. See
          :meth:`_normalize_fabric_data_agent_reference`.
        - ``remoteSharePoint`` (Microsoft 365 Copilot Retrieval API) sources
          return ``sourceData.webUrl`` and ``sourceData.resourceMetadata``
          with SharePoint properties (Title, LastModifiedTime, ...) plus
          ``extracts`` snippets. See :meth:`_normalize_sharepoint_remote_reference`.

        We accept the union with explicit priority so the downstream
        ``{title, link, content}`` contract is identical across backends,
        and so a future small rename does not silently drop references.
        Title falls back to the link's filename when the source does not
        carry one (true for ``azureBlob`` references).
        """
        records: List[Dict[str, str]] = []
        for ref in payload.get("references", []) or []:
            source = ref.get("sourceData") or {}

            # Fabric Data Agent shape (fabricDataAgent): distinct fields,
            # no overlap with fabricOntology / workIQ.
            if source.get("dataAgentAnswer") or source.get("dataAgentRawData"):
                data_agent_record = cls._normalize_fabric_data_agent_reference(source)
                if data_agent_record is not None:
                    records.append(data_agent_record)
                continue

            # SharePoint remote shape (remoteSharePoint / Copilot Retrieval
            # API): keyed on webUrl + resourceMetadata. Check before Work IQ
            # because both may carry extracts, but only SharePoint remote
            # carries a webUrl deep link.
            if source.get("webUrl") or source.get("resourceMetadata"):
                sp_record = cls._normalize_sharepoint_remote_reference(source)
                if sp_record is not None:
                    records.append(sp_record)
                continue

            # Fabric IQ shape (fabricOntology): distinct fields, no
            # attributions/extracts overlap with Work IQ.
            if source.get("fabricAnswer") or source.get("fabricRawData"):
                fabric_record = cls._normalize_fabric_iq_reference(source)
                if fabric_record is not None:
                    records.append(fabric_record)
                continue

            # Work IQ shape (workIQ): attributions + extracts.
            if source.get("attributions") or source.get("extracts"):
                record = cls._normalize_work_iq_reference(source)
                if record is not None:
                    records.append(record)
                continue

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

    def _build_conversation_source_params(
        self,
        *,
        conversation_id: Optional[str],
        user_context: Optional[Mapping[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Build the sidecar searchIndex source that carries runtime uploads.

        Returns ``None`` when the hybrid file-upload sidecar does not apply:

        - the feature is disabled, or
        - the primary source is not the native ``azureBlob`` corpus (Pattern B
          already trims its single searchIndex source by conversationId), or
        - no conversation knowledge source name was provisioned.

        When it does apply, the source is a ``searchIndex`` knowledge source over
        the existing GPT-RAG index, always trimmed by a simple conversationId
        ``filterAddOn`` accepted by Foundry IQ. ``failOnError`` is ``false`` so
        an empty or missing upload index degrades gracefully to the shared corpus
        instead of failing the whole retrieve.
        """
        if not self.conversation_upload_enabled:
            return None
        if self.knowledge_source_kind != "azureBlob":
            # Pattern B (searchIndex primary) already applies a conversationId
            # filterAddOn on its own source; a second source would double-count.
            return None
        if not self.conversation_knowledge_source_name:
            logging.warning(
                "[FoundryIQClient] FOUNDRY_IQ_CONVERSATION_UPLOAD_ENABLED=true but "
                "FOUNDRY_IQ_CONVERSATION_KNOWLEDGE_SOURCE_NAME is empty; skipping the "
                "file-upload sidecar source."
            )
            return None
        cid = (conversation_id or "").strip()
        if not cid:
            logging.warning(
                "[FoundryIQClient] FOUNDRY_IQ_CONVERSATION_UPLOAD_ENABLED=true but "
                "conversation_id is empty; skipping the file-upload sidecar source."
            )
            return None
        if self.api_version != DEFAULT_FOUNDRY_IQ_API_VERSION:
            raise ValueError(
                "Foundry IQ conversation-upload filterAddOn requires "
                f"{DEFAULT_FOUNDRY_IQ_API_VERSION}; current "
                f"FOUNDRY_IQ_API_VERSION={self.api_version}."
            )
        params: Dict[str, Any] = {
            "knowledgeSourceName": self.conversation_knowledge_source_name,
            "kind": "searchIndex",
            "includeReferences": True,
            "includeReferenceSourceData": True,
            "failOnError": False,
            "filterAddOn": build_conversation_upload_filter_add_on(cid),
        }
        logging.info(
            "[FoundryIQClient][Upload] Adding conversation-upload source %s "
            "(conversation_scoped filterAddOn)",
            self.conversation_knowledge_source_name,
        )
        return params

    def _build_work_iq_source_params(
        self, *, obo_token: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """Build the Work IQ (Microsoft 365) knowledge source entry.

        Returns ``None`` when Work IQ does not apply:

        - the feature is disabled, or
        - no knowledge source name was provisioned, or
        - no per-user OBO token is available (remote kinds strictly require
          impersonation - MI fallback is never used).

        Work IQ enforces ACL natively via the M365 user token, so no
        ``filterAddOn`` is emitted.
        """
        if not self.work_iq_enabled:
            return None
        if not self.work_iq_knowledge_source_name:
            logging.warning(
                "[FoundryIQClient] WORK_IQ_ENABLED=true but "
                "WORK_IQ_KNOWLEDGE_SOURCE_NAME is empty; skipping Work IQ source."
            )
            return None
        if not obo_token:
            logging.warning(
                "[FoundryIQClient] WORK_IQ_ENABLED=true but no OBO token "
                "(x-ms-query-source-authorization) available; skipping Work IQ "
                "source. Managed-identity fallback is never used for remote "
                "knowledge source kinds."
            )
            return None

        logging.info(
            "[FoundryIQClient][WorkIQ] Adding workIQ source %s (ACL native, no filterAddOn)",
            self.work_iq_knowledge_source_name,
        )
        return {
            "knowledgeSourceName": self.work_iq_knowledge_source_name,
            "kind": "workIQ",
            "includeReferences": True,
            "includeReferenceSourceData": True,
        }

    def _build_fabric_iq_source_params(
        self, *, obo_token: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """Build the Fabric IQ (Microsoft Fabric ontology) knowledge source entry.

        Returns ``None`` when Fabric IQ does not apply:

        - the feature is disabled, or
        - no knowledge source name was provisioned, or
        - no per-user OBO token is available (remote kinds strictly require
          impersonation; MI fallback is never used).

        Fabric IQ enforces ACL natively via the forwarded per-user token
        (workspace / ontology / underlying-item permissions), so no
        ``filterAddOn`` is emitted. The fabricOntology KS itself carries the
        workspaceId and ontologyId at registration time; the retrieve-time
        entry only needs the knowledge source name and reference flags.
        """
        if not self.fabric_iq_enabled:
            return None
        if not self.fabric_iq_knowledge_source_name:
            logging.warning(
                "[FoundryIQClient] FABRIC_IQ_ENABLED=true but "
                "FABRIC_IQ_KNOWLEDGE_SOURCE_NAME is empty; skipping Fabric IQ source."
            )
            return None
        if not obo_token:
            logging.warning(
                "[FoundryIQClient] FABRIC_IQ_ENABLED=true but no OBO token "
                "(x-ms-query-source-authorization) available; skipping Fabric IQ "
                "source. Managed-identity fallback is never used for remote "
                "knowledge source kinds."
            )
            return None

        logging.info(
            "[FoundryIQClient][FabricIQ] Adding fabricOntology source %s (ACL native, no filterAddOn)",
            self.fabric_iq_knowledge_source_name,
        )
        return {
            "knowledgeSourceName": self.fabric_iq_knowledge_source_name,
            "kind": "fabricOntology",
            "includeReferences": True,
            "includeReferenceSourceData": True,
        }

    def _build_fabric_data_agent_source_params(
        self, *, obo_token: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """Build the Fabric Data Agent knowledge source entry.

        Returns ``None`` when Fabric Data Agent does not apply:

        - the feature is disabled, or
        - no knowledge source name was provisioned, or
        - no per-user OBO token is available (remote kinds strictly require
          impersonation; MI fallback is never used).

        Fabric Data Agent enforces ACL natively via the forwarded per-user
        token (workspace / data agent / underlying-item permissions), so no
        ``filterAddOn`` is emitted. The fabricDataAgent KS itself carries
        the workspaceId and dataAgentId at registration time; the
        retrieve-time entry only needs the knowledge source name and
        reference flags.
        """
        if not self.fabric_data_agent_enabled:
            return None
        if not self.fabric_data_agent_knowledge_source_name:
            logging.warning(
                "[FoundryIQClient] FABRIC_DATA_AGENT_ENABLED=true but "
                "FABRIC_DATA_AGENT_KNOWLEDGE_SOURCE_NAME is empty; skipping "
                "Fabric Data Agent source."
            )
            return None
        if not obo_token:
            logging.warning(
                "[FoundryIQClient] FABRIC_DATA_AGENT_ENABLED=true but no OBO "
                "token (x-ms-query-source-authorization) available; skipping "
                "Fabric Data Agent source. Managed-identity fallback is never "
                "used for remote knowledge source kinds."
            )
            return None

        logging.info(
            "[FoundryIQClient][FabricDataAgent] Adding fabricDataAgent source "
            "%s (ACL native, no filterAddOn)",
            self.fabric_data_agent_knowledge_source_name,
        )
        return {
            "knowledgeSourceName": self.fabric_data_agent_knowledge_source_name,
            "kind": "fabricDataAgent",
            "includeReferences": True,
            "includeReferenceSourceData": True,
        }

    def _build_sharepoint_remote_source_params(
        self, *, obo_token: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """Build the SharePoint remote (Copilot Retrieval API) KS entry.

        Returns ``None`` when SharePoint remote does not apply:

        - the feature is disabled, or
        - no knowledge source name was provisioned, or
        - no per-user OBO token is available (remote kinds strictly require
          impersonation; MI fallback is never used).

        SharePoint remote enforces ACL natively via the forwarded per-user
        token (item-level M365 permissions), so no ``filterAddOn`` is
        emitted. An optional retrieval-time KQL scope can be forwarded via
        ``filterExpressionAddOn`` when configured.
        """
        if not self.sharepoint_remote_enabled:
            return None
        if not self.sharepoint_remote_knowledge_source_name:
            logging.warning(
                "[FoundryIQClient] SHAREPOINT_REMOTE_ENABLED=true but "
                "SHAREPOINT_REMOTE_KNOWLEDGE_SOURCE_NAME is empty; skipping "
                "SharePoint remote source."
            )
            return None
        if not obo_token:
            logging.warning(
                "[FoundryIQClient] SHAREPOINT_REMOTE_ENABLED=true but no OBO "
                "token (x-ms-query-source-authorization) available; skipping "
                "SharePoint remote source. Managed-identity fallback is "
                "never used for remote knowledge source kinds."
            )
            return None

        logging.info(
            "[FoundryIQClient][SharePointRemote] Adding remoteSharePoint "
            "source %s (ACL native, no filterAddOn)",
            self.sharepoint_remote_knowledge_source_name,
        )
        params: Dict[str, Any] = {
            "knowledgeSourceName": self.sharepoint_remote_knowledge_source_name,
            "kind": "remoteSharePoint",
            "includeReferences": True,
            "includeReferenceSourceData": True,
        }
        if self.sharepoint_remote_filter_expression_add_on:
            params["filterExpressionAddOn"] = (
                self.sharepoint_remote_filter_expression_add_on
            )
        return params

    def _remote_kinds_enabled(self) -> bool:
        """Return True when any remote (M365 / Fabric) knowledge source is on."""
        return (
            self.work_iq_enabled
            or self.fabric_iq_enabled
            or self.fabric_data_agent_enabled
            or self.sharepoint_remote_enabled
        )

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
            logging.info(
                "[FoundryIQClient][Trimming] x-ms-query-source-authorization=present "
                "token_audience=%s source=obo",
                _SEARCH_SCOPE,
            )
        elif self.forward_source_auth:
            # Anonymous/unauth chat path: forward the service MI Search-audience
            # token so Foundry IQ can evaluate RBAC-scope permission filters on
            # the bound knowledge source. The MI itself must hold the relevant
            # data-plane role on the storage container (or other source) for
            # the filter to admit documents.
            headers["x-ms-query-source-authorization"] = token
            logging.info(
                "[FoundryIQClient][Trimming] x-ms-query-source-authorization=present "
                "token_audience=%s source=managed_identity",
                _SEARCH_SCOPE,
            )
        else:
            logging.info(
                "[FoundryIQClient][Trimming] x-ms-query-source-authorization=absent "
                "reason=FOUNDRY_IQ_FORWARD_SOURCE_AUTH=false and no OBO token"
            )

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

        conversation_source_params = self._build_conversation_source_params(
            conversation_id=conversation_id,
            user_context=user_context,
        )
        if conversation_source_params:
            knowledge_source_params.append(conversation_source_params)

        work_iq_source_params = self._build_work_iq_source_params(obo_token=obo_token)
        if work_iq_source_params:
            knowledge_source_params.append(work_iq_source_params)

        fabric_iq_source_params = self._build_fabric_iq_source_params(obo_token=obo_token)
        if fabric_iq_source_params:
            knowledge_source_params.append(fabric_iq_source_params)

        fabric_data_agent_source_params = self._build_fabric_data_agent_source_params(
            obo_token=obo_token
        )
        if fabric_data_agent_source_params:
            knowledge_source_params.append(fabric_data_agent_source_params)

        sharepoint_remote_source_params = self._build_sharepoint_remote_source_params(
            obo_token=obo_token
        )
        if sharepoint_remote_source_params:
            knowledge_source_params.append(sharepoint_remote_source_params)

        if knowledge_source_params:
            body["knowledgeSourceParams"] = knowledge_source_params

        # Remote kinds (workIQ, future fabric*) can take 40-60s to fan out.
        # Emit the runtime ceiling only when a remote kind is enabled so the
        # default Pattern A / Pattern B request body stays byte-identical.
        if self._remote_kinds_enabled():
            body["maxRuntimeInSeconds"] = self.max_runtime_seconds

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
