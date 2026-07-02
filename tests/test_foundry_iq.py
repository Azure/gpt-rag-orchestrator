"""Tests for the Foundry IQ retrieval client, provider, and connector branch.

Covers Azure/GPT-RAG#526 Step 1:

- :class:`FoundryIQClient.retrieve` forwards the OBO token in the
  ``x-ms-query-source-authorization`` header and normalizes the ``references``
  response into the shared ``{title, link, content}`` contract.
- :class:`FoundryIQContextProvider.invoking` returns context that is
  byte-identical in shape to the Azure AI Search provider and forwards the OBO
  token to the client.
- ``SearchClient.search_knowledge_base`` foundry_iq branch returns the same
  citation JSON contract that ``single_agent_rag`` consumes.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import json

import pytest
from agent_framework import ChatMessage, Role

from strategies.context_shaping import build_context_text, format_context_part


# ---------------------------------------------------------------------------
# Helpers: a fake aiohttp session whose post() captures headers/body.
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status = status

    async def text(self):
        return json.dumps(self._payload)

    async def json(self):
        return self._payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    def __init__(self, payload, status=200):
        self._payload = payload
        self._status = status
        self.captured = {}

    def post(self, url, headers=None, json=None):  # noqa: A002
        self.captured = {"url": url, "headers": headers, "json": json}
        return _FakeResponse(self._payload, self._status)


def _build_client(payload, status=200, config_overrides=None):
    """Build a FoundryIQClient with a stubbed config and fake session."""
    from connectors import foundry_iq

    values = {
        "KNOWLEDGE_BASE_ENDPOINT": "https://fake-search.search.windows.net",
        "SEARCH_SERVICE_QUERY_ENDPOINT": "https://fake-search.search.windows.net",
        "KNOWLEDGE_BASE_NAME": "kb-test",
        "FOUNDRY_IQ_API_VERSION": "2026-05-01-preview",
        "FOUNDRY_IQ_KNOWLEDGE_SOURCE_NAME": "",
        "FOUNDRY_IQ_KNOWLEDGE_SOURCE_KIND": "",
        "FOUNDRY_IQ_PATTERN": "",
        "FOUNDRY_IQ_FILTER_ADD_ON_ENABLED": False,
        "FOUNDRY_IQ_SECURITY_FIELD_NAME": "metadata_security_id",
        "FOUNDRY_IQ_MAX_OUTPUT_DOCUMENTS": None,
        "FOUNDRY_IQ_FORWARD_SOURCE_AUTH": True,
        "FOUNDRY_IQ_CONVERSATION_UPLOAD_ENABLED": False,
        "FOUNDRY_IQ_CONVERSATION_KNOWLEDGE_SOURCE_NAME": "",
    }
    values.update(config_overrides or {})
    cfg = MagicMock()
    cfg.get.side_effect = lambda key, default=None, type=str: values.get(key, default)  # noqa: A002
    cfg.aiocredential = MagicMock()
    cfg.aiocredential.get_token = AsyncMock(return_value=SimpleNamespace(token="svc-token"))

    with patch("connectors.foundry_iq.get_config", return_value=cfg):
        client = foundry_iq.FoundryIQClient()
    session = _FakeSession(payload, status)
    client._get_session = AsyncMock(return_value=session)
    return client, session


_SAMPLE_PAYLOAD = {
    "references": [
        {
            "docKey": "doc-1",
            "sourceData": {
                "id": "1",
                "title": "Doc One",
                "filepath": "doc1.pdf",
                "content": "First content",
            },
        },
        {
            "docKey": "doc-2",
            "sourceData": {
                "id": "2",
                "title": "Doc Two",
                "url": "https://example/doc2",
                "content": "Second content",
            },
        },
        {
            # Empty content is skipped.
            "docKey": "doc-3",
            "sourceData": {"id": "3", "title": "Empty", "content": ""},
        },
    ]
}


# ---------------------------------------------------------------------------
# FoundryIQClient.retrieve
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_retrieve_forwards_obo_header_and_normalizes():
    client, session = _build_client(_SAMPLE_PAYLOAD)
    records = await client.retrieve("hello", obo_token="user-obo-token")

    # OBO token forwarded in the dedicated per-user security header.
    assert session.captured["headers"]["x-ms-query-source-authorization"] == "user-obo-token"
    assert session.captured["headers"]["Authorization"] == "Bearer svc-token"
    # Endpoint shape and pinned preview api-version.
    assert "/knowledgebases/kb-test/retrieve" in session.captured["url"]
    assert "api-version=2026-05-01-preview" in session.captured["url"]
    # Minimal-reasoning knowledge bases require explicit intents.
    assert session.captured["json"]["intents"] == [{"search": "hello", "type": "semantic"}]

    # Normalized to {title, link, content}; empty-content reference dropped.
    assert records == [
        {"title": "Doc One", "link": "doc1.pdf", "content": "First content"},
        {"title": "Doc Two", "link": "https://example/doc2", "content": "Second content"},
    ]


def test_pattern_b_filter_add_on_uses_security_fields_and_conversation_scope():
    from connectors.foundry_iq import build_pattern_b_filter_add_on

    filter_add_on = build_pattern_b_filter_add_on(
        conversation_id="conv-1",
        user_context={
            "principal_id": "user-1",
            "principal_name": "ada@example.com",
            "groups": ["group-a", "group-b"],
        },
    )

    assert "metadata_security_id/any(g:search.in(g, 'user-1,ada@example.com,group-a,group-b'))" in filter_add_on
    assert "or not metadata_security_id/any()" in filter_add_on
    assert "conversationId eq 'conv-1'" in filter_add_on
    assert "conversationId eq 'NaN'" in filter_add_on


@pytest.mark.asyncio
async def test_retrieve_uses_native_blob_knowledge_source_by_default():
    client, session = _build_client(
        _SAMPLE_PAYLOAD,
        config_overrides={"FOUNDRY_IQ_KNOWLEDGE_SOURCE_NAME": "documents-blob-ks"},
    )

    await client.retrieve("hello")

    assert session.captured["json"]["knowledgeSourceParams"] == [
        {
            "knowledgeSourceName": "documents-blob-ks",
            "kind": "azureBlob",
            "includeReferences": True,
            "includeReferenceSourceData": True,
        }
    ]


@pytest.mark.asyncio
async def test_retrieve_adds_pattern_b_filter_add_on_when_enabled():
    client, session = _build_client(
        _SAMPLE_PAYLOAD,
        config_overrides={
            "FOUNDRY_IQ_KNOWLEDGE_SOURCE_NAME": "ragindex-ks",
            "FOUNDRY_IQ_KNOWLEDGE_SOURCE_KIND": "searchIndex",
            "FOUNDRY_IQ_FILTER_ADD_ON_ENABLED": True,
            "FOUNDRY_IQ_MAX_OUTPUT_DOCUMENTS": 7,
        },
    )

    await client.retrieve(
        "hello",
        conversation_id="conv-1",
        user_context={"principal_id": "user-1"},
    )

    captured_body = session.captured["json"]
    assert captured_body["maxOutputDocuments"] == 7
    assert captured_body["knowledgeSourceParams"] == [
        {
            "knowledgeSourceName": "ragindex-ks",
            "kind": "searchIndex",
            "includeReferences": True,
            "includeReferenceSourceData": True,
            "filterAddOn": (
                "((metadata_security_id/any(g:search.in(g, 'user-1')) "
                "or not metadata_security_id/any())) and "
                "(conversationId eq 'conv-1' or (conversationId eq 'NaN' or conversationId eq null))"
            ),
        }
    ]


@pytest.mark.asyncio
async def test_pattern_b_filter_add_on_requires_preview_api():
    client, _ = _build_client(
        _SAMPLE_PAYLOAD,
        config_overrides={
            "FOUNDRY_IQ_API_VERSION": "2026-04-01",
            "FOUNDRY_IQ_KNOWLEDGE_SOURCE_NAME": "ragindex-ks",
            "FOUNDRY_IQ_KNOWLEDGE_SOURCE_KIND": "searchIndex",
            "FOUNDRY_IQ_FILTER_ADD_ON_ENABLED": True,
        },
    )

    with pytest.raises(ValueError, match="filterAddOn requires 2026-05-01-preview"):
        await client.retrieve("hello")


@pytest.mark.asyncio
async def test_retrieve_omits_obo_header_when_no_token():
    """When forwarding is disabled and no OBO token is supplied, the
    ``x-ms-query-source-authorization`` header must be omitted entirely."""
    client, session = _build_client(
        _SAMPLE_PAYLOAD,
        config_overrides={"FOUNDRY_IQ_FORWARD_SOURCE_AUTH": False},
    )
    await client.retrieve("hello")
    assert "x-ms-query-source-authorization" not in session.captured["headers"]


@pytest.mark.asyncio
async def test_retrieve_forwards_managed_identity_token_when_no_obo():
    """Anonymous chat path: the service MI Search-audience token must be
    forwarded as ``x-ms-query-source-authorization`` so RBAC-scoped permission
    filters on the bound knowledge source can be evaluated."""
    client, session = _build_client(_SAMPLE_PAYLOAD)
    await client.retrieve("hello")
    headers = session.captured["headers"]
    assert headers["Authorization"] == "Bearer svc-token"
    # MI token reused as the source-auth token when no OBO token is present.
    assert headers["x-ms-query-source-authorization"] == "svc-token"


@pytest.mark.asyncio
async def test_retrieve_raises_on_http_error():
    client, _ = _build_client({"error": "boom"}, status=403)
    with pytest.raises(RuntimeError):
        await client.retrieve("hello")


# ---------------------------------------------------------------------------
# Hybrid file-upload sidecar (Pattern A + UI upload)
# ---------------------------------------------------------------------------

def _conv_upload_overrides(**extra):
    base = {
        "FOUNDRY_IQ_KNOWLEDGE_SOURCE_NAME": "documents-blob-ks",
        "FOUNDRY_IQ_KNOWLEDGE_SOURCE_KIND": "azureBlob",
        "FOUNDRY_IQ_CONVERSATION_UPLOAD_ENABLED": True,
        "FOUNDRY_IQ_CONVERSATION_KNOWLEDGE_SOURCE_NAME": "ragindex-conv-ks",
    }
    base.update(extra)
    return base


@pytest.mark.asyncio
async def test_conversation_upload_adds_second_source_in_pattern_a():
    """Pattern A (azureBlob primary) + upload enabled: the retrieve body carries
    two knowledge sources. The native blob source is untouched (no filterAddOn);
    the sidecar searchIndex source is conversation-scoped and tolerant of an
    empty/missing upload index (failOnError=False)."""
    client, session = _build_client(
        _SAMPLE_PAYLOAD,
        config_overrides=_conv_upload_overrides(),
    )

    await client.retrieve(
        "hello",
        conversation_id="conv-1",
        user_context={"principal_id": "user-1"},
    )

    sources = session.captured["json"]["knowledgeSourceParams"]
    assert len(sources) == 2

    primary, sidecar = sources[0], sources[1]
    # Native blob corpus is never trimmed by the Pattern B filter.
    assert primary == {
        "knowledgeSourceName": "documents-blob-ks",
        "kind": "azureBlob",
        "includeReferences": True,
        "includeReferenceSourceData": True,
    }
    # Sidecar is a searchIndex source, degrades gracefully, and is scoped.
    assert sidecar["knowledgeSourceName"] == "ragindex-conv-ks"
    assert sidecar["kind"] == "searchIndex"
    assert sidecar["failOnError"] is False
    assert "conversationId eq 'conv-1'" in sidecar["filterAddOn"]
    assert "metadata_security_id/any(g:search.in(g, 'user-1'))" in sidecar["filterAddOn"]


@pytest.mark.asyncio
async def test_conversation_upload_sidecar_always_carries_filter_add_on():
    """CONTRACT: the runtime-upload sidecar source must never be queried without
    a filterAddOn, even when there is no conversation and no user context. A bare
    security-only filter (public-or-owned) is still applied, so uploads can never
    leak across users or conversations."""
    client, session = _build_client(
        _SAMPLE_PAYLOAD,
        config_overrides=_conv_upload_overrides(),
    )

    await client.retrieve("hello")

    sidecar = session.captured["json"]["knowledgeSourceParams"][1]
    assert sidecar["knowledgeSourceName"] == "ragindex-conv-ks"
    assert sidecar["filterAddOn"], "sidecar upload source must always be filtered"


@pytest.mark.asyncio
async def test_conversation_upload_skipped_for_pattern_b_primary():
    """Pattern B already scopes its single searchIndex source by conversationId;
    enabling the upload flag must not add a duplicate second source."""
    client, session = _build_client(
        _SAMPLE_PAYLOAD,
        config_overrides=_conv_upload_overrides(
            FOUNDRY_IQ_KNOWLEDGE_SOURCE_KIND="searchIndex",
            FOUNDRY_IQ_FILTER_ADD_ON_ENABLED=True,
        ),
    )

    await client.retrieve("hello", conversation_id="conv-1")

    sources = session.captured["json"]["knowledgeSourceParams"]
    assert len(sources) == 1
    assert sources[0]["kind"] == "searchIndex"


@pytest.mark.asyncio
async def test_conversation_upload_skipped_when_name_missing():
    """Upload enabled but no sidecar source name provisioned: skip gracefully
    (shared corpus only), never emit a nameless source."""
    client, session = _build_client(
        _SAMPLE_PAYLOAD,
        config_overrides=_conv_upload_overrides(
            FOUNDRY_IQ_CONVERSATION_KNOWLEDGE_SOURCE_NAME="",
        ),
    )

    await client.retrieve("hello", conversation_id="conv-1")

    sources = session.captured["json"]["knowledgeSourceParams"]
    assert len(sources) == 1
    assert sources[0]["knowledgeSourceName"] == "documents-blob-ks"


@pytest.mark.asyncio
async def test_conversation_upload_disabled_by_default():
    """Default config (flag off) keeps the single-source Pattern A behavior."""
    client, session = _build_client(
        _SAMPLE_PAYLOAD,
        config_overrides={"FOUNDRY_IQ_KNOWLEDGE_SOURCE_NAME": "documents-blob-ks"},
    )

    await client.retrieve("hello", conversation_id="conv-1")

    assert len(session.captured["json"]["knowledgeSourceParams"]) == 1


@pytest.mark.asyncio
async def test_conversation_upload_requires_preview_api():
    """The sidecar filterAddOn is a preview capability; a non-preview api-version
    must fail loudly rather than silently drop conversation scoping."""
    client, _ = _build_client(
        _SAMPLE_PAYLOAD,
        config_overrides=_conv_upload_overrides(
            FOUNDRY_IQ_API_VERSION="2026-04-01",
        ),
    )

    with pytest.raises(ValueError, match="conversation-upload filterAddOn requires"):
        await client.retrieve("hello", conversation_id="conv-1")


@pytest.mark.asyncio
async def test_retrieve_normalizes_azure_blob_snippet_shape():
    """Real azureBlob payloads (both minimal and standard extraction modes)
    return ``sourceData.snippet`` and ``sourceData.blob_url`` and do not
    include ``title``; the parser must map those to the shared contract."""
    payload = {
        "references": [
            {
                "type": "azureBlob",
                "id": "0",
                "sourceData": {
                    "uid": "abc_pages_0",
                    "blob_url": "https://acct.blob.core.windows.net/documents/vw-fuel-system.pdf",
                    "snippet": "# Fuel System\n\nThe fuel system, as covered...",
                },
                "blobUrl": "https://acct.blob.core.windows.net/documents/vw-fuel-system.pdf",
            },
            {
                "type": "azureBlob",
                "id": "1",
                "sourceData": {
                    "uid": "def_pages_0",
                    "blob_url": "https://acct.blob.core.windows.net/documents/foundry-iq-validation.txt",
                    "snippet": "Foundry IQ validation marker.",
                },
            },
            {
                # Empty snippet is dropped (matches the empty-content rule).
                "sourceData": {"blob_url": "https://x/empty.txt", "snippet": ""},
            },
        ]
    }
    client, _ = _build_client(payload)
    records = await client.retrieve("fuel system")
    assert records == [
        {
            "title": "vw-fuel-system.pdf",
            "link": "https://acct.blob.core.windows.net/documents/vw-fuel-system.pdf",
            "content": "# Fuel System\n\nThe fuel system, as covered...",
        },
        {
            "title": "foundry-iq-validation.txt",
            "link": "https://acct.blob.core.windows.net/documents/foundry-iq-validation.txt",
            "content": "Foundry IQ validation marker.",
        },
    ]


# ---------------------------------------------------------------------------
# FoundryIQContextProvider.invoking
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_provider_invoking_shape_and_forwards_obo():
    from strategies.foundry_iq_context_provider import FoundryIQContextProvider

    records = [
        {"title": "Doc One", "link": "doc1.pdf", "content": "First content"},
        {"title": "Doc Two", "link": "https://example/doc2", "content": "Second content"},
    ]
    fake_client = MagicMock()
    fake_client.retrieve = AsyncMock(return_value=records)

    async def _get_obo():
        return "user-obo-token"

    provider = FoundryIQContextProvider(
        conversation_id="conv-1", top_k=3, get_obo_token=_get_obo
    )

    with patch(
        "strategies.foundry_iq_context_provider.get_foundry_iq_client",
        return_value=fake_client,
    ):
        ctx = await provider.invoking([ChatMessage(role=Role.USER, text="what is x?")])

    # OBO token forwarded to the client.
    fake_client.retrieve.assert_awaited_once()
    _, kwargs = fake_client.retrieve.call_args
    assert kwargs["obo_token"] == "user-obo-token"
    assert kwargs["conversation_id"] == "conv-1"
    assert kwargs["user_context"] == {}

    # Context shape is byte-identical to the shared shaping helpers.
    expected_parts = [format_context_part(r["title"], r["link"], r["content"]) for r in records]
    expected_text = build_context_text(expected_parts)
    assert len(ctx.messages) == 1
    assert ctx.messages[0].role == Role.SYSTEM
    assert ctx.messages[0].text == expected_text


@pytest.mark.asyncio
async def test_provider_no_user_message_returns_empty_context():
    from strategies.foundry_iq_context_provider import FoundryIQContextProvider

    provider = FoundryIQContextProvider()
    ctx = await provider.invoking([ChatMessage(role=Role.SYSTEM, text="system only")])
    assert not ctx.messages


@pytest.mark.asyncio
async def test_provider_retrieval_failure_returns_empty_context():
    from strategies.foundry_iq_context_provider import FoundryIQContextProvider

    fake_client = MagicMock()
    fake_client.retrieve = AsyncMock(side_effect=RuntimeError("boom"))
    provider = FoundryIQContextProvider()
    with patch(
        "strategies.foundry_iq_context_provider.get_foundry_iq_client",
        return_value=fake_client,
    ):
        ctx = await provider.invoking([ChatMessage(role=Role.USER, text="q")])
    assert not ctx.messages


# ---------------------------------------------------------------------------
# SearchClient.search_knowledge_base foundry_iq branch
# ---------------------------------------------------------------------------

@pytest.fixture()
def foundry_search_client(patch_dependencies, mock_config):
    from connectors.search import SearchClient

    mock_config.get.side_effect = lambda key, default=None, type=str: {  # noqa: A002
        "SEARCH_SERVICE_QUERY_ENDPOINT": "https://fake-search.search.windows.net",
        "SEARCH_RAG_INDEX_NAME": "ragindex",
        "ALLOW_ANONYMOUS": "true",
    }.get(key, default)
    with patch("connectors.search.get_config", return_value=mock_config):
        client = SearchClient()
    client._get_search_user_token_for_trimming = AsyncMock(return_value="user-obo-token")
    return client


@pytest.mark.asyncio
async def test_search_knowledge_base_foundry_iq_branch_contract(foundry_search_client):
    records = [
        {"title": "Doc One", "link": "doc1.pdf", "content": "First content"},
        {"title": "Doc Two", "link": "https://example/doc2", "content": "Second content"},
    ]
    fake_client = MagicMock()
    fake_client.retrieve = AsyncMock(return_value=records)

    with (
        patch("connectors.search.get_retrieval_backend", return_value="foundry_iq"),
        patch("connectors.search.get_foundry_iq_client", return_value=fake_client),
    ):
        result = await foundry_search_client.search_knowledge_base("hello")

    parsed = json.loads(result)
    assert parsed["query"] == "hello"
    assert parsed["results"] == [
        {"title": "Doc One", "link": "doc1.pdf", "content": "First content"},
        {"title": "Doc Two", "link": "https://example/doc2", "content": "Second content"},
    ]
    # OBO token acquired and forwarded the same way as the AI Search path.
    _, kwargs = fake_client.retrieve.call_args
    assert kwargs["obo_token"] == "user-obo-token"


@pytest.mark.asyncio
async def test_search_knowledge_base_default_backend_does_not_call_foundry(foundry_search_client):
    fake_client = MagicMock()
    fake_client.retrieve = AsyncMock()
    foundry_search_client.search = AsyncMock(return_value=[])

    with (
        patch("connectors.search.get_retrieval_backend", return_value="ai_search"),
        patch("connectors.search.get_foundry_iq_client", return_value=fake_client),
    ):
        await foundry_search_client.search_knowledge_base("hello")

    fake_client.retrieve.assert_not_called()
    foundry_search_client.search.assert_awaited()
