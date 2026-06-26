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
        "FOUNDRY_IQ_FILTER_ADD_ON_ENABLED": False,
        "FOUNDRY_IQ_SECURITY_FIELD_NAME": "metadata_security_id",
        "FOUNDRY_IQ_MAX_OUTPUT_DOCUMENTS": None,
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
    # Minimal retrieve body.
    assert session.captured["json"]["messages"][0]["content"][0]["text"] == "hello"

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
async def test_retrieve_adds_pattern_b_filter_add_on_when_enabled():
    client, session = _build_client(
        _SAMPLE_PAYLOAD,
        config_overrides={
            "FOUNDRY_IQ_KNOWLEDGE_SOURCE_NAME": "ragindex-ks",
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
            "FOUNDRY_IQ_FILTER_ADD_ON_ENABLED": True,
        },
    )

    with pytest.raises(ValueError, match="filterAddOn requires 2026-05-01-preview"):
        await client.retrieve("hello")


@pytest.mark.asyncio
async def test_retrieve_omits_obo_header_when_no_token():
    client, session = _build_client(_SAMPLE_PAYLOAD)
    await client.retrieve("hello")
    assert "x-ms-query-source-authorization" not in session.captured["headers"]


@pytest.mark.asyncio
async def test_retrieve_raises_on_http_error():
    client, _ = _build_client({"error": "boom"}, status=403)
    with pytest.raises(RuntimeError):
        await client.retrieve("hello")


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
