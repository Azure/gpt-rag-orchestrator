"""Tests for the SharePoint Indexed (indexedSharePoint) knowledge source path.

Covers the preview-flagged ``indexedSharePoint`` server-side knowledge source
wired into :mod:`connectors.foundry_iq`:

- ``indexedSharePoint`` is NOT a remote kind. It reads a pre-built AI Search
  index, so it never triggers ``maxRuntimeInSeconds`` and never requires an
  OBO token at retrieve time. Auth is baked into the KS connectionString via
  a Federated Identity Credential (managed identity + Graph Sites.Selected)
  at registration.
- The knowledge source is appended only when
  ``SHAREPOINT_INDEXED_ENABLED`` is true and
  ``SHAREPOINT_INDEXED_KNOWLEDGE_SOURCE_NAME`` is set.
- The ``indexedSharePoint`` entry never carries a ``filterAddOn``.
- ``_normalize_references`` extracts a SharePoint deep link and a
  human-readable title from the SharePoint-oriented sourceData fields.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fakes (kept in-file so this module is self-contained; matches the style of
# tests/test_foundry_iq_work_iq.py).
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status = status

    async def text(self):
        import json
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
        "FOUNDRY_IQ_MAX_RUNTIME_SECONDS": 120,
        "WORK_IQ_ENABLED": False,
        "WORK_IQ_KNOWLEDGE_SOURCE_NAME": "",
        "FABRIC_IQ_ENABLED": False,
        "FABRIC_IQ_KNOWLEDGE_SOURCE_NAME": "",
        "FABRIC_DATA_AGENT_ENABLED": False,
        "FABRIC_DATA_AGENT_KNOWLEDGE_SOURCE_NAME": "",
        "SHAREPOINT_INDEXED_ENABLED": False,
        "SHAREPOINT_INDEXED_KNOWLEDGE_SOURCE_NAME": "",
        "SHAREPOINT_INDEXED_INDEX_NAME": "",
        "SHAREPOINT_INDEXED_SITE_URL": "",
        "SHAREPOINT_INDEXED_TENANT_ID": "",
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


_SAMPLE_PAYLOAD = {"references": []}


# ---------------------------------------------------------------------------
# Config wiring
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sharepoint_indexed_not_emitted_when_disabled():
    """Disabled feature must not append an indexedSharePoint source."""
    client, session = _build_client(
        _SAMPLE_PAYLOAD,
        config_overrides={
            "FOUNDRY_IQ_KNOWLEDGE_SOURCE_NAME": "documents-blob-ks",
            "SHAREPOINT_INDEXED_ENABLED": False,
            "SHAREPOINT_INDEXED_KNOWLEDGE_SOURCE_NAME": "spo-idx-ks",
        },
    )
    await client.retrieve("hello")
    sources = session.captured["json"].get("knowledgeSourceParams", [])
    assert not any(s.get("kind") == "indexedSharePoint" for s in sources)


@pytest.mark.asyncio
async def test_sharepoint_indexed_not_emitted_when_name_missing():
    """Enabling without a KS name must warn and skip, not fail."""
    client, session = _build_client(
        _SAMPLE_PAYLOAD,
        config_overrides={
            "SHAREPOINT_INDEXED_ENABLED": True,
            "SHAREPOINT_INDEXED_KNOWLEDGE_SOURCE_NAME": "",
        },
    )
    await client.retrieve("hello")
    sources = session.captured["json"].get("knowledgeSourceParams", [])
    assert not any(s.get("kind") == "indexedSharePoint" for s in sources)


@pytest.mark.asyncio
async def test_sharepoint_indexed_emitted_without_obo():
    """indexedSharePoint must emit without an OBO token: it is a server-side KS."""
    client, session = _build_client(
        _SAMPLE_PAYLOAD,
        config_overrides={
            "SHAREPOINT_INDEXED_ENABLED": True,
            "SHAREPOINT_INDEXED_KNOWLEDGE_SOURCE_NAME": "spo-idx-ks",
        },
    )
    await client.retrieve("hello")
    sources = session.captured["json"]["knowledgeSourceParams"]
    spo = [s for s in sources if s.get("kind") == "indexedSharePoint"]
    assert spo == [
        {
            "knowledgeSourceName": "spo-idx-ks",
            "kind": "indexedSharePoint",
            "includeReferences": True,
            "includeReferenceSourceData": True,
        }
    ]


@pytest.mark.asyncio
async def test_sharepoint_indexed_emitted_with_obo_too():
    """OBO token is optional for indexedSharePoint; presence must not block emission."""
    client, session = _build_client(
        _SAMPLE_PAYLOAD,
        config_overrides={
            "SHAREPOINT_INDEXED_ENABLED": True,
            "SHAREPOINT_INDEXED_KNOWLEDGE_SOURCE_NAME": "spo-idx-ks",
        },
    )
    await client.retrieve("hello", obo_token="user-obo")
    sources = session.captured["json"]["knowledgeSourceParams"]
    assert any(s.get("kind") == "indexedSharePoint" for s in sources)


@pytest.mark.asyncio
async def test_sharepoint_indexed_never_carries_filter_add_on():
    """SharePoint Indexed does not participate in Pattern B filterAddOn trimming."""
    client, session = _build_client(
        _SAMPLE_PAYLOAD,
        config_overrides={
            "SHAREPOINT_INDEXED_ENABLED": True,
            "SHAREPOINT_INDEXED_KNOWLEDGE_SOURCE_NAME": "spo-idx-ks",
        },
    )
    await client.retrieve("hello", conversation_id="conv-1")
    sources = session.captured["json"]["knowledgeSourceParams"]
    spo = next(s for s in sources if s.get("kind") == "indexedSharePoint")
    assert "filterAddOn" not in spo


@pytest.mark.asyncio
async def test_sharepoint_indexed_does_not_trigger_max_runtime():
    """indexedSharePoint reads a local index; it must not bump maxRuntimeInSeconds."""
    client, session = _build_client(
        _SAMPLE_PAYLOAD,
        config_overrides={
            "SHAREPOINT_INDEXED_ENABLED": True,
            "SHAREPOINT_INDEXED_KNOWLEDGE_SOURCE_NAME": "spo-idx-ks",
        },
    )
    await client.retrieve("hello")
    assert "maxRuntimeInSeconds" not in session.captured["json"]


# ---------------------------------------------------------------------------
# _normalize_references shape mapping
# ---------------------------------------------------------------------------

def test_normalize_sharepoint_indexed_empty_content_returns_none():
    """No usable content in sourceData yields None, not a bare reference."""
    from connectors import foundry_iq

    record = foundry_iq.FoundryIQClient._normalize_sharepoint_indexed_reference({
        "webUrl": "https://contoso.sharepoint.com/sites/knowledge/policy.docx",
    })
    assert record is None


def test_normalize_sharepoint_indexed_extracts_webUrl_and_title():
    """SharePoint deep link + human title must be preferred over blob URLs."""
    from connectors import foundry_iq

    record = foundry_iq.FoundryIQClient._normalize_sharepoint_indexed_reference({
        "content": "Policy body text.",
        "webUrl": "https://contoso.sharepoint.com/sites/knowledge/policy.docx",
        "title": "Corporate Policy",
        "blob_url": "https://acct.blob.core.windows.net/idx/policy.docx",
    })
    assert record == {
        "title": "Corporate Policy",
        "link": "https://contoso.sharepoint.com/sites/knowledge/policy.docx",
        "content": "Policy body text.",
    }


def test_normalize_sharepoint_indexed_falls_back_to_link_tail():
    """When no title is projected, derive one from the SharePoint URL tail."""
    from connectors import foundry_iq

    record = foundry_iq.FoundryIQClient._normalize_sharepoint_indexed_reference({
        "content": "Doc body.",
        "webUrl": "https://contoso.sharepoint.com/sites/knowledge/deck.pptx?web=1",
    })
    assert record["title"] == "deck.pptx"
    assert record["link"] == (
        "https://contoso.sharepoint.com/sites/knowledge/deck.pptx?web=1"
    )


def test_normalize_references_routes_multiple_sharepoint_hits():
    """Multi-result response with SharePoint-shaped sourceData must be routed
    to the indexedSharePoint normalizer, not the generic azureBlob path."""
    from connectors import foundry_iq

    payload = {
        "references": [
            {
                "sourceData": {
                    "content": "First doc body.",
                    "webUrl": "https://contoso.sharepoint.com/sites/kb/first.docx",
                    "title": "First doc",
                }
            },
            {
                "sourceData": {
                    "content": "Second doc body.",
                    "siteUrl": "https://contoso.sharepoint.com/sites/kb",
                    "driveItemId": "01ABCDEF234567890",
                    "name": "Second doc",
                }
            },
        ]
    }
    records = foundry_iq.FoundryIQClient._normalize_references(payload)
    assert records == [
        {
            "title": "First doc",
            "link": "https://contoso.sharepoint.com/sites/kb/first.docx",
            "content": "First doc body.",
        },
        {
            "title": "Second doc",
            "link": "https://contoso.sharepoint.com/sites/kb",
            "content": "Second doc body.",
        },
    ]


def test_normalize_references_generic_hits_still_flow_to_generic_path():
    """Generic searchIndex hits without SharePoint fields must not be
    hijacked by the indexedSharePoint branch."""
    from connectors import foundry_iq

    payload = {
        "references": [
            {
                "sourceData": {
                    "snippet": "Generic blob snippet.",
                    "blob_url": "https://acct.blob.core.windows.net/idx/plain.pdf",
                    "title": "Plain PDF",
                }
            }
        ]
    }
    records = foundry_iq.FoundryIQClient._normalize_references(payload)
    assert records == [
        {
            "title": "Plain PDF",
            "link": "https://acct.blob.core.windows.net/idx/plain.pdf",
            "content": "Generic blob snippet.",
        }
    ]
