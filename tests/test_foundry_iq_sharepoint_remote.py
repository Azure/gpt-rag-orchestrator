"""Tests for the SharePoint remote (Copilot Retrieval API) knowledge source.

Covers the 5th Foundry IQ knowledge source kind, ``remoteSharePoint``:

- ``maxRuntimeInSeconds`` is emitted when the SharePoint remote source is
  enabled (same contract as the other remote kinds).
- ``remoteSharePoint`` knowledge source params are appended only when
  ``SHAREPOINT_REMOTE_ENABLED`` is true and
  ``SHAREPOINT_REMOTE_KNOWLEDGE_SOURCE_NAME`` is set.
- Optional ``filterExpressionAddOn`` (KQL) is forwarded verbatim when set,
  and omitted when empty.
- ``remoteSharePoint`` params never carry a ``filterAddOn``. ACL is
  enforced natively by SharePoint via the forwarded per-user token.
- The MI fallback controlled by ``FOUNDRY_IQ_FORWARD_SOURCE_AUTH`` still
  applies to local ``azureBlob`` / ``searchIndex`` sources and never
  substitutes for OBO on SharePoint remote.
- ``_normalize_references`` maps the ``webUrl`` + ``resourceMetadata``
  shape (and tolerant content fallbacks) into the shared
  ``{title, link, content}`` contract.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


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
        "SHAREPOINT_REMOTE_ENABLED": False,
        "SHAREPOINT_REMOTE_KNOWLEDGE_SOURCE_NAME": "",
        "SHAREPOINT_REMOTE_FILTER_EXPRESSION_ADD_ON": "",
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


@pytest.mark.asyncio
async def test_max_runtime_emitted_when_sharepoint_remote_enabled():
    client, session = _build_client(
        _SAMPLE_PAYLOAD,
        config_overrides={
            "SHAREPOINT_REMOTE_ENABLED": True,
            "SHAREPOINT_REMOTE_KNOWLEDGE_SOURCE_NAME": "sharepoint-remote-ks",
        },
    )
    await client.retrieve("hello", obo_token="user-obo")
    assert session.captured["json"]["maxRuntimeInSeconds"] == 120


@pytest.mark.asyncio
async def test_sharepoint_remote_not_emitted_when_disabled():
    client, session = _build_client(
        _SAMPLE_PAYLOAD,
        config_overrides={
            "FOUNDRY_IQ_KNOWLEDGE_SOURCE_NAME": "documents-blob-ks",
            "SHAREPOINT_REMOTE_ENABLED": False,
            "SHAREPOINT_REMOTE_KNOWLEDGE_SOURCE_NAME": "sharepoint-remote-ks",
        },
    )
    await client.retrieve("hello", obo_token="user-obo")
    sources = session.captured["json"].get("knowledgeSourceParams", [])
    assert not any(s.get("kind") == "remoteSharePoint" for s in sources)


@pytest.mark.asyncio
async def test_sharepoint_remote_not_emitted_when_name_missing():
    client, session = _build_client(
        _SAMPLE_PAYLOAD,
        config_overrides={
            "SHAREPOINT_REMOTE_ENABLED": True,
            "SHAREPOINT_REMOTE_KNOWLEDGE_SOURCE_NAME": "",
        },
    )
    await client.retrieve("hello", obo_token="user-obo")
    sources = session.captured["json"].get("knowledgeSourceParams", [])
    assert not any(s.get("kind") == "remoteSharePoint" for s in sources)


@pytest.mark.asyncio
async def test_sharepoint_remote_emitted_when_enabled_with_obo():
    client, session = _build_client(
        _SAMPLE_PAYLOAD,
        config_overrides={
            "SHAREPOINT_REMOTE_ENABLED": True,
            "SHAREPOINT_REMOTE_KNOWLEDGE_SOURCE_NAME": "sharepoint-remote-ks",
        },
    )
    await client.retrieve("hello", obo_token="user-obo")
    sources = session.captured["json"]["knowledgeSourceParams"]
    sp_sources = [s for s in sources if s.get("kind") == "remoteSharePoint"]
    assert sp_sources == [
        {
            "knowledgeSourceName": "sharepoint-remote-ks",
            "kind": "remoteSharePoint",
            "includeReferences": True,
            "includeReferenceSourceData": True,
        }
    ]


@pytest.mark.asyncio
async def test_sharepoint_remote_forwards_filter_expression_add_on():
    client, session = _build_client(
        _SAMPLE_PAYLOAD,
        config_overrides={
            "SHAREPOINT_REMOTE_ENABLED": True,
            "SHAREPOINT_REMOTE_KNOWLEDGE_SOURCE_NAME": "sharepoint-remote-ks",
            "SHAREPOINT_REMOTE_FILTER_EXPRESSION_ADD_ON": "filetype:docx",
        },
    )
    await client.retrieve("hello", obo_token="user-obo")
    sources = session.captured["json"]["knowledgeSourceParams"]
    sp_sources = [s for s in sources if s.get("kind") == "remoteSharePoint"]
    assert sp_sources == [
        {
            "knowledgeSourceName": "sharepoint-remote-ks",
            "kind": "remoteSharePoint",
            "includeReferences": True,
            "includeReferenceSourceData": True,
            "filterExpressionAddOn": "filetype:docx",
        }
    ]


@pytest.mark.asyncio
async def test_sharepoint_remote_never_carries_filter_add_on():
    client, session = _build_client(
        _SAMPLE_PAYLOAD,
        config_overrides={
            "SHAREPOINT_REMOTE_ENABLED": True,
            "SHAREPOINT_REMOTE_KNOWLEDGE_SOURCE_NAME": "sharepoint-remote-ks",
        },
    )
    await client.retrieve(
        "hello",
        obo_token="user-obo",
        conversation_id="conv-1",
        user_context={"principal_id": "user-1", "groups": ["g-a"]},
    )
    sp_sources = [
        s for s in session.captured["json"]["knowledgeSourceParams"]
        if s.get("kind") == "remoteSharePoint"
    ]
    assert len(sp_sources) == 1
    assert "filterAddOn" not in sp_sources[0]


@pytest.mark.asyncio
async def test_sharepoint_remote_skipped_when_obo_missing():
    client, session = _build_client(
        _SAMPLE_PAYLOAD,
        config_overrides={
            "FOUNDRY_IQ_KNOWLEDGE_SOURCE_NAME": "documents-blob-ks",
            "SHAREPOINT_REMOTE_ENABLED": True,
            "SHAREPOINT_REMOTE_KNOWLEDGE_SOURCE_NAME": "sharepoint-remote-ks",
        },
    )
    await client.retrieve("hello")
    sources = session.captured["json"].get("knowledgeSourceParams", [])
    assert not any(s.get("kind") == "remoteSharePoint" for s in sources)
    assert any(s.get("kind") == "azureBlob" for s in sources)


@pytest.mark.asyncio
async def test_normalize_sharepoint_remote_reference_shape():
    payload = {
        "references": [
            {
                "type": "remoteSharePoint",
                "sourceData": {
                    "webUrl": "https://contoso.sharepoint.com/sites/x/plan.docx",
                    "resourceMetadata": {
                        "Title": "Q4 rollout plan",
                        "Author": "Sarah",
                    },
                    "extracts": [
                        {"text": "Milestone A ships in October."},
                        {"text": "Milestone B ships in November."},
                    ],
                },
            },
            {
                "type": "remoteSharePoint",
                "sourceData": {
                    "webUrl": "https://contoso.sharepoint.com/sites/x/notes.docx",
                    "resourceMetadata": {"Title": "Notes"},
                    "text": "Weekly notes, no headings.",
                },
            },
            {
                "type": "remoteSharePoint",
                "sourceData": {
                    "webUrl": "https://contoso.sharepoint.com/sites/x/deck.pptx",
                    "resourceMetadata": {"Author": "Sarah"},
                    "extracts": [{"text": "Deck section 1."}],
                },
            },
            {
                "type": "remoteSharePoint",
                "sourceData": {
                    "webUrl": "https://contoso.sharepoint.com/sites/x/empty.docx",
                    "resourceMetadata": {"Title": "Empty"},
                    "extracts": [],
                },
            },
        ]
    }
    client, _ = _build_client(payload)
    records = await client.retrieve("hello", obo_token="user-obo")
    assert records == [
        {
            "title": "Q4 rollout plan",
            "link": "https://contoso.sharepoint.com/sites/x/plan.docx",
            "content": "Milestone A ships in October.\n\nMilestone B ships in November.",
        },
        {
            "title": "Notes",
            "link": "https://contoso.sharepoint.com/sites/x/notes.docx",
            "content": "Weekly notes, no headings.",
        },
        {
            "title": "deck.pptx",
            "link": "https://contoso.sharepoint.com/sites/x/deck.pptx",
            "content": "Deck section 1.",
        },
    ]


@pytest.mark.asyncio
async def test_mi_fallback_still_works_when_sharepoint_remote_off():
    client, session = _build_client(
        _SAMPLE_PAYLOAD,
        config_overrides={"FOUNDRY_IQ_KNOWLEDGE_SOURCE_NAME": "documents-blob-ks"},
    )
    await client.retrieve("hello")
    headers = session.captured["headers"]
    assert headers["x-ms-query-source-authorization"] == "svc-token"
