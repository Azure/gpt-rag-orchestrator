"""Tests for the Work IQ (Microsoft 365) knowledge source path.

Covers Azure/GPT-RAG#543 Phase 0 plumbing + Phase 1 Work IQ kind:

- ``maxRuntimeInSeconds`` is emitted only when a remote knowledge source kind
  is enabled, so today's Pattern A / Pattern B request bodies remain
  byte-identical.
- ``workIQ`` knowledge source params are appended only when
  ``WORK_IQ_ENABLED`` is true and ``WORK_IQ_KNOWLEDGE_SOURCE_NAME`` is set.
- ``workIQ`` params never carry a ``filterAddOn`` — ACL is enforced natively
  by Microsoft 365 via the forwarded user token.
- The MI fallback controlled by ``FOUNDRY_IQ_FORWARD_SOURCE_AUTH`` still
  applies to local ``azureBlob`` / ``searchIndex`` sources and never
  substitutes for OBO on remote kinds.
- ``_normalize_references`` maps the Work IQ ``attributions[].seeMoreWebUrl`` +
  ``extracts[].text`` shape to the shared ``{title, link, content}`` contract.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fakes (kept in-file so this module is self-contained; matches the style of
# tests/test_foundry_iq.py).
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
# Phase 0: maxRuntimeInSeconds
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_max_runtime_not_emitted_by_default():
    """Default Pattern A / Pattern B request body must stay byte-identical:
    no maxRuntimeInSeconds when no remote kind is enabled."""
    client, session = _build_client(
        _SAMPLE_PAYLOAD,
        config_overrides={"FOUNDRY_IQ_KNOWLEDGE_SOURCE_NAME": "documents-blob-ks"},
    )
    await client.retrieve("hello")
    assert "maxRuntimeInSeconds" not in session.captured["json"]


@pytest.mark.asyncio
async def test_max_runtime_emitted_when_work_iq_enabled():
    client, session = _build_client(
        _SAMPLE_PAYLOAD,
        config_overrides={
            "WORK_IQ_ENABLED": True,
            "WORK_IQ_KNOWLEDGE_SOURCE_NAME": "workiq-ks",
        },
    )
    await client.retrieve("hello", obo_token="user-obo")
    assert session.captured["json"]["maxRuntimeInSeconds"] == 120


@pytest.mark.asyncio
async def test_max_runtime_override_respected():
    client, session = _build_client(
        _SAMPLE_PAYLOAD,
        config_overrides={
            "WORK_IQ_ENABLED": True,
            "WORK_IQ_KNOWLEDGE_SOURCE_NAME": "workiq-ks",
            "FOUNDRY_IQ_MAX_RUNTIME_SECONDS": 240,
        },
    )
    await client.retrieve("hello", obo_token="user-obo")
    assert session.captured["json"]["maxRuntimeInSeconds"] == 240


# ---------------------------------------------------------------------------
# Phase 1: Work IQ knowledge source emission
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_work_iq_not_emitted_when_disabled():
    client, session = _build_client(
        _SAMPLE_PAYLOAD,
        config_overrides={
            "FOUNDRY_IQ_KNOWLEDGE_SOURCE_NAME": "documents-blob-ks",
            "WORK_IQ_ENABLED": False,
            "WORK_IQ_KNOWLEDGE_SOURCE_NAME": "workiq-ks",
        },
    )
    await client.retrieve("hello", obo_token="user-obo")
    sources = session.captured["json"].get("knowledgeSourceParams", [])
    assert not any(s.get("kind") == "workIQ" for s in sources)


@pytest.mark.asyncio
async def test_work_iq_not_emitted_when_name_missing():
    client, session = _build_client(
        _SAMPLE_PAYLOAD,
        config_overrides={
            "WORK_IQ_ENABLED": True,
            "WORK_IQ_KNOWLEDGE_SOURCE_NAME": "",
        },
    )
    await client.retrieve("hello", obo_token="user-obo")
    sources = session.captured["json"].get("knowledgeSourceParams", [])
    assert not any(s.get("kind") == "workIQ" for s in sources)


@pytest.mark.asyncio
async def test_work_iq_emitted_when_enabled_with_obo():
    client, session = _build_client(
        _SAMPLE_PAYLOAD,
        config_overrides={
            "WORK_IQ_ENABLED": True,
            "WORK_IQ_KNOWLEDGE_SOURCE_NAME": "workiq-ks",
        },
    )
    await client.retrieve("hello", obo_token="user-obo")
    sources = session.captured["json"]["knowledgeSourceParams"]
    work_iq = [s for s in sources if s.get("kind") == "workIQ"]
    assert work_iq == [
        {
            "knowledgeSourceName": "workiq-ks",
            "kind": "workIQ",
            "includeReferences": True,
            "includeReferenceSourceData": True,
        }
    ]


@pytest.mark.asyncio
async def test_work_iq_never_carries_filter_add_on():
    """M365 enforces ACL natively; a filterAddOn on workIQ would be wrong."""
    client, session = _build_client(
        _SAMPLE_PAYLOAD,
        config_overrides={
            "WORK_IQ_ENABLED": True,
            "WORK_IQ_KNOWLEDGE_SOURCE_NAME": "workiq-ks",
        },
    )
    await client.retrieve(
        "hello",
        obo_token="user-obo",
        conversation_id="conv-1",
        user_context={"principal_id": "user-1", "groups": ["g-a"]},
    )
    work_iq = [
        s for s in session.captured["json"]["knowledgeSourceParams"]
        if s.get("kind") == "workIQ"
    ]
    assert len(work_iq) == 1
    assert "filterAddOn" not in work_iq[0]


@pytest.mark.asyncio
async def test_work_iq_skipped_when_obo_missing():
    """CONTRACT: remote kinds require OBO. MI fallback must not be used for
    Work IQ. When no OBO token is available the Work IQ source is dropped
    (with a warning) so local sources still serve the request."""
    client, session = _build_client(
        _SAMPLE_PAYLOAD,
        config_overrides={
            "FOUNDRY_IQ_KNOWLEDGE_SOURCE_NAME": "documents-blob-ks",
            "WORK_IQ_ENABLED": True,
            "WORK_IQ_KNOWLEDGE_SOURCE_NAME": "workiq-ks",
        },
    )
    await client.retrieve("hello")

    sources = session.captured["json"].get("knowledgeSourceParams", [])
    assert not any(s.get("kind") == "workIQ" for s in sources)
    # Local source still present.
    assert any(s.get("kind") == "azureBlob" for s in sources)


# ---------------------------------------------------------------------------
# Regression guard: MI fallback still works for local kinds.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mi_fallback_still_works_for_azure_blob():
    """Baseline behavior for Pattern A must be preserved when Work IQ is off."""
    client, session = _build_client(
        _SAMPLE_PAYLOAD,
        config_overrides={"FOUNDRY_IQ_KNOWLEDGE_SOURCE_NAME": "documents-blob-ks"},
    )
    await client.retrieve("hello")
    headers = session.captured["headers"]
    assert headers["x-ms-query-source-authorization"] == "svc-token"


@pytest.mark.asyncio
async def test_mi_fallback_still_works_for_search_index():
    """Pattern B still uses the MI-forwarded header when no OBO token is
    supplied and forwarding is enabled."""
    client, session = _build_client(
        _SAMPLE_PAYLOAD,
        config_overrides={
            "FOUNDRY_IQ_KNOWLEDGE_SOURCE_NAME": "ragindex-ks",
            "FOUNDRY_IQ_KNOWLEDGE_SOURCE_KIND": "searchIndex",
        },
    )
    await client.retrieve("hello")
    headers = session.captured["headers"]
    assert headers["x-ms-query-source-authorization"] == "svc-token"


# ---------------------------------------------------------------------------
# _normalize_references for Work IQ shape
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_normalize_work_iq_reference_shape():
    """Work IQ sourceData carries attributions[].seeMoreWebUrl for links,
    extracts[].text for content, and a subject/name attribution for titles."""
    payload = {
        "references": [
            {
                "type": "workIQ",
                "sourceData": {
                    "attributions": [
                        {
                            "subject": "Q4 roadmap discussion",
                            "seeMoreWebUrl": "https://outlook.office.com/mail/id/abc",
                        }
                    ],
                    "extracts": [
                        {"text": "Ship Work IQ integration this quarter."},
                        {"text": "Follow-up: enable admin consent."},
                    ],
                },
            },
            {
                # Attribution with only a link — title falls back to URL tail.
                "type": "workIQ",
                "sourceData": {
                    "attributions": [
                        {"seeMoreWebUrl": "https://contoso.sharepoint.com/sites/x/plan.docx"}
                    ],
                    "extracts": [{"text": "Draft plan section."}],
                },
            },
            {
                # Empty extracts must be dropped, not surfaced as a bare title.
                "type": "workIQ",
                "sourceData": {
                    "attributions": [{"seeMoreWebUrl": "https://x/empty"}],
                    "extracts": [],
                },
            },
        ]
    }
    client, _ = _build_client(payload)
    records = await client.retrieve("hello", obo_token="user-obo")
    assert records == [
        {
            "title": "Q4 roadmap discussion",
            "link": "https://outlook.office.com/mail/id/abc",
            "content": "Ship Work IQ integration this quarter.\n\nFollow-up: enable admin consent.",
        },
        {
            "title": "plan.docx",
            "link": "https://contoso.sharepoint.com/sites/x/plan.docx",
            "content": "Draft plan section.",
        },
    ]
