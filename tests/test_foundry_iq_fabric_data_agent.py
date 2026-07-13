"""Tests for the Fabric Data Agent (Microsoft Fabric) knowledge source path.

Covers fabricDataAgent enablement:

- ``maxRuntimeInSeconds`` is emitted when a remote knowledge source kind is
  enabled (fabricDataAgent counts).
- ``fabricDataAgent`` knowledge source params are appended only when
  ``FABRIC_DATA_AGENT_ENABLED`` is true and
  ``FABRIC_DATA_AGENT_KNOWLEDGE_SOURCE_NAME`` is set.
- ``fabricDataAgent`` params never carry a ``filterAddOn``; ACL is enforced
  natively by Fabric via the forwarded per-user OBO token.
- The MI fallback controlled by ``FOUNDRY_IQ_FORWARD_SOURCE_AUTH`` never
  substitutes for OBO on Fabric Data Agent.
- ``_normalize_references`` maps the Fabric Data Agent ``dataAgentAnswer`` /
  ``dataAgentRawData`` shape to the shared ``{title, link, content}`` contract.
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
# maxRuntimeInSeconds: fabricDataAgent counts as a remote kind.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_max_runtime_emitted_when_fabric_data_agent_enabled():
    client, session = _build_client(
        _SAMPLE_PAYLOAD,
        config_overrides={
            "FABRIC_DATA_AGENT_ENABLED": True,
            "FABRIC_DATA_AGENT_KNOWLEDGE_SOURCE_NAME": "fda-ks",
        },
    )
    await client.retrieve("hello", obo_token="user-obo")
    assert session.captured["json"]["maxRuntimeInSeconds"] == 120


# ---------------------------------------------------------------------------
# Fabric Data Agent knowledge source emission
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fabric_data_agent_not_emitted_when_disabled():
    client, session = _build_client(
        _SAMPLE_PAYLOAD,
        config_overrides={
            "FOUNDRY_IQ_KNOWLEDGE_SOURCE_NAME": "documents-blob-ks",
            "FABRIC_DATA_AGENT_ENABLED": False,
            "FABRIC_DATA_AGENT_KNOWLEDGE_SOURCE_NAME": "fda-ks",
        },
    )
    await client.retrieve("hello", obo_token="user-obo")
    sources = session.captured["json"].get("knowledgeSourceParams", [])
    assert not any(s.get("kind") == "fabricDataAgent" for s in sources)


@pytest.mark.asyncio
async def test_fabric_data_agent_not_emitted_when_name_missing():
    client, session = _build_client(
        _SAMPLE_PAYLOAD,
        config_overrides={
            "FABRIC_DATA_AGENT_ENABLED": True,
            "FABRIC_DATA_AGENT_KNOWLEDGE_SOURCE_NAME": "",
        },
    )
    await client.retrieve("hello", obo_token="user-obo")
    sources = session.captured["json"].get("knowledgeSourceParams", [])
    assert not any(s.get("kind") == "fabricDataAgent" for s in sources)


@pytest.mark.asyncio
async def test_fabric_data_agent_emitted_when_enabled_with_obo():
    client, session = _build_client(
        _SAMPLE_PAYLOAD,
        config_overrides={
            "FABRIC_DATA_AGENT_ENABLED": True,
            "FABRIC_DATA_AGENT_KNOWLEDGE_SOURCE_NAME": "fda-ks",
        },
    )
    await client.retrieve("hello", obo_token="user-obo")
    sources = session.captured["json"]["knowledgeSourceParams"]
    fda = [s for s in sources if s.get("kind") == "fabricDataAgent"]
    assert fda == [
        {
            "knowledgeSourceName": "fda-ks",
            "kind": "fabricDataAgent",
            "includeReferences": True,
            "includeReferenceSourceData": True,
        }
    ]


@pytest.mark.asyncio
async def test_fabric_data_agent_never_carries_filter_add_on():
    """Fabric enforces ACL natively via OBO; a filterAddOn would be wrong."""
    client, session = _build_client(
        _SAMPLE_PAYLOAD,
        config_overrides={
            "FABRIC_DATA_AGENT_ENABLED": True,
            "FABRIC_DATA_AGENT_KNOWLEDGE_SOURCE_NAME": "fda-ks",
        },
    )
    await client.retrieve(
        "hello",
        obo_token="user-obo",
        conversation_id="conv-1",
        user_context={"principal_id": "user-1", "groups": ["g-a"]},
    )
    fda = [
        s for s in session.captured["json"]["knowledgeSourceParams"]
        if s.get("kind") == "fabricDataAgent"
    ]
    assert len(fda) == 1
    assert "filterAddOn" not in fda[0]


@pytest.mark.asyncio
async def test_fabric_data_agent_skipped_when_obo_missing():
    """CONTRACT: remote kinds require OBO. MI fallback must not be used for
    Fabric Data Agent. When no OBO token is available the Fabric Data Agent
    source is dropped (with a warning) so local sources still serve the
    request."""
    client, session = _build_client(
        _SAMPLE_PAYLOAD,
        config_overrides={
            "FOUNDRY_IQ_KNOWLEDGE_SOURCE_NAME": "documents-blob-ks",
            "FABRIC_DATA_AGENT_ENABLED": True,
            "FABRIC_DATA_AGENT_KNOWLEDGE_SOURCE_NAME": "fda-ks",
        },
    )
    await client.retrieve("hello")

    sources = session.captured["json"].get("knowledgeSourceParams", [])
    assert not any(s.get("kind") == "fabricDataAgent" for s in sources)
    # Local source still present.
    assert any(s.get("kind") == "azureBlob" for s in sources)


@pytest.mark.asyncio
async def test_all_three_remote_kinds_coexist():
    """workIQ + fabricOntology + fabricDataAgent must be emitted together."""
    client, session = _build_client(
        _SAMPLE_PAYLOAD,
        config_overrides={
            "WORK_IQ_ENABLED": True,
            "WORK_IQ_KNOWLEDGE_SOURCE_NAME": "workiq-ks",
            "FABRIC_IQ_ENABLED": True,
            "FABRIC_IQ_KNOWLEDGE_SOURCE_NAME": "fabric-ks",
            "FABRIC_DATA_AGENT_ENABLED": True,
            "FABRIC_DATA_AGENT_KNOWLEDGE_SOURCE_NAME": "fda-ks",
        },
    )
    await client.retrieve("hello", obo_token="user-obo")
    kinds = [s.get("kind") for s in session.captured["json"]["knowledgeSourceParams"]]
    assert "workIQ" in kinds
    assert "fabricOntology" in kinds
    assert "fabricDataAgent" in kinds


# ---------------------------------------------------------------------------
# _normalize_references for Fabric Data Agent shape
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_normalize_fabric_data_agent_reference_shape():
    """Fabric Data Agent sourceData carries dataAgentAnswer and/or
    dataAgentRawData plus workspaceId / dataAgentId. Both fields are
    concatenated when present, and the title falls back to workspace / data
    agent ids when no deep link is available."""
    payload = {
        "references": [
            {
                "type": "fabricDataAgent",
                "sourceData": {
                    "dataAgentAnswer": "Sales grew 12% quarter over quarter.",
                    "dataAgentRawData": "Region,QoQ\nEMEA,0.14\nAMER,0.11",
                    "workspaceId": "ws-abc",
                    "dataAgentId": "da-123",
                },
            },
            {
                # Only dataAgentAnswer.
                "type": "fabricDataAgent",
                "sourceData": {
                    "dataAgentAnswer": "Top product: SKU-42.",
                    "dataAgentId": "da-123",
                },
            },
            {
                # Empty payload must be dropped.
                "type": "fabricDataAgent",
                "sourceData": {
                    "dataAgentAnswer": "",
                    "dataAgentRawData": "",
                    "workspaceId": "ws-abc",
                },
            },
        ]
    }
    client, _ = _build_client(payload)
    records = await client.retrieve("hello", obo_token="user-obo")
    assert records == [
        {
            "title": "Fabric data agent da-123 (workspace ws-abc)",
            "link": "",
            "content": "Sales grew 12% quarter over quarter.\n\nRegion,QoQ\nEMEA,0.14\nAMER,0.11",
        },
        {
            "title": "Fabric data agent da-123",
            "link": "",
            "content": "Top product: SKU-42.",
        },
    ]
