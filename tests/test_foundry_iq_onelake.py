"""Tests for the indexedOneLake (Microsoft Fabric OneLake) knowledge source path.

Covers the OneLake indexed knowledge source, a native kind: Foundry IQ owns
the underlying Azure AI Search index for the KS, so OneLake does not require
a per-user OBO token and does not trigger the ``maxRuntimeInSeconds`` bump.

- ``indexedOneLake`` knowledge source params are appended only when
  ``ONELAKE_KS_ENABLED`` is true and ``ONELAKE_KNOWLEDGE_SOURCE_NAME`` is set.
- ``indexedOneLake`` params never carry a ``filterAddOn``; ACL is enforced by
  Foundry IQ against the underlying AI Search index.
- OneLake does not count as a remote kind: it must not, on its own, cause
  ``maxRuntimeInSeconds`` to be emitted.
- OneLake does not require an OBO token: it must be emitted even when no
  ``x-ms-query-source-authorization`` header is available.
- ``_normalize_references`` maps the OneLake ``sourceData`` projection
  (snippet + OneLake file pointer, optionally annotated with workspace /
  lakehouse) to the shared ``{title, link, content}`` contract.
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
        "ONELAKE_KS_ENABLED": False,
        "ONELAKE_KNOWLEDGE_SOURCE_NAME": "",
        "ONELAKE_WORKSPACE_ID": "",
        "ONELAKE_LAKEHOUSE_ID": "",
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
# OneLake knowledge source emission
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_onelake_not_emitted_when_disabled():
    client, session = _build_client(
        _SAMPLE_PAYLOAD,
        config_overrides={
            "FOUNDRY_IQ_KNOWLEDGE_SOURCE_NAME": "documents-blob-ks",
            "ONELAKE_KS_ENABLED": False,
            "ONELAKE_KNOWLEDGE_SOURCE_NAME": "onelake-ks",
        },
    )
    await client.retrieve("hello")
    sources = session.captured["json"].get("knowledgeSourceParams", [])
    assert not any(s.get("kind") == "indexedOneLake" for s in sources)


@pytest.mark.asyncio
async def test_onelake_not_emitted_when_name_missing():
    client, session = _build_client(
        _SAMPLE_PAYLOAD,
        config_overrides={
            "ONELAKE_KS_ENABLED": True,
            "ONELAKE_KNOWLEDGE_SOURCE_NAME": "",
        },
    )
    await client.retrieve("hello")
    sources = session.captured["json"].get("knowledgeSourceParams", [])
    assert not any(s.get("kind") == "indexedOneLake" for s in sources)


@pytest.mark.asyncio
async def test_onelake_emitted_when_enabled():
    client, session = _build_client(
        _SAMPLE_PAYLOAD,
        config_overrides={
            "ONELAKE_KS_ENABLED": True,
            "ONELAKE_KNOWLEDGE_SOURCE_NAME": "onelake-ks",
            "ONELAKE_WORKSPACE_ID": "ws-abc",
            "ONELAKE_LAKEHOUSE_ID": "lh-xyz",
        },
    )
    await client.retrieve("hello")
    sources = session.captured["json"]["knowledgeSourceParams"]
    onelake = [s for s in sources if s.get("kind") == "indexedOneLake"]
    assert onelake == [
        {
            "knowledgeSourceName": "onelake-ks",
            "kind": "indexedOneLake",
            "includeReferences": True,
            "includeReferenceSourceData": True,
        }
    ]


@pytest.mark.asyncio
async def test_onelake_never_carries_filter_add_on():
    """OneLake ACL is enforced against the underlying AI Search index; the
    Pattern B filterAddOn is only valid for the primary searchIndex source."""
    client, session = _build_client(
        _SAMPLE_PAYLOAD,
        config_overrides={
            "ONELAKE_KS_ENABLED": True,
            "ONELAKE_KNOWLEDGE_SOURCE_NAME": "onelake-ks",
        },
    )
    await client.retrieve(
        "hello",
        conversation_id="conv-1",
        user_context={"principal_id": "user-1", "groups": ["g-a"]},
    )
    onelake = [
        s for s in session.captured["json"]["knowledgeSourceParams"]
        if s.get("kind") == "indexedOneLake"
    ]
    assert len(onelake) == 1
    assert "filterAddOn" not in onelake[0]


@pytest.mark.asyncio
async def test_onelake_emitted_without_obo_token():
    """OneLake is a native kind (Foundry IQ owns the AI Search index) and
    must not require an OBO token, unlike workIQ / fabricOntology /
    fabricDataAgent."""
    client, session = _build_client(
        _SAMPLE_PAYLOAD,
        config_overrides={
            "ONELAKE_KS_ENABLED": True,
            "ONELAKE_KNOWLEDGE_SOURCE_NAME": "onelake-ks",
        },
    )
    await client.retrieve("hello")  # no obo_token
    kinds = [
        s.get("kind")
        for s in session.captured["json"].get("knowledgeSourceParams", [])
    ]
    assert "indexedOneLake" in kinds


# ---------------------------------------------------------------------------
# maxRuntimeInSeconds: OneLake is native, must not trigger the bump alone.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_max_runtime_not_emitted_when_only_onelake_enabled():
    """indexedOneLake is a native kind (index is local to Foundry IQ), so on
    its own it must not cause the maxRuntimeInSeconds bump used for M365 /
    Fabric remote kinds."""
    client, session = _build_client(
        _SAMPLE_PAYLOAD,
        config_overrides={
            "ONELAKE_KS_ENABLED": True,
            "ONELAKE_KNOWLEDGE_SOURCE_NAME": "onelake-ks",
        },
    )
    await client.retrieve("hello")
    assert "maxRuntimeInSeconds" not in session.captured["json"]


# ---------------------------------------------------------------------------
# _normalize_references for indexedOneLake shape
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_normalize_onelake_reference_shape():
    """OneLake sourceData looks like an azureBlob projection: a snippet plus
    a OneLake file pointer, optionally annotated with workspace / lakehouse
    identifiers. The normalizer must map it to the shared
    ``{title, link, content}`` contract and drop empty entries."""
    payload = {
        "references": [
            {
                "type": "indexedOneLake",
                "sourceData": {
                    "title": "sales-fy25.parquet",
                    "snippet": "Q1 revenue was 42.1M.",
                    "oneLakeFilePath": (
                        "https://onelake.dfs.fabric.microsoft.com/ws-abc/"
                        "lh-xyz.Lakehouse/Files/sales/sales-fy25.parquet"
                    ),
                    "workspaceId": "ws-abc",
                    "lakehouseId": "lh-xyz",
                },
            },
            {
                # No explicit title: derive from file name; workspace only.
                "type": "indexedOneLake",
                "sourceData": {
                    "snippet": "Top product: SKU-42.",
                    "webUrl": (
                        "https://onelake.dfs.fabric.microsoft.com/ws-abc/"
                        "lh-xyz.Lakehouse/Files/products/top.parquet?token=xyz"
                    ),
                    "workspaceId": "ws-abc",
                    "lakehouseId": "lh-xyz",
                },
            },
            {
                # No content: must be dropped.
                "type": "indexedOneLake",
                "sourceData": {
                    "snippet": "",
                    "oneLakeFilePath": "abfss://foo/bar",
                    "lakehouseId": "lh-xyz",
                },
            },
        ]
    }
    client, _ = _build_client(payload)
    records = await client.retrieve("hello")
    assert records == [
        {
            "title": "sales-fy25.parquet (workspace ws-abc, lakehouse lh-xyz)",
            "link": (
                "https://onelake.dfs.fabric.microsoft.com/ws-abc/"
                "lh-xyz.Lakehouse/Files/sales/sales-fy25.parquet"
            ),
            "content": "Q1 revenue was 42.1M.",
        },
        {
            "title": "top.parquet (workspace ws-abc, lakehouse lh-xyz)",
            "link": (
                "https://onelake.dfs.fabric.microsoft.com/ws-abc/"
                "lh-xyz.Lakehouse/Files/products/top.parquet?token=xyz"
            ),
            "content": "Top product: SKU-42.",
        },
    ]
