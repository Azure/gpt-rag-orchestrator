"""Tests for optional generic MCP Server knowledge sources in Foundry IQ."""

from __future__ import annotations

import hashlib
import json
import logging
import traceback
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agent_framework import ChatMessage, Role
from pydantic import ValidationError

from connectors.foundry_iq_mcp import (
    McpConfigurationError,
    McpCredentialError,
    McpQueryHeader,
    McpRuntimeConfig,
    build_mcp_control_headers,
    redact_mcp_tool_arguments,
)


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
        self.payload = payload
        self.status = status
        self.captured = {}

    def post(self, url, headers=None, json=None, data=None):  # noqa: A002
        self.captured = {
            "url": url,
            "headers": headers,
            "json": json,
            "data": data,
        }
        return _FakeResponse(self.payload, self.status)


def _source(
    *,
    name="monitor-mcp",
    fail_on_error=True,
    max_documents=4,
    query_headers=None,
    tools=None,
):
    return {
        "name": name,
        "serverURL": "https://mcp.contoso.com/mcp",
        "failOnError": fail_on_error,
        "maxOutputDocuments": max_documents,
        "tools": tools
        or [
            {
                "name": "query_logs",
                "maxOutputTokens": 2048,
                "inclusionMode": "reranked",
                "outputParsing": {
                    "kind": "json",
                    "jsonParameters": {"documentsPath": "$.results[*]"},
                },
            }
        ],
        "queryHeaders": query_headers or [],
    }


def _canonical_source_fixture() -> dict:
    fixture_path = (
        Path(__file__).parent
        / "fixtures"
        / "foundry_iq_mcp_canonical_source.json"
    )
    return json.loads(fixture_path.read_text(encoding="utf-8"))


def test_cross_schema_fixture_uses_the_canonical_search_contract():
    """Keep the orchestrator parser aligned with Azure/gpt-rag PR #568."""
    source = _canonical_source_fixture()

    config = McpRuntimeConfig.parse(
        enabled=True,
        sources_json=[source],
        reasoning_effort="low",
        trusted_hosts="mcp.contoso.com",
        log_tool_arguments=False,
        api_version="2026-05-01-preview",
        max_runtime_seconds=120,
    )

    assert "serverURL" in source
    assert "serverUrl" not in source
    assert config.sources[0].server_url == source["serverURL"]
    assert config.sources[0].tools[0].output_parsing.kind == "json"
    assert config.sources[0].tools[0].output_parsing.json_parameters.include_context
    assert config.sources[0].tools[1].output_parsing.kind == "auto"
    assert config.sources[0].tools[1].inclusion_mode == "always"


def _build_client(
    payload=None,
    *,
    status=200,
    enabled=True,
    sources=None,
    config_overrides=None,
):
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
        "FOUNDRY_IQ_MAX_OUTPUT_DOCUMENTS": 5,
        "FOUNDRY_IQ_FORWARD_SOURCE_AUTH": True,
        "FOUNDRY_IQ_CONVERSATION_UPLOAD_ENABLED": False,
        "FOUNDRY_IQ_CONVERSATION_KNOWLEDGE_SOURCE_NAME": "",
        "FOUNDRY_IQ_MAX_RUNTIME_SECONDS": 120,
        "FOUNDRY_IQ_MCP_ENABLED": enabled,
        "FOUNDRY_IQ_MCP_SOURCES_JSON": json.dumps(sources or [_source()]),
        "FOUNDRY_IQ_MCP_REASONING_EFFORT": "low",
        "FOUNDRY_IQ_MCP_TRUSTED_HOSTS": "mcp.contoso.com",
        "FOUNDRY_IQ_MCP_LOG_TOOL_ARGUMENTS": False,
    }
    values.update(config_overrides or {})
    cfg = MagicMock()
    cfg.get.side_effect = lambda key, default=None, type=str: values.get(  # noqa: A002
        key, default
    )
    cfg.aiocredential = MagicMock()
    cfg.aiocredential.get_token = AsyncMock(
        return_value=SimpleNamespace(token="search-service-token")
    )

    with patch("connectors.foundry_iq.get_config", return_value=cfg):
        client = foundry_iq.FoundryIQClient()
    session = _FakeSession(payload or {"references": [], "activity": []}, status)
    client._get_session = AsyncMock(return_value=session)
    return client, session


@pytest.mark.asyncio
async def test_disabled_preserves_exact_legacy_request_and_headers():
    client, session = _build_client(enabled=False)

    await client.retrieve("hello")

    assert session.captured["json"] == {
        "intents": [{"search": "hello", "type": "semantic"}],
        "maxOutputDocuments": 5,
    }
    assert session.captured["headers"]["Content-Type"] == "application/json"
    assert session.captured["headers"]["Authorization"].endswith(
        "search-service-token"
    )
    assert (
        session.captured["headers"]["x-ms-query-source-authorization"]
        == "search-service-token"
    )


@pytest.mark.asyncio
async def test_enabled_uses_messages_reasoning_activity_runtime_and_source_names():
    client, session = _build_client(
        config_overrides={"FOUNDRY_IQ_MCP_REASONING_EFFORT": "medium"}
    )

    await client.retrieve("show failures")

    body = session.captured["json"]
    assert "intents" not in body
    assert body["messages"] == [
        {
            "role": "user",
            "content": [{"type": "text", "text": "show failures"}],
        }
    ]
    assert body["retrievalReasoningEffort"] == {"kind": "medium"}
    assert body["outputMode"] == "extractiveData"
    assert body["includeActivity"] is True
    assert body["maxRuntimeInSeconds"] == 120
    assert body["knowledgeSourceParams"] == [
        {
            "knowledgeSourceName": "monitor-mcp",
            "kind": "mcpServer",
            "includeReferences": True,
            "includeReferenceSourceData": True,
            "failOnError": True,
            "maxOutputDocuments": 4,
        }
    ]
    serialized = json.dumps(body)
    assert "serverURL" not in serialized
    assert "tools" not in serialized
    assert "queryHeaders" not in serialized
    assert "alwaysQuerySource" not in serialized


@pytest.mark.asyncio
async def test_enabled_sources_coexist_with_existing_source_kinds():
    client, session = _build_client(
        sources=[_source(name="one"), _source(name="two")],
        config_overrides={
            "FOUNDRY_IQ_KNOWLEDGE_SOURCE_NAME": "documents",
            "FOUNDRY_IQ_KNOWLEDGE_SOURCE_KIND": "azureBlob",
        },
    )

    await client.retrieve("hello")

    params = session.captured["json"]["knowledgeSourceParams"]
    assert [(item["knowledgeSourceName"], item["kind"]) for item in params] == [
        ("documents", "azureBlob"),
        ("one", "mcpServer"),
        ("two", "mcpServer"),
    ]


@pytest.mark.asyncio
async def test_control_headers_are_paired_deterministically_for_all_auth_modes():
    config = McpRuntimeConfig.parse(
        enabled=True,
        sources_json=[
            _source(
                query_headers=[
                    {
                        "name": "Authorization",
                        "valueFrom": {
                            "kind": "managedIdentity",
                            "scope": "api://monitor/.default",
                        },
                    },
                    {
                        "name": "x-user-token",
                        "valueFrom": {
                            "kind": "obo",
                            "scope": "api://mcp/user_impersonation",
                        },
                    },
                    {
                        "name": "x-api-key",
                        "valueFrom": {
                            "kind": "keyVaultSecret",
                            "secretName": "monitor-api-key",
                        },
                    },
                    {"name": "x-none", "valueFrom": {"kind": "none"}},
                ]
            )
        ],
        reasoning_effort="low",
        trusted_hosts="mcp.contoso.com",
        log_tool_arguments=False,
        api_version="2026-05-01-preview",
        max_runtime_seconds=120,
    )
    credential = MagicMock()
    credential.get_token = AsyncMock(return_value=SimpleNamespace(token="mi-token"))
    exchange = AsyncMock(return_value="obo-token")
    secret_loader = AsyncMock(return_value="secret-value")

    headers, modes = await build_mcp_control_headers(
        config,
        credential=credential,
        incoming_token="incoming-token",
        acquire_obo_token=exchange,
        get_secret=secret_loader,
    )

    assert headers["monitor-mcp-header-name"] == "Authorization"
    assert headers["monitor-mcp-header-value"].endswith("mi-token")
    assert headers["monitor-mcp-header-name1"] == "x-user-token"
    assert headers["monitor-mcp-header-value1"] == "obo-token"
    assert headers["monitor-mcp-header-name2"] == "x-api-key"
    assert headers["monitor-mcp-header-value2"] == "secret-value"
    assert modes == {
        "monitor-mcp": ("managedIdentity", "obo", "keyVaultSecret")
    }
    credential.get_token.assert_awaited_once_with("api://monitor/.default")
    exchange.assert_awaited_once_with(
        "incoming-token", "api://mcp/user_impersonation"
    )
    secret_loader.assert_awaited_once_with("monitor-api-key")


@pytest.mark.asyncio
async def test_retrieve_keeps_search_and_mcp_authorization_separate():
    client, session = _build_client(
        sources=[
            _source(
                query_headers=[
                    {
                        "name": "Authorization",
                        "valueFrom": {
                            "kind": "obo",
                            "scope": "api://mcp/user_impersonation",
                        },
                    }
                ]
            )
        ]
    )

    with (
        patch(
            "connectors.search.acquire_obo_token",
            new=AsyncMock(return_value="mcp-obo-token"),
        ) as exchange,
        patch(
            "connectors.keyvault.get_secret", new=AsyncMock(return_value=None)
        ),
    ):
        await client.retrieve(
            "hello",
            obo_token="search-obo-token",
            incoming_token="incoming-api-token",
        )

    headers = session.captured["headers"]
    assert headers["Authorization"].endswith("search-service-token")
    assert headers["x-ms-query-source-authorization"] == "search-obo-token"
    assert headers["monitor-mcp-header-name"] == "Authorization"
    assert headers["monitor-mcp-header-value"].endswith("mcp-obo-token")
    exchange.assert_awaited_once_with(
        "incoming-api-token", "api://mcp/user_impersonation"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("source", "incoming_token", "secret_value", "match"),
    [
        (
            _source(
                query_headers=[
                    {
                        "name": "Authorization",
                        "valueFrom": {
                            "kind": "obo",
                            "scope": "api://mcp/user_impersonation",
                        },
                    }
                ]
            ),
            None,
            None,
            "requires an incoming user token",
        ),
        (
            _source(
                query_headers=[
                    {
                        "name": "x-api-key",
                        "valueFrom": {
                            "kind": "keyVaultSecret",
                            "secretName": "missing-key",
                        },
                    }
                ]
            ),
            "incoming",
            None,
            "Key Vault credential is missing",
        ),
    ],
)
async def test_missing_credentials_fail_clearly(
    source, incoming_token, secret_value, match
):
    config = McpRuntimeConfig.parse(
        enabled=True,
        sources_json=[source],
        reasoning_effort="low",
        trusted_hosts="mcp.contoso.com",
        log_tool_arguments=False,
        api_version="2026-05-01-preview",
        max_runtime_seconds=120,
    )

    with pytest.raises(McpCredentialError, match=match):
        await build_mcp_control_headers(
            config,
            credential=MagicMock(),
            incoming_token=incoming_token,
            acquire_obo_token=AsyncMock(return_value="unused"),
            get_secret=AsyncMock(return_value=secret_value),
        )


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"trusted_hosts": ""}, "TRUSTED_HOSTS is required"),
        ({"reasoning_effort": "minimal"}, "must be 'low' or 'medium'"),
        ({"max_runtime_seconds": 29}, "between 30 and 600"),
        (
            {"sources_json": [_source()] * 2},
            "source names must be unique",
        ),
        (
            {
                "sources_json": [
                    {**_source(), "serverURL": "http://mcp.contoso.com/mcp"}
                ]
            },
            "must be HTTPS",
        ),
        (
            {
                "sources_json": [
                    {
                        **_source(),
                        "serverURL": "https://mcp.contoso.com/mcp?tenant=secret",
                    }
                ]
            },
            "query string",
        ),
        (
            {
                "sources_json": [
                    {**_source(), "serverURL": "https://127.0.0.1/mcp"}
                ],
                "trusted_hosts": "127.0.0.1",
            },
            "IP literal",
        ),
        (
            {
                "sources_json": [
                    {**_source(), "serverURL": "https://127.1/mcp"}
                ],
                "trusted_hosts": "127.1",
            },
            "reserved host",
        ),
        (
            {
                "sources_json": [
                    {
                        **_source(),
                        "serverURL": "https://metadata.internal/mcp",
                    }
                ],
                "trusted_hosts": "metadata.internal",
            },
            "reserved host",
        ),
        (
            {
                "sources_json": [
                    {**_source(), "serverURL": "https://other.contoso.com/mcp"}
                ]
            },
            "is not in",
        ),
        (
            {
                "sources_json": [
                    {**_source(), "tools": []}
                ]
            },
            "at least one tool",
        ),
        (
            {
                "sources_json": [
                    {
                        **_source(),
                        "tools": [
                            {
                                "name": "query",
                                "maxOutputTokens": 8193,
                            }
                        ],
                    }
                ]
            },
            "between 1 and 8192",
        ),
        (
            {
                "sources_json": [
                    {
                        **_source(),
                        "tools": [
                            {
                                "name": "query",
                                "outputParsing": {"kind": "json"},
                            }
                        ],
                    }
                ]
            },
            "documentsPath",
        ),
        (
            {
                "sources_json": [
                    {
                        **_source(),
                        "tools": [
                            {
                                "name": "query",
                                "outputParsing": "json",
                            }
                        ],
                    }
                ]
            },
            "outputParsing",
        ),
        (
            {
                "sources_json": [
                    {
                        **_source(),
                        "tools": [
                            {
                                "name": "query",
                                "outputParsing": {
                                    "kind": "split",
                                    "splitParameters": {"maximumPagesToTake": 3},
                                },
                            }
                        ],
                    }
                ]
            },
            "splitParameters",
        ),
        (
            {
                "sources_json": [
                    {
                        **_source(),
                        "tools": [
                            {
                                "name": "query",
                                "inclusionMode": "all",
                            }
                        ],
                    }
                ]
            },
            "inclusionMode",
        ),
        (
            {"sources_json": "not-json"},
            "not valid JSON",
        ),
    ],
)
def test_invalid_enabled_configuration_fails_closed(overrides, match):
    values = {
        "enabled": True,
        "sources_json": [_source()],
        "reasoning_effort": "low",
        "trusted_hosts": "mcp.contoso.com",
        "log_tool_arguments": False,
        "api_version": "2026-05-01-preview",
        "max_runtime_seconds": 120,
    }
    values.update(overrides)

    with pytest.raises(McpConfigurationError, match=match):
        McpRuntimeConfig.parse(**values)


@pytest.mark.parametrize("name", ["Host", "Content-Length", "Connection", "bad\nname"])
def test_query_header_rejects_control_and_hop_by_hop_names(name):
    with pytest.raises(ValidationError):
        McpQueryHeader.model_validate(
            {
                "name": name,
                "valueFrom": {
                    "kind": "managedIdentity",
                    "scope": "api://mcp/.default",
                },
            }
        )


def test_query_header_rejects_literal_values_and_secrets():
    with pytest.raises(ValidationError):
        McpQueryHeader.model_validate(
            {
                "name": "Authorization",
                "value": "literal-secret",
                "valueFrom": {
                    "kind": "managedIdentity",
                    "scope": "api://mcp/.default",
                },
            }
        )


def test_configuration_error_does_not_echo_rejected_literal_secret():
    source = _source(
        query_headers=[
            {
                "name": "Authorization",
                "value": "literal-secret-marker",
                "valueFrom": {
                    "kind": "managedIdentity",
                    "scope": "api://mcp/.default",
                },
            }
        ]
    )

    with pytest.raises(McpConfigurationError) as exc_info:
        McpRuntimeConfig.parse(
            enabled=True,
            sources_json=[source],
            reasoning_effort="low",
            trusted_hosts="mcp.contoso.com",
            log_tool_arguments=False,
            api_version="2026-05-01-preview",
            max_runtime_seconds=120,
        )

    assert "literal-secret-marker" not in str(exc_info.value)
    rendered_traceback = "".join(
        traceback.format_exception(exc_info.type, exc_info.value, exc_info.tb)
    )
    assert "literal-secret-marker" not in rendered_traceback


@pytest.mark.parametrize(
    "credential_field",
    ["access_token", "api-key", "password", "clientSecret", "headers"],
)
def test_configuration_rejects_nested_literal_credential_fields(credential_field):
    source = _source(
        tools=[
            {
                "name": "query_logs",
                "outputParsing": {
                    "kind": "json",
                    "jsonParameters": {
                        "documentsPath": "$.results[*]",
                        "unexpected": {credential_field: "literal-secret-marker"},
                    },
                },
            }
        ]
    )

    with pytest.raises(McpConfigurationError) as exc_info:
        McpRuntimeConfig.parse(
            enabled=True,
            sources_json=[source],
            reasoning_effort="low",
            trusted_hosts="mcp.contoso.com",
            log_tool_arguments=False,
            api_version="2026-05-01-preview",
            max_runtime_seconds=120,
        )

    assert "literal-secret-marker" not in str(exc_info.value)
    assert credential_field in str(exc_info.value)


def test_tool_argument_redaction_is_recursive_and_bounded():
    deeply_nested: object = "too-deep-to-log"
    for _ in range(10):
        deeply_nested = {"level": deeply_nested}
    arguments = {
        "query": "safe query",
        "Authorization": "authorization-secret",
        "nested": [
            {
                "access_token": "nested-token",
                "details": {
                    "api-key": "nested-api-key",
                    "password": "nested-password",
                },
            },
            {
                "monitor-mcp-header-value": "paired-header-secret",
                "public": "visible",
            },
        ],
        "deep": deeply_nested,
        "large": "x" * 5000,
    }

    rendered = redact_mcp_tool_arguments(arguments)

    for secret in (
        "authorization-secret",
        "nested-token",
        "nested-api-key",
        "nested-password",
        "paired-header-secret",
        "too-deep-to-log",
    ):
        assert secret not in rendered
    assert "<redacted>" in rendered
    assert "<truncated>" in rendered
    assert len(rendered) <= 1011


@pytest.mark.asyncio
async def test_enabled_tool_argument_logging_never_logs_paired_header_values(caplog):
    secrets = {
        "Authorization": "authorization-secret",
        "nested": [
            {
                "accessToken": "nested-token",
                "headers": {
                    "monitor-mcp-header-value": "paired-header-secret",
                },
            }
        ],
    }
    payload = {
        "activity": [
            {
                "type": "mcpServer",
                "knowledgeSourceName": "monitor-mcp",
                "status": "succeeded",
                "mcpServerArguments": {
                    "toolName": "query_logs",
                    "toolArguments": secrets,
                },
            }
        ],
        "references": [],
    }
    client, _ = _build_client(
        payload,
        config_overrides={"FOUNDRY_IQ_MCP_LOG_TOOL_ARGUMENTS": True},
    )
    caplog.set_level(logging.DEBUG)

    await client.retrieve("hello")

    for secret in (
        "authorization-secret",
        "nested-token",
        "paired-header-secret",
    ):
        assert secret not in caplog.text
    assert "<redacted>" in caplog.text


@pytest.mark.asyncio
async def test_partial_optional_failure_preserves_successful_references(caplog):
    payload = {
        "activity": [
            {
                "type": "mcpServer",
                "knowledgeSourceName": "monitor-mcp",
                "status": "failed",
                "error": {
                    "code": "ToolTimeout",
                    "message": "sensitive raw detail",
                },
                "mcpServerArguments": {
                    "toolName": "query_logs",
                    "toolArguments": {"query": "secret query"},
                },
            }
        ],
        "references": [
            {
                "type": "mcpServer",
                "toolName": "query_logs",
                "sourceData": {"title": "Result", "content": "safe result"},
            }
        ],
    }
    client, _ = _build_client(
        payload,
        status=206,
        sources=[_source(fail_on_error=False)],
    )

    records = await client.retrieve("hello")

    assert records == [{"title": "Result", "link": "", "content": "safe result"}]
    assert "ToolTimeout" in caplog.text
    assert "secret query" not in caplog.text
    assert "sensitive raw detail" not in caplog.text


@pytest.mark.asyncio
async def test_partial_required_source_failure_is_surfaced():
    from connectors.foundry_iq import McpSourceError

    payload = {
        "activity": [
            {
                "type": "mcpServer",
                "knowledgeSourceName": "monitor-mcp",
                "status": "failed",
                "error": {"code": "ToolTimeout"},
            }
        ],
        "references": [],
    }
    client, _ = _build_client(payload, status=206)

    with pytest.raises(McpSourceError, match="monitor-mcp.*ToolTimeout"):
        await client.retrieve("hello")


@pytest.mark.asyncio
async def test_206_without_mcp_failure_preserves_mixed_source_results():
    payload = {
        "activity": [
            {
                "type": "workIQ",
                "knowledgeSourceName": "work-iq",
                "status": "failed",
            }
        ],
        "references": [
            {
                "type": "mcpServer",
                "sourceData": {"title": "MCP result", "content": "usable"},
            }
        ],
    }
    client, _ = _build_client(payload, status=206)

    assert await client.retrieve("hello") == [
        {"title": "MCP result", "link": "", "content": "usable"}
    ]


def test_mcp_reference_normalization_supports_flat_json_text_and_drops_empty():
    from connectors.foundry_iq import FoundryIQClient

    payload = {
        "references": [
            {
                "type": "mcpServer",
                "sourceData": {
                    "title": "Flat",
                    "url": "https://example/flat",
                    "content": "flat content",
                },
            },
            {
                "type": "mcpServer",
                "toolName": "json_tool",
                "sourceData": {"name": "JSON", "payload": {"b": 2, "a": 1}},
            },
            {
                "type": "mcpServer",
                "toolName": "text_tool",
                "sourceData": "plain text",
            },
            {
                "type": "mcpServer",
                "sourceData": {},
            },
        ]
    }

    records = FoundryIQClient._normalize_references(payload)

    assert records == [
        {
            "title": "Flat",
            "link": "https://example/flat",
            "content": "flat content",
        },
        {
            "title": "JSON",
            "link": "",
            "content": '{"name":"JSON","payload":{"a":1,"b":2}}',
        },
        {
            "title": "text_tool",
            "link": "",
            "content": "plain text",
        },
    ]


@pytest.mark.asyncio
async def test_scope_aware_obo_cache_does_not_reuse_token_across_scopes():
    from connectors import search

    search._obo_cache.clear()
    cfg = MagicMock()
    cfg.get_value.side_effect = lambda key, default=None, allow_none=True: {
        "OAUTH_AZURE_AD_TENANT_ID": "tenant",
        "OAUTH_AZURE_AD_CLIENT_ID": "client",
        "OAUTH_AZURE_AD_CLIENT_SECRET": "secret",
    }[key]
    calls = []

    class _TokenSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def post(self, url, data=None):
            calls.append(data)
            return _FakeResponse(
                {
                    "access_token": f"token-{len(calls)}",
                    "expires_in": 3600,
                }
            )

    with (
        patch("connectors.search.get_config", return_value=cfg),
        patch("connectors.search.aiohttp.ClientSession", _TokenSession),
    ):
        first = await search.acquire_obo_token("incoming", "api://one/.default")
        cached = await search.acquire_obo_token("incoming", "api://one/.default")
        second_scope = await search.acquire_obo_token(
            "incoming", "api://two/.default"
        )

    assert first == cached == "token-1"
    assert second_scope == "token-2"
    assert [call["scope"] for call in calls] == [
        "api://one/.default",
        "api://two/.default",
    ]
    expected_fingerprint = hashlib.sha256(b"incoming").hexdigest()
    assert {
        cache_key.partition(":")[0] for cache_key in search._obo_cache
    } == {expected_fingerprint}


@pytest.mark.asyncio
async def test_singleton_search_path_preserves_disabled_retrieval_arguments():
    from connectors import search

    search_client = search.SearchClient.__new__(search.SearchClient)
    search_client._request_api_access_token = "request-a"
    search_client._allow_anonymous = True
    search_client._conversation_id = "conversation-a"
    search_client._user_context = {"principal_id": "user-a"}
    search_client.search_top_k = 3
    foundry_client = MagicMock()
    foundry_client.mcp_config = SimpleNamespace(enabled=False)
    foundry_client.retrieve = AsyncMock(return_value=[])
    search_client._get_search_user_token_for_trimming = AsyncMock(
        return_value="search-token-a"
    )

    with patch(
        "connectors.search.get_foundry_iq_client",
        return_value=foundry_client,
    ):
        await search_client._search_knowledge_base_foundry_iq("hello")

    foundry_client.retrieve.assert_awaited_once_with(
        "hello",
        obo_token="search-token-a",
        conversation_id="conversation-a",
        user_context={"principal_id": "user-a"},
    )
    search_client._get_search_user_token_for_trimming.assert_awaited_once()


@pytest.mark.asyncio
async def test_singleton_search_path_never_swallows_enabled_mcp_failures():
    from connectors import search

    search_client = search.SearchClient.__new__(search.SearchClient)
    search_client._request_api_access_token = "incoming"
    search_client._allow_anonymous = True
    search_client._conversation_id = None
    search_client._user_context = {}
    search_client.search_top_k = 3
    foundry_client = MagicMock()
    foundry_client.mcp_config = SimpleNamespace(enabled=True)
    foundry_client.retrieve = AsyncMock(
        side_effect=McpCredentialError("credential failed")
    )

    with (
        patch(
            "connectors.search.get_foundry_iq_client",
            return_value=foundry_client,
        ),
        patch(
            "connectors.search.acquire_obo_search_token",
            new=AsyncMock(return_value="search-obo-token"),
        ),
    ):
        with pytest.raises(McpCredentialError, match="credential failed"):
            await search_client._search_knowledge_base_foundry_iq("hello")


@pytest.mark.asyncio
async def test_enabled_credential_transport_and_http_errors_are_mcp_failures():
    from connectors.foundry_iq import McpSourceError

    mi_source = _source(
        query_headers=[
            {
                "name": "Authorization",
                "valueFrom": {
                    "kind": "managedIdentity",
                    "scope": "api://mcp/.default",
                },
            }
        ]
    )
    client, _ = _build_client(sources=[mi_source])
    client.credential.get_token = AsyncMock(
        side_effect=[
            SimpleNamespace(token="search-token"),
            RuntimeError("identity unavailable"),
        ]
    )
    with pytest.raises(McpCredentialError, match="Failed to resolve"):
        await client.retrieve("hello")

    client, _ = _build_client()
    client.credential.get_token = AsyncMock(
        side_effect=RuntimeError("service identity unavailable")
    )
    with pytest.raises(McpCredentialError, match="service token"):
        await client.retrieve("hello")

    client, _ = _build_client(status=503)
    with pytest.raises(McpSourceError, match="status=503"):
        await client.retrieve("hello")


@pytest.mark.asyncio
async def test_context_provider_logs_and_continues_after_disabled_search_obo_failure(
    caplog,
):
    from strategies.foundry_iq_context_provider import FoundryIQContextProvider

    get_token = AsyncMock(side_effect=RuntimeError("OBO unavailable"))
    foundry_client = MagicMock()
    foundry_client.mcp_config = SimpleNamespace(enabled=False)
    foundry_client.retrieve = AsyncMock(return_value=[])
    provider = FoundryIQContextProvider(
        get_obo_token=get_token,
        request_access_token="incoming",
        allow_anonymous=False,
        mcp_enabled=False,
    )
    caplog.set_level(logging.WARNING)

    with patch(
        "strategies.foundry_iq_context_provider.get_foundry_iq_client",
        return_value=foundry_client,
    ):
        context = await provider.invoking(
            [ChatMessage(role=Role.USER, text="question")]
        )

    assert not context.messages
    assert "OBO token acquisition failed" in caplog.text
    foundry_client.retrieve.assert_awaited_once_with(
        "question",
        obo_token=None,
        conversation_id=None,
        user_context={},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "token_result",
    [None, RuntimeError("OBO unavailable")],
)
async def test_context_provider_fails_closed_for_enabled_mcp_obo_requirements(
    token_result,
):
    from strategies.foundry_iq_context_provider import FoundryIQContextProvider

    get_token = AsyncMock()
    if isinstance(token_result, Exception):
        get_token.side_effect = token_result
    else:
        get_token.return_value = token_result
    foundry_client = MagicMock()
    foundry_client.mcp_config = SimpleNamespace(enabled=True)
    foundry_client.retrieve = AsyncMock()
    provider = FoundryIQContextProvider(
        get_obo_token=get_token,
        request_access_token="incoming",
        allow_anonymous=False,
        mcp_enabled=True,
    )

    with patch(
        "strategies.foundry_iq_context_provider.get_foundry_iq_client",
        return_value=foundry_client,
    ):
        with pytest.raises(McpCredentialError, match="OBO token"):
            await provider.invoking(
                [ChatMessage(role=Role.USER, text="question")]
            )

    foundry_client.retrieve.assert_not_awaited()


@pytest.mark.asyncio
async def test_context_provider_propagates_enabled_mcp_source_failure():
    from connectors.foundry_iq import McpSourceError
    from strategies.foundry_iq_context_provider import FoundryIQContextProvider

    foundry_client = MagicMock()
    foundry_client.mcp_config = SimpleNamespace(enabled=True)
    foundry_client.retrieve = AsyncMock(side_effect=McpSourceError("source failed"))
    provider = FoundryIQContextProvider(
        mcp_enabled=True,
        request_access_token="incoming",
    )

    with patch(
        "strategies.foundry_iq_context_provider.get_foundry_iq_client",
        return_value=foundry_client,
    ):
        with pytest.raises(McpSourceError, match="source failed"):
            await provider.invoking(
                [ChatMessage(role=Role.USER, text="question")]
            )

    foundry_client.retrieve.assert_awaited_once_with(
        "question",
        obo_token=None,
        incoming_token="incoming",
        conversation_id=None,
        user_context={},
    )


@pytest.mark.asyncio
async def test_generic_obo_fails_when_token_response_has_no_access_token():
    from connectors import search

    search._obo_cache.clear()
    cfg = MagicMock()
    cfg.get_value.side_effect = lambda key, default=None, allow_none=True: {
        "OAUTH_AZURE_AD_TENANT_ID": "tenant",
        "OAUTH_AZURE_AD_CLIENT_ID": "client",
        "OAUTH_AZURE_AD_CLIENT_SECRET": "secret",
    }[key]

    class _MissingTokenSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def post(self, url, data=None):
            return _FakeResponse({"expires_in": 3600})

    with (
        patch("connectors.search.get_config", return_value=cfg),
        patch("connectors.search.aiohttp.ClientSession", _MissingTokenSession),
    ):
        with pytest.raises(RuntimeError, match="missing access_token"):
            await search.acquire_obo_token("incoming", "api://mcp/.default")
