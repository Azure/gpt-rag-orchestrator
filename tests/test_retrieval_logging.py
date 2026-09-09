"""Regression tests for swallowed retrieval/auth failure logging.

Issue Azure/GPT-RAG#508: when retrieval fails (auth, transient, etc.) the
orchestrator returns empty results so the model can still respond, but the
underlying error must be obvious to operators looking at App Insights.

These tests pin the contract for the standardized log markers emitted from
``SearchClient.search_knowledge_base``:

- ``[Retrieval][AUTH_FAILURE]`` at ERROR level when the failure mentions a
  401/403 status (the operator-actionable case).
- ``[Retrieval][ERROR]`` at WARNING level for any other swallowed failure.
"""

import asyncio
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agent_framework import ChatMessage, Context, Role
from azure.core.exceptions import ClientAuthenticationError

from connectors.search import (
    _RETRIEVAL_AUTH_FAILURE_MARKER,
    _RETRIEVAL_ERROR_MARKER,
    SearchClient,
    build_conversation_filter,
)
from strategies.composite_context_provider import CompositeContextProvider
from strategies.search_context_provider import SearchContextProvider
from telemetry.audit import AuditEmitter, begin_audit_request, end_audit_request
from telemetry.audit_contract import AuditSettings


def _configure_audit(enabled):
    AuditEmitter._default = AuditEmitter(
        AuditSettings(
            enabled=enabled,
            sensitive_content_enabled=False,
            sensitive_content_fields=frozenset(),
            actor_pseudonym_enabled=False,
            source_event_limit=25,
            hmac_key_id="v1",
            hmac_key=b"k" * 32 if enabled else None,
            additional_redacted_keys=frozenset(),
        ),
        service_name="gpt-rag-orchestrator",
        service_version="3.7.0",
        environment="test",
    )


@pytest.fixture()
def search_client(patch_dependencies, mock_config):
    """Build a SearchClient with a fake endpoint and term-only search."""
    mock_config.get.side_effect = lambda key, default=None, type=str: {
        "SEARCH_SERVICE_QUERY_ENDPOINT": "https://fake-search.search.windows.net",
        "SEARCH_RAG_INDEX_NAME": "ragindex",
        "SEARCH_APPROACH": "term",
        "ALLOW_ANONYMOUS": "true",
    }.get(key, default)
    # ``connectors.search`` does ``from dependencies import get_config`` so
    # patching the module-local binding is required for the constructor to
    # pick up our fake config.
    with patch("connectors.search.get_config", return_value=mock_config):
        client = SearchClient()
    # Token acquisition is exercised by other tests; short-circuit it here.
    client._get_search_user_token_for_trimming = AsyncMock(return_value=None)
    return client


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["search", "get_document"])
@pytest.mark.parametrize("failure_type", [ClientAuthenticationError, RuntimeError, asyncio.CancelledError])
async def test_low_level_search_token_failure_propagates_without_request_or_raw_log(
    search_client, operation, failure_type, caplog,
):
    marker = "synthetic-private-token-provider-detail"
    failure = failure_type(marker)
    search_client.credential = SimpleNamespace(get_token=AsyncMock(side_effect=failure))
    search_client._get_session = AsyncMock()
    call = (
        search_client.search("documents", {"search": "question"})
        if operation == "search"
        else search_client.get_document("documents", "document-1")
    )
    with pytest.raises(failure_type) as raised:
        await call
    assert raised.value is failure
    search_client.credential.get_token.assert_awaited_once_with(
        "https://search.azure.com/.default"
    )
    search_client._get_session.assert_not_awaited()
    assert marker not in caplog.text
    if failure_type is ClientAuthenticationError:
        assert "failed to acquire token" in caplog.text


@pytest.mark.asyncio
async def test_403_emits_auth_failure_marker_at_error(search_client, caplog):
    caplog.set_level(logging.WARNING)
    with patch.object(
        search_client, "search", AsyncMock(side_effect=RuntimeError("HTTP 403 Forbidden"))
    ):
        await search_client.search_knowledge_base("hello")

    matches = [r for r in caplog.records if _RETRIEVAL_AUTH_FAILURE_MARKER in r.getMessage()]
    assert matches, "expected an AUTH_FAILURE marker in logs"
    assert all(r.levelno == logging.ERROR for r in matches)


@pytest.mark.asyncio
async def test_generic_error_emits_error_marker_at_warning(search_client, caplog):
    caplog.set_level(logging.WARNING)
    with patch.object(
        search_client, "search", AsyncMock(side_effect=RuntimeError("connection reset"))
    ):
        await search_client.search_knowledge_base("hello")

    matches = [r for r in caplog.records if _RETRIEVAL_ERROR_MARKER in r.getMessage()]
    assert matches, "expected a generic Retrieval ERROR marker in logs"
    assert all(r.levelno == logging.WARNING for r in matches)
    assert not any(_RETRIEVAL_AUTH_FAILURE_MARKER in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
@pytest.mark.parametrize("anonymous", [False, True], ids=["strict", "anonymous"])
async def test_connector_failure_preserves_full_error_contract_and_trimming_inputs(
    anonymous, search_client, caplog,
):
    search_client._allow_anonymous = anonymous
    search_client._conversation_id = "conversation'quoted"
    token = "synthetic-delegated-token"
    search_client._get_search_user_token_for_trimming = AsyncMock(return_value=token)
    marker = "synthetic-connector-error-detail"
    failure = RuntimeError(marker)
    search_client.search = AsyncMock(side_effect=failure)
    if anonymous:
        response = json.loads(await search_client.search_knowledge_base("policy"))
        assert response == {"results": [], "query": "policy", "error": "search_failed"}
        assert marker not in json.dumps(response)
        assert token not in json.dumps(response)
    else:
        with pytest.raises(RuntimeError) as caught:
            await search_client.search_knowledge_base("policy")
        assert caught.value is failure
    search_client._get_search_user_token_for_trimming.assert_awaited_once()
    sent = search_client.search.await_args.kwargs
    assert sent["index_name"] == search_client.index_name
    assert sent["body"]["search"] == "policy"
    assert sent["body"]["filter"] == build_conversation_filter("conversation'quoted")
    assert sent["search_user_token"] == token
    assert marker not in caplog.text
    assert token not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_boundary", ["embedding", "search", "sibling"])
async def test_real_search_provider_and_composite_preserve_distinct_failure_outcomes(
    failed_boundary, patch_dependencies, mock_config, caplog,
):
    marker = "synthetic-provider-error-detail"
    failure = RuntimeError(marker)
    token = "synthetic-delegated-token"
    embed = AsyncMock(
        return_value=[0.1, 0.2], side_effect=failure if failed_boundary == "embedding" else None)
    obo = AsyncMock(return_value=token)
    provider = SearchContextProvider(
        endpoint="https://search.example.invalid", index_name="test-index",
        credential=MagicMock(), conversation_id="conversation'quoted",
        embed_fn=embed, get_obo_token=obo)
    sibling = SimpleNamespace(invoking=AsyncMock(
        return_value=Context(instructions="Independent provider context"),
        side_effect=failure if failed_boundary == "sibling" else None))

    async def documents():
        yield {"id": "doc-1", "title": "Policy", "content": "Grounded content",
               "filepath": "policy.txt"}

    sdk = MagicMock()
    sdk.__aenter__ = AsyncMock(return_value=sdk)
    sdk.__aexit__ = AsyncMock(return_value=False)
    sdk.search = AsyncMock(
        return_value=documents(), side_effect=failure if failed_boundary == "search" else None)
    with (
        patch("strategies.search_context_provider.get_config", return_value=mock_config),
        patch("strategies.search_context_provider.SearchClient", return_value=sdk),
    ):
        composite = CompositeContextProvider([provider, sibling], required_providers=[provider])
        if failed_boundary == "search":
            with pytest.raises(RuntimeError) as raised:
                await composite.invoking(ChatMessage(role=Role.USER, text="policy"))
            assert raised.value is failure
            context = Context()
        else:
            context = await composite.invoking(ChatMessage(role=Role.USER, text="policy"))
    sent = sdk.search.await_args.kwargs
    assert sent["search_text"] == "policy"
    assert sent["filter"] == build_conversation_filter("conversation'quoted")
    assert sent["x_ms_query_source_authorization"] == token
    assert ("vector_queries" in sent) is (failed_boundary != "embedding")
    embed.assert_awaited_once_with("policy")
    obo.assert_awaited_once()
    messages = [message.text for message in context.messages or []]
    assert bool(messages) is (failed_boundary != "search")
    if messages:
        assert "Grounded content" in messages[0]
    assert context.instructions == (
        None if failed_boundary in {"sibling", "search"} else "Independent provider context")
    assert marker not in str(messages)
    assert token not in str(messages)
    assert marker not in caplog.text
    assert token not in caplog.text


@pytest.mark.parametrize("failed_boundary", ["embedding", "search"])
async def test_search_provider_cancellation_is_not_empty_context(
    failed_boundary, mock_config, monkeypatch,
):
    import asyncio

    cancelled = asyncio.CancelledError("synthetic cancellation")
    embed = AsyncMock(return_value=[0.1, 0.2],
                      side_effect=cancelled if failed_boundary == "embedding" else None)
    provider = SearchContextProvider(
        endpoint="https://search.example.invalid", index_name="test-index",
        credential=MagicMock(), conversation_id="conversation",
        get_obo_token=AsyncMock(return_value="synthetic-delegated"),
        embed_fn=embed)
    sdk = MagicMock()
    sdk.__aenter__ = AsyncMock(return_value=sdk)
    sdk.__aexit__ = AsyncMock(return_value=False)
    sdk.search = AsyncMock(side_effect=cancelled)
    monkeypatch.setattr("strategies.search_context_provider.get_config", lambda: mock_config)
    monkeypatch.setattr("strategies.search_context_provider.SearchClient", MagicMock(return_value=sdk))
    with pytest.raises(asyncio.CancelledError) as caught:
        await provider.invoking(ChatMessage(role=Role.USER, text="policy"))
    assert caught.value is cancelled
    if failed_boundary == "embedding":
        sdk.search.assert_not_awaited()
    else:
        sdk.__aexit__.assert_awaited_once()
        assert sdk.__aexit__.await_args.args[0] is asyncio.CancelledError
        assert sdk.__aexit__.await_args.args[1] is cancelled


@pytest.mark.asyncio
@pytest.mark.parametrize("audit_enabled", [False, True])
async def test_empty_ai_search_result_is_returned_regardless_of_audit(
    search_client, audit_enabled, caplog
):
    _configure_audit(audit_enabled)
    caplog.set_level(logging.INFO, logger="gptrag.audit")
    token = None
    if audit_enabled:
        context, token = begin_audit_request()
        context.request_started_event_id = "evt_" + ("1" * 32)
    search_client.search = AsyncMock(
        return_value={
            "value": [
                {
                    "title": "Empty reference",
                    "filepath": "empty.txt",
                    "content": "",
                    "chunk_id": "empty-1",
                }
            ]
        }
    )
    try:
        result = json.loads(await search_client.search_knowledge_base("hello"))
    finally:
        if token is not None:
            end_audit_request(token)

    assert result["results"] == [
        {"title": "Empty reference", "link": "empty.txt", "content": ""}
    ]
    audit_types = [
        record.event_type
        for record in caplog.records
        if hasattr(record, "event_type")
    ]
    assert audit_types == (
        ["grounding.source.rejected"] if audit_enabled else []
    )
