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

import json
import logging
from unittest.mock import AsyncMock, patch

import pytest

from connectors.search import (
    _RETRIEVAL_AUTH_FAILURE_MARKER,
    _RETRIEVAL_ERROR_MARKER,
    SearchClient,
)
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
