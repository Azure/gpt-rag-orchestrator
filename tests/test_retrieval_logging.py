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

import logging
from unittest.mock import AsyncMock, patch

import pytest

from connectors.search import (
    _RETRIEVAL_AUTH_FAILURE_MARKER,
    _RETRIEVAL_ERROR_MARKER,
    SearchClient,
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
