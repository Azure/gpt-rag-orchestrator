"""Shared test fixtures for the GPT-RAG Orchestrator test suite."""

import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Make sure ``src/`` is on sys.path (pytest.ini sets pythonpath=["src"] but
# conftest loads before that in some setups).
# ---------------------------------------------------------------------------
_src = os.path.join(os.path.dirname(__file__), os.pardir, "src")
if os.path.isdir(_src) and _src not in sys.path:
    sys.path.insert(0, os.path.abspath(_src))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_config():
    """Return a fake AppConfigClient-like object pre-loaded with safe defaults."""
    cfg = MagicMock()
    cfg.get.side_effect = lambda key, default=None, type=str: {  # noqa: A002
        "AI_FOUNDRY_PROJECT_ENDPOINT": "https://fake-project.openai.azure.com",
        "AI_FOUNDRY_ACCOUNT_ENDPOINT": "https://fake-account.openai.azure.com",
        "CHAT_DEPLOYMENT_NAME": "gpt-4o",
        "EMBEDDING_DEPLOYMENT_NAME": "text-embedding-3-large",
        "OPENAI_API_VERSION": "2025-04-01-preview",
        "PROMPT_SOURCE": "file",
        "SEARCH_SERVICE_QUERY_ENDPOINT": "",
        "SEARCH_RAG_INDEX_NAME": "",
        "KNOWLEDGE_BASE_NAME": "",
        "AGENT_STRATEGY": "single_agent_rag",
        "AGENT_ID": "",
        "SEARCH_RETRIEVAL_ENABLED": "true",
        "BING_RETRIEVAL_ENABLED": "false",
        "BING_CONNECTION_ID": "",
        "LOG_LEVEL": "INFO",
        "ALLOW_ANONYMOUS": "true",
    }.get(key, default)
    cfg.get_value = lambda key, allow_none=False: cfg.get(key)
    cfg.credential = MagicMock()
    cfg.aiocredential = MagicMock()
    cfg.auth_failed = False
    return cfg


@pytest.fixture()
def mock_cosmos():
    """Return a fake CosmosDBClient with async helpers."""
    cosmos = MagicMock()
    cosmos.get_document = AsyncMock(return_value=None)
    cosmos.upsert_document = AsyncMock()
    cosmos.create_document = AsyncMock()
    return cosmos


@pytest.fixture()
def mock_identity_manager():
    """Return a fake identity manager that provides mock credentials."""
    im = MagicMock()
    im.get_credential.return_value = MagicMock()
    im.get_aio_credential.return_value = MagicMock()
    return im


@pytest.fixture()
def patch_dependencies(mock_config, mock_cosmos, mock_identity_manager):
    """Patch the main singletons so strategy constructors don't hit Azure."""
    with (
        patch("dependencies.get_config", return_value=mock_config),
        patch("connectors.cosmosdb.get_cosmosdb_client", return_value=mock_cosmos),
        patch("connectors.identity_manager.get_identity_manager", return_value=mock_identity_manager),
        # BaseAgentStrategy.__init__ imports these directly
        patch("strategies.base_agent_strategy.get_config", return_value=mock_config),
        patch("strategies.base_agent_strategy.get_cosmosdb_client", return_value=mock_cosmos),
        patch("strategies.base_agent_strategy.get_identity_manager", return_value=mock_identity_manager),
        # AIProjectClient is instantiated in base __init__
        patch("strategies.base_agent_strategy.AIProjectClient"),
    ):
        yield
