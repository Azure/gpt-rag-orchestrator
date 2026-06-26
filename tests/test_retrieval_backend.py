"""Tests for the RETRIEVAL_BACKEND selector and the strategy seams.

Covers Azure/GPT-RAG#526 Step 1:

- ``resolve_retrieval_backend`` / ``get_retrieval_backend`` resolution: default
  ``ai_search``, explicit ``foundry_iq``, and unknown values falling back to
  ``ai_search`` with a warning.
- The four ``_create_search_provider`` strategy seams build the AI Search
  provider on the default backend (no behavior change) and the Foundry IQ
  provider when ``foundry_iq`` is selected.
"""

import logging
from unittest.mock import patch

import pytest

from strategies.foundry_iq_context_provider import FoundryIQContextProvider
from strategies.search_context_provider import SearchContextProvider
from strategies.multimodal_search_context_provider import MultimodalSearchContextProvider
import util.retrieval_backend as rb


@pytest.fixture(autouse=True)
def _reset_backend_cache():
    rb.reset_retrieval_backend_cache()
    yield
    rb.reset_retrieval_backend_cache()


def _cfg(value):
    """Minimal config stub whose .get returns ``value`` for RETRIEVAL_BACKEND."""
    class _Cfg:
        def get(self, key, default=None, type=str):  # noqa: A002
            if key == "RETRIEVAL_BACKEND":
                return value
            return default
    return _Cfg()


# ---------------------------------------------------------------------------
# Selector resolution
# ---------------------------------------------------------------------------

def test_default_is_ai_search():
    assert rb.resolve_retrieval_backend(_cfg(None)) == "ai_search"
    assert rb.resolve_retrieval_backend(_cfg("")) == "ai_search"


def test_explicit_foundry_iq():
    assert rb.resolve_retrieval_backend(_cfg("foundry_iq")) == "foundry_iq"


def test_value_is_normalized_case_and_whitespace():
    assert rb.resolve_retrieval_backend(_cfg("  Foundry_IQ ")) == "foundry_iq"


def test_unknown_value_falls_back_with_warning(caplog):
    caplog.set_level(logging.WARNING)
    assert rb.resolve_retrieval_backend(_cfg("elasticsearch")) == "ai_search"
    assert any("Unknown RETRIEVAL_BACKEND" in r.getMessage() for r in caplog.records)


def test_get_retrieval_backend_is_cached():
    with patch.object(rb, "resolve_retrieval_backend", return_value="foundry_iq") as resolver:
        assert rb.get_retrieval_backend() == "foundry_iq"
        assert rb.get_retrieval_backend() == "foundry_iq"
        resolver.assert_called_once()


# ---------------------------------------------------------------------------
# Strategy seam regression: default backend keeps the AI Search providers
# ---------------------------------------------------------------------------

async def _build_provider(module_path, strategy_cls_name, backend, mock_config):
    """Construct a strategy under mock_config and build its search provider.

    The strategy constructor and the seam both call ``get_config``; patch it
    per-module with ``mock_config`` so no live App Configuration is contacted.
    """
    import importlib

    module = importlib.import_module(module_path)
    strategy_cls = getattr(module, strategy_cls_name)
    with patch(f"{module_path}.get_config", return_value=mock_config):
        s = strategy_cls()
        s.search_endpoint = "https://fake-search.search.windows.net"
        s.search_index_name = "ragindex"
        with patch(f"{module_path}.get_retrieval_backend", return_value=backend):
            return await s._create_search_provider()


@pytest.mark.asyncio
async def test_maf_lite_default_backend_builds_search_provider(patch_dependencies, mock_config):
    provider = await _build_provider(
        "strategies.maf_lite_strategy", "MafLiteStrategy", "ai_search", mock_config
    )
    assert isinstance(provider, SearchContextProvider)


@pytest.mark.asyncio
async def test_maf_lite_foundry_iq_builds_foundry_provider(patch_dependencies, mock_config):
    provider = await _build_provider(
        "strategies.maf_lite_strategy", "MafLiteStrategy", "foundry_iq", mock_config
    )
    assert isinstance(provider, FoundryIQContextProvider)


@pytest.mark.asyncio
async def test_maf_agent_service_default_backend_builds_search_provider(patch_dependencies, mock_config):
    provider = await _build_provider(
        "strategies.maf_agent_service_strategy", "MafAgentServiceStrategy", "ai_search", mock_config
    )
    assert isinstance(provider, SearchContextProvider)


@pytest.mark.asyncio
async def test_maf_agent_service_foundry_iq_builds_foundry_provider(patch_dependencies, mock_config):
    provider = await _build_provider(
        "strategies.maf_agent_service_strategy", "MafAgentServiceStrategy", "foundry_iq", mock_config
    )
    assert isinstance(provider, FoundryIQContextProvider)


@pytest.mark.asyncio
async def test_multimodal_default_backend_builds_multimodal_provider(patch_dependencies, mock_config):
    provider = await _build_provider(
        "strategies.multimodal_strategy", "MultimodalStrategy", "ai_search", mock_config
    )
    assert isinstance(provider, MultimodalSearchContextProvider)


@pytest.mark.asyncio
async def test_multimodal_foundry_iq_builds_foundry_provider_text_only(patch_dependencies, mock_config):
    provider = await _build_provider(
        "strategies.multimodal_strategy", "MultimodalStrategy", "foundry_iq", mock_config
    )
    # Decision #5: multimodal stays on ai_search for v3.0.0 image grounding, but
    # the selector still routes foundry_iq to the text-only Foundry IQ provider.
    assert isinstance(provider, FoundryIQContextProvider)
    assert not isinstance(provider, MultimodalSearchContextProvider)
