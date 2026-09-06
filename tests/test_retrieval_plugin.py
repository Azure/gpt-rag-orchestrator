"""The legacy plugin must resolve its current connector and await embeddings."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from plugins.retrieval.plugin import RetrievalPlugin


@pytest.fixture
def plugin(mock_config):
    model = MagicMock()
    model.get_embeddings = AsyncMock(return_value=[0.1, 0.2])
    with (
        patch("plugins.retrieval.plugin.get_config", return_value=mock_config),
        patch("plugins.retrieval.plugin.get_genai_client", return_value=model),
    ):
        result = RetrievalPlugin()
    result._get_azure_search_token = AsyncMock(return_value="synthetic-search-token")
    result._perform_search = AsyncMock(return_value={"value": []})
    return result


@pytest.mark.parametrize("method", ["vector_index_retrieve", "multimodal_vector_index_retrieve"])
async def test_plugin_uses_resolved_async_embeddings_and_keeps_security_filter(plugin, method):
    result = await getattr(plugin, method)("question", "group-a")
    plugin.aoai.get_embeddings.assert_awaited_once_with("question")
    body = plugin._perform_search.await_args.args[2]
    assert body["vectorQueries"][0]["vector"] == [0.1, 0.2]
    assert body["filter"] == (
        "metadata_security_id/any(g:search.in(g, 'group-a')) or not metadata_security_id/any()"
    )
    assert result.error is None


@pytest.mark.parametrize("method", ["vector_index_retrieve", "multimodal_vector_index_retrieve"])
async def test_embedding_failure_remains_an_explicit_error_result(plugin, method):
    plugin.aoai.get_embeddings.side_effect = RuntimeError("embedding unavailable")
    result = await getattr(plugin, method)("question")
    assert result.error
    assert "embedding unavailable" in result.error
    plugin._perform_search.assert_not_awaited()
