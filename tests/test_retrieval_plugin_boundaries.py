"""Real legacy retrieval boundaries preserve typed errors and valid partial results."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
from azure.core.exceptions import ClientAuthenticationError

from plugins.retrieval.plugin import RetrievalPlugin


@pytest.fixture
def plugin(mock_config):
    model = SimpleNamespace(get_embeddings=AsyncMock(return_value=[0.1, 0.2]))
    with (
        patch("plugins.retrieval.plugin.get_config", return_value=mock_config),
        patch("plugins.retrieval.plugin.get_genai_client", return_value=model),
    ):
        return RetrievalPlugin()


@pytest.mark.parametrize("mode", ["sdk", "unexpected", "cancelled", "success"])
async def test_search_credential_boundary_translates_only_sdk_errors(plugin, mode, caplog):
    marker = "synthetic-private-credential-detail"
    failure = {
        "sdk": ClientAuthenticationError(marker),
        "unexpected": RuntimeError(marker),
        "cancelled": asyncio.CancelledError(marker),
    }.get(mode)
    credential = SimpleNamespace(
        get_token=MagicMock(side_effect=failure, return_value=SimpleNamespace(token="synthetic-token"))
    )
    identity = SimpleNamespace(get_credential=MagicMock(return_value=credential))
    with patch("plugins.retrieval.plugin.get_identity_manager", return_value=identity):
        if failure:
            with pytest.raises(Exception if mode == "sdk" else type(failure)) as raised:
                await plugin._get_azure_search_token()
            if mode == "sdk":
                assert str(raised.value) == "Failed to obtain Azure Search token."
                assert raised.value.__cause__ is failure
            else:
                assert raised.value is failure
        else:
            assert await plugin._get_azure_search_token() == "synthetic-token"
    credential.get_token.assert_called_once_with("https://search.azure.com/.default")
    assert marker not in caplog.text


@pytest.mark.parametrize("mode", ["400", "401", "403", "500", "connection", "json", "unexpected", "cancelled", "success"])
async def test_search_http_boundary_bounds_details_and_closes_contexts(plugin, mode, caplog):
    marker = "synthetic-private-http-detail"
    response = MagicMock()
    response.status = int(mode) if mode.isdigit() else 200
    failure = (
        aiohttp.ClientResponseError(SimpleNamespace(real_url="https://search.invalid"), (),
                                    status=response.status, message=marker)
        if mode.isdigit()
        else {
            "connection": aiohttp.ClientConnectionError(marker),
            "json": ValueError(marker),
            "unexpected": RuntimeError(marker),
            "cancelled": asyncio.CancelledError(marker),
        }.get(mode)
    )
    response.raise_for_status.side_effect = failure if mode.isdigit() else None
    response.text = AsyncMock(return_value=marker)
    response.json = AsyncMock(
        return_value={"value": []},
        side_effect=failure if mode in {"json", "unexpected", "cancelled"} else None,
    )
    request = MagicMock()
    request.__aenter__ = AsyncMock(return_value=response)
    request.__aexit__ = AsyncMock(return_value=False)
    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    session.post = MagicMock(return_value=request, side_effect=failure if mode == "connection" else None)
    headers, body = {"test-header": "value"}, {"filter": "existing-filter"}
    with patch("plugins.retrieval.plugin.aiohttp.ClientSession", return_value=session):
        if failure:
            with pytest.raises(type(failure) if mode in {"unexpected", "cancelled"} else Exception) as raised:
                await plugin._perform_search("https://search.invalid", headers, body)
            if mode in {"unexpected", "cancelled"}:
                assert raised.value is failure
            else:
                assert str(raised.value) == "Failed to execute search query."
                assert raised.value.__cause__ is failure
        else:
            assert await plugin._perform_search("https://search.invalid", headers, body) == {"value": []}
    session.post.assert_called_once_with("https://search.invalid", headers=headers, json=body)
    session.__aexit__.assert_awaited_once()
    if mode != "connection":
        request.__aexit__.assert_awaited_once()
    response.text.assert_not_awaited()
    assert marker not in caplog.text


@pytest.mark.parametrize("method", ["vector_index_retrieve", "multimodal_vector_index_retrieve"], ids=["vector", "multimodal"])
@pytest.mark.parametrize("stage", [
    "embedding", "token", "search", "malformed", "success",
    "cancelled-embedding", "cancelled-token", "cancelled-search",
])
async def test_retrieval_tools_preserve_filtered_partial_results_and_explicit_errors(plugin, method, stage, caplog):
    marker = "synthetic-private-retrieval-detail"
    cancelled = stage.startswith("cancelled-")
    site = stage.removeprefix("cancelled-")
    failure = asyncio.CancelledError(marker) if cancelled else RuntimeError(marker)
    image = "https://storage.invalid/image.png"
    doc = {
        "url": "https://example.blob.core.windows.net/docs/item.pdf",
        "content": " content ",
        "imageCaptions": "[image]: caption",
        "relatedImages": [image],
    }
    docs = [doc]
    if stage == "malformed":
        docs.append({**doc, "content": {"detail": marker}})
    plugin.aoai.get_embeddings.side_effect = failure if site == "embedding" else None
    plugin._get_azure_search_token = AsyncMock(
        return_value="synthetic-search-token", side_effect=failure if site == "token" else None
    )
    plugin._perform_search = AsyncMock(
        return_value={"value": docs}, side_effect=failure if site == "search" else None
    )
    if cancelled:
        with pytest.raises(asyncio.CancelledError) as raised:
            await getattr(plugin, method)("question", "group-a,group-b")
        assert raised.value is failure
    else:
        result = await getattr(plugin, method)("question", "group-a,group-b")
        assert (result.error is None) == (stage == "success")
        assert marker not in result.model_dump_json()
        if stage in {"embedding", "token", "search"}:
            assert "RuntimeError" in result.error
        has_valid_doc = stage in {"malformed", "success"}
        if method == "vector_index_retrieve":
            assert result.result == ("/docs/item.pdf: content\n" if has_valid_doc else "")
        else:
            assert result.texts == (["/docs/item.pdf: content"] if has_valid_doc else [])
            assert result.images == ([[image]] if has_valid_doc else [])
            assert result.captions == ([["caption"]] if has_valid_doc else [])
    if site in {"embedding", "token"}:
        plugin._perform_search.assert_not_awaited()
    else:
        body = plugin._perform_search.await_args.args[2]
        assert body["filter"] == (
            "metadata_security_id/any(g:search.in(g, 'group-a,group-b')) or not metadata_security_id/any()"
        )
        assert body["vectorQueries"][0]["vector"] == [0.1, 0.2]
    assert marker not in caplog.text


@pytest.mark.parametrize("method", ["vector_index_retrieve", "multimodal_vector_index_retrieve"], ids=["vector", "multimodal"])
@pytest.mark.parametrize("stage", ["credential", "unexpected-credential", "http", "success"])
async def test_real_retrieval_chain_keeps_failed_credentials_off_the_network(plugin, method, stage, caplog):
    marker = "synthetic-private-chain-detail"
    token_failure = (
        ClientAuthenticationError(marker) if stage == "credential"
        else RuntimeError(marker) if stage == "unexpected-credential"
        else None
    )
    credential = SimpleNamespace(get_token=MagicMock(
        return_value=SimpleNamespace(token="synthetic-search-token"), side_effect=token_failure
    ))
    identity = SimpleNamespace(get_credential=MagicMock(return_value=credential))
    response = MagicMock()
    response.raise_for_status.side_effect = (
        aiohttp.ClientResponseError(SimpleNamespace(real_url="https://search.invalid"), (), status=403, message=marker)
        if stage == "http" else None
    )
    response.json = AsyncMock(return_value={"value": []})
    request = MagicMock()
    request.__aenter__ = AsyncMock(return_value=response)
    request.__aexit__ = AsyncMock(return_value=False)
    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    session.post = MagicMock(return_value=request)
    with (
        patch("plugins.retrieval.plugin.get_identity_manager", return_value=identity),
        patch("plugins.retrieval.plugin.aiohttp.ClientSession", return_value=session),
    ):
        result = await getattr(plugin, method)("question", "group-a")
    assert (result.error is None) == (stage == "success")
    assert marker not in result.model_dump_json()
    assert marker not in caplog.text
    if token_failure:
        session.post.assert_not_called()
        session.__aenter__.assert_not_awaited()
    else:
        assert session.post.call_args.kwargs["headers"]["Authorization"] == "Bearer synthetic-search-token"
        request.__aexit__.assert_awaited_once()
        session.__aexit__.assert_awaited_once()
