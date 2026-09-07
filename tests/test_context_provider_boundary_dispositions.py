"""Legacy provider recovery is characterized, not approved as strict OBO enforcement."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agent_framework import ChatMessage, Role

from connectors.foundry_iq_mcp import McpConfigurationError, McpCredentialError
from connectors.search import build_conversation_filter
from strategies import foundry_iq_context_provider as foundry
from strategies import maf_agent_service_strategy as service
from strategies import maf_lite_strategy as lite
from strategies import multimodal_search_context_provider as vision
from strategies import multimodal_strategy as multimodal
from strategies import search_context_provider as text


def _sdk_search(documents, fail):
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)

    async def rows():
        for document in documents:
            yield document

    async def search(**kwargs):
        failure = fail(client.search.await_count)
        if failure:
            raise failure
        return rows()

    client.search = AsyncMock(side_effect=search)
    return client


@pytest.mark.parametrize("provider_kind", ["text", "vision"])
@pytest.mark.parametrize("mode", [
    "obo-error", "obo-none", "success", "search-no-obo", "search-with-obo",
    "retry-error", "cancelled-obo", "cancelled-search", "embedding-error", "cancelled-embedding",
])
async def test_search_provider_keeps_legacy_service_identity_recovery(
    mock_config, provider_kind, mode, caplog,
):
    marker = "synthetic-private-provider-detail"
    failure = asyncio.CancelledError(marker) if mode.startswith("cancelled-") else RuntimeError(marker)
    obo = AsyncMock(
        return_value=None if mode in {"obo-none", "search-no-obo"} else "synthetic-delegated",
        side_effect=failure if mode in {"obo-error", "cancelled-obo"} else None,
    )
    embed = AsyncMock(return_value=[0.1], side_effect=failure if mode in {"embedding-error", "cancelled-embedding"} else None)

    def fail(attempt):
        if mode in {"search-no-obo", "retry-error", "cancelled-search"}:
            return failure
        if mode == "search-with-obo" and attempt == 1:
            return failure
        return None

    sdk = _sdk_search([{"id": "one", "title": "Document", "filepath": "file.pdf", "content": "service identity context"}], fail)
    module = text if provider_kind == "text" else vision
    kwargs = dict(
        endpoint="https://search.invalid", index_name="documents", credential=MagicMock(),
        conversation_id="chat'quoted", get_obo_token=obo, embed_fn=embed,
    )
    if provider_kind == "vision":
        kwargs["blob_credential"] = MagicMock()
        provider = vision.MultimodalSearchContextProvider(**kwargs)
    else:
        provider = text.SearchContextProvider(**kwargs)
    with (
        patch.object(module, "get_config", return_value=mock_config),
        patch.object(module, "SearchClient", return_value=sdk),
    ):
        if mode.startswith("cancelled-"):
            with pytest.raises(asyncio.CancelledError) as raised:
                await provider.invoking(ChatMessage(role=Role.USER, text="question"))
            assert raised.value is failure
        else:
            context = await provider.invoking(ChatMessage(role=Role.USER, text="question"))
            empty = mode in {"search-no-obo", "retry-error"} or (mode == "search-with-obo" and provider_kind == "text")
            assert bool(context.messages) is not empty
            if not empty:
                assert "service identity context" in context.messages[0].text
    if mode in {"cancelled-obo", "cancelled-embedding"}:
        sdk.search.assert_not_awaited()
    else:
        first = sdk.search.await_args_list[0].kwargs
        assert first["filter"] == build_conversation_filter("chat'quoted")
        assert ("vector_queries" in first) == (mode != "embedding-error")
        if mode in {"obo-error", "obo-none", "search-no-obo"}:
            assert "x_ms_query_source_authorization" not in first
        else:
            assert first["x_ms_query_source_authorization"] == (
                "synthetic-delegated" if provider_kind == "text" else "Bearer synthetic-delegated"
            )
        retries = provider_kind == "vision" and mode in {"search-with-obo", "retry-error"}
        assert sdk.search.await_count == (2 if retries else 1)
        if retries:
            assert "x_ms_query_source_authorization" not in sdk.search.await_args_list[1].kwargs
        assert sdk.__aexit__.await_count == sdk.search.await_count
    assert marker not in caplog.text
    assert not any(record.exc_info for record in caplog.records)


@pytest.mark.parametrize("module,strategy_type", [
    (lite, lite.MafLiteStrategy), (service, service.MafAgentServiceStrategy),
    (multimodal, multimodal.MultimodalStrategy),
], ids=["lite", "service", "vision"])
@pytest.mark.parametrize("callback_failure", [False, True])
async def test_actual_non_mcp_strategy_callback_does_not_enforce_strict_obo(
    patch_dependencies, mock_config, module, strategy_type, callback_failure, caplog,
):
    marker = "synthetic-private-obo-detail"
    read = mock_config.get.side_effect
    mock_config.get.side_effect = lambda key, default=None, type=str: False if key == "ALLOW_ANONYMOUS" else read(key, default, type)
    obo = AsyncMock(return_value=None, side_effect=RuntimeError(marker) if callback_failure else None)
    sdk = _sdk_search([{"id": "one", "title": "Document", "filepath": "file.pdf", "content": "service identity context"}], lambda _: None)
    with (
        patch.object(module, "get_config", return_value=mock_config),
        patch.object(module, "get_retrieval_backend", return_value="ai_search"),
        patch.object(module, "acquire_obo_search_token", obo),
        patch.object(text, "get_config", return_value=mock_config),
        patch.object(text, "SearchClient", return_value=sdk),
        patch.object(vision, "get_config", return_value=mock_config),
        patch.object(vision, "SearchClient", return_value=sdk),
    ):
        strategy = strategy_type()
        strategy.search_endpoint = "https://search.invalid"
        strategy.search_index_name = "documents"
        strategy.embedding_deployment = None
        strategy.request_access_token = "synthetic-incoming"
        provider = await strategy._create_search_provider()
        assert provider is not None
        context = await provider.invoking(ChatMessage(role=Role.USER, text="question"))
    obo.assert_awaited_once_with("synthetic-incoming", allow_anonymous=True)
    assert "x_ms_query_source_authorization" not in sdk.search.await_args.kwargs
    assert "service identity context" in context.messages[0].text
    assert marker not in caplog.text
    assert not any(record.exc_info for record in caplog.records)


@pytest.mark.parametrize("mcp", [False, True])
@pytest.mark.parametrize("anonymous", [False, True], ids=["strict-setting", "anonymous-setting"])
@pytest.mark.parametrize("mode", ["obo-error", "obo-none", "provider-error", "configuration", "malformed", "cancelled", "success"])
async def test_foundry_context_retains_mcp_specific_failure_rules(mcp, anonymous, mode, caplog):
    marker = "synthetic-private-foundry-context-detail"
    failure = (
        asyncio.CancelledError(marker) if mode == "cancelled"
        else McpConfigurationError(marker) if mode == "configuration"
        else RuntimeError(marker)
    )
    obo = AsyncMock(
        return_value=None if mode == "obo-none" else "synthetic-delegated",
        side_effect=failure if mode in {"obo-error", "cancelled"} else None,
    )
    client = SimpleNamespace(
        mcp_config=SimpleNamespace(enabled=mcp),
        retrieve=AsyncMock(
            return_value=[None] if mode == "malformed" else [{"title": "Document", "link": "file.pdf", "content": "grounding"}],
            side_effect=failure if mode in {"provider-error", "configuration"} else None,
        ),
    )
    provider = foundry.FoundryIQContextProvider(
        get_obo_token=obo, conversation_id="chat", request_access_token="synthetic-incoming",
        allow_anonymous=anonymous, mcp_enabled=mcp, user_context={"principal_id": "principal"},
    )
    credential_failure = mcp and (mode == "obo-error" or (mode == "obo-none" and not anonymous))
    with patch.object(foundry, "get_foundry_iq_client", return_value=client):
        if mode == "cancelled" or credential_failure or (mcp and mode == "configuration"):
            expected = McpCredentialError if credential_failure else type(failure)
            with pytest.raises(expected) as raised:
                await provider.invoking(ChatMessage(role=Role.USER, text="question"))
            if credential_failure:
                assert marker not in str(raised.value)
            else:
                assert raised.value is failure
        else:
            context = await provider.invoking(ChatMessage(role=Role.USER, text="question"))
            assert bool(context.messages) == (mode not in {"provider-error", "configuration", "malformed"})
    if mode == "cancelled" or credential_failure:
        client.retrieve.assert_not_awaited()
    else:
        kwargs = {
            "obo_token": None if mode in {"obo-error", "obo-none"} else "synthetic-delegated",
            "conversation_id": "chat", "user_context": {"principal_id": "principal"},
        }
        if mcp:
            kwargs["incoming_token"] = "synthetic-incoming"
        client.retrieve.assert_awaited_once_with("question", **kwargs)
    assert marker not in caplog.text
    assert not any(record.exc_info for record in caplog.records)


@pytest.mark.parametrize("mode", [
    "classifier-error", "rejected", "cancelled-classifier", "success", "no-classifier",
    "construct", "download", "read", "cleanup", "invalid-bytes", "cancelled-download",
])
async def test_multimodal_optional_images_keep_text_and_close_blob_context(mock_config, mode, caplog):
    marker = "synthetic-private-image-detail"
    url_marker = "synthetic-private-url-signature"
    blob_url = f"https://example.blob.core.windows.net/images/diagram.png?sig={url_marker}"
    failure = asyncio.CancelledError(marker) if mode.startswith("cancelled-") else RuntimeError(marker)
    classify = AsyncMock(
        return_value=mode != "rejected", side_effect=failure if mode in {"classifier-error", "cancelled-classifier"} else None,
    )
    provider = vision.MultimodalSearchContextProvider(
        endpoint="https://search.invalid", index_name="documents", credential=MagicMock(),
        blob_credential=MagicMock(), conversation_id="chat",
        classify_images_fn=None if mode == "no-classifier" else classify,
    )
    sdk = _sdk_search([{
        "id": "one", "title": "Document", "filepath": "file.pdf",
        "content": "procedure <figure>images/diagram.png</figure> diagram",
        "relatedImages": [blob_url], "imageCaptions": "[images/diagram.png]: procedure diagram",
    }], lambda _: None)
    blob = MagicMock()
    blob.__aenter__ = AsyncMock(return_value=blob)
    blob.__aexit__ = AsyncMock(return_value=False, side_effect=failure if mode == "cleanup" else None)
    blob.download_blob = AsyncMock(
        return_value=SimpleNamespace(readall=AsyncMock(
            return_value=None if mode == "invalid-bytes" else b"synthetic-image-bytes",
            side_effect=failure if mode == "read" else None,
        )),
        side_effect=failure if mode in {"download", "cancelled-download"} else None,
    )
    with (
        patch.object(vision, "get_config", return_value=mock_config),
        patch.object(vision, "SearchClient", return_value=sdk),
        patch.object(vision.AzureBlobClient, "from_blob_url", return_value=blob,
                     side_effect=failure if mode == "construct" else None) as create,
    ):
        if mode.startswith("cancelled-"):
            with pytest.raises(asyncio.CancelledError) as raised:
                await provider.invoking(ChatMessage(role=Role.USER, text="procedure diagram"))
            assert raised.value is failure
        else:
            context = await provider.invoking(ChatMessage(role=Role.USER, text="procedure diagram"))
            assert context.messages
            has_image = mode in {"classifier-error", "success", "no-classifier"}
            assert context.messages[0].text.startswith(vision.MULTIMODAL_PREFIX) is has_image
            assert bool(provider.image_data) is has_image
    create.assert_called_once_with(blob_url, credential=provider._blob_credential)
    if mode == "construct":
        blob.__aexit__.assert_not_awaited()
    else:
        blob.__aexit__.assert_awaited_once()
    if mode in {"construct", "download", "read", "cleanup", "invalid-bytes", "cancelled-download", "no-classifier"}:
        classify.assert_not_awaited()
    else:
        classify.assert_awaited_once()
    assert marker not in caplog.text
    assert url_marker not in caplog.text
    assert not any(record.exc_info for record in caplog.records)


async def test_multimodal_retry_cancellation_propagates_without_empty_result(mock_config, caplog):
    marker = "synthetic-private-retry-cancellation"
    cancellation = asyncio.CancelledError(marker)
    sdk = _sdk_search([], lambda attempt: RuntimeError(marker) if attempt == 1 else cancellation)
    provider = vision.MultimodalSearchContextProvider(
        endpoint="https://search.invalid", index_name="documents",
        credential=MagicMock(), blob_credential=MagicMock(),
        get_obo_token=AsyncMock(return_value="synthetic-delegated"),
    )
    with (
        patch.object(vision, "get_config", return_value=mock_config),
        patch.object(vision, "SearchClient", return_value=sdk),
    ):
        with pytest.raises(asyncio.CancelledError) as raised:
            await provider.invoking(ChatMessage(role=Role.USER, text="question"))
    assert raised.value is cancellation
    assert sdk.search.await_count == sdk.__aexit__.await_count == 2
    assert "x_ms_query_source_authorization" not in sdk.search.await_args.kwargs
    assert marker not in caplog.text
