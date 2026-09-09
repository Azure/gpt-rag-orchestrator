"""ADR-0006: real factory/provider authorization, with no application ACL bypass."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agent_framework import ChatMessage, Role

from connectors.obo import RetrievalAuthorizationMode, resolve_retrieval_authorization
from orchestration.orchestrator import Orchestrator
from strategies import foundry_iq_context_provider as foundry
from strategies import multimodal_search_context_provider as vision
from strategies import search_context_provider as text
from test_context_provider_boundary_dispositions import _sdk_search
from test_strategy_helper_boundaries import strategy


@pytest.mark.parametrize("backend", ["ai_search", "foundry_iq", "mcp"])
@pytest.mark.parametrize("anonymous", [False, True])
@pytest.mark.parametrize("assertion", [None, "synthetic-user-assertion"])
@pytest.mark.parametrize("outcome", ["token", "none", "failure", "cancelled"])
async def test_factory_authorization_matrix(
    strategy, mock_config, backend, anonymous, assertion, outcome, monkeypatch, caplog,
):
    module, instance = strategy
    original_get = mock_config.get.side_effect
    mock_config.get.side_effect = lambda key, default=None, type=str: (
        anonymous if key == "ALLOW_ANONYMOUS"
        else backend == "mcp" if key == "FOUNDRY_IQ_MCP_ENABLED"
        else original_get(key, default, type)
    )
    instance.request_access_token = assertion
    # Untrusted request body identity is irrelevant to the policy decision.
    instance.user_context = {"principal_id": "spoofed", "user_id": "spoofed"}
    marker = "synthetic-sensitive-provider-marker"
    failure = asyncio.CancelledError(marker) if outcome == "cancelled" else RuntimeError(marker)
    obo = AsyncMock(
        return_value="synthetic-delegated" if outcome == "token" else None,
        side_effect=failure if outcome in {"failure", "cancelled"} else None,
    )
    documents = [{"id": "one", "title": "Authorized", "content": "authorized content"}]
    sdk = _sdk_search(documents, lambda _: None)
    remote = SimpleNamespace(
        mcp_config=SimpleNamespace(enabled=backend == "mcp", sources=()),
        retrieve=AsyncMock(return_value=documents),
    )
    with (
        patch.object(module, "get_retrieval_backend", return_value="ai_search" if backend == "ai_search" else "foundry_iq"),
        patch.object(module, "acquire_obo_search_token", obo),
        patch.object(text, "SearchClient", return_value=sdk),
        patch.object(vision, "SearchClient", return_value=sdk),
        patch.object(foundry, "get_foundry_iq_client", return_value=remote),
    ):
        provider = await instance._create_search_provider()
        required = bool(assertion) or not anonymous
        assert provider._authorization_mode is (
            RetrievalAuthorizationMode.USER_REQUIRED if required else RetrievalAuthorizationMode.SERVICE_ONLY
        )
        denied = required and (assertion is None or outcome != "token")
        if denied:
            expected = asyncio.CancelledError if assertion and outcome == "cancelled" else RuntimeError
            with pytest.raises(expected):
                await provider.invoking(ChatMessage(role=Role.USER, text="question"))
            sdk.search.assert_not_awaited()
            remote.retrieve.assert_not_awaited()
            if isinstance(provider, vision.MultimodalSearchContextProvider):
                assert provider.image_data == {}
        else:
            result = await provider.invoking(ChatMessage(role=Role.USER, text="question"))
            assert "authorized content" in result.messages[0].text
            if backend == "ai_search":
                assert ("x_ms_query_source_authorization" in sdk.search.await_args.kwargs) is required
            else:
                assert remote.retrieve.await_args.kwargs["obo_token"] == ("synthetic-delegated" if required else None)
        if assertion and required:
            obo.assert_awaited_once_with(assertion, allow_anonymous=False)
        else:
            obo.assert_not_awaited()
    assert marker not in caplog.text
    assert "synthetic-delegated" not in caplog.text


@pytest.mark.parametrize("kind", ["text", "vision", "foundry"])
@pytest.mark.parametrize("outcome", ["absent", "none", "blank", "failure", "cancelled"])
async def test_omitted_mode_defaults_closed_before_content_or_images(kind, outcome, mock_config, caplog):
    marker = "synthetic-protected-content"
    failure = asyncio.CancelledError(marker) if outcome == "cancelled" else RuntimeError(marker)
    callback = None if outcome == "absent" else AsyncMock(
        return_value=" " if outcome == "blank" else None,
        side_effect=failure if outcome in {"failure", "cancelled"} else None,
    )
    sdk = _sdk_search([{
        "content": marker, "filepath": marker, "relatedImages": [marker],
    }], lambda _: None)
    remote = SimpleNamespace(mcp_config=None, retrieve=AsyncMock())
    common = dict(get_obo_token=callback)
    if kind == "foundry":
        provider = foundry.FoundryIQContextProvider(**common)
    else:
        common.update(endpoint="https://search.invalid", index_name="documents", credential=MagicMock())
        provider = (
            text.SearchContextProvider(**common) if kind == "text"
            else vision.MultimodalSearchContextProvider(**common, blob_credential=MagicMock())
        )
    with (
        patch.object(text, "SearchClient", return_value=sdk),
        patch.object(vision, "SearchClient", return_value=sdk),
        patch.object(foundry, "get_foundry_iq_client", return_value=remote),
        patch.object(vision.AzureBlobClient, "from_blob_url") as blob,
    ):
        if kind == "vision":
            provider.image_data["old-user-image"] = marker
        with pytest.raises(asyncio.CancelledError if outcome == "cancelled" else RuntimeError):
            await provider.invoking(ChatMessage(role=Role.USER, text="question"))
        sdk.search.assert_not_awaited()
        remote.retrieve.assert_not_awaited()
        blob.assert_not_called()
        if kind == "vision":
            assert provider.image_data == {}
    assert marker not in caplog.text


@pytest.mark.parametrize("source", [
    "work_iq_enabled", "fabric_iq_enabled", "fabric_data_agent_enabled",
    "sharepoint_remote_enabled", "mcp-obo", "mixed-mcp",
])
async def test_source_required_user_cannot_be_bypassed_by_service_mode(source, strategy):
    module, instance = strategy
    def configured_source(kind):
        return SimpleNamespace(query_headers=[SimpleNamespace(value_from=SimpleNamespace(kind=kind))])

    sources = [configured_source("obo")]
    if source == "mixed-mcp":
        sources.insert(0, configured_source("managedIdentity"))
    remote = SimpleNamespace(
        mcp_config=SimpleNamespace(enabled=source.endswith("mcp") or source == "mcp-obo", sources=sources),
        retrieve=AsyncMock(return_value=[{"content": "protected"}]),
    )
    if not source.startswith("mcp") and source != "mixed-mcp":
        setattr(remote, source, True)
    with (
        patch.object(foundry, "get_foundry_iq_client", return_value=remote),
        patch.object(module, "get_retrieval_backend", return_value="foundry_iq"),
    ):
        provider = await instance._create_search_provider()
        assert provider._authorization_mode is RetrievalAuthorizationMode.SERVICE_ONLY
        with pytest.raises(RuntimeError, match="authorization unavailable"):
            await provider.invoking(ChatMessage(role=Role.USER, text="question"))
    remote.retrieve.assert_not_awaited()


@pytest.mark.parametrize("kind", ["text", "vision"])
async def test_iterator_failure_discards_partial_context_without_retry(kind):
    failure = RuntimeError("synthetic-sensitive-enumeration")
    sdk = _sdk_search([], lambda _: None)

    async def rows():
        yield {"id": "one", "title": "Protected", "content": "must not reach model", "relatedImages": ["protected-image"]}
        raise failure

    sdk.search.side_effect = None
    sdk.search.return_value = rows()
    module = text if kind == "text" else vision
    kwargs = dict(
        endpoint="https://search.invalid", index_name="documents", credential=MagicMock(),
        get_obo_token=AsyncMock(return_value="synthetic-delegated"),
    )
    provider = text.SearchContextProvider(**kwargs) if kind == "text" else vision.MultimodalSearchContextProvider(
        **kwargs, blob_credential=MagicMock(),
    )
    with patch.object(module, "SearchClient", return_value=sdk), patch.object(vision.AzureBlobClient, "from_blob_url") as blob:
        with pytest.raises(RuntimeError) as caught:
            await provider.invoking(ChatMessage(role=Role.USER, text="question"))
    assert caught.value is failure
    sdk.search.assert_awaited_once()
    sdk.__aexit__.assert_awaited_once()
    blob.assert_not_called()


@pytest.mark.parametrize("bad", ["unknown", None, True])
async def test_unknown_mode_is_not_an_anonymous_bypass(bad):
    provider = text.SearchContextProvider(
        endpoint="https://search.invalid", index_name="documents", credential=MagicMock(),
        authorization_mode=bad,
    )
    with patch.object(text, "SearchClient") as sdk:
        with pytest.raises(ValueError):
            await provider.invoking(ChatMessage(role=Role.USER, text="question"))
    sdk.assert_not_called()


@pytest.mark.parametrize("assertion,anonymous,expected", [
    ("invalid-but-present", True, RetrievalAuthorizationMode.USER_REQUIRED),
    (None, False, RetrievalAuthorizationMode.USER_REQUIRED),
    (None, True, RetrievalAuthorizationMode.SERVICE_ONLY),
    ("  ", True, RetrievalAuthorizationMode.SERVICE_ONLY),
])
def test_trusted_policy_does_not_treat_present_assertion_as_anonymous(assertion, anonymous, expected):
    assert resolve_retrieval_authorization(assertion, anonymous) is expected


async def test_creation_clears_previous_request_token_and_context(patch_dependencies, mock_config, mock_cosmos):
    instance = SimpleNamespace(request_access_token="prior-user", user_context={"principal_id": "prior"})
    instance.set_context = MagicMock()
    with (
        patch("orchestration.orchestrator.AgentStrategyFactory.get_strategy", AsyncMock(return_value=instance)),
        patch("orchestration.orchestrator.get_config", return_value=mock_config),
        patch("orchestration.orchestrator.get_cosmosdb_client", return_value=mock_cosmos),
    ):
        first = await Orchestrator.create(
            conversation_id="first", user_context={"principal_id": "first"}, request_access_token="first-token",
        )
        assert first.agentic_strategy.request_access_token == "first-token"
        second = await Orchestrator.create(conversation_id=None, user_context={}, request_access_token=None)
    assert second.agentic_strategy.request_access_token is None
    assert second.agentic_strategy.user_context == {}
    assert instance.set_context.call_args.args == (None,)
