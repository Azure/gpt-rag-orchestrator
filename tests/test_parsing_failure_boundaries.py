"""Malformed input recovery is distinct from unexpected implementation failure."""

import base64
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from connectors.search import SearchClient
from strategies.multimodal_search_context_provider import _extract_blob_relative_path
from util.jwt_utils import decode_jwt_claims_unverified


def _token(payload):
    encoded = base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")
    return f"header.{encoded}.signature"


@pytest.mark.parametrize("token", [
    "", "header", "header.x.signature", "header.\ud800.signature",
    _token(b"\xff"), _token(b"not-json"), _token(b"null"), _token(b"[]"),
    _token(b"[" * 2000 + b"]" * 2000), _token(b"9" * 5000),
], ids=["empty", "no-payload", "base64", "surrogate", "utf8", "json", "null",
        "array", "nested", "integer-limit"])
def test_unverified_claims_retain_malformed_input_recovery(token):
    assert decode_jwt_claims_unverified(token) == {}


def test_unverified_claims_parse_without_authorizing():
    assert decode_jwt_claims_unverified(_token(b'{"sub":"synthetic-subject"}')) == {
        "sub": "synthetic-subject"}


def test_unverified_claims_do_not_hide_unexpected_decoder_failure(monkeypatch):
    error = RuntimeError("decoder implementation failure")
    monkeypatch.setattr("util.jwt_utils.json.loads", MagicMock(side_effect=error))
    with pytest.raises(RuntimeError) as raised:
        decode_jwt_claims_unverified(_token(b"{}"))
    assert raised.value is error


def test_unverified_claims_preserve_json_recursion_recovery(monkeypatch):
    monkeypatch.setattr("util.jwt_utils.json.loads", MagicMock(side_effect=RecursionError))
    assert decode_jwt_claims_unverified(_token(b"{}")) == {}


@pytest.mark.parametrize("url,expected", [
    ("https://account.example.invalid/images/path/image.png", "path/image.png"),
    ("https://account.example.invalid/images", None),
    ("https://[invalid/images/path.png", None),
    ("https://example.invalid\uff0f/images/path.png", None),
    ("", None),
])
def test_blob_relative_path_keeps_malformed_url_recovery(url, expected):
    assert _extract_blob_relative_path(url) == expected


def test_blob_relative_path_does_not_hide_unexpected_parser_failure(monkeypatch):
    error = RuntimeError("parser implementation failure")
    monkeypatch.setattr("urllib.parse.urlparse", MagicMock(side_effect=error))
    with pytest.raises(RuntimeError) as raised:
        _extract_blob_relative_path("https://example.invalid/images/path.png")
    assert raised.value is error


@pytest.fixture
def search_token_client():
    client = SearchClient.__new__(SearchClient)
    values = {
        "OAUTH_AZURE_AD_TENANT_ID": "synthetic-tenant",
        "OAUTH_AZURE_AD_CLIENT_ID": "synthetic-client",
        "OAUTH_AZURE_AD_CLIENT_SECRET": "synthetic-secret",
    }
    client.cfg = MagicMock()
    client.cfg.get_value.side_effect = lambda key, **kwargs: values.get(key)
    client.index_name = "synthetic-index"
    return client


def _respond(client, raw, status=200):
    response = MagicMock()
    response.status = status
    response.text = AsyncMock(return_value=raw)
    response.__aenter__ = AsyncMock(return_value=response)
    response.__aexit__ = AsyncMock(return_value=False)
    session = MagicMock()
    session.post.return_value = response
    client._get_session = AsyncMock(return_value=session)
    return session


@pytest.mark.parametrize("raw", ["not-json", "[" * 2000, "9" * 5000],
                         ids=["json", "incomplete-nesting", "integer-limit"])
async def test_search_token_malformed_json_remains_explicitly_unavailable(search_token_client, raw):
    _respond(search_token_client, raw)
    assert await search_token_client._acquire_search_user_token_via_obo("synthetic-assertion") is None


@pytest.mark.parametrize("expiry", ["invalid", {}, [], float("inf")])
async def test_search_token_invalid_expiry_does_not_reuse_token(search_token_client, expiry, monkeypatch):
    monkeypatch.setattr("connectors.search.time.time", lambda: 100.0)
    session = _respond(search_token_client, json.dumps({
        "access_token": "synthetic-delegated-token", "expires_in": expiry}))
    token = await search_token_client._acquire_search_user_token_via_obo("synthetic-assertion")
    assert token == "synthetic-delegated-token"
    assert search_token_client._cached_search_user_token_expires_at == 100.0
    assert session.post.call_args.kwargs["data"]["assertion"] == "synthetic-assertion"
    assert session.post.call_args.kwargs["data"]["scope"] == "https://search.azure.com/user_impersonation"


@pytest.mark.parametrize("status", [200, 400])
async def test_search_token_does_not_hide_unexpected_json_decoder_failure(
    search_token_client, monkeypatch, status,
):
    _respond(search_token_client, "{}", status=status)
    error = RuntimeError("decoder implementation failure")
    monkeypatch.setattr("connectors.search.json.loads", MagicMock(side_effect=error))
    with pytest.raises(RuntimeError) as raised:
        await search_token_client._acquire_search_user_token_via_obo("synthetic-assertion")
    assert raised.value is error


@pytest.mark.parametrize("raw", ['{"error":"synthetic-rejection"}', "null", "[]", "invalid-json"])
async def test_search_token_endpoint_rejection_retains_explicit_error(search_token_client, raw):
    _respond(search_token_client, raw, status=400)
    assert await search_token_client._acquire_search_user_token_via_obo("synthetic-assertion") is None
    assert search_token_client._last_obo_error.startswith("status=400 ")
    if raw.startswith("{"):
        assert "error=synthetic-rejection" in search_token_client._last_obo_error
    else:
        assert f"body={raw}" in search_token_client._last_obo_error


async def test_search_token_preserves_json_recursion_recovery(search_token_client, monkeypatch):
    _respond(search_token_client, "{}")
    monkeypatch.setattr("connectors.search.json.loads", MagicMock(side_effect=RecursionError))
    assert await search_token_client._acquire_search_user_token_via_obo("synthetic-assertion") is None


@pytest.mark.parametrize("token,expected", [(None, "<none>"), ("\ud800", "<unknown>")])
def test_search_token_fingerprint_keeps_invalid_unicode_recovery(search_token_client, token, expected):
    assert search_token_client._token_fingerprint(token) == expected
