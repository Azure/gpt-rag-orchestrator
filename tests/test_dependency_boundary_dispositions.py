"""Dependency failure contracts with actual JWT, AppConfig and HTTPX boundaries."""

import asyncio
import base64
import logging
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import jwt
import pytest
from fastapi import HTTPException

import dependencies
from test_auth_failure_boundaries import _claims, _jwks, auth_config, signing_keys


MARKER = "synthetic-private-dependency-detail"


@pytest.mark.parametrize("operation,key", [
    ("jwt", "OAUTH_AZURE_AD_TENANT_ID"),
    ("jwt", "OAUTH_AZURE_AD_CLIENT_ID"),
    ("groups", "OAUTH_AZURE_AD_CLIENT_ID"),
    ("groups", "CLIENT_ID"),
    ("groups", "OAUTH_AZURE_AD_CLIENT_SECRET"),
    ("groups", "OAUTH_AZURE_AD_TENANT_ID"),
])
async def test_unexpected_config_read_is_not_reported_as_missing(
    auth_config, signing_keys, monkeypatch, operation, key,
):
    auth_config.client["OAUTH_AZURE_AD_CLIENT_SECRET"] = "synthetic-secret"
    if key == "CLIENT_ID":
        del auth_config.client["OAUTH_AZURE_AD_CLIENT_ID"]
    failure = RuntimeError(MARKER)
    original = auth_config.get_config_with_retry

    def read(name):
        if name == key:
            raise failure
        return original(name)

    monkeypatch.setattr(auth_config, "get_config_with_retry", read)
    token = jwt.encode(_claims(), signing_keys[0], algorithm="RS256", headers={"kid": "test-kid"})
    with patch.object(dependencies.httpx, "AsyncClient") as client:
        with pytest.raises(RuntimeError) as caught:
            if operation == "jwt":
                await dependencies.validate_access_token(token)
            else:
                await dependencies.get_user_groups_from_graph("test-user")
    assert caught.value is failure
    client.assert_not_called()


@pytest.mark.parametrize("outcome", ["match", "missing", "mismatch", "cancelled"])
async def test_api_key_provider_failure_preserves_legacy_env_fallback(
    auth_config, monkeypatch, caplog, outcome,
):
    monkeypatch.delenv("DISABLE_AUTH", raising=False)
    if outcome == "missing":
        monkeypatch.delenv("ORCHESTRATOR_APP_APIKEY", raising=False)
    else:
        monkeypatch.setenv("ORCHESTRATOR_APP_APIKEY", "synthetic-key")
    failure = asyncio.CancelledError(MARKER) if outcome == "cancelled" else RuntimeError(MARKER)
    monkeypatch.setattr(auth_config, "get_value", MagicMock(side_effect=failure))
    if outcome == "match":
        assert await dependencies.validate_auth(None, "synthetic-key") is True
    elif outcome == "cancelled":
        with pytest.raises(asyncio.CancelledError) as caught:
            await dependencies.validate_auth(None, "synthetic-key")
        assert caught.value is failure
    else:
        with pytest.raises(HTTPException) as caught:
            await dependencies.validate_auth(None, "wrong-key")
        assert caught.value.status_code == 401
    assert MARKER not in caplog.text
    if outcome != "cancelled":
        assert "environment fallback" in caplog.text


@pytest.mark.parametrize("enabled", [False, True])
async def test_api_key_and_dapr_precedence_is_unchanged(auth_config, monkeypatch, enabled):
    monkeypatch.setenv("DISABLE_AUTH", "true" if enabled else "false")
    monkeypatch.setenv("APP_API_TOKEN", "synthetic-dapr")
    monkeypatch.delenv("DAPR_API_TOKEN", raising=False)
    auth_config.client["ORCHESTRATOR_APP_APIKEY"] = "synthetic-api"
    assert await dependencies.validate_auth("synthetic-dapr", "wrong-api") is True
    if enabled:
        assert await dependencies.validate_auth("wrong-dapr", "synthetic-api") is True
    else:
        with pytest.raises(HTTPException) as caught:
            await dependencies.validate_auth("wrong-dapr", "synthetic-api")
        assert caught.value.status_code == 401
        assert await dependencies.validate_auth(None, "synthetic-api") is True
        with pytest.raises(HTTPException) as caught:
            await dependencies.validate_auth(None, None)
        assert caught.value.status_code == 401


@pytest.mark.parametrize("stage", ["cache", "segment_lengths", "base64", "claims", "graph_hint", "signature_hint"])
async def test_optional_jwt_diagnostics_never_log_raw_failure_or_change_validation(
    auth_config, signing_keys, monkeypatch, caplog, stage,
):
    caplog.set_level(logging.DEBUG)
    claims = _claims()
    if stage in {"graph_hint", "signature_hint"}:
        claims["aud"] = "00000003-0000-0000-c000-000000000000"
    key = signing_keys[1] if stage in {"cache", "signature_hint"} else signing_keys[0]
    token = jwt.encode(claims, key, algorithm="RS256", headers={"kid": "test-kid"})
    original_debug = logging.debug
    original_warning = logging.warning
    prefix = {
        "cache": "[Auth] Forced JWKS cache refresh",
        "segment_lengths": "[Auth] Token segment lengths:",
        "claims": "[Auth] Token claims (unverified):",
        "graph_hint": "[Auth] Incoming token audience indicates",
        "signature_hint": "[Auth] Token audience indicates",
    }.get(stage, "")
    triggered = []

    def log(original, message, *args, **kwargs):
        if prefix and message.startswith(prefix):
            triggered.append(True)
            raise RuntimeError(MARKER)
        return original(message, *args, **kwargs)

    monkeypatch.setattr(logging, "debug", lambda message, *a, **k: log(original_debug, message, *a, **k))
    monkeypatch.setattr(logging, "warning", lambda message, *a, **k: log(original_warning, message, *a, **k))
    if stage == "base64":
        original_decode = base64.urlsafe_b64decode

        def decode(value):
            if not triggered:
                triggered.append(True)
                raise RuntimeError(MARKER)
            return original_decode(value)

        monkeypatch.setattr(base64, "urlsafe_b64decode", decode)
    if stage == "cache":
        urls = dependencies._jwks_urls_for_tenant("tenant-id")
        monkeypatch.setattr(dependencies, "__cached_public_keys", {
            f"tenant-id|{urls['v2']}": {},
        })
    key_source = AsyncMock(
        return_value=_jwks(signing_keys[0]),
        side_effect=[_jwks(signing_keys[0]), _jwks(signing_keys[1])] if stage == "cache" else None,
    )
    with patch.object(dependencies, "_get_cached_public_keys", key_source):
        if stage in {"graph_hint", "signature_hint"}:
            with pytest.raises(HTTPException) as caught:
                await dependencies.validate_access_token(token)
            assert caught.value.status_code == 401
        else:
            result = await dependencies.validate_access_token(token)
            assert result["oid"] == "user-id"
    if stage != "base64":
        assert triggered
    if stage in {"graph_hint", "signature_hint"}:
        assert "Failed to emit Graph audience hint" in caplog.text
    if stage == "base64":
        assert "Failed to collect token diagnostics" in caplog.text
    assert MARKER not in caplog.text
    assert token not in caplog.text


@pytest.mark.parametrize("missing", ["client", "secret", "tenant"])
async def test_missing_optional_graph_credentials_skip_network(auth_config, missing):
    auth_config.client["OAUTH_AZURE_AD_CLIENT_SECRET"] = "synthetic-secret"
    key = {"client": "CLIENT_ID", "secret": "CLIENT_SECRET", "tenant": "TENANT_ID"}[missing]
    del auth_config.client[f"OAUTH_AZURE_AD_{key}"]
    with patch.object(dependencies.httpx, "AsyncClient") as client:
        assert await dependencies.get_user_groups_from_graph("test-user") == []
    client.assert_not_called()


@pytest.mark.parametrize("stage", ["token", "groups"])
@pytest.mark.parametrize("outcome", ["success", "http", "transport", "json", "unexpected", "cancelled"])
async def test_actual_graph_http_boundary_preserves_optional_result_and_cleanup(
    auth_config, monkeypatch, caplog, stage, outcome,
):
    auth_config.client["OAUTH_AZURE_AD_CLIENT_SECRET"] = "synthetic-secret"
    failure = {
        "transport": httpx.ConnectError(MARKER),
        "unexpected": RuntimeError(MARKER),
        "cancelled": asyncio.CancelledError(MARKER),
    }.get(outcome)
    requests = []
    clients = []
    original_client = httpx.AsyncClient

    async def send(request):
        requests.append(request)
        current = "token" if request.method == "POST" else "groups"
        if current == stage:
            if failure is not None:
                raise failure
            if outcome == "http":
                return httpx.Response(403, text=MARKER)
            if outcome == "json":
                return httpx.Response(200, text=MARKER)
        if current == "token":
            return httpx.Response(200, json={"access_token": "synthetic-graph-token"})
        return httpx.Response(200, json={"value": [{"displayName": "group-one"}, {}]})

    def make_client():
        client = original_client(transport=httpx.MockTransport(send))
        clients.append(client)
        return client

    monkeypatch.setattr(dependencies.httpx, "AsyncClient", make_client)
    if outcome == "cancelled":
        with pytest.raises(asyncio.CancelledError) as caught:
            await dependencies.get_user_groups_from_graph("test-user")
        assert caught.value is failure
    else:
        result = await dependencies.get_user_groups_from_graph("test-user")
        assert result == (["group-one", "unknown-group"] if outcome == "success" else [])
    assert clients and all(client.is_closed for client in clients)
    assert requests[0].method == "POST"
    if len(requests) > 1:
        assert requests[1].headers["authorization"] == "Bearer synthetic-graph-token"
    assert MARKER not in caplog.text
    assert "synthetic-graph-token" not in caplog.text


@pytest.mark.parametrize("outcome", ["success", "http", "malformed", "cancelled"])
async def test_actual_jwks_http_cache_only_stores_successful_fetches(
    auth_config, monkeypatch, outcome,
):
    clients = []
    requests = []
    original_client = httpx.AsyncClient
    failure = asyncio.CancelledError(MARKER)

    async def send(request):
        requests.append(request)
        if outcome == "cancelled":
            raise failure
        if outcome == "http":
            return httpx.Response(503)
        if outcome == "malformed":
            return httpx.Response(200, text="not-json")
        return httpx.Response(200, headers={"cache-control": "max-age=60"}, json={"keys": []})

    def make_client():
        client = original_client(transport=httpx.MockTransport(send))
        clients.append(client)
        return client

    monkeypatch.setattr(dependencies.httpx, "AsyncClient", make_client)
    if outcome == "success":
        assert await dependencies._get_cached_public_keys("tenant-id") == {"keys": []}
        assert await dependencies._get_cached_public_keys("tenant-id") == {"keys": []}
        assert len(requests) == 1
        entry = next(iter(dependencies.__cached_public_keys.values()))
        assert datetime.now() < entry["expires_at"] <= datetime.now() + timedelta(seconds=60)
    else:
        error = {"http": httpx.HTTPStatusError, "malformed": ValueError, "cancelled": asyncio.CancelledError}[outcome]
        with pytest.raises(error):
            await dependencies._get_cached_public_keys("tenant-id")
        assert dependencies.__cached_public_keys == {}
    assert clients and all(client.is_closed for client in clients)


@pytest.mark.parametrize("status", [400, 500, 503])
def test_legacy_http_exception_helper_preserves_status_without_provider_detail(caplog, status):
    failure = RuntimeError(MARKER)
    with pytest.raises(HTTPException) as caught:
        dependencies.handle_exception(failure, status)
    assert caught.value.status_code == status
    assert MARKER not in str(caught.value.detail) + caplog.text


@pytest.mark.parametrize("outcome", ["success", "invalid_signature", "transport", "cancelled"])
async def test_real_jwks_provider_to_jwt_consumer_preserves_verification_and_cleanup(
    auth_config, signing_keys, monkeypatch, caplog, outcome,
):
    clients = []
    original_client = httpx.AsyncClient
    key = signing_keys[1] if outcome == "invalid_signature" else signing_keys[0]
    token = jwt.encode(_claims(), key, algorithm="RS256", headers={"kid": "test-kid"})
    cancellation = asyncio.CancelledError(MARKER)

    async def send(request):
        if outcome == "transport":
            raise httpx.ConnectError(MARKER)
        if outcome == "cancelled":
            raise cancellation
        return httpx.Response(200, json=_jwks(signing_keys[0]))

    def make_client():
        client = original_client(transport=httpx.MockTransport(send))
        clients.append(client)
        return client

    monkeypatch.setattr(dependencies.httpx, "AsyncClient", make_client)
    if outcome == "success":
        identity = await dependencies.validate_access_token(token)
        assert identity["oid"] == "user-id"
        assert identity["roles"] == ["Admin"]
    else:
        with pytest.raises(asyncio.CancelledError if outcome == "cancelled" else HTTPException) as caught:
            await dependencies.validate_access_token(token)
        if outcome == "cancelled":
            assert caught.value is cancellation
        else:
            assert caught.value.status_code == 401
            assert caught.value.detail == "Invalid token"
    assert clients and all(client.is_closed for client in clients)
    assert MARKER not in caplog.text
    assert token not in caplog.text
