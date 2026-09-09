"""Real JWT validation and bounded provider-failure translation, without Azure."""

import asyncio
import json
from unittest.mock import AsyncMock, patch

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import HTTPException

import dependencies
from connectors.appconfig import AppConfigClient


@pytest.fixture(scope="module")
def signing_keys():
    return [rsa.generate_private_key(public_exponent=65537, key_size=2048) for _ in range(2)]


def _jwks(key):
    public = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(key.public_key()))
    return {"keys": [{**public, "kid": "test-kid", "use": "sig"}]}


def _claims():
    return {
        "tid": "tenant-id",
        "iss": "https://login.microsoftonline.com/tenant-id/v2.0",
        "aud": "client-id",
        "oid": "user-id",
        "preferred_username": "user@example.invalid",
        "name": "Test User",
        "roles": ["Admin"],
        "exp": 4102444800,
    }


@pytest.fixture
def auth_config(monkeypatch, mock_identity_manager):
    monkeypatch.delenv("APP_CONFIG_ENDPOINT", raising=False)
    monkeypatch.setenv("allow_environment_variables", "false")
    with patch("connectors.appconfig.get_identity_manager", return_value=mock_identity_manager):
        config = AppConfigClient()
    config.disabled = False
    config.client = {
        "OAUTH_AZURE_AD_TENANT_ID": "tenant-id",
        "OAUTH_AZURE_AD_CLIENT_ID": "client-id",
    }
    monkeypatch.setattr(dependencies, "__cached_public_keys", {})
    with patch.object(dependencies, "get_config", return_value=config):
        yield config


@pytest.mark.parametrize("scenario", [
    "v2", "v1", "expired", "audience", "issuer", "tenant", "signature",
    "algorithm", "malformed", "roles", "rotation", "alternate",
])
async def test_real_jwt_preserves_identity_rejection_and_rotation(
    signing_keys, auth_config, scenario, caplog,
):
    caplog.set_level("DEBUG")
    claims = _claims()
    if scenario == "v1":
        claims["iss"] = "https://sts.windows.net/tenant-id/"
    elif scenario == "expired":
        claims["exp"] = 1
    elif scenario == "audience":
        claims["aud"] = "00000003-0000-0000-c000-000000000000"
    elif scenario == "issuer":
        claims["iss"] = "https://issuer.example.invalid"
    elif scenario == "tenant":
        claims["tid"] = "other-tenant"
    elif scenario == "roles":
        claims["roles"] = "Admin"
    key = signing_keys[1] if scenario in {"signature", "rotation", "alternate"} else signing_keys[0]
    token = jwt.encode(
        claims, key, algorithm="RS512" if scenario == "algorithm" else "RS256",
        headers={"kid": "test-kid"},
    )
    if scenario == "malformed":
        token = "broken.payload.signature"
    if scenario == "rotation":
        responses = [_jwks(signing_keys[0]), _jwks(signing_keys[1])]
    elif scenario == "alternate":
        responses = [_jwks(signing_keys[0]), _jwks(signing_keys[0]), _jwks(signing_keys[1])]
    else:
        responses = None
    public_keys = AsyncMock(return_value=_jwks(signing_keys[0]), side_effect=responses)
    with patch.object(dependencies, "_get_cached_public_keys", new=public_keys):
        if scenario in {"v2", "v1", "roles", "rotation", "alternate"}:
            result = await dependencies.validate_access_token(token)
            assert result == {
                "oid": "user-id", "preferred_username": "user@example.invalid",
                "name": "Test User", "roles": [] if scenario == "roles" else ["Admin"],
            }
        else:
            with pytest.raises(HTTPException) as raised:
                await dependencies.validate_access_token(token)
            assert raised.value.status_code == 401
            assert raised.value.detail == "Invalid token"
    assert token not in caplog.text
    if scenario == "rotation":
        assert public_keys.await_count == 2
    elif scenario in {"signature", "alternate"}:
        assert public_keys.await_count == 3
        assert public_keys.await_args.kwargs["jwks_url"].endswith("/discovery/keys")
    elif scenario in {"tenant", "algorithm", "malformed"}:
        public_keys.assert_not_awaited()
    else:
        public_keys.assert_awaited_once()
    if scenario == "v1":
        assert public_keys.await_args.kwargs["jwks_url"].endswith("/discovery/keys")


@pytest.mark.parametrize("failure_type", [httpx.ConnectError, RuntimeError, asyncio.CancelledError])
async def test_real_jwt_provider_failure_is_denied_without_raw_diagnostic(
    signing_keys, auth_config, failure_type, caplog,
):
    marker = "synthetic-private-jwks-provider-detail"
    failure = failure_type(marker)
    token = jwt.encode(_claims(), signing_keys[0], algorithm="RS256", headers={"kid": "test-kid"})
    with patch.object(dependencies, "_get_cached_public_keys", new=AsyncMock(side_effect=failure)):
        if failure_type is asyncio.CancelledError:
            with pytest.raises(asyncio.CancelledError) as raised:
                await dependencies.validate_access_token(token)
            assert raised.value is failure
        else:
            with pytest.raises(HTTPException) as raised:
                await dependencies.validate_access_token(token)
            assert raised.value.status_code == 401
            assert raised.value.detail == "Invalid token"
    assert marker not in caplog.text
    assert token not in caplog.text
    if failure_type is not asyncio.CancelledError:
        assert failure_type.__name__ in caplog.text


@pytest.mark.parametrize("version", [None, "v1", "v2"])
def test_jwks_refresh_is_exactly_tenant_and_endpoint_scoped(auth_config, version):
    urls = dependencies._jwks_urls_for_tenant("tenant-id")
    cache = {f"tenant-id|{url}": {"keys": []} for url in urls.values()}
    cache["other-tenant|unrelated-url"] = {"keys": []}
    with patch.object(dependencies, "__cached_public_keys", cache):
        dependencies._force_refresh_jwks_cache("tenant-id", urls[version] if version else None)
    assert "other-tenant|unrelated-url" in cache
    for name, url in urls.items():
        assert (f"tenant-id|{url}" in cache) is (version is not None and version != name)


@pytest.mark.parametrize("key", ["OAUTH_AZURE_AD_TENANT_ID", "OAUTH_AZURE_AD_CLIENT_ID"])
async def test_real_jwt_required_config_remains_server_error(signing_keys, auth_config, key):
    del auth_config.client[key]
    token = jwt.encode(_claims(), signing_keys[0], algorithm="RS256", headers={"kid": "test-kid"})
    with patch.object(dependencies, "_get_cached_public_keys", new=AsyncMock()) as public_keys:
        with pytest.raises(HTTPException) as raised:
            await dependencies.validate_access_token(token)
    assert raised.value.status_code == 500
    assert raised.value.detail == f"Authentication not configured (missing {key})"
    public_keys.assert_not_awaited()


async def test_unexpected_verifier_failure_still_denies_without_fallback_or_raw_log(
    signing_keys, auth_config, caplog,
):
    marker = "synthetic-private-verifier-detail"
    original_decode = jwt.decode
    token = jwt.encode(_claims(), signing_keys[0], algorithm="RS256", headers={"kid": "test-kid"})

    def decode(*args, **kwargs):
        if kwargs.get("options", {}).get("verify_signature") is False:
            return original_decode(*args, **kwargs)
        raise RuntimeError(marker)

    with (
        patch.object(jwt, "decode", side_effect=decode),
        patch.object(dependencies, "_get_cached_public_keys",
                     new=AsyncMock(return_value=_jwks(signing_keys[0]))) as public_keys,
    ):
        with pytest.raises(HTTPException) as raised:
            await dependencies.validate_access_token(token)
    assert raised.value.status_code == 401
    assert raised.value.detail == "Invalid token"
    public_keys.assert_awaited_once()
    assert marker not in caplog.text
