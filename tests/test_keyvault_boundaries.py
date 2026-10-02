"""Key Vault unavailable outcomes use SDK/configuration errors, not arbitrary failures."""

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from azure.core.exceptions import (
    ClientAuthenticationError,
    HttpResponseError,
    ResourceNotFoundError,
    ServiceRequestError,
)

from connectors.keyvault import get_secret


@pytest.fixture
def vault(caplog):
    caplog.set_level(logging.INFO)
    cfg = MagicMock()
    cfg.get = MagicMock(return_value="https://vault.invalid")
    credential = MagicMock()
    credential.__aenter__ = AsyncMock(return_value=credential)
    credential.__aexit__ = AsyncMock(return_value=False)
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.get_secret = AsyncMock(return_value=SimpleNamespace(value="synthetic-secret"))
    with (
        patch("connectors.keyvault.get_config", return_value=cfg),
        patch("connectors.keyvault.ManagedIdentityCredential"),
        patch("connectors.keyvault.AzureCliCredential"),
        patch("connectors.keyvault.ChainedTokenCredential", return_value=credential),
        patch("connectors.keyvault.AsyncSecretClient", return_value=client),
    ):
        yield cfg, credential, client


@pytest.mark.parametrize("stage", ["config", "read", "client-exit", "credential-exit"])
@pytest.mark.parametrize("failure_kind", ["expected", "unexpected", "cancelled", "success"])
async def test_secret_lookup_preserves_unavailable_results_and_context_cleanup(vault, stage, failure_kind, caplog):
    marker = "synthetic-private-vault-detail"
    failure = {
        "expected": ValueError(marker) if stage == "config" else ServiceRequestError(marker),
        "unexpected": RuntimeError(marker),
        "cancelled": asyncio.CancelledError(marker),
    }.get(failure_kind)
    cfg, credential, client = vault
    cfg.get.side_effect = failure if stage == "config" else None
    credential.__aexit__.side_effect = failure if stage == "credential-exit" else None
    client.__aexit__.side_effect = failure if stage == "client-exit" else None
    client.get_secret.side_effect = failure if stage == "read" else None
    if failure_kind in {"unexpected", "cancelled"}:
        with pytest.raises(type(failure)) as raised:
            await get_secret("example-secret")
        assert raised.value is failure
    else:
        result = await get_secret("example-secret")
        assert result == ("synthetic-secret" if failure_kind == "success" else None)
    if stage == "config" and failure:
        credential.__aenter__.assert_not_awaited()
        client.get_secret.assert_not_awaited()
    else:
        client.get_secret.assert_awaited_once_with("example-secret")
        client.__aexit__.assert_awaited_once()
        credential.__aexit__.assert_awaited_once()
    assert marker not in caplog.text


@pytest.mark.parametrize("failure_type", [ClientAuthenticationError, ResourceNotFoundError, HttpResponseError])
async def test_secret_sdk_failures_still_return_unavailable_without_diagnostics(vault, failure_type, caplog):
    marker = "synthetic-private-vault-detail"
    _, credential, client = vault
    client.get_secret.side_effect = failure_type(marker)
    assert await get_secret(marker) is None
    client.__aexit__.assert_awaited_once()
    credential.__aexit__.assert_awaited_once()
    assert marker not in caplog.text
