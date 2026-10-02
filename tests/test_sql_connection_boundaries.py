"""Driver/auth failures propagate without exposing connection diagnostics."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pyodbc
import pytest
from azure.core.exceptions import ClientAuthenticationError

from connectors.fabric import SQLEndpointClient, SemanticModelClient
from connectors.sqldbs import SQLDBClient


@pytest.fixture
def connection_dependencies(monkeypatch):
    secret = AsyncMock(return_value="synthetic-secret")
    monkeypatch.setattr("connectors.fabric.get_secret", secret)
    monkeypatch.setattr("connectors.sqldbs.get_secret", secret)
    credential = MagicMock()
    credential.get_token.return_value = SimpleNamespace(token="synthetic-token")
    monkeypatch.setattr("connectors.sqldbs.ChainedTokenCredential",
                        MagicMock(return_value=credential))
    monkeypatch.setattr("connectors.sqldbs.ManagedIdentityCredential", MagicMock())
    monkeypatch.setattr("connectors.sqldbs.AzureCliCredential", MagicMock())
    connect = MagicMock()
    monkeypatch.setattr("pyodbc.connect", connect)
    return secret, credential, connect


def _sql_client(kind):
    config = {
        "id": "synthetic-source", "description": "test", "type": "test",
        "server": "database.example.invalid", "database": "test",
    }
    if kind == "endpoint":
        return SQLEndpointClient({
            **config, "tenant_id": "synthetic-tenant", "client_id": "synthetic-client",
        })
    return SQLDBClient({
        **config, "uid": "synthetic-user" if kind == "database-password" else None,
    })


@pytest.mark.parametrize("kind", ["endpoint", "database-password", "database-identity"])
@pytest.mark.parametrize("outcome", ["success", "driver", "unexpected"])
async def test_sql_connection_retains_result_or_identical_failure_without_raw_diagnostics(
    kind, outcome, connection_dependencies, caplog,
):
    secret, credential, connect = connection_dependencies
    marker = "synthetic-private-connection-detail"
    error = {
        "driver": pyodbc.OperationalError(marker),
        "unexpected": RuntimeError(marker),
    }.get(outcome)
    connect.side_effect = error
    client = _sql_client(kind)
    if error is None:
        assert await client.create_connection() is connect.return_value
    else:
        with pytest.raises(type(error)) as raised:
            await client.create_connection()
        assert raised.value is error
    connect.assert_called_once()
    assert marker not in caplog.text
    if kind == "database-identity":
        secret.assert_not_awaited()
        credential.get_token.assert_called_once_with("https://database.windows.net/.default")
        assert set(connect.call_args.kwargs["attrs_before"]) == {1256}
    else:
        secret.assert_awaited_once()
        credential.get_token.assert_not_called()


@pytest.mark.parametrize("error_type", [ClientAuthenticationError, RuntimeError])
async def test_sql_identity_failure_does_not_attempt_driver_or_expose_token_details(
    error_type, connection_dependencies, caplog,
):
    _, credential, connect = connection_dependencies
    error = error_type("synthetic-private-identity-detail")
    credential.get_token.side_effect = error
    with pytest.raises(error_type) as raised:
        await _sql_client("database-identity").create_connection()
    assert raised.value is error
    connect.assert_not_called()
    assert str(error) not in caplog.text


@pytest.mark.parametrize("outcome", ["success", "authentication", "unexpected", "cancelled"])
async def test_semantic_model_token_outcome_preserves_credential_cleanup(
    outcome, connection_dependencies, monkeypatch, caplog,
):
    marker = "synthetic-private-identity-detail"
    error = {
        "authentication": ClientAuthenticationError(marker),
        "unexpected": RuntimeError(marker),
        "cancelled": asyncio.CancelledError(marker),
    }.get(outcome)
    credential = MagicMock()
    credential.get_token = AsyncMock(
        return_value=SimpleNamespace(token="synthetic-token"), side_effect=error,
    )
    credential.close = AsyncMock()
    factory = MagicMock(return_value=credential)
    monkeypatch.setattr("connectors.fabric.ClientSecretCredential", factory)
    client = SemanticModelClient({
        "id": "synthetic-source", "description": "test", "type": "test",
        "organization": "test", "workspace": "test", "dataset": "test",
        "tenant_id": "synthetic-tenant", "client_id": "synthetic-client",
    })
    if error is None:
        assert await client._get_restapi_access_token() == "synthetic-token"
    else:
        with pytest.raises(type(error)) as raised:
            await client._get_restapi_access_token()
        assert raised.value is error
    credential.close.assert_awaited_once()
    credential.get_token.assert_awaited_once_with(
        "https://analysis.windows.net/powerbi/api/.default",
    )
    factory.assert_called_once_with(
        tenant_id="synthetic-tenant", client_id="synthetic-client",
        client_secret="synthetic-secret",
    )
    assert marker not in caplog.text
