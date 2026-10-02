"""Tests for the ``validate_auth`` request authentication dependency."""

import pytest
from fastapi import HTTPException

from dependencies import validate_auth


@pytest.fixture(autouse=True)
def _auth_env(monkeypatch):
    monkeypatch.delenv("DISABLE_AUTH", raising=False)
    monkeypatch.delenv("APP_API_TOKEN", raising=False)
    monkeypatch.delenv("DAPR_API_TOKEN", raising=False)


@pytest.mark.parametrize("configured", [None, "", "   "])
async def test_dev_token_rejected_when_no_token_configured(monkeypatch, configured):
    if configured is not None:
        monkeypatch.setenv("APP_API_TOKEN", configured)
        monkeypatch.setenv("DAPR_API_TOKEN", configured)

    with pytest.raises(HTTPException) as exc:
        await validate_auth(dapr_api_token="dev-token", x_api_key=None)

    assert exc.value.status_code == 401


async def test_dev_token_rejected_when_tokens_configured(monkeypatch):
    monkeypatch.setenv("APP_API_TOKEN", "configured-app-token")
    monkeypatch.setenv("DAPR_API_TOKEN", "configured-dapr-token")

    with pytest.raises(HTTPException) as exc:
        await validate_auth(dapr_api_token="dev-token", x_api_key=None)

    assert exc.value.status_code == 401


async def test_wrong_dapr_token_rejected(monkeypatch):
    monkeypatch.setenv("DAPR_API_TOKEN", "configured-dapr-token")

    with pytest.raises(HTTPException) as exc:
        await validate_auth(dapr_api_token="not-the-token", x_api_key=None)

    assert exc.value.status_code == 401


@pytest.mark.parametrize("env_var", ["APP_API_TOKEN", "DAPR_API_TOKEN"])
async def test_configured_token_accepted(monkeypatch, env_var):
    monkeypatch.setenv(env_var, "configured-token")

    assert await validate_auth(dapr_api_token="configured-token", x_api_key=None) is True
