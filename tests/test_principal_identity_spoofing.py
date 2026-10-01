"""Client-supplied identity in user_context must never become the trusted principal."""

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import Response
from starlette.requests import Request

from connectors.appconfig import AppConfigClient
from schemas import OrchestratorRequest
from test_primary_strategy_failure_boundaries import main_module


SPOOFED = "synthetic-spoofed-victim-principal"
IDENTITY_KEYS = (
    "principal_id", "principal_name", "user_name", "oid", "user_id",
    "security_ids", "groups", "group_ids", "client_group_names",
)


@pytest.fixture
def actual_config(monkeypatch):
    monkeypatch.setenv("allow_environment_variables", "false")
    cfg = AppConfigClient.__new__(AppConfigClient)
    cfg.disabled = False
    cfg.client = {}
    return cfg


def _spoofed_context():
    context = {key: SPOOFED for key in IDENTITY_KEYS}
    context["security_ids"] = [SPOOFED]
    context["groups"] = [SPOOFED]
    context["group_ids"] = [SPOOFED]
    context["client_group_names"] = [SPOOFED]
    context["locale"] = "pt-BR"
    return context


def _request():
    return Request({"type": "http", "method": "POST", "path": "/orchestrator", "headers": []})


async def _ask(main, body, authorization=None):
    return await main.orchestrator_endpoint(_request(), Response(), body, authorization=authorization)


def _patch_turn_factory(main, monkeypatch):
    factory = AsyncMock(return_value=SimpleNamespace())
    monkeypatch.setattr(main.Orchestrator, "from_turn_request", factory)
    return factory


def _assert_anonymous(context):
    assert context["principal_id"] == "anonymous"
    assert context["principal_name"] == "anonymous"
    for key in IDENTITY_KEYS:
        if key not in ("principal_id", "principal_name"):
            assert key not in context
    assert context["locale"] == "pt-BR"


@pytest.mark.parametrize("authorization", [None, "Bearer synthetic"])
async def test_anonymous_request_ignores_client_supplied_identity(
    main_module, actual_config, monkeypatch, authorization,
):
    monkeypatch.setattr(main_module, "cfg", actual_config)
    factory = _patch_turn_factory(main_module, monkeypatch)

    result = await _ask(
        main_module,
        OrchestratorRequest(ask="test", user_context=_spoofed_context()),
        authorization,
    )

    assert result.status_code == 200
    _assert_anonymous(factory.await_args.args[0].user_context)


async def test_missing_user_context_becomes_anonymous(main_module, actual_config, monkeypatch):
    monkeypatch.setattr(main_module, "cfg", actual_config)
    factory = _patch_turn_factory(main_module, monkeypatch)

    result = await _ask(main_module, OrchestratorRequest(ask="test"))

    assert result.status_code == 200
    context = factory.await_args.args[0].user_context
    assert context["principal_id"] == "anonymous"
    assert context["principal_name"] == "anonymous"


async def test_authenticated_request_uses_token_identity_not_client_identity(
    main_module, actual_config, monkeypatch,
):
    actual_config.client.update({
        "OAUTH_AZURE_AD_TENANT_ID": "tenant",
        "OAUTH_AZURE_AD_CLIENT_ID": "client",
    })
    monkeypatch.setattr(main_module, "cfg", actual_config)
    monkeypatch.setattr(
        main_module,
        "validate_access_token",
        AsyncMock(return_value={"oid": "token-oid", "preferred_username": "user@contoso", "name": "User"}),
    )
    factory = _patch_turn_factory(main_module, monkeypatch)

    result = await _ask(
        main_module,
        OrchestratorRequest(ask="test", user_context=_spoofed_context()),
        "Bearer synthetic",
    )

    assert result.status_code == 200
    context = factory.await_args.args[0].user_context
    assert context["principal_id"] == "token-oid"
    assert context["principal_name"] == "user@contoso"
    assert context["user_name"] == "User"
    for key in ("oid", "user_id", "security_ids", "groups", "group_ids", "client_group_names"):
        assert key not in context
    assert SPOOFED not in repr(context)


async def test_feedback_ignores_client_supplied_identity(main_module, actual_config, monkeypatch):
    monkeypatch.setattr(main_module, "cfg", actual_config)
    orchestrator = SimpleNamespace(save_feedback=AsyncMock())
    create = AsyncMock(return_value=orchestrator)
    monkeypatch.setattr(main_module.Orchestrator, "create", create)

    result = await _ask(
        main_module,
        OrchestratorRequest(
            type="feedback", conversation_id="c1", is_positive=True, user_context=_spoofed_context(),
        ),
    )

    assert result["status"] == "success"
    _assert_anonymous(create.await_args.kwargs["user_context"])
    orchestrator.save_feedback.assert_awaited_once()


async def test_spoofed_identity_values_are_not_logged_and_body_is_not_mutated(
    main_module, actual_config, monkeypatch, caplog,
):
    monkeypatch.setattr(main_module, "cfg", actual_config)
    _patch_turn_factory(main_module, monkeypatch)
    original = _spoofed_context()
    body = OrchestratorRequest(ask="test", user_context=original)

    with caplog.at_level(logging.DEBUG):
        await _ask(main_module, body)

    assert SPOOFED not in caplog.text
    assert body.user_context == _spoofed_context()
