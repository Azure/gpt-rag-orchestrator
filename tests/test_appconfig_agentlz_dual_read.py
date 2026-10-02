"""Agent Landing Zone dual-read (Azure/GPT-RAG#695): agent-lz/AGENTLZ_ win, gpt-rag/GPT_RAG_ fall back."""

from unittest.mock import patch

import pytest

import constants
from connectors.appconfig import AppConfigClient, candidate_keys


def _client(monkeypatch, mock_identity_manager, values, env_enabled=False):
    monkeypatch.setenv("APP_CONFIG_ENDPOINT", "https://config.example.invalid")
    monkeypatch.setenv("allow_environment_variables", str(env_enabled).lower())
    with (
        patch("connectors.appconfig.get_identity_manager", return_value=mock_identity_manager),
        patch("connectors.appconfig.load", return_value=values) as load,
    ):
        return AppConfigClient(), load


def test_label_selectors_prefer_agent_lz_before_gpt_rag(monkeypatch, mock_identity_manager):
    _, load = _client(monkeypatch, mock_identity_manager, {})
    labels = [s.label_filter for s in load.call_args.kwargs["selects"]]
    assert labels == ["orchestrator", "gpt-rag-orchestrator", "agent-lz", "gpt-rag", None]
    assert labels.index("agent-lz") < labels.index("gpt-rag")


@pytest.mark.parametrize(
    "key, expected",
    [
        ("GPT_RAG_FOO", ["AGENTLZ_FOO", "GPT_RAG_FOO"]),
        ("AGENTLZ_FOO", ["AGENTLZ_FOO", "GPT_RAG_FOO"]),
        ("SEARCH_SERVICE_NAME", ["SEARCH_SERVICE_NAME"]),
    ],
)
def test_candidate_keys_order(key, expected):
    assert candidate_keys(key) == expected


@pytest.mark.parametrize("requested", ["GPT_RAG_FOO", "AGENTLZ_FOO"])
def test_appconfig_prefers_agentlz_key(monkeypatch, mock_identity_manager, requested):
    cfg, _ = _client(monkeypatch, mock_identity_manager, {"AGENTLZ_FOO": "new", "GPT_RAG_FOO": "old"})
    assert cfg.get(requested) == "new"


@pytest.mark.parametrize("requested", ["GPT_RAG_FOO", "AGENTLZ_FOO"])
def test_appconfig_falls_back_to_gpt_rag_key(monkeypatch, mock_identity_manager, requested):
    cfg, _ = _client(monkeypatch, mock_identity_manager, {"GPT_RAG_FOO": "old"})
    assert cfg.get(requested) == "old"


def test_env_prefers_agentlz_then_falls_back(monkeypatch, mock_identity_manager):
    cfg, _ = _client(monkeypatch, mock_identity_manager, {}, env_enabled=True)
    monkeypatch.setenv("GPT_RAG_BAR", "old")
    assert cfg.get("AGENTLZ_BAR") == "old"
    monkeypatch.setenv("AGENTLZ_BAR", "new")
    assert cfg.get("GPT_RAG_BAR") == "new"


def test_missing_in_both_prefixes_still_raises(monkeypatch, mock_identity_manager):
    cfg, _ = _client(monkeypatch, mock_identity_manager, {})
    assert cfg.get("AGENTLZ_MISSING", default="d") == "d"
    with pytest.raises(Exception, match="AGENTLZ_MISSING not found"):
        cfg.get("AGENTLZ_MISSING")


def test_telemetry_service_name_uses_agentlz_prefix():
    assert constants.TELEMETRY_SERVICE_NAME.startswith("agentlz.")
    assert constants.APP_NAME == "gpt-rag-orchestrator"
