"""Provider-to-consumer characterization without approving legacy fallback policy."""

import runpy
from unittest.mock import MagicMock, patch

import pytest
from azure.core.exceptions import ClientAuthenticationError

from connectors.appconfig import AppConfigClient
from strategies.maf_lite_strategy import MafLiteStrategy


@pytest.mark.parametrize("provider_state", ["missing", "authentication", "unavailable", "loaded"])
@pytest.mark.parametrize("environment_enabled", [False, True])
def test_real_appconfig_provider_to_startup_and_required_strategy_consumers(
    provider_state, environment_enabled, monkeypatch, mock_identity_manager, caplog,
):
    values = {
        "AI_FOUNDRY_PROJECT_ENDPOINT": "https://project.example.invalid",
        "AI_FOUNDRY_ACCOUNT_ENDPOINT": "https://account.example.invalid",
        "CHAT_DEPLOYMENT_NAME": "test-model",
    }
    monkeypatch.setenv("allow_environment_variables", str(environment_enabled).lower())
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    if provider_state == "missing":
        monkeypatch.delenv("APP_CONFIG_ENDPOINT", raising=False)
    else:
        monkeypatch.setenv("APP_CONFIG_ENDPOINT", "https://config.example.invalid")
    marker = "synthetic-provider-error-detail"
    error = {
        "authentication": ClientAuthenticationError(marker),
        "unavailable": RuntimeError(marker),
    }.get(provider_state)
    with (
        patch("connectors.appconfig.get_identity_manager", return_value=mock_identity_manager),
        patch("connectors.appconfig.load", return_value=values, side_effect=error) as load,
    ):
        config = AppConfigClient()
    assert load.call_count == (provider_state != "missing")
    assert config.disabled is (provider_state != "loaded")
    assert config.auth_failed is (provider_state == "authentication")
    assert config.get("OPTIONAL_UNSET", default="fallback") == "fallback"
    assert config.get_value("OPTIONAL_UNSET", allow_none=True) is None
    with pytest.raises(Exception, match="configuration variable REQUIRED_UNSET not found"):
        config.get("REQUIRED_UNSET")

    with (
        patch("dependencies.get_config", return_value=config),
        patch("dotenv.load_dotenv", return_value=False),
        patch("telemetry.Telemetry.configure_basic"),
        patch("telemetry.Telemetry.log_log_level_diagnostics"),
        patch("opentelemetry.instrumentation.fastapi.FastAPIInstrumentor.instrument_app"),
        patch("opentelemetry.instrumentation.httpx.HTTPXClientInstrumentor.instrument"),
        patch("logging.shutdown"),
        patch("os._exit", side_effect=SystemExit(1)) as exit_process,
    ):
        if config.auth_failed:
            with pytest.raises(SystemExit) as stopped:
                runpy.run_module("main", run_name="quality_startup_probe")
            assert stopped.value.code == 1
            exit_process.assert_called_once_with(1)
        else:
            main = runpy.run_module("main", run_name="quality_startup_probe")
            assert main["cfg"] is config
            exit_process.assert_not_called()

    if not config.auth_failed:
        with (
            patch("strategies.base_agent_strategy.get_config", return_value=config),
            patch("strategies.maf_lite_strategy.get_config", return_value=config),
            patch("strategies.base_agent_strategy.get_identity_manager",
                  return_value=mock_identity_manager),
            patch("strategies.base_agent_strategy.AIProjectClient"),
            patch("strategies.base_agent_strategy.get_cosmosdb_client", return_value=MagicMock()),
        ):
            if provider_state == "loaded" or environment_enabled:
                strategy = MafLiteStrategy()
                assert strategy.model_name == "test-model"
                assert strategy.search_endpoint is None
            else:
                with pytest.raises(Exception, match="AI_FOUNDRY_PROJECT_ENDPOINT not found"):
                    MafLiteStrategy()
    if provider_state == "unavailable":
        assert marker in caplog.text  # Existing diagnostic exposure remains unapproved.
    else:
        assert marker not in caplog.text


@pytest.mark.parametrize("endpoint,host", [
    ("https://config.example.invalid/path", "config.example.invalid"),
    ("http://config.example.invalid", "config.example.invalid"),
    ("config.example.invalid", "config.example.invalid"),
])
def test_appconfig_endpoint_host_formatting_needs_no_broad_recovery(
    endpoint, host, monkeypatch, mock_identity_manager, caplog,
):
    monkeypatch.setenv("APP_CONFIG_ENDPOINT", endpoint)
    caplog.set_level("INFO")
    with (
        patch("connectors.appconfig.get_identity_manager", return_value=mock_identity_manager),
        patch("connectors.appconfig.load", return_value={}) as load,
    ):
        config = AppConfigClient()
    assert config.disabled is False
    assert f"endpoint_host={host} " in caplog.text
    assert load.call_args.kwargs["endpoint"] == endpoint
