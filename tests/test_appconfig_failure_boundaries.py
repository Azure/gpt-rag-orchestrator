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
    assert marker not in caplog.text
    if provider_state == "unavailable":
        assert "unavailable (RuntimeError)" in caplog.text


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


@pytest.mark.parametrize("endpoint,host", [
    (None, "None"), ("", "None"), ("https://config.example.invalid/path", "config.example.invalid"),
])
def test_auth_config_diagnostics_use_environment_string_contract(endpoint, host, monkeypatch, caplog):
    from dependencies import _log_app_config_state

    if endpoint is None:
        monkeypatch.delenv("APP_CONFIG_ENDPOINT", raising=False)
    else:
        monkeypatch.setenv("APP_CONFIG_ENDPOINT", endpoint)
    config = MagicMock(disabled=True, auth_failed=False, allow_env_vars=False, client={})
    _log_app_config_state(config, keys_to_check=["MISSING_KEY"])
    assert f"endpoint_host={host} " in caplog.text
    assert "keys_loaded=0" in caplog.text


@pytest.mark.parametrize("required", [False, True])
def test_real_provider_retries_preserve_optional_and_required_outcomes(
    required, monkeypatch, mock_identity_manager,
):
    from types import MethodType
    from tenacity import stop_after_attempt, wait_none

    class UnavailableProvider(dict):
        attempts = 0

        def __getitem__(self, key):
            self.attempts += 1
            raise RuntimeError("synthetic transient provider failure")

    monkeypatch.setenv("APP_CONFIG_ENDPOINT", "https://config.example.invalid")
    monkeypatch.setenv("allow_environment_variables", "false")
    with (
        patch("connectors.appconfig.get_identity_manager", return_value=mock_identity_manager),
        patch("connectors.appconfig.load", return_value=UnavailableProvider(configured=True)),
    ):
        config = AppConfigClient()
    fast_retry = AppConfigClient.get_config_with_retry.retry_with(
        wait=wait_none(), stop=stop_after_attempt(2))
    config.get_config_with_retry = MethodType(fast_retry, config)
    if required:
        with pytest.raises(Exception, match="configuration variable REQUIRED not found"):
            config.get("REQUIRED")
    else:
        assert config.get("OPTIONAL", default="fallback") == "fallback"
        assert config.get_value("OPTIONAL", allow_none=True) is None
    assert config.client.attempts == (2 if required else 4)


def test_appconfig_does_not_hide_unexpected_retry_callback_failure(monkeypatch):
    from types import MethodType
    from tenacity import stop_after_attempt, wait_none

    monkeypatch.delenv("APP_CONFIG_ENDPOINT", raising=False)
    monkeypatch.setenv("allow_environment_variables", "false")
    config = AppConfigClient()
    config.disabled = False
    config.client = MagicMock()
    config.client.__getitem__.side_effect = RuntimeError("provider failure")
    error = RuntimeError("retry callback failure")
    fast_retry = AppConfigClient.get_config_with_retry.retry_with(
        wait=wait_none(), stop=stop_after_attempt(2),
        before_sleep=MagicMock(side_effect=error))
    config.get_config_with_retry = MethodType(fast_retry, config)
    with pytest.raises(RuntimeError) as raised:
        config.get("OPTIONAL", default="fallback")
    assert raised.value is error


def test_appconfig_retries_nested_provider_retry_error_instead_of_masking_value(monkeypatch):
    from types import MethodType
    from tenacity import Future, RetryError, stop_after_attempt, wait_none

    monkeypatch.delenv("APP_CONFIG_ENDPOINT", raising=False)
    monkeypatch.setenv("allow_environment_variables", "false")
    config = AppConfigClient()
    config.disabled = False
    attempt = Future(1)
    attempt.set_exception(RuntimeError("nested provider failure"))
    config.client = MagicMock()
    config.client.__getitem__.side_effect = [RetryError(attempt), "configured"]
    fast_retry = AppConfigClient.get_config_with_retry.retry_with(
        wait=wait_none(), stop=stop_after_attempt(2))
    config.get_config_with_retry = MethodType(fast_retry, config)
    assert config.get("OPTIONAL", default="fallback") == "configured"
    assert config.client.__getitem__.call_count == 2
