import logging
from types import MethodType
from unittest.mock import MagicMock, patch

import pytest

from opentelemetry.sdk.resources import SERVICE_VERSION

from telemetry import Telemetry


class Config:
    def __init__(self, values):
        self.values = values

    def get(self, key, default=None, **_kwargs):
        return self.values.get(key, default)

    def get_value(self, key, default=None, **_kwargs):
        return self.values.get(key, default)


def test_runtime_version_is_used_for_azure_monitor_resource():
    config = Config(
        {
            "APPLICATIONINSIGHTS_CONNECTION_STRING": "InstrumentationKey=test",
            "AZURE_MONITOR_DISABLE_LOGGING": "true",
            "AUDIT_EVENTS_ENABLED": "false",
        }
    )
    with (
        patch("telemetry.telemetry.configure_azure_monitor") as configure,
        patch.object(Telemetry, "configure_logging"),
    ):
        Telemetry.configure_monitoring(
            config,
            "APPLICATIONINSIGHTS_CONNECTION_STRING",
            "gpt-rag-orchestrator",
            "3.7.0",
        )

    assert configure.call_args.kwargs["resource"].attributes[SERVICE_VERSION] == "3.7.0"


def test_audit_only_export_uses_pinned_logger_namespace():
    config = Config(
        {
            "APPLICATIONINSIGHTS_CONNECTION_STRING": "InstrumentationKey=test",
            "AZURE_MONITOR_DISABLE_LOGGING": "true",
            "AUDIT_EVENTS_ENABLED": "true",
        }
    )
    with (
        patch("telemetry.telemetry.configure_azure_monitor") as configure,
        patch.object(Telemetry, "configure_logging"),
    ):
        Telemetry.configure_monitoring(
            config,
            "APPLICATIONINSIGHTS_CONNECTION_STRING",
            "gpt-rag-orchestrator",
            "3.7.0",
        )

    assert configure.call_args.kwargs["disable_logging"] is False
    assert configure.call_args.kwargs["logger_name"] == "gptrag.audit"


def test_disabled_audit_does_not_enable_log_export():
    config = Config(
        {
            "APPLICATIONINSIGHTS_CONNECTION_STRING": "InstrumentationKey=test",
            "AZURE_MONITOR_DISABLE_LOGGING": "true",
            "AUDIT_EVENTS_ENABLED": "false",
        }
    )
    with (
        patch("telemetry.telemetry.configure_azure_monitor") as configure,
        patch.object(Telemetry, "configure_logging"),
    ):
        Telemetry.configure_monitoring(
            config,
            "APPLICATIONINSIGHTS_CONNECTION_STRING",
            "gpt-rag-orchestrator",
            "3.7.0",
        )

    assert configure.call_args.kwargs["disable_logging"] is True
    assert configure.call_args.kwargs["logger_name"] == ""


@pytest.mark.parametrize("fails", [False, True])
def test_monitor_setup_restores_logger_levels_and_preserves_original_failure(fails):
    names = [
        "azure.monitor.opentelemetry", "azure.monitor.opentelemetry._configure",
        "opentelemetry", "azure.core.pipeline.policies.http_logging_policy",
    ]
    loggers = [logging.getLogger(name) for name in names]
    levels = [logger.level for logger in loggers]
    error = RuntimeError("synthetic monitor initialization failure")
    config = Config({"APPLICATIONINSIGHTS_CONNECTION_STRING": "InstrumentationKey=test"})
    with (
        patch("telemetry.telemetry.configure_azure_monitor", side_effect=error if fails else None),
        patch.object(Telemetry, "configure_logging"),
    ):
        if fails:
            with pytest.raises(RuntimeError) as raised:
                Telemetry.configure_monitoring(
                    config, "APPLICATIONINSIGHTS_CONNECTION_STRING", "gpt-rag-orchestrator", "test")
            assert raised.value is error
        else:
            Telemetry.configure_monitoring(
                config, "APPLICATIONINSIGHTS_CONNECTION_STRING", "gpt-rag-orchestrator", "test")
    assert [logger.level for logger in loggers] == levels


@pytest.mark.parametrize("consumer", ["configure_basic", "configure_logging", "configure_monitoring"])
@pytest.mark.parametrize("provider_state", ["missing", "loaded", "unavailable", "callback_failure"])
def test_real_optional_telemetry_config_preserves_defaults_and_provider_failures(
    consumer, provider_state, monkeypatch, mock_identity_manager,
):
    from connectors.appconfig import AppConfigClient
    from tenacity import stop_after_attempt, wait_none

    monkeypatch.delenv("APP_CONFIG_ENDPOINT", raising=False)
    monkeypatch.delenv("APPLICATIONINSIGHTS_CONNECTION_STRING", raising=False)
    monkeypatch.setenv("allow_environment_variables", "false")
    with patch("connectors.appconfig.get_identity_manager", return_value=mock_identity_manager):
        config = AppConfigClient()
    optional_key = (
        "APPLICATIONINSIGHTS_CONNECTION_STRING"
        if consumer == "configure_monitoring" else "AZURE_HTTP_LOG_LEVEL"
    )
    configured = "InstrumentationKey=test" if consumer == "configure_monitoring" else "DEBUG"
    values = {"LOG_LEVEL": "INFO", "AZURE_LOG_LEVEL": "WARNING"}
    if provider_state == "loaded":
        values[optional_key] = configured
    error = RuntimeError("synthetic telemetry retry callback failure")
    provider = MagicMock()
    lookups = []

    def read(key):
        lookups.append(key)
        if key == optional_key and provider_state in {"unavailable", "callback_failure"}:
            raise RuntimeError("synthetic provider failure")
        return values[key]

    provider.__getitem__.side_effect = read
    config.client = provider
    config.disabled = False
    fast_retry = AppConfigClient.get_config_with_retry.retry_with(
        wait=wait_none(), stop=stop_after_attempt(2),
        before_sleep=MagicMock(side_effect=error if provider_state == "callback_failure" else None),
    )
    config.get_config_with_retry = MethodType(fast_retry, config)
    loggers = {}
    with (
        patch("logging.getLogger", side_effect=lambda name=None: loggers.setdefault(name, MagicMock())),
        patch("logging.basicConfig"),
        patch("logging.config.dictConfig") as configure_logs,
        patch("telemetry.telemetry.configure_azure_monitor") as configure_monitor,
        patch("telemetry.telemetry.silence_context_detach_noise"),
    ):
        def configure():
            if consumer == "configure_monitoring":
                with patch.object(Telemetry, "configure_logging"):
                    Telemetry.configure_monitoring(
                        config, optional_key, "gpt-rag-orchestrator", "test")
            else:
                getattr(Telemetry, consumer)(config)

        if provider_state == "callback_failure":
            with pytest.raises(RuntimeError) as raised:
                configure()
            assert raised.value is error
        else:
            configure()
            if consumer == "configure_monitoring":
                assert configure_monitor.call_count == (provider_state == "loaded")
            elif consumer == "configure_basic":
                assert loggers["azure.core.pipeline.policies.http_logging_policy"].disabled is (
                    provider_state != "loaded")
            else:
                http = configure_logs.call_args.args[0]["loggers"][
                    "azure.core.pipeline.policies.http_logging_policy"]
                assert http["propagate"] is (provider_state == "loaded")
    expected_attempts = 2 if provider_state == "unavailable" else 1
    assert lookups.count(optional_key) == expected_attempts


@pytest.mark.parametrize("environment_enabled", [False, True])
@pytest.mark.parametrize("remote_value", [None, "InstrumentationKey=remote-test"])
def test_monitoring_keeps_environment_default_and_configured_precedence(
    environment_enabled, remote_value, monkeypatch, mock_identity_manager,
):
    from connectors.appconfig import AppConfigClient

    key = "APPLICATIONINSIGHTS_CONNECTION_STRING"
    monkeypatch.delenv("APP_CONFIG_ENDPOINT", raising=False)
    monkeypatch.setenv(key, "InstrumentationKey=environment-test")
    monkeypatch.setenv("allow_environment_variables", str(environment_enabled).lower())
    with patch("connectors.appconfig.get_identity_manager", return_value=mock_identity_manager):
        config = AppConfigClient()
    config.disabled = False
    config.client = {key: remote_value} if remote_value else {}
    with (
        patch("telemetry.telemetry.configure_azure_monitor") as configure,
        patch.object(Telemetry, "configure_logging"),
    ):
        Telemetry.configure_monitoring(config, key, "gpt-rag-orchestrator", "test")
    expected = (
        "InstrumentationKey=environment-test"
        if environment_enabled or remote_value is None else remote_value
    )
    assert configure.call_args.kwargs["connection_string"] == expected
