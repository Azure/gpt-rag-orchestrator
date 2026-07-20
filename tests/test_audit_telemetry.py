from unittest.mock import patch

from opentelemetry.sdk.resources import SERVICE_VERSION

from telemetry import Telemetry


class Config:
    def __init__(self, values):
        self.values = values

    def get(self, key, default=None, **_kwargs):
        return self.values.get(key, default)

    def get_value(self, _key, default=None, **_kwargs):
        return default


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
