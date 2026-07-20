import base64
import json
import re
from pathlib import Path

import jsonschema
import pytest

from telemetry.audit_contract import (
    MAX_EVENT_BYTES,
    AuditConfigurationError,
    AuditSettings,
    format_utc,
    new_correlation_id,
    new_event_id,
    utc_now,
)
from telemetry.audit_sanitizer import REDACTED, sanitize_event


ROOT = Path(__file__).resolve().parents[1]


class Config:
    def __init__(self, values=None):
        self.values = values or {}

    def get(self, key, default=None, **_kwargs):
        return self.values.get(key, default)


def _base_event():
    return {
        "schema_version": 1,
        "event_id": new_event_id(),
        "event_type": "request.started",
        "event_time_utc": format_utc(utc_now()),
        "correlation_id": new_correlation_id(),
        "trace_id": "0" * 32,
        "span_id": "0" * 16,
        "parent_event_id": None,
        "service_name": "gpt-rag-orchestrator",
        "service_version": "3.7.0",
        "environment": "test",
        "operation": "test",
        "status": "started",
        "reason_code": "request_received",
        "capture_mode": "metadata_only",
        "redaction_applied": False,
        "omitted_fields": [],
        "truncated_fields": [],
    }


def test_golden_event_validates_against_shared_schema():
    schema = json.loads(
        (ROOT / "contracts" / "audit-event-v1.schema.json").read_text()
    )
    golden = json.loads(
        (ROOT / "tests" / "golden" / "audit_event_v1.json").read_text()
    )

    jsonschema.Draft202012Validator(schema).validate(golden)


def test_ids_and_timestamp_have_canonical_shapes():
    assert re.fullmatch(r"evt_[0-9a-f]{32}", new_event_id())
    assert re.fullmatch(r"req_[0-9a-f]{32}", new_correlation_id())
    assert re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z",
        format_utc(utc_now()),
    )


def test_disabled_settings_need_no_hmac_key():
    settings = AuditSettings.from_config(Config())

    assert settings.enabled is False
    assert settings.hmac_key is None
    assert settings.sensitive_content_fields == frozenset()
    assert settings.source_event_limit == 25


def test_disabled_settings_ignore_invalid_inactive_values():
    settings = AuditSettings.from_config(
        Config(
            {
                "AUDIT_EVENTS_ENABLED": "false",
                "AUDIT_HMAC_KEY": "invalid",
                "AUDIT_SOURCE_EVENT_LIMIT": "not-an-integer",
                "AUDIT_SENSITIVE_CONTENT_FIELDS": "unknown",
            }
        )
    )

    assert settings.enabled is False


def test_config_provider_controls_environment_precedence(monkeypatch):
    encoded = base64.urlsafe_b64encode(b"k" * 32).decode()
    monkeypatch.setenv("AUDIT_EVENTS_ENABLED", "false")
    settings = AuditSettings.from_config(
        Config(
            {
                "AUDIT_EVENTS_ENABLED": "true",
                "AUDIT_HMAC_KEY": encoded,
            }
        )
    )

    assert settings.enabled is True


@pytest.mark.parametrize("key", ["", "short", base64.b64encode(b"x" * 31).decode()])
def test_enabled_settings_reject_non_256_bit_hmac_keys(key):
    with pytest.raises(AuditConfigurationError):
        AuditSettings.from_config(
            Config({"AUDIT_EVENTS_ENABLED": "true", "AUDIT_HMAC_KEY": key})
        )


def test_enabled_settings_accept_base64url_256_bit_key():
    encoded = base64.urlsafe_b64encode(b"k" * 32).decode().rstrip("=")
    settings = AuditSettings.from_config(
        Config({"AUDIT_EVENTS_ENABLED": "true", "AUDIT_HMAC_KEY": encoded})
    )

    assert settings.hmac_key == b"k" * 32


def test_sensitive_allowlist_rejects_unknown_field_even_when_empty_is_safe():
    empty = AuditSettings.from_config(
        Config(
            {
                "AUDIT_SENSITIVE_CONTENT_ENABLED": "true",
                "AUDIT_SENSITIVE_CONTENT_FIELDS": "",
            }
        )
    )
    assert empty.capture_mode.value == "metadata_only"

    with pytest.raises(AuditConfigurationError):
        AuditSettings.from_config(
            Config(
                {
                    "AUDIT_EVENTS_ENABLED": "true",
                    "AUDIT_HMAC_KEY": base64.urlsafe_b64encode(
                        b"k" * 32
                    ).decode(),
                    "AUDIT_SENSITIVE_CONTENT_FIELDS": "prompt,unknown",
                }
            )
        )


def test_recursive_redaction_handles_cycles_unknown_objects_and_bounds():
    cycle = {}
    cycle["self"] = cycle
    event = _base_event()
    event.update(
        {
            "tool_arguments": {
                "Authorization": "Bearer definitely-secret-token",
                "nested": {
                    "client-secret": "s3cr3t",
                    "safe": "x" * 4000,
                    "cycle": cycle,
                    "unknown": object(),
                },
            },
            "decision_value": "y" * 900,
        }
    )

    result = sanitize_event(
        event,
        additional_redacted_keys=frozenset({"tenant-private-value"}),
    )

    assert "definitely-secret-token" not in result.serialized
    assert "s3cr3t" not in result.serialized
    assert REDACTED in result.serialized
    assert len(result.serialized.encode()) <= MAX_EVENT_BYTES
    assert result.attributes["redaction_applied"] is True
    assert result.attributes["truncated_fields"]
    assert any("cycle" in field or "unknown" in field for field in result.attributes["omitted_fields"])


@pytest.mark.parametrize(
    "secret",
    [
        "Bearer abcdefghijklmnopqrstuvwxyz",
        "AccountKey=abcdefghijklmnopqrstuvwxyz012345",
        "SharedAccessSignature=abcdefghijklmnopqrstuvwxyz",
        "https://example.test/path?sig=abcdefghijklmnopqrstuvwxyz&sv=2026",
        "-----BEGIN PRIVATE KEY-----\nsecret\n-----END PRIVATE KEY-----",
        "eyJabcdefgh.ijklmnop.qrstuvwxyz",
        '{"api_key":"supersecret123"}',
        "Authorization: Basic dXNlcjpwYXNzd29yZA==",
        "Cookie: session=supersecret123",
        "https://user:password@example.test/path",
        '{"password":"abc"}',
        '{"api_key":"xyz"}',
    ],
)
def test_prohibited_values_never_reach_serialized_exporter_input(secret):
    event = _base_event()
    event["source_excerpt"] = {"value": secret}

    result = sanitize_event(
        event,
        additional_redacted_keys=frozenset(),
    )

    assert secret not in result.serialized
    assert REDACTED in result.serialized


def test_unknown_optional_fields_are_omitted_for_major_version_one():
    event = _base_event()
    event["future_field"] = "reader must ignore this"

    result = sanitize_event(event, additional_redacted_keys=frozenset())

    assert "future_field" not in result.attributes
    assert "future_field" in result.attributes["omitted_fields"]


def test_recursive_key_denylist_covers_prohibited_credential_classes():
    event = _base_event()
    event["tool_arguments"] = {
        "proxy-auth": "secret-a",
        "cookies": "secret-b",
        "refresh_token": "secret-c",
        "id-token": "secret-d",
        "api.client.password": "secret-e",
        "database_connection_string": "secret-f",
        "sas": "secret-g",
        "certificate_credential": "secret-h",
    }

    result = sanitize_event(event, additional_redacted_keys=frozenset())

    for secret in "abcdefgh":
        assert f"secret-{secret}" not in result.serialized
    assert result.serialized.count(REDACTED) >= 8


def test_additional_redaction_keys_cannot_corrupt_required_identifiers():
    event = _base_event()
    event["tool_arguments"] = {"nested_id": "redact-me"}

    result = sanitize_event(
        event,
        additional_redacted_keys=frozenset({"id"}),
    )

    assert result.attributes["event_id"].startswith("evt_")
    assert result.attributes["correlation_id"].startswith("req_")
    assert "redact-me" not in result.serialized
