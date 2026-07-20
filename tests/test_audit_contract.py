import base64
import hashlib
import json
import re
import tracemalloc
from collections.abc import Mapping, Sequence
from pathlib import Path

import jsonschema
import pytest

from telemetry.audit_contract import (
    INGESTION_EVENT_TYPES,
    MAX_EVENT_BYTES,
    ROOT_PARENT_EVENT_ID,
    AuditConfigurationError,
    AuditSettings,
    EventType,
    format_utc,
    logical_parent_to_wire,
    new_correlation_id,
    new_event_id,
    utc_now,
    wire_parent_to_logical,
)
from telemetry.audit_sanitizer import REDACTED, sanitize_event


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_INGESTION_EVENT_TYPES = frozenset(
    {
        "ingestion.run.started",
        "ingestion.run.completed",
        "ingestion.run.failed",
        "ingestion.run.cancelled",
        "ingestion.document.indexed",
        "ingestion.document.rejected",
        "ingestion.document.deleted",
    }
)


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


def _as_application_insights_event(event):
    def stringify(value):
        if value is None:
            return ROOT_PARENT_EVENT_ID
        if isinstance(value, bool):
            return str(value).lower()
        if isinstance(value, (list, dict)):
            return json.dumps(value, separators=(",", ":"))
        return str(value)

    return {
        "name": f"gptrag.audit.{event['event_type']}",
        "properties": {key: stringify(value) for key, value in event.items()},
    }


def test_golden_event_validates_against_shared_schema():
    schema = json.loads(
        (ROOT / "contracts" / "audit-event-v1.schema.json").read_text()
    )
    golden = json.loads(
        (ROOT / "tests" / "golden" / "audit_event_v1.json").read_text()
    )

    jsonschema.Draft202012Validator(schema).validate(golden)


def test_root_golden_validates_and_translates_only_at_wire_boundary():
    schema = json.loads(
        (ROOT / "contracts" / "audit-event-v1.schema.json").read_text()
    )
    golden = json.loads(
        (ROOT / "tests" / "golden" / "audit_event_v1_root.json").read_text()
    )

    jsonschema.Draft202012Validator(schema).validate(golden)
    assert golden["parent_event_id"] is None
    assert logical_parent_to_wire(golden["parent_event_id"]) == ROOT_PARENT_EVENT_ID


@pytest.mark.parametrize(
    "fixture_name",
    [
        "audit_event_v1_ingestion_run.json",
        "audit_event_v1_ingestion_document.json",
    ],
)
def test_ingestion_goldens_validate_against_logical_and_wire_schemas(fixture_name):
    logical_schema = json.loads(
        (ROOT / "contracts" / "audit-event-v1.schema.json").read_text()
    )
    wire_schema = json.loads(
        (
            ROOT
            / "contracts"
            / "audit-event-v1.application-insights.schema.json"
        ).read_text()
    )
    golden = json.loads((ROOT / "tests" / "golden" / fixture_name).read_text())

    jsonschema.Draft202012Validator(logical_schema).validate(golden)
    jsonschema.Draft202012Validator(wire_schema).validate(
        _as_application_insights_event(golden)
    )


def test_ingestion_taxonomy_is_exact_across_python_and_both_schemas():
    logical_schema = json.loads(
        (ROOT / "contracts" / "audit-event-v1.schema.json").read_text()
    )
    wire_schema = json.loads(
        (
            ROOT
            / "contracts"
            / "audit-event-v1.application-insights.schema.json"
        ).read_text()
    )
    orchestrator_event_types = {event_type.value for event_type in EventType}
    expected_event_types = orchestrator_event_types | EXPECTED_INGESTION_EVENT_TYPES

    assert INGESTION_EVENT_TYPES == EXPECTED_INGESTION_EVENT_TYPES
    assert set(logical_schema["properties"]["event_type"]["enum"]) == expected_event_types
    assert (
        set(wire_schema["properties"]["properties"]["properties"]["event_type"]["enum"])
        == expected_event_types
    )
    assert {
        name.removeprefix("gptrag.audit.")
        for name in wire_schema["properties"]["name"]["enum"]
    } == expected_event_types


def test_legacy_ingestion_aliases_are_rejected_by_both_schemas():
    logical_schema = json.loads(
        (ROOT / "contracts" / "audit-event-v1.schema.json").read_text()
    )
    wire_schema = json.loads(
        (
            ROOT
            / "contracts"
            / "audit-event-v1.application-insights.schema.json"
        ).read_text()
    )
    legacy_aliases = {
        f"ingestion.{scope}.{action}"
        for scope, action in (
            ("request", "started"),
            ("request", "completed"),
            ("request", "failed"),
            ("request", "cancelled"),
            ("document", "selected"),
            ("outcome", "produced"),
            ("outcome", "rejected"),
        )
    }

    for alias in legacy_aliases:
        event = _base_event()
        event["event_type"] = alias
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.Draft202012Validator(logical_schema).validate(event)
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.Draft202012Validator(wire_schema).validate(
                _as_application_insights_event(event)
            )


def test_published_contract_hashes_match_artifacts():
    expected = {}
    for line in (ROOT / "contracts" / "audit-event-v1.sha256").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        expected[name] = digest

    for name, digest in expected.items():
        content = (ROOT / "contracts" / name).read_bytes().replace(b"\r\n", b"\n")
        assert hashlib.sha256(content).hexdigest() == digest


def test_ids_and_timestamp_have_canonical_shapes():
    assert re.fullmatch(r"evt_[0-9a-f]{32}", new_event_id())
    assert re.fullmatch(r"req_[0-9a-f]{32}", new_correlation_id())
    assert re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z",
        format_utc(utc_now()),
    )


def test_root_parent_logical_wire_conversion_is_lossless():
    child = "evt_" + ("1" * 32)

    assert logical_parent_to_wire(None) == ROOT_PARENT_EVENT_ID
    assert logical_parent_to_wire(child) == child
    assert wire_parent_to_logical(ROOT_PARENT_EVENT_ID) is None
    assert wire_parent_to_logical(child) == child


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


def test_sanitizer_uses_bounded_iteration_for_virtual_containers():
    class CountingSequence(Sequence):
        def __init__(self):
            self.iterations = 0

        def __len__(self):
            return 10**12

        def __getitem__(self, index):
            self.iterations += 1
            if self.iterations > 33:
                raise AssertionError("sequence was iterated beyond its bound")
            return index

    class CountingMapping(Mapping):
        def __init__(self):
            self.iterations = 0

        def __len__(self):
            return 10**12

        def __getitem__(self, key):
            raise KeyError(key)

        def __iter__(self):
            raise AssertionError("items() must be used")

        def items(self):
            for index in range(10**12):
                self.iterations += 1
                if self.iterations > 65:
                    raise AssertionError("mapping was iterated beyond its bound")
                yield str(index), index

    class UnknownInfiniteIterable:
        def __iter__(self):
            raise AssertionError("unknown iterables must not be inspected")

    sequence = CountingSequence()
    mapping = CountingMapping()
    event = _base_event()
    event["tool_arguments"] = {
        "sequence": sequence,
        "mapping": mapping,
        "unknown": UnknownInfiniteIterable(),
    }

    tracemalloc.start()
    try:
        result = sanitize_event(event, additional_redacted_keys=frozenset())
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert sequence.iterations == 33
    assert mapping.iterations == 65
    assert peak_bytes < 2 * 1024 * 1024
    assert "tool_arguments.unknown" in result.attributes["omitted_fields"]
    assert "tool_arguments.sequence" in result.attributes["truncated_fields"]
    assert "tool_arguments.mapping" in result.attributes["truncated_fields"]


def test_sanitizer_bounds_work_before_scanning_oversized_strings():
    event = _base_event()
    event["source_excerpt"] = "\x01" * (8 * 1024 * 1024)

    tracemalloc.start()
    try:
        result = sanitize_event(event, additional_redacted_keys=frozenset())
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert result.attributes["source_excerpt"] == ""
    assert "source_excerpt" in result.attributes["truncated_fields"]
    assert peak_bytes < 2 * 1024 * 1024


def test_oversized_nested_key_is_omitted_before_redaction_classification():
    event = _base_event()
    key = ("a" * 512) + "_password"
    event["tool_arguments"] = {key: "short-sensitive-credential"}

    result = sanitize_event(event, additional_redacted_keys=frozenset())

    assert "short-sensitive-credential" not in result.serialized
    assert any(
        field.startswith("tool_arguments.")
        for field in result.attributes["omitted_fields"]
    )
    assert all(
        len(field) <= 512
        for field in (
            result.attributes["omitted_fields"]
            + result.attributes["truncated_fields"]
        )
    )
