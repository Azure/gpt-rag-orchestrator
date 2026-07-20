"""Versioned contract and configuration for GPT-RAG audit events."""

from __future__ import annotations

import base64
import binascii
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any


SCHEMA_VERSION = 1
AUDIT_LOGGER_NAME = "gptrag.audit"
AUDIT_EVENT_PREFIX = "gptrag.audit."
AUDIT_LOG_BODY = "GPT-RAG audit event"
ROOT_PARENT_EVENT_ID = f"evt_{'0' * 32}"

MAX_EVENT_BYTES = 16 * 1024
# Leave room for the custom-event marker and the three source-code attributes
# added by the pinned OpenTelemetry LoggingHandler.
MAX_ATTRIBUTES = 60
MAX_METADATA_STRING = 512
MAX_SENSITIVE_STRING = 2048
MAX_DEPTH = 6
MAX_COLLECTION_ITEMS = 64
MAX_EMITTED_ARRAY_ITEMS = 32
MAX_SANITIZER_NODES = 256
MAX_SOURCE_EVENTS = 25
MAX_AUDIT_EVENTS = 64
MAX_TOOL_INVOCATIONS = 16
RESERVED_REQUEST_TERMINAL_EVENTS = 1
RESERVED_EMISSION_FAILURE_EVENTS = 1
MAX_AUDIT_DURATION_MS = 86_400_000.0
MAX_ENVIRONMENT_LENGTH = 64


class AuditConfigurationError(ValueError):
    """Raised when enabled audit configuration is unsafe or incomplete."""


class EventType(StrEnum):
    REQUEST_STARTED = "request.started"
    REQUEST_COMPLETED = "request.completed"
    REQUEST_FAILED = "request.failed"
    REQUEST_CANCELLED = "request.cancelled"
    ROUTE_SELECTED = "route.selected"
    SOURCE_SELECTED = "grounding.source.selected"
    SOURCE_REJECTED = "grounding.source.rejected"
    TOOL_STARTED = "tool.invocation.started"
    TOOL_COMPLETED = "tool.invocation.completed"
    TOOL_FAILED = "tool.invocation.failed"
    TOOL_CANCELLED = "tool.invocation.cancelled"
    OUTCOME_PRODUCED = "outcome.produced"
    OUTCOME_REJECTED = "outcome.rejected"
    EMISSION_FAILED = "audit.emission.failed"


# The shared JSON Schema also reserves these event names for the ingestion
# component. The orchestrator deliberately does not emit them.
INGESTION_EVENT_TYPES = frozenset(
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


class AuditStatus(StrEnum):
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SELECTED = "selected"
    REJECTED = "rejected"
    PRODUCED = "produced"


class ReasonCode(StrEnum):
    NONE = "none"
    REQUEST_RECEIVED = "request_received"
    REQUEST_COMPLETED = "request_completed"
    REQUEST_FAILED = "request_failed"
    REQUEST_CANCELLED = "request_cancelled"
    CLIENT_DISCONNECTED = "client_disconnected"
    PARTIAL_OUTPUT = "partial_output"
    STRATEGY_CONFIGURED = "strategy_configured"
    DIRECT_MODEL_SELECTED = "direct_model_selected"
    AGENT_SELECTED = "agent_selected"
    SOURCE_SELECTED = "source_selected"
    SOURCE_REJECTED = "source_rejected"
    SOURCE_EMPTY = "source_empty"
    SOURCE_LIMIT_REACHED = "source_limit_reached"
    TOOL_INVOKED = "tool_invoked"
    TOOL_COMPLETED = "tool_completed"
    TOOL_FAILED = "tool_failed"
    TOOL_CANCELLED = "tool_cancelled"
    TIMEOUT = "timeout"
    OUTCOME_PRODUCED = "outcome_produced"
    OUTCOME_REJECTED = "outcome_rejected"
    VALIDATION_FAILED = "validation_failed"
    REDACTION_FAILURE = "redaction_failure"
    SERIALIZATION_FAILURE = "serialization_failure"
    EVENT_TOO_LARGE = "event_too_large"
    ATTRIBUTE_LIMIT_EXCEEDED = "attribute_limit_exceeded"
    EXPORT_FAILURE = "export_failure"
    UNKNOWN = "unknown"


class CaptureMode(StrEnum):
    METADATA_ONLY = "metadata_only"
    SENSITIVE_ALLOWLIST = "sensitive_allowlist"


SENSITIVE_FIELDS = frozenset(
    {
        "prompt",
        "response",
        "source_excerpt",
        "tool_arguments",
        "tool_result",
    }
)

REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "event_id",
        "event_type",
        "event_time_utc",
        "correlation_id",
        "trace_id",
        "span_id",
        "parent_event_id",
        "service_name",
        "service_version",
        "environment",
        "operation",
        "status",
        "reason_code",
        "capture_mode",
        "redaction_applied",
        "omitted_fields",
        "truncated_fields",
    }
)

OPTIONAL_FIELDS = frozenset(
    {
        "started_at_utc",
        "duration_ms",
        "decision_type",
        "decision_value",
        "source_id",
        "source_type",
        "source_rank",
        "tool_name",
        "tool_id",
        "tool_invocation_id",
        "outcome_type",
        "failure_type",
        "input_count",
        "output_count",
        "source_count",
        "partial_output",
        "http_status_code",
        "transport",
        "actor_id",
        "conversation_id",
        "question_id",
        "thread_id",
        "hmac_key_id",
        "timing_source",
        "audit_events_omitted",
        "source_events_omitted",
        "tool_invocations_omitted",
        *SENSITIVE_FIELDS,
    }
)


def new_event_id() -> str:
    return f"evt_{uuid.uuid4().hex}"


def new_correlation_id() -> str:
    return f"req_{uuid.uuid4().hex}"


def is_safe_correlation_id(value: Any) -> bool:
    """Return whether a value has the server-generated correlation ID shape."""
    return (
        isinstance(value, str)
        and len(value) == 36
        and value.startswith("req_")
        and all(character in "0123456789abcdef" for character in value[4:])
    )


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def logical_parent_to_wire(parent_event_id: str | None) -> str:
    """Encode a logical root parent for Azure Monitor string properties."""
    return ROOT_PARENT_EVENT_ID if parent_event_id is None else parent_event_id


def wire_parent_to_logical(parent_event_id: str) -> str | None:
    """Decode the Azure Monitor root sentinel to the logical null parent."""
    return None if parent_event_id == ROOT_PARENT_EVENT_ID else parent_event_id


def _config_value(config: Any, key: str, default: Any) -> Any:
    try:
        return config.get(key, default=default)
    except TypeError:
        return config.get(key, default)
    except Exception:
        return default


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _decode_hmac_key(value: str) -> bytes:
    candidate = value.strip()
    if not candidate:
        raise AuditConfigurationError(
            "AUDIT_EVENTS_ENABLED requires AUDIT_HMAC_KEY as a 256-bit secret."
        )

    if len(candidate) == 64:
        try:
            decoded = bytes.fromhex(candidate)
        except ValueError:
            decoded = b""
        if len(decoded) == 32:
            return decoded

    padded = candidate + ("=" * (-len(candidate) % 4))
    for decoder in (base64.b64decode, base64.urlsafe_b64decode):
        try:
            decoded = decoder(padded.encode("ascii"))
        except (ValueError, UnicodeEncodeError, binascii.Error):
            continue
        if len(decoded) == 32:
            return decoded

    raise AuditConfigurationError(
        "AUDIT_HMAC_KEY must be Base64, Base64URL, or hexadecimal encoding of "
        "exactly 32 random bytes."
    )


@dataclass(frozen=True, slots=True)
class AuditSettings:
    enabled: bool
    sensitive_content_enabled: bool
    sensitive_content_fields: frozenset[str]
    actor_pseudonym_enabled: bool
    source_event_limit: int
    hmac_key_id: str
    hmac_key: bytes | None
    additional_redacted_keys: frozenset[str]

    @property
    def capture_mode(self) -> CaptureMode:
        if self.sensitive_content_enabled and self.sensitive_content_fields:
            return CaptureMode.SENSITIVE_ALLOWLIST
        return CaptureMode.METADATA_ONLY

    @classmethod
    def from_config(cls, config: Any) -> "AuditSettings":
        enabled = _as_bool(_config_value(config, "AUDIT_EVENTS_ENABLED", "false"))
        if not enabled:
            return cls(
                enabled=False,
                sensitive_content_enabled=False,
                sensitive_content_fields=frozenset(),
                actor_pseudonym_enabled=False,
                source_event_limit=MAX_SOURCE_EVENTS,
                hmac_key_id="v1",
                hmac_key=None,
                additional_redacted_keys=frozenset(),
            )

        sensitive_enabled = _as_bool(
            _config_value(config, "AUDIT_SENSITIVE_CONTENT_ENABLED", "false")
        )
        configured_fields = frozenset(
            part.strip()
            for part in str(
                _config_value(config, "AUDIT_SENSITIVE_CONTENT_FIELDS", "")
            ).split(",")
            if part.strip()
        )
        unknown_fields = configured_fields - SENSITIVE_FIELDS
        if unknown_fields:
            raise AuditConfigurationError(
                "AUDIT_SENSITIVE_CONTENT_FIELDS contains unsupported values: "
                + ", ".join(sorted(unknown_fields))
            )

        try:
            source_event_limit = int(
                _config_value(config, "AUDIT_SOURCE_EVENT_LIMIT", MAX_SOURCE_EVENTS)
            )
        except (TypeError, ValueError) as exc:
            raise AuditConfigurationError(
                "AUDIT_SOURCE_EVENT_LIMIT must be an integer between 1 and 25."
            ) from exc
        if not 1 <= source_event_limit <= MAX_SOURCE_EVENTS:
            raise AuditConfigurationError(
                f"AUDIT_SOURCE_EVENT_LIMIT must be between 1 and {MAX_SOURCE_EVENTS}."
            )

        hmac_key_id = str(
            _config_value(config, "AUDIT_HMAC_KEY_ID", "v1")
        ).strip()
        if enabled and (not hmac_key_id or len(hmac_key_id) > MAX_METADATA_STRING):
            raise AuditConfigurationError(
                "AUDIT_HMAC_KEY_ID is required and must not exceed 512 characters."
            )

        key_value = str(_config_value(config, "AUDIT_HMAC_KEY", "") or "")
        hmac_key = _decode_hmac_key(key_value) if enabled else None
        additional_keys = frozenset(
            part.strip()
            for part in str(
                _config_value(config, "AUDIT_ADDITIONAL_REDACTED_KEYS", "")
            ).split(",")
            if part.strip()
        )

        return cls(
            enabled=enabled,
            sensitive_content_enabled=sensitive_enabled,
            sensitive_content_fields=configured_fields,
            actor_pseudonym_enabled=_as_bool(
                _config_value(config, "AUDIT_ACTOR_PSEUDONYM_ENABLED", "false")
            ),
            source_event_limit=source_event_limit,
            hmac_key_id=hmac_key_id or "v1",
            hmac_key=hmac_key,
            additional_redacted_keys=additional_keys,
        )
