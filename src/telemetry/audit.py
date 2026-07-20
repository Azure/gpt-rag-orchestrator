"""Best-effort GPT-RAG audit event emission over OpenTelemetry logging."""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import hmac
import inspect
import logging
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from opentelemetry import trace

from .audit_contract import (
    AUDIT_EVENT_PREFIX,
    AUDIT_LOG_BODY,
    AUDIT_LOGGER_NAME,
    SCHEMA_VERSION,
    AuditSettings,
    AuditStatus,
    EventType,
    ReasonCode,
    ROOT_PARENT_EVENT_ID,
    format_utc,
    new_correlation_id,
    new_event_id,
    utc_now,
)
from .audit_sanitizer import AuditSanitizationError, sanitize_event


T = TypeVar("T")
_logger = logging.getLogger(AUDIT_LOGGER_NAME)
_warning_logger = logging.getLogger("gptrag.audit_warning")
_SOURCE_TYPE_ALIASES = {
    "azure_ai_search": "azure_ai_search",
    "azure_ai_search_multimodal": "azure_ai_search_multimodal",
    "foundry_iq": "foundry_iq",
    "foundry_iq_mcp": "foundry_iq_mcp",
    "mcpserver": "foundry_iq_mcp",
    "foundryiq": "foundry_iq",
    "azureblob": "foundry_iq",
    "searchindex": "azure_ai_search",
    "web": "web_grounding",
    "web_grounding": "web_grounding",
    "workiq": "work_iq",
    "work_iq": "work_iq",
    "fabricontology": "fabric_iq",
    "fabric_iq": "fabric_iq",
    "fabricdataagent": "fabric_data_agent",
    "fabric_data_agent": "fabric_data_agent",
    "remotesharepoint": "sharepoint_remote",
    "sharepoint_remote": "sharepoint_remote",
    "indexedonelake": "onelake",
    "onelake": "onelake",
    "indexedsharepoint": "sharepoint_indexed",
    "sharepoint_indexed": "sharepoint_indexed",
    "nl2sql_datasource": "nl2sql_datasource",
    "test": "test",
}


@dataclass(slots=True)
class AuditRequestContext:
    correlation_id: str
    request_started_event_id: str | None = None
    source_events_emitted: int = 0
    partial_output: bool = False
    output_count: int = 0
    started_at: Any = field(default_factory=utc_now)


_current_context: contextvars.ContextVar[AuditRequestContext | None] = (
    contextvars.ContextVar("gptrag_audit_context", default=None)
)


class AuditEmitter:
    """Create and emit safe v1 audit custom events."""

    _default: "AuditEmitter | None" = None

    def __init__(
        self,
        settings: AuditSettings,
        *,
        service_name: str,
        service_version: str,
        environment: str,
    ) -> None:
        self.settings = settings
        self.service_name = service_name
        self.service_version = service_version
        self.environment = (environment or "unknown")[:64]

    @classmethod
    def configure(
        cls,
        config: Any,
        *,
        service_name: str,
        service_version: str,
    ) -> "AuditEmitter":
        settings = AuditSettings.from_config(config)
        try:
            environment = config.get(
                "ENVIRONMENT_NAME",
                default=config.get("AZURE_ENV_NAME", default="unknown"),
            )
        except Exception:
            environment = "unknown"
        cls._default = cls(
            settings,
            service_name=service_name,
            service_version=service_version,
            environment=str(environment or "unknown"),
        )
        _logger.setLevel(logging.INFO)
        return cls._default

    @classmethod
    def default(cls) -> "AuditEmitter":
        if cls._default is None:
            cls._default = cls(
                AuditSettings(
                    enabled=False,
                    sensitive_content_enabled=False,
                    sensitive_content_fields=frozenset(),
                    actor_pseudonym_enabled=False,
                    source_event_limit=25,
                    hmac_key_id="v1",
                    hmac_key=None,
                    additional_redacted_keys=frozenset(),
                ),
                service_name="gpt-rag-orchestrator",
                service_version="0.0.0",
                environment="unknown",
            )
        return cls._default

    @property
    def enabled(self) -> bool:
        return self.settings.enabled

    def pseudonymize(self, kind: str, value: str | None) -> str | None:
        if not value or not self.settings.hmac_key:
            return None
        digest = hmac.new(
            self.settings.hmac_key,
            f"{kind}:{value}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()[:32]
        return f"hmac_{digest}"

    def _trace_fields(self) -> tuple[str, str]:
        span_context = trace.get_current_span().get_span_context()
        if not span_context.is_valid:
            return "0" * 32, "0" * 16
        return f"{span_context.trace_id:032x}", f"{span_context.span_id:016x}"

    def emit(
        self,
        event_type: EventType,
        *,
        operation: str,
        status: AuditStatus,
        reason_code: ReasonCode,
        correlation_id: str | None = None,
        parent_event_id: str | None = None,
        started_at: Any | None = None,
        duration_ms: float | None = None,
        metadata: dict[str, Any] | None = None,
        sensitive: dict[str, Any] | None = None,
    ) -> str | None:
        if not self.enabled:
            return None

        context = _current_context.get()
        event_id = new_event_id()
        event_time = utc_now()
        trace_id, span_id = self._trace_fields()
        event: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "event_id": event_id,
            "event_type": event_type.value,
            "event_time_utc": format_utc(event_time),
            "correlation_id": correlation_id
            or (context.correlation_id if context else new_correlation_id()),
            "trace_id": trace_id,
            "span_id": span_id,
            # The Azure Monitor exporter drops None-valued custom properties.
            # A reserved all-zero event ID keeps the required root field
            # transport-safe while the shared schema remains nullable.
            "parent_event_id": parent_event_id or ROOT_PARENT_EVENT_ID,
            "service_name": self.service_name,
            "service_version": self.service_version,
            "environment": self.environment,
            "operation": operation,
            "status": status.value,
            "reason_code": reason_code.value,
            "capture_mode": self.settings.capture_mode.value,
            "redaction_applied": False,
            "omitted_fields": [],
            "truncated_fields": [],
            "hmac_key_id": self.settings.hmac_key_id,
        }
        if started_at is not None:
            event["started_at_utc"] = format_utc(started_at)
        if duration_ms is not None:
            event["duration_ms"] = max(0.0, round(duration_ms, 3))
        if metadata:
            event.update(metadata)

        sensitive_values = sensitive or {}
        for field_name, value in sensitive_values.items():
            if (
                self.settings.sensitive_content_enabled
                and field_name in self.settings.sensitive_content_fields
            ):
                event[field_name] = value
            else:
                event["omitted_fields"].append(field_name)

        try:
            sanitized = sanitize_event(
                event,
                additional_redacted_keys=self.settings.additional_redacted_keys,
            )
            _logger.info(
                AUDIT_LOG_BODY,
                extra={
                    "microsoft.custom_event.name": (
                        f"{AUDIT_EVENT_PREFIX}{event_type.value}"
                    ),
                    **sanitized.attributes,
                },
            )
        except AuditSanitizationError as exc:
            failure = str(exc).casefold()
            if "serialization" in failure:
                reason = ReasonCode.SERIALIZATION_FAILURE
            elif "16 kib" in failure:
                reason = ReasonCode.EVENT_TOO_LARGE
            elif "attribute" in failure:
                reason = ReasonCode.ATTRIBUTE_LIMIT_EXCEEDED
            else:
                reason = ReasonCode.REDACTION_FAILURE
            self._emit_failure(reason)
            return None
        except BaseException:
            self._emit_failure(ReasonCode.EXPORT_FAILURE)
            return None
        return event_id

    def _emit_failure(self, reason: ReasonCode) -> None:
        """Emit a payload-free failure event, then degrade to a fixed warning."""
        event_time = utc_now()
        trace_id, span_id = self._trace_fields()
        context = _current_context.get()
        attributes = {
            "schema_version": SCHEMA_VERSION,
            "event_id": new_event_id(),
            "event_type": EventType.EMISSION_FAILED.value,
            "event_time_utc": format_utc(event_time),
            "correlation_id": (
                context.correlation_id if context else new_correlation_id()
            ),
            "trace_id": trace_id,
            "span_id": span_id,
            "parent_event_id": ROOT_PARENT_EVENT_ID,
            "service_name": self.service_name,
            "service_version": self.service_version,
            "environment": self.environment,
            "operation": "audit.emit",
            "status": AuditStatus.FAILED.value,
            "reason_code": reason.value,
            "capture_mode": self.settings.capture_mode.value,
            "redaction_applied": True,
            "omitted_fields": [],
            "truncated_fields": [],
        }
        try:
            _logger.info(
                AUDIT_LOG_BODY,
                extra={
                    "microsoft.custom_event.name": (
                        f"{AUDIT_EVENT_PREFIX}{EventType.EMISSION_FAILED.value}"
                    ),
                    **attributes,
                },
            )
        except BaseException:
            try:
                _warning_logger.warning(
                    "GPT-RAG audit event emission failed; request processing continues."
                )
            except BaseException:
                pass

    def emit_source(
        self,
        *,
        selected: bool,
        source_type: str,
        source_reference: str | None = None,
        source_rank: int | None = None,
        reason_code: ReasonCode | None = None,
        source_excerpt: Any = None,
    ) -> str | None:
        context = _current_context.get()
        if not context or not self.enabled:
            return None
        if context.source_events_emitted >= self.settings.source_event_limit:
            return None
        context.source_events_emitted += 1
        normalized_source_type = "".join(
            character
            for character in source_type.casefold()
            if character.isalnum() or character == "_"
        )
        metadata: dict[str, Any] = {
            "source_type": _SOURCE_TYPE_ALIASES.get(
                normalized_source_type, "unknown"
            )
        }
        if source_reference:
            metadata["source_id"] = self.pseudonymize("source", source_reference)
        if source_rank is not None:
            metadata["source_rank"] = source_rank
        return self.emit(
            EventType.SOURCE_SELECTED if selected else EventType.SOURCE_REJECTED,
            operation="grounding.select",
            status=AuditStatus.SELECTED if selected else AuditStatus.REJECTED,
            reason_code=reason_code
            or (ReasonCode.SOURCE_SELECTED if selected else ReasonCode.SOURCE_REJECTED),
            parent_event_id=context.request_started_event_id,
            metadata=metadata,
            sensitive={"source_excerpt": source_excerpt},
        )


def current_audit_context() -> AuditRequestContext | None:
    return _current_context.get()


def begin_audit_request(
    correlation_id: str | None = None,
) -> tuple[AuditRequestContext, contextvars.Token]:
    context = AuditRequestContext(correlation_id=correlation_id or new_correlation_id())
    return context, _current_context.set(context)


def end_audit_request(token: contextvars.Token) -> None:
    _current_context.reset(token)


async def invoke_audited_tool(
    tool_name: str,
    invocation: Callable[[], Awaitable[T] | T],
    *,
    tool_kind: str = "in_process",
    sensitive_arguments: Any = None,
) -> T:
    emitter = AuditEmitter.default()
    context = _current_context.get()
    if not emitter.enabled or context is None:
        result = invocation()
        return await result if inspect.isawaitable(result) else result

    started_at = utc_now()
    started_monotonic = time.monotonic()
    invocation_id = emitter.pseudonymize(
        "tool-invocation", f"{tool_name}:{new_event_id()}"
    )
    tool_id = emitter.pseudonymize("tool", tool_name)
    started_event_id = emitter.emit(
        EventType.TOOL_STARTED,
        operation="tool.invoke",
        status=AuditStatus.STARTED,
        reason_code=ReasonCode.TOOL_INVOKED,
        parent_event_id=context.request_started_event_id,
        metadata={
            "tool_name": tool_name,
            "tool_id": tool_id,
            "tool_invocation_id": invocation_id,
            "decision_type": tool_kind,
        },
        sensitive={"tool_arguments": sensitive_arguments},
    )
    try:
        result = invocation()
        value = await result if inspect.isawaitable(result) else result
    except asyncio.CancelledError:
        emitter.emit(
            EventType.TOOL_CANCELLED,
            operation="tool.invoke",
            status=AuditStatus.CANCELLED,
            reason_code=ReasonCode.TOOL_CANCELLED,
            parent_event_id=started_event_id,
            started_at=started_at,
            duration_ms=(time.monotonic() - started_monotonic) * 1000,
            metadata={
                "tool_name": tool_name,
                "tool_id": tool_id,
                "tool_invocation_id": invocation_id,
                "failure_type": "cancelled",
            },
        )
        raise
    except TimeoutError:
        emitter.emit(
            EventType.TOOL_FAILED,
            operation="tool.invoke",
            status=AuditStatus.FAILED,
            reason_code=ReasonCode.TIMEOUT,
            parent_event_id=started_event_id,
            started_at=started_at,
            duration_ms=(time.monotonic() - started_monotonic) * 1000,
            metadata={
                "tool_name": tool_name,
                "tool_id": tool_id,
                "tool_invocation_id": invocation_id,
                "failure_type": "timeout",
            },
        )
        raise
    except Exception:
        emitter.emit(
            EventType.TOOL_FAILED,
            operation="tool.invoke",
            status=AuditStatus.FAILED,
            reason_code=ReasonCode.TOOL_FAILED,
            parent_event_id=started_event_id,
            started_at=started_at,
            duration_ms=(time.monotonic() - started_monotonic) * 1000,
            metadata={
                "tool_name": tool_name,
                "tool_id": tool_id,
                "tool_invocation_id": invocation_id,
                "failure_type": "exception",
            },
        )
        raise

    emitter.emit(
        EventType.TOOL_COMPLETED,
        operation="tool.invoke",
        status=AuditStatus.COMPLETED,
        reason_code=ReasonCode.TOOL_COMPLETED,
        parent_event_id=started_event_id,
        started_at=started_at,
        duration_ms=(time.monotonic() - started_monotonic) * 1000,
        metadata={
            "tool_name": tool_name,
            "tool_id": tool_id,
            "tool_invocation_id": invocation_id,
        },
        sensitive={"tool_result": value},
    )
    return value


def wrap_ai_functions(
    functions: Iterable[Any],
    *,
    tool_kind: str = "mcp",
) -> list[Any]:
    """Return public MAF AIFunction proxies with audited invocation delegates."""
    from agent_framework import AIFunction

    wrapped: list[Any] = []
    for original in functions:
        if not isinstance(original, AIFunction):
            wrapped.append(original)
            continue

        async def audited_function(
            _original: Any = original,
            _tool_name: str = original.name,
            **kwargs: Any,
        ) -> Any:
            return await invoke_audited_tool(
                _tool_name,
                lambda: _original.invoke(**kwargs),
                tool_kind=tool_kind,
                sensitive_arguments=kwargs,
            )

        wrapped.append(
            AIFunction(
                name=original.name,
                description=original.description,
                approval_mode=original.approval_mode,
                max_invocations=original.max_invocations,
                max_invocation_exceptions=original.max_invocation_exceptions,
                additional_properties=original.additional_properties,
                func=audited_function,
                input_model=original.input_model,
            )
        )
    return wrapped
