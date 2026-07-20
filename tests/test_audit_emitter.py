import asyncio
import json
import logging
from contextlib import contextmanager
from pathlib import Path

import jsonschema
import pytest
from azure.monitor.opentelemetry.exporter.export.logs._exporter import (
    _convert_log_to_envelope,
    _log_data_is_event,
)
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import (
    InMemoryLogRecordExporter,
    SimpleLogRecordProcessor,
)
from opentelemetry.sdk.trace import TracerProvider

from telemetry.audit import (
    AuditEmitter,
    begin_audit_request,
    end_audit_request,
    invoke_audited_tool,
    wrap_ai_functions,
)
from telemetry.audit_contract import (
    AUDIT_LOG_BODY,
    AuditSettings,
    AuditStatus,
    EventType,
    ReasonCode,
    ROOT_PARENT_EVENT_ID,
)


class CaptureHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


class RaisingHandler(logging.Handler):
    def emit(self, record):
        raise RuntimeError("exporter failed")


def enabled_emitter(*, source_limit=25, sensitive_fields=frozenset()):
    return AuditEmitter(
        AuditSettings(
            enabled=True,
            sensitive_content_enabled=bool(sensitive_fields),
            sensitive_content_fields=sensitive_fields,
            actor_pseudonym_enabled=False,
            source_event_limit=source_limit,
            hmac_key_id="v1",
            hmac_key=b"k" * 32,
            additional_redacted_keys=frozenset(),
        ),
        service_name="gpt-rag-orchestrator",
        service_version="3.7.0",
        environment="test",
    )


@contextmanager
def capture_audit_logs(handler=None):
    logger = logging.getLogger("gptrag.audit")
    previous = (list(logger.handlers), logger.propagate, logger.level)
    handler = handler or CaptureHandler()
    logger.handlers = [handler]
    logger.propagate = False
    logger.setLevel(logging.INFO)
    try:
        yield handler
    finally:
        logger.handlers, logger.propagate, logger.level = previous


def event_types(handler):
    return [
        getattr(record, "event_type")
        for record in handler.records
        if hasattr(record, "event_type")
    ]


def test_fixed_body_custom_event_name_and_metadata_only_default():
    emitter = enabled_emitter()
    AuditEmitter._default = emitter

    with capture_audit_logs() as capture:
        emitter.emit(
            EventType.REQUEST_STARTED,
            operation="test",
            status=AuditStatus.STARTED,
            reason_code=ReasonCode.REQUEST_RECEIVED,
            sensitive={"prompt": "must not be exported"},
        )

    record = capture.records[0]
    assert record.getMessage() == AUDIT_LOG_BODY
    assert (
        getattr(record, "microsoft.custom_event.name")
        == "gptrag.audit.request.started"
    )
    assert not hasattr(record, "prompt")
    assert "prompt" in record.omitted_fields
    assert record.parent_event_id == ROOT_PARENT_EVENT_ID


def test_emitter_failure_never_escapes_to_user_operation():
    emitter = enabled_emitter()

    with capture_audit_logs(RaisingHandler()):
        assert (
            emitter.emit(
                EventType.REQUEST_STARTED,
                operation="test",
                status=AuditStatus.STARTED,
                reason_code=ReasonCode.REQUEST_RECEIVED,
            )
            is None
        )


def test_redaction_failure_discards_payload_and_emits_minimal_failure():
    emitter = enabled_emitter()

    with capture_audit_logs() as capture:
        result = emitter.emit(
            EventType.REQUEST_STARTED,
            operation="test",
            status=AuditStatus.STARTED,
            reason_code=ReasonCode.REQUEST_RECEIVED,
            metadata={"duration_ms": float("nan")},
            sensitive={"prompt": "payload-to-discard"},
        )

    assert result is None
    assert event_types(capture) == ["audit.emission.failed"]
    assert "payload-to-discard" not in str(capture.records[0].__dict__)


def test_enabled_sensitive_allowlist_still_redacts_prohibited_values():
    emitter = enabled_emitter(sensitive_fields=frozenset({"prompt"}))

    with capture_audit_logs() as capture:
        emitter.emit(
            EventType.REQUEST_STARTED,
            operation="test",
            status=AuditStatus.STARTED,
            reason_code=ReasonCode.REQUEST_RECEIVED,
            sensitive={
                "prompt": {
                    "question": "safe text",
                    "access_token": "must-not-export",
                }
            },
        )

    record = capture.records[0]
    assert "safe text" in record.prompt
    assert "must-not-export" not in record.prompt
    assert record.redaction_applied is True


def test_in_memory_otel_log_has_current_trace_and_span_context():
    emitter = enabled_emitter()
    log_provider = LoggerProvider()
    log_exporter = InMemoryLogRecordExporter()
    log_provider.add_log_record_processor(SimpleLogRecordProcessor(log_exporter))
    trace_provider = TracerProvider()

    with capture_audit_logs(LoggingHandler(logger_provider=log_provider)):
        with trace_provider.get_tracer("audit-test").start_as_current_span(
            "request"
        ) as span:
            emitter.emit(
                EventType.REQUEST_STARTED,
                operation="test",
                status=AuditStatus.STARTED,
                reason_code=ReasonCode.REQUEST_RECEIVED,
            )
            expected_trace = span.get_span_context().trace_id
            expected_span = span.get_span_context().span_id

    readable = log_exporter.get_finished_logs()[0]
    assert readable.log_record.trace_id == expected_trace
    assert readable.log_record.span_id == expected_span
    assert _log_data_is_event(readable) is True
    assert len(readable.log_record.attributes) <= 64
    envelope = _convert_log_to_envelope(readable)
    assert (
        envelope.data.base_data.properties["parent_event_id"]
        == ROOT_PARENT_EVENT_ID
    )
    wire_schema = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "contracts"
            / "audit-event-v1.application-insights.schema.json"
        ).read_text()
    )
    jsonschema.Draft202012Validator(wire_schema).validate(
        {
            "name": envelope.data.base_data.name,
            "properties": dict(envelope.data.base_data.properties),
        }
    )


@pytest.mark.asyncio
async def test_tool_success_failure_timeout_and_cancellation_events():
    emitter = enabled_emitter()
    AuditEmitter._default = emitter
    context, token = begin_audit_request()
    context.request_started_event_id = "evt_" + ("1" * 32)

    async def fail():
        raise ValueError("raw failure must not be exported")

    async def timeout():
        raise TimeoutError()

    async def cancel():
        raise asyncio.CancelledError()

    try:
        with capture_audit_logs() as capture:
            assert await invoke_audited_tool("ok", lambda: "done") == "done"
            with pytest.raises(ValueError):
                await invoke_audited_tool("fail", fail)
            with pytest.raises(TimeoutError):
                await invoke_audited_tool("timeout", timeout)
            with pytest.raises(asyncio.CancelledError):
                await invoke_audited_tool("cancel", cancel)
    finally:
        end_audit_request(token)

    assert event_types(capture) == [
        "tool.invocation.started",
        "tool.invocation.completed",
        "tool.invocation.started",
        "tool.invocation.failed",
        "tool.invocation.started",
        "tool.invocation.failed",
        "tool.invocation.started",
        "tool.invocation.cancelled",
    ]
    serialized = " ".join(str(record.__dict__) for record in capture.records)
    assert "raw failure must not be exported" not in serialized


def test_source_events_are_hmac_pseudonymized_and_hard_limited():
    emitter = enabled_emitter(source_limit=2)
    AuditEmitter._default = emitter
    context, token = begin_audit_request()
    context.request_started_event_id = "evt_" + ("1" * 32)
    try:
        with capture_audit_logs() as capture:
            for rank in range(10):
                emitter.emit_source(
                    selected=True,
                    source_type="test",
                    source_reference=f"https://secret.example/{rank}?sig=nope",
                    source_rank=rank,
                )
    finally:
        end_audit_request(token)

    assert event_types(capture) == [
        "grounding.source.selected",
        "grounding.source.selected",
    ]
    assert all(record.source_id.startswith("hmac_") for record in capture.records)
    assert "secret.example" not in str([record.__dict__ for record in capture.records])


def test_source_excerpt_is_captured_only_when_explicitly_allowlisted():
    emitter = enabled_emitter(
        sensitive_fields=frozenset({"source_excerpt"})
    )
    AuditEmitter._default = emitter
    context, token = begin_audit_request()
    context.request_started_event_id = "evt_" + ("1" * 32)
    try:
        with capture_audit_logs() as capture:
            emitter.emit_source(
                selected=True,
                source_type="azure_ai_search",
                source_reference="document",
                source_excerpt="approved excerpt",
            )
    finally:
        end_audit_request(token)

    assert capture.records[0].source_excerpt == "approved excerpt"


@pytest.mark.asyncio
async def test_public_maf_function_proxy_preserves_contract_and_audits_call():
    from agent_framework import AIFunction

    async def echo(value: str) -> str:
        return value

    original = AIFunction(name="echo", description="Echo", func=echo)
    wrapped = wrap_ai_functions([original], tool_kind="mcp")
    emitter = enabled_emitter()
    AuditEmitter._default = emitter
    context, token = begin_audit_request()
    context.request_started_event_id = "evt_" + ("1" * 32)
    try:
        with capture_audit_logs() as capture:
            result = await wrapped[0].invoke(value="hello")
    finally:
        end_audit_request(token)

    assert result == "hello"
    assert wrapped[0].name == original.name
    assert wrapped[0].input_model is original.input_model
    assert event_types(capture) == [
        "tool.invocation.started",
        "tool.invocation.completed",
    ]
