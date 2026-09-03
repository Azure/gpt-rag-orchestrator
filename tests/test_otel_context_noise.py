"""Unit tests for the OpenTelemetry context-detach noise suppression.

``opentelemetry.context.detach`` logs ``Failed to detach context`` at ERROR
whenever a context token is detached from a different asyncio task than the one
that attached it. Instrumented async generators -- which is how every streamed
turn is produced -- hit that by design when the generator is closed by the
garbage collector or by the ASGI server from a foreign task.

These tests pin the two properties that make the suppression safe: the specific
noise is dropped, and every other ``opentelemetry.context`` record still gets
through.
"""

import logging

import pytest

from util.otel_context_noise import (
    AsyncGeneratorContextDetachFilter,
    silence_context_detach_noise,
)

CONTEXT_LOGGER_NAME = "opentelemetry.context"


def _record(message: str) -> logging.LogRecord:
    return logging.LogRecord(
        name=CONTEXT_LOGGER_NAME,
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=(),
        exc_info=None,
    )


@pytest.fixture
def clean_context_logger():
    """Restore the real logger's filters so tests do not leak into each other."""
    context_logger = logging.getLogger(CONTEXT_LOGGER_NAME)
    original = list(context_logger.filters)
    context_logger.filters = []
    try:
        yield context_logger
    finally:
        context_logger.filters = original


class TestAsyncGeneratorContextDetachFilter:
    def test_drops_the_detach_failure_record(self):
        assert AsyncGeneratorContextDetachFilter().filter(
            _record("Failed to detach context")
        ) is False

    def test_match_is_case_insensitive(self):
        assert AsyncGeneratorContextDetachFilter().filter(
            _record("FAILED TO DETACH CONTEXT")
        ) is False

    def test_keeps_unrelated_context_errors(self):
        assert AsyncGeneratorContextDetachFilter().filter(
            _record("Failed to attach context")
        ) is True

    def test_keeps_records_that_merely_mention_context(self):
        assert AsyncGeneratorContextDetachFilter().filter(
            _record("context propagation misconfigured")
        ) is True

    def test_applies_to_lazily_formatted_records(self):
        record = logging.LogRecord(
            name=CONTEXT_LOGGER_NAME,
            level=logging.ERROR,
            pathname=__file__,
            lineno=1,
            msg="Failed to %s context",
            args=("detach",),
            exc_info=None,
        )
        assert AsyncGeneratorContextDetachFilter().filter(record) is False

    def test_never_breaks_logging_when_formatting_fails(self):
        record = logging.LogRecord(
            name=CONTEXT_LOGGER_NAME,
            level=logging.ERROR,
            pathname=__file__,
            lineno=1,
            msg="missing %s %s placeholder",
            args=("only-one",),
            exc_info=None,
        )
        assert AsyncGeneratorContextDetachFilter().filter(record) is True


class TestSilenceContextDetachNoise:
    def test_installs_the_filter_on_the_context_logger(self, clean_context_logger):
        assert silence_context_detach_noise() is True
        assert any(
            isinstance(f, AsyncGeneratorContextDetachFilter)
            for f in clean_context_logger.filters
        )

    def test_is_idempotent(self, clean_context_logger):
        silence_context_detach_noise()
        assert silence_context_detach_noise() is False
        installed = [
            f
            for f in clean_context_logger.filters
            if isinstance(f, AsyncGeneratorContextDetachFilter)
        ]
        assert len(installed) == 1

    def test_preserves_filters_installed_by_others(self, clean_context_logger):
        unrelated = logging.Filter()
        clean_context_logger.addFilter(unrelated)
        silence_context_detach_noise()
        assert unrelated in clean_context_logger.filters

    def test_suppresses_the_record_end_to_end(self, clean_context_logger, caplog):
        silence_context_detach_noise()
        with caplog.at_level(logging.ERROR, logger=CONTEXT_LOGGER_NAME):
            clean_context_logger.error("Failed to detach context")
            clean_context_logger.error("A genuine context failure")
        messages = [r.getMessage() for r in caplog.records]
        assert "Failed to detach context" not in messages
        assert "A genuine context failure" in messages
