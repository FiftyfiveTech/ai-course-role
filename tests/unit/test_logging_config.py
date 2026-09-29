"""Unit tests for role.logging_config — every log record carries pid always,
and a real trace id whenever a span is active.

The "real trace id" case uses a private, local TracerProvider().get_tracer()
instance rather than the global trace.set_tracer_provider() — sidesteps
test_tracing.py's documented problem (OpenTelemetry only honors the FIRST
set_tracer_provider() call per process) entirely: get_current_span() reads
the same contextvar regardless of which provider produced the span, so a
locally-created span is exactly as "current" as a globally-configured one,
without ever mutating global state.
"""

import logging

from opentelemetry.sdk.trace import TracerProvider

from role.logging_config import _CorrelationFilter


def _make_record() -> logging.LogRecord:
    return logging.LogRecord(
        name="test", level=logging.INFO, pathname=__file__, lineno=1,
        msg="hello", args=(), exc_info=None,
    )


def test_filter_sets_pid_and_dash_trace_id_with_no_active_span():
    record = _make_record()
    assert _CorrelationFilter().filter(record) is True
    assert isinstance(record.pid, int)
    assert record.trace_id == "-"


def test_filter_sets_real_trace_id_when_a_span_is_active():
    provider = TracerProvider(shutdown_on_exit=False)
    tracer = provider.get_tracer("test")

    with tracer.start_as_current_span("x"):
        record = _make_record()
        _CorrelationFilter().filter(record)

    assert len(record.trace_id) == 32
    assert record.trace_id != "0" * 32
    int(record.trace_id, 16)  # valid hex


def test_filter_reverts_to_dash_after_span_closes():
    provider = TracerProvider(shutdown_on_exit=False)
    tracer = provider.get_tracer("test")

    with tracer.start_as_current_span("x"):
        pass

    record = _make_record()
    _CorrelationFilter().filter(record)
    assert record.trace_id == "-"
