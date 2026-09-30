"""Unit tests for role.tracing — opt-in tracer configuration.

get_tracer() must stay a safe no-op with zero global side effects when
OTEL_EXPORTER_OTLP_ENDPOINT isn't set — this is what keeps `make demo` /
`make web` / the rest of this test suite working with no Docker running.

These tests never call the real trace.set_tracer_provider(): OpenTelemetry
only honors the FIRST call per process ("Overriding of current TracerProvider
is not allowed"), so calling it for real here would permanently change
global tracer state for every other test in this pytest run. Every assertion
mocks at that boundary instead.
"""

from unittest.mock import patch

import pytest

import role.tracing as tracing_mod


@pytest.fixture(autouse=True)
def _reset_configured_latch():
    """Each test gets its own not-yet-called get_tracer() state."""
    tracing_mod._configured = False
    yield
    tracing_mod._configured = False


def test_get_tracer_is_noop_without_endpoint(monkeypatch):
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)

    with patch.object(tracing_mod.trace, "set_tracer_provider") as mock_set:
        tracer = tracing_mod.get_tracer()
        mock_set.assert_not_called()

    with tracer.start_as_current_span("x") as span:
        assert span.is_recording() is False


def test_get_tracer_is_noop_on_blank_endpoint(monkeypatch):
    """Blank (whitespace-only) is treated the same as unset, not '' as a URL."""
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "   ")

    with patch.object(tracing_mod.trace, "set_tracer_provider") as mock_set:
        tracing_mod.get_tracer()
        mock_set.assert_not_called()


def test_get_tracer_configures_real_provider_when_endpoint_set(monkeypatch):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://example:1234")

    with patch(
        "opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter"
    ) as mock_exporter_cls, patch.object(tracing_mod.trace, "set_tracer_provider") as mock_set:
        tracing_mod.get_tracer()

    mock_exporter_cls.assert_called_once_with(endpoint="http://example:1234/v1/traces")
    mock_set.assert_called_once()
    provider = mock_set.call_args[0][0]
    assert provider.resource.attributes["service.name"] == "role"


def test_get_tracer_strips_trailing_slash_from_endpoint(monkeypatch):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://example:1234/")

    with patch(
        "opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter"
    ) as mock_exporter_cls, patch.object(tracing_mod.trace, "set_tracer_provider"):
        tracing_mod.get_tracer()

    mock_exporter_cls.assert_called_once_with(endpoint="http://example:1234/v1/traces")


def test_get_tracer_only_configures_once(monkeypatch):
    """A second get_tracer() call must not reconfigure — _configured latches."""
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://example:1234")

    with patch(
        "opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter"
    ), patch.object(tracing_mod.trace, "set_tracer_provider") as mock_set:
        tracing_mod.get_tracer()
        tracing_mod.get_tracer()

    mock_set.assert_called_once()
