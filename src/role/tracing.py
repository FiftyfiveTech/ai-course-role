"""OpenTelemetry tracing — opt-in via OTEL_EXPORTER_OTLP_ENDPOINT.

Every model call (persona/evaluator/coach) and every webapp request wraps
itself in a span via get_tracer(), unconditionally — there is no
`if tracing_enabled:` branching at any call site. The safety lives entirely
here: without OTEL_EXPORTER_OTLP_ENDPOINT set, opentelemetry.trace.get_tracer()
returns the API's built-in no-op tracer (verified: no TracerProvider is ever
constructed, so no atexit shutdown/force-flush hook is ever registered —
`make demo` / `make web` / the test suite behave exactly as before, whether
or not a Tempo container exists). Only when the endpoint is configured does
this module build a real TracerProvider exporting to it, viewable in Grafana
via `make otel-up` (see docker-compose.yml).
"""

import os

from opentelemetry import trace

SERVICE_NAME = "role"

_configured = False


def get_tracer() -> trace.Tracer:
    """Return the module-wide tracer, configuring a real exporter on first
    call if OTEL_EXPORTER_OTLP_ENDPOINT is set — otherwise the OpenTelemetry
    API's default no-op tracer, untouched."""
    global _configured
    if not _configured:
        _configured = True
        endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
        if endpoint:
            _configure_provider(endpoint)
    return trace.get_tracer(SERVICE_NAME)


def _configure_provider(endpoint: str) -> None:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(resource=Resource.create({"service.name": SERVICE_NAME}))
    exporter = OTLPSpanExporter(endpoint=f"{endpoint.rstrip('/')}/v1/traces")
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
