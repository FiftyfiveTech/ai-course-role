"""Structured logging: every line tagged with a trace id or a process id.

configure_logging() is called once per process, early, by each real
entrypoint (scripts/roleplay.py's main(), role/webapp.py at module level).
Every module just does the standard `logger = logging.getLogger(__name__)`
— no wrapper needed, this module only configures the root logger's handlers.

The trace id is the same one role.tracing hands to Tempo/Grafana — when a
real span is active, a log line and its full trace are one click apart, no
separate correlation id scheme. When tracing is off (the default —
role.tracing.get_tracer() stays a no-op), every line still carries the OS
process id, so "check live logs" never depends on tracing being configured.
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

from opentelemetry import trace

LOGS_DIR = Path(__file__).parent.parent.parent / "logs"
LOG_FORMAT = "%(asctime)s %(levelname)-7s [pid=%(pid)s trace=%(trace_id)s] %(name)s: %(message)s"

_configured = False


class _CorrelationFilter(logging.Filter):
    """Attach pid (always) and trace_id (when a real span is active) to
    every record — '-' for trace_id otherwise, never a missing field."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.pid = os.getpid()
        span_context = trace.get_current_span().get_span_context()
        record.trace_id = format(span_context.trace_id, "032x") if span_context.is_valid else "-"
        return True


def configure_logging(level: int = logging.INFO) -> None:
    """Attach a console + rotating-file handler to the root logger, both
    carrying the correlation filter above. Safe to call more than once —
    only the first call attaches handlers."""
    global _configured
    if _configured:
        return
    _configured = True

    LOGS_DIR.mkdir(exist_ok=True)
    formatter = logging.Formatter(LOG_FORMAT)
    correlation_filter = _CorrelationFilter()

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.addFilter(correlation_filter)

    file_handler = RotatingFileHandler(
        LOGS_DIR / "role.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.addFilter(correlation_filter)

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(console_handler)
    root.addHandler(file_handler)
