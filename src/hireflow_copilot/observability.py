"""Structured JSON logs and OpenTelemetry traces (console, OTLP, or Azure Application Insights)."""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

from hireflow_copilot.config import Settings

_STANDARD = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    """One JSON object per line, with ``extra={...}`` fields and the active trace id merged in."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update({k: v for k, v in record.__dict__.items() if k not in _STANDARD})
        span = trace.get_current_span().get_span_context()
        if span.is_valid:
            payload["trace_id"] = format(span.trace_id, "032x")
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup(settings: Settings) -> TracerProvider | None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=settings.log_level, handlers=[handler], force=True)

    if settings.otel_exporter == "none":
        return None

    if settings.otel_exporter == "azure":
        # Reads APPLICATIONINSIGHTS_CONNECTION_STRING. Install with: uv sync --extra azure
        from azure.monitor.opentelemetry import configure_azure_monitor

        configure_azure_monitor(logger_name="hireflow_copilot")
        provider = trace.get_tracer_provider()
        return provider if isinstance(provider, TracerProvider) else None

    provider = TracerProvider(resource=Resource.create({"service.name": "hireflow-copilot"}))
    if settings.otel_exporter == "otlp":
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        # Endpoint from OTEL_EXPORTER_OTLP_ENDPOINT (default http://localhost:4318), e.g. Jaeger or Aspire.
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    else:
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter(out=sys.stderr)))
    trace.set_tracer_provider(provider)
    os.environ.setdefault("OTEL_SERVICE_NAME", "hireflow-copilot")
    return provider
