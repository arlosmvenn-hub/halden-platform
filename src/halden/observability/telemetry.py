"""OpenTelemetry setup, redaction, and the instruments we record."""

import hashlib
import re

from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import MetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExporter,
)
from opentelemetry.sdk.trace.sampling import (
    ParentBased,
    TraceIdRatioBased,
)

tracer = trace.get_tracer("halden")
meter = metrics.get_meter("halden")


# [start:setup]
def setup_telemetry(
    service_name: str,
    exporter: SpanExporter,
    *,
    sample_ratio: float = 1.0,
    metric_reader: MetricReader | None = None,
    batch: bool = True,
) -> TracerProvider:
    """Install global tracer and meter providers.

    Sampling is decided once, at the root span, and inherited by
    every child (ParentBased), so a trace is kept or dropped whole.
    Head sampling can't know whether a request will be slow or
    fail; keep 100% here and let a collector do tail sampling
    once volume makes that too expensive.
    """
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(
        resource=resource,
        sampler=ParentBased(TraceIdRatioBased(sample_ratio)),
    )
    processor = (
        BatchSpanProcessor(exporter)
        if batch
        else SimpleSpanProcessor(exporter)
    )
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    if metric_reader is not None:
        metrics.set_meter_provider(
            MeterProvider(
                resource=resource, metric_readers=[metric_reader]
            )
        )
    return provider


# [end:setup]

# [start:redact]
PATTERNS = [
    (re.compile(r"sk-ant-[A-Za-z0-9_\-]+"), "[API_KEY]"),
    (re.compile(r"hk_[A-Za-z0-9_\-]+"), "[API_KEY]"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[EMAIL]"),
    (re.compile(r"\+?\d[\d ()-]{7,}\d"), "[PHONE]"),
]


def redact(text: str, limit: int = 500) -> str:
    """Mask secrets and contact details, and truncate. Used only
    when content capture is switched on; by default no question
    or answer text is recorded at all."""
    for pattern, mask in PATTERNS:
        text = pattern.sub(mask, text)
    return text if len(text) <= limit else text[:limit] + "…"


def pseudonym(user_id: str, salt: str) -> str:
    """A stable, non-reversible user reference. Traces from one
    user can be grouped without the trace store holding who it is.
    Rotating the salt breaks the link to old traces."""
    digest = hashlib.sha256(f"{salt}:{user_id}".encode()).hexdigest()
    return digest[:16]


# [end:redact]

# [start:instruments]
llm_duration = meter.create_histogram(
    "gen_ai.client.operation.duration",
    unit="s",
    description="Duration of model calls",
)
input_tokens = meter.create_counter(
    "gen_ai.client.inference.usage.input_tokens",
    unit="{token}",
    description="Input tokens, including cached tokens",
)
output_tokens = meter.create_counter(
    "gen_ai.client.inference.usage.output_tokens",
    unit="{token}",
    description="Output tokens",
)
llm_cost = meter.create_counter(
    "halden.llm.cost",
    unit="USD",
    description="Estimated model spend at list prices",
)
ask_duration = meter.create_histogram(
    "halden.ask.duration",
    unit="s",
    description="End-to-end /ask latency by outcome",
)
# [end:instruments]
