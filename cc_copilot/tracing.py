from openinference.instrumentation.pydantic_ai import OpenInferenceSpanProcessor
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter
from pydantic_ai import Agent

from cc_copilot.env import PHOENIX_URL


def setup_tracing(exporter: SpanExporter | None = None) -> TracerProvider:
    """
    Send every agent call to Phoenix (or to `exporter`, e.g. an in-memory one in tests).

    Chat and policy prompts are PII-masked before the agent runs, so their traces are too.
    Call audio and document images are sent as-is, so those traces can contain personal data.
    """
    tracer_provider = TracerProvider()
    trace.set_tracer_provider(tracer_provider)
    tracer_provider.add_span_processor(OpenInferenceSpanProcessor())
    exporter = exporter or OTLPSpanExporter(endpoint=f"{PHOENIX_URL}/v1/traces")
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    # pydantic-ai creates no spans at all until instrumentation is switched on.
    Agent.instrument_all()
    return tracer_provider
