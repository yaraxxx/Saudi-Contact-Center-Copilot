from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from cc_copilot.agents.chat_review_agent import review_chat
from cc_copilot.tracing import setup_tracing
from cc_copilot.types.model_choice import ModelChoice


async def test_agent_calls_are_traced_without_raw_pii():
    exporter = InMemorySpanExporter()
    setup_tracing(exporter)
    try:
        await review_chat(ModelChoice(model=TestModel()), "Agent: أبشر، رقم هويتك 1000000008 صح؟")
    finally:
        Agent.instrument_all(False)

    spans = exporter.get_finished_spans()
    kinds = {s.attributes.get("openinference.span.kind") for s in spans}
    assert {"AGENT", "LLM"} <= kinds
    assert not any("1000000008" in str(v) for s in spans for v in s.attributes.values())
