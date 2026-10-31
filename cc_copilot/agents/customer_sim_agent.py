from dataclasses import dataclass

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage

from cc_copilot.types.model_choice import ModelChoice

DIALECT_HINTS = {
    "najdi": 'Najdi (Riyadh): e.g. "أبي"، "وش السالفة"، "الحين"',
    "hijazi": 'Hijazi (Jeddah/Makkah): e.g. "أبغى"، "إيش"، "دحين"',
    "gulf": 'Eastern Province / Gulf: e.g. "أبي"، "شلون"، "زين"',
    "english": "English with occasional Arabic words such as yalla or wallah",
}


@dataclass
class CustomerPersona:
    dialect: str = "najdi"
    issue: str = "My fibre internet has been cutting out every evening for a week, and this is my third complaint."
    mood: str = "frustrated but willing to calm down if treated with respect"


customer_agent = Agent(output_type=str, deps_type=CustomerPersona)


@customer_agent.instructions
def persona_instructions(ctx: RunContext[CustomerPersona]) -> str:
    p = ctx.deps
    return f"""
You are role-playing a customer of Rimal Telecom, a Saudi telecom operator, to train new support agents.
Speak only in {DIALECT_HINTS.get(p.dialect, p.dialect)}. Keep each message short, like a real chat.
Your problem: {p.issue}
Your mood: {p.mood}
If the agent is rude or unhelpful, get angrier and ask for a supervisor. If the agent verifies your identity,
shows empathy and gives a clear plan, calm down and thank them.
Never reveal that you are an AI. Never invent real ID numbers; say you will send them if asked.
"""


async def customer_reply(
    model_choice: ModelChoice,
    persona: CustomerPersona,
    agent_message: str,
    history: list[ModelMessage] | None = None,
) -> tuple[str, list[ModelMessage]]:
    result = await customer_agent.run(
        agent_message,
        deps=persona,
        message_history=history,
        model=model_choice.model,
        model_settings=model_choice.model_settings,
    )
    return result.output, result.all_messages()
