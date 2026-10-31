from pydantic_ai import Agent

from cc_copilot.pii.saudi_patterns import mask_pii
from cc_copilot.types.model_choice import ModelChoice
from cc_copilot.types.results import ChatReviewOutput, ChatReviewResult, PIIFinding

REVIEW_INSTRUCTIONS = """
<context>
Rimal Telecom is a Saudi telecom operator. Customers write in Saudi dialects (Najdi, Hijazi, Gulf),
Modern Standard Arabic, English, or Arabizi (Arabic in Latin letters, e.g. "7ayak allah").
Agents must be courteous, professional and follow company policy in every dialect.
</context>

<role>
You are a quality assurance reviewer for the customer service team.
</role>

<input>
A chat transcript with lines prefixed "Customer:" or "Agent:". Personal data such as national IDs, IBANs
and phone numbers has already been replaced with placeholders like [NATIONAL_ID].
</input>

<instructions>
Review only the Agent's messages and detect whether:
- the tone is unfriendly
- the tone or wording is unprofessional (judge by the norms of the dialect used, not by MSA formality)
- the customer asked for a supervisor or complained repeatedly and the agent did not escalate
- the agent promised something it likely cannot guarantee
- the agent revealed personal data that is not already masked
</instructions>

<output>
Quote the exact phrases that triggered each flag in your rationale.
</output>
"""

chat_review_agent = Agent(
    output_type=ChatReviewOutput,
    instructions=REVIEW_INSTRUCTIONS,
)


async def review_chat(model_choice: ModelChoice, transcript: str) -> ChatReviewResult:
    # Mask first: raw identifiers never reach the model provider.
    masked, matches = mask_pii(transcript)
    result = await chat_review_agent.run(
        masked,
        model=model_choice.model,
        model_settings=model_choice.model_settings,
    )
    return ChatReviewResult(
        **result.output.model_dump(),
        masked_transcript=masked,
        pii_findings=[PIIFinding.from_match(m) for m in matches],
    )
