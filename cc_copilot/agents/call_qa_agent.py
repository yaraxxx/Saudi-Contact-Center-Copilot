from pydantic_ai import Agent
from pydantic_ai.messages import BinaryContent

from cc_copilot.pii.saudi_patterns import mask_pii
from cc_copilot.types.model_choice import ModelChoice
from cc_copilot.types.results import CallQAOutput, CallQAResult, PIIFinding, TranscriptTurn

QA_INSTRUCTIONS = """
CONTEXT
Rimal Telecom is a Saudi telecom operator. Calls are usually in Saudi dialects, often mixed with English.

ROLE
You are a call-center quality assurance reviewer.

TASK
1. Transcribe the call in the original language and dialect. Do not translate. Write numbers as digits.
   Label each turn as "agent" or "customer".
2. Identify the main dialect.
3. Score the agent from 0 to 5 on: greeting, identity verification, empathy, resolution, closing.
4. Flag unprofessional behaviour and record the customer's sentiment at the end of the call.

OUTPUT
Return the transcript, scores, flags and a rationale that quotes the turns behind each score.
"""

call_qa_agent = Agent(
    instructions=QA_INSTRUCTIONS,
    output_type=CallQAOutput,
)


async def review_call(model_choice: ModelChoice, audio: bytes, media_type: str) -> CallQAResult:
    # Audio cannot be masked before transcription, so the provider sees it. Mask the transcript
    # before it is stored or shown, and see the README section on data residency.
    result = await call_qa_agent.run(
        ["Review this customer service call.", BinaryContent(data=audio, media_type=media_type)],
        model=model_choice.model,
        model_settings=model_choice.model_settings,
    )
    output = result.output

    masked_turns: list[TranscriptTurn] = []
    findings: list[PIIFinding] = []
    for i, turn in enumerate(output.transcript):
        masked_text, matches = mask_pii(turn.text)
        masked_turns.append(TranscriptTurn(speaker=turn.speaker, text=masked_text))
        findings.extend(PIIFinding.from_match(m, turn=i) for m in matches)

    return CallQAResult(
        **output.model_dump(exclude={"transcript"}),
        transcript=masked_turns,
        qa_score=output.rubric.average,
        pii_findings=findings,
    )
