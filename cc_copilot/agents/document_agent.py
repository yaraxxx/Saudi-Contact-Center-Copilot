from pydantic_ai import Agent
from pydantic_ai.messages import BinaryContent

from cc_copilot.types.model_choice import ModelChoice
from cc_copilot.types.results import DocumentReviewResult

DOCUMENT_INSTRUCTIONS = """
CONTEXT
Customers of Rimal Telecom, a Saudi operator, send photos and screenshots during support chats:
national ID cards, Iqama cards, bank cards, invoices, or screenshots of errors.

ROLE
You are a data-protection reviewer. You classify the image and decide how it must be handled.

INSTRUCTIONS
- Classify the document type.
- Detect whether it shows personal data: faces, ID or Iqama numbers, card numbers, IBANs, names, addresses.
- Judge whether it is legible enough to process.
- Identity documents and bank cards must be deleted after verification.
Do not transcribe any personal numbers in your rationale; describe them instead (e.g. "a 10-digit ID number").
"""

document_agent = Agent(
    instructions=DOCUMENT_INSTRUCTIONS,
    output_type=DocumentReviewResult,
)


async def review_document(model_choice: ModelChoice, image: bytes, media_type: str) -> DocumentReviewResult:
    result = await document_agent.run(
        ["Review this image sent by a customer.", BinaryContent(data=image, media_type=media_type)],
        model=model_choice.model,
        model_settings=model_choice.model_settings,
    )
    return result.output
