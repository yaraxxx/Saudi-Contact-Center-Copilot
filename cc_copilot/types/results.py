from typing import Literal

from pydantic import BaseModel, Field

from cc_copilot.pii.saudi_patterns import PIIMatch, PIIType

Dialect = Literal["najdi", "hijazi", "gulf", "msa", "english", "arabizi", "mixed", "other"]


class PIIFinding(BaseModel):
    """A deterministic (regex + checksum) PII hit. Only the position is returned, never the raw value."""

    type: PIIType
    start: int
    end: int
    turn: int | None = Field(default=None, description="Transcript turn index, for call reviews")

    @classmethod
    def from_match(cls, match: PIIMatch, turn: int | None = None) -> "PIIFinding":
        return cls(type=match.type, start=match.start, end=match.end, turn=turn)


class ReviewResult(BaseModel):
    rationale: str = Field(description="Explanation of the assessment, quoting the phrases that triggered each flag")


# ---------- Chat review ----------


class ChatReviewOutput(ReviewResult):
    """What the LLM returns for a chat transcript."""

    dialect: Dialect = Field(description="Main dialect or language the agent used")
    is_unfriendly: bool = Field(description="Whether the agent's tone is unfriendly")
    is_unprofessional: bool = Field(description="Whether the agent's tone or wording is unprofessional")
    missed_escalation: bool = Field(
        description="Customer asked for a supervisor or complained repeatedly, and the agent did not escalate"
    )
    unauthorized_promise: bool = Field(
        description="Agent promised something it likely cannot guarantee (same-day refund, big compensation)"
    )
    contains_other_pii: bool = Field(
        description="Whether the agent's messages contain unmasked personal data, e.g. full names or home addresses"
    )


class ChatReviewResult(ChatReviewOutput):
    masked_transcript: str = Field(description="Transcript with Saudi PII replaced by placeholders")
    pii_findings: list[PIIFinding] = Field(default_factory=list)


# ---------- Call QA ----------


class TranscriptTurn(BaseModel):
    speaker: Literal["agent", "customer"]
    text: str


class QARubric(BaseModel):
    greeting: int = Field(ge=0, le=5, description="Greets the customer and introduces themselves")
    identity_verification: int = Field(ge=0, le=5, description="Verifies identity before discussing the account")
    empathy: int = Field(ge=0, le=5, description="Acknowledges the customer's frustration")
    resolution: int = Field(ge=0, le=5, description="Resolves the issue or gives clear next steps")
    closing: int = Field(ge=0, le=5, description="Summarises and closes politely")

    @property
    def average(self) -> float:
        scores = [self.greeting, self.identity_verification, self.empathy, self.resolution, self.closing]
        return sum(scores) / len(scores)


class CallQAOutput(ReviewResult):
    transcript: list[TranscriptTurn] = Field(description="Diarised transcript in the original language and dialect")
    dialect: Dialect
    rubric: QARubric
    is_unprofessional: bool = Field(description="Whether the agent was unprofessional or rude at any point")
    customer_sentiment_end: Literal["positive", "neutral", "negative"] = Field(
        description="Customer sentiment at the end of the call"
    )


class CallQAResult(CallQAOutput):
    """Same as the LLM output, except `transcript` is masked before it leaves the service."""

    qa_score: float = Field(description="Average rubric score, 0-5")
    pii_findings: list[PIIFinding] = Field(default_factory=list)


# ---------- Document / screenshot review ----------


class DocumentReviewResult(ReviewResult):
    document_type: Literal["national_id", "iqama", "passport", "bank_card", "invoice", "screenshot", "other"]
    contains_pii: bool = Field(description="Whether the image shows personal data (faces, ID numbers, card numbers)")
    is_legible: bool = Field(description="Whether the document is readable enough to process")
    must_be_deleted_after_use: bool = Field(
        description="Whether policy requires the image to be deleted after verification (true for identity documents)"
    )


# ---------- Policy compliance (RAG) ----------


class PolicyCheckResult(ReviewResult):
    is_compliant: bool = Field(description="Whether the agent's reply follows the retrieved policies")
    violated_policy_ids: list[str] = Field(description="IDs (e.g. REF-2) of policies the reply violates")
    cited_policy_ids: list[str] = Field(description="IDs of the policies the assessment relied on")
    suggested_reply: str = Field(
        description="A compliant rewrite of the agent's reply in the same language and dialect"
    )


# ---------- Combined report ----------


class InteractionReport(BaseModel):
    risk_level: Literal["low", "medium", "high"]
    blocking_issues: list[str]
    warnings: list[str]
    qa_score: float | None = None
