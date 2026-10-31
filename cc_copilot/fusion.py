"""Late fusion: combine per-modality reviews into one decision, like the original moderation app."""

from cc_copilot.types.results import (
    CallQAResult,
    ChatReviewResult,
    DocumentReviewResult,
    InteractionReport,
    PolicyCheckResult,
)

LOW_QA_SCORE = 3.0


def build_report(
    chat: ChatReviewResult | None = None,
    call: CallQAResult | None = None,
    documents: list[DocumentReviewResult] | None = None,
    policy: PolicyCheckResult | None = None,
) -> InteractionReport:
    blocking: list[str] = []
    warnings: list[str] = []

    if chat:
        if chat.pii_findings:
            types = sorted({f.type.value for f in chat.pii_findings})
            blocking.append(f"Chat exposes personal data: {', '.join(types)} (policy IDV-2)")
        if chat.contains_other_pii:
            warnings.append("Chat may contain names or addresses")
        if chat.missed_escalation:
            blocking.append("Customer asked to escalate and the agent did not (policy ESC-1/ESC-2)")
        if chat.unauthorized_promise:
            blocking.append("Agent made a promise it cannot guarantee (policy REF-2/REF-3)")
        if chat.is_unprofessional:
            warnings.append("Unprofessional wording")
        if chat.is_unfriendly:
            warnings.append("Unfriendly tone")

    if call:
        if call.pii_findings:
            blocking.append("Call transcript contains personal data read out loud (policy IDV-2)")
        if call.is_unprofessional:
            warnings.append("Unprofessional behaviour on the call")
        if call.qa_score < LOW_QA_SCORE:
            warnings.append(f"Low QA score: {call.qa_score:.1f}/5")

    for doc in documents or []:
        if doc.must_be_deleted_after_use:
            warnings.append(f"{doc.document_type} image must be deleted after verification (policy IDV-3)")
        if not doc.is_legible:
            warnings.append(f"{doc.document_type} image is not legible; ask the customer to resend")

    if policy and not policy.is_compliant:
        ids = ", ".join(policy.violated_policy_ids) or "unspecified"
        blocking.append(f"Reply violates policy: {ids}")

    risk = "high" if blocking else "medium" if warnings else "low"
    return InteractionReport(
        risk_level=risk,
        blocking_issues=blocking,
        warnings=warnings,
        qa_score=call.qa_score if call else None,
    )
