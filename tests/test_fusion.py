from cc_copilot.fusion import build_report
from cc_copilot.pii.saudi_patterns import PIIType
from cc_copilot.types.results import ChatReviewResult, PIIFinding, PolicyCheckResult


def make_chat(**overrides) -> ChatReviewResult:
    fields = dict(
        rationale="ok",
        dialect="najdi",
        is_unfriendly=False,
        is_unprofessional=False,
        missed_escalation=False,
        unauthorized_promise=False,
        contains_other_pii=False,
        masked_transcript="",
        pii_findings=[],
    )
    return ChatReviewResult(**(fields | overrides))


def test_clean_chat_is_low_risk():
    assert build_report(chat=make_chat()).risk_level == "low"


def test_tone_issue_is_a_warning_not_a_block():
    report = build_report(chat=make_chat(is_unfriendly=True))
    assert report.risk_level == "medium"
    assert report.blocking_issues == []


def test_pii_and_missed_escalation_block():
    chat = make_chat(missed_escalation=True, pii_findings=[PIIFinding(type=PIIType.IBAN, start=0, end=24)])
    report = build_report(chat=chat)
    assert report.risk_level == "high"
    assert len(report.blocking_issues) == 2


def test_policy_violation_blocks():
    policy = PolicyCheckResult(
        rationale="x", is_compliant=False, violated_policy_ids=["REF-3"], cited_policy_ids=["REF-3"], suggested_reply=""
    )
    report = build_report(policy=policy)
    assert report.risk_level == "high"
    assert "REF-3" in report.blocking_issues[0]
