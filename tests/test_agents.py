from pydantic_ai import capture_run_messages
from pydantic_ai.models.test import TestModel

from cc_copilot.agents.chat_review_agent import review_chat
from cc_copilot.agents.policy_agent import check_policy
from cc_copilot.env import POLICY_DIR
from cc_copilot.pii.saudi_patterns import PIIType
from cc_copilot.rag.retriever import BM25Retriever, load_policy_chunks
from cc_copilot.types.model_choice import ModelChoice

TRANSCRIPT = "Customer: ابغى اتأكد من حسابي\nAgent: أبشر، رقم هويتك 1000000008 صح؟"


async def test_chat_review_masks_pii_before_calling_the_model():
    with capture_run_messages() as messages:
        result = await review_chat(ModelChoice(model=TestModel()), TRANSCRIPT)

    sent_to_model = str(messages[0])
    assert "1000000008" not in sent_to_model
    assert "[NATIONAL_ID]" in sent_to_model
    assert [f.type for f in result.pii_findings] == [PIIType.NATIONAL_ID]
    assert "1000000008" not in result.model_dump_json()


async def test_policy_check_drops_citations_that_were_not_retrieved():
    retriever = BM25Retriever(load_policy_chunks(POLICY_DIR))
    model = TestModel(
        custom_output_args={
            "rationale": "Promised a same-day refund.",
            "is_compliant": False,
            "violated_policy_ids": ["REF-2", "FAKE-9"],
            "cited_policy_ids": ["REF-2", "REF-1", "FAKE-9"],
            "suggested_reply": "أبشر، المبلغ يرجع لبطاقتك خلال ٥ إلى ١٠ أيام عمل.",
        }
    )
    result = await check_policy(
        ModelChoice(model=model), retriever, "أبي أسترجع المبلغ", "ولا يهمك، برجع لك المبلغ اليوم"
    )
    assert result.violated_policy_ids == ["REF-2"]
    assert "FAKE-9" not in result.cited_policy_ids
