import pytest

from cc_copilot.env import POLICY_DIR
from cc_copilot.rag.arabic_text import normalize_arabic, tokenize
from cc_copilot.rag.retriever import BM25Retriever, load_policy_chunks


@pytest.fixture(scope="module")
def retriever() -> BM25Retriever:
    return BM25Retriever(load_policy_chunks(POLICY_DIR))


def test_normalize_unifies_alef_and_removes_diacritics():
    assert normalize_arabic("إِسْتِرْجَاع") == "استرجاع"
    assert normalize_arabic("مكتبة") == "مكتبه"


def test_tokenize_strips_article_and_attached_letters():
    assert tokenize("بالتعويض والمبلغ بتعويض") == ["تعويض", "مبلغ", "تعويض"]


def test_policy_sections_are_loaded_with_ids(retriever):
    ids = {c.policy_id for c in retriever.chunks}
    assert {"REF-1", "REF-2", "REF-3", "IDV-1", "IDV-2", "IDV-3", "ESC-1", "ESC-2", "ESC-3"} <= ids


@pytest.mark.parametrize(
    "query,expected",
    [
        ("أبي أسترجع المبلغ", "REF-1"),  # Najdi
        ("ابغى اكلم المدير", "ESC-1"),  # Hijazi
        ("الموظف وعدني بتعويض ٢٠٠ ريال", "REF-3"),
        ("can you read my IBAN back to me", "IDV-2"),
    ],
)
def test_dialect_queries_find_the_right_policy(retriever, query, expected):
    assert retriever.search(query)[0].chunk.policy_id == expected


def test_irrelevant_query_returns_nothing(retriever):
    assert retriever.search("weather tomorrow in Abha") == []
