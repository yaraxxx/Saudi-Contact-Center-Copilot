import pytest

from cc_copilot.pii.saudi_patterns import PIIType, detect_pii, is_valid_iban, is_valid_saudi_id, mask_pii

VALID_ID = "1000000008"
VALID_IQAMA = "2000000006"
VALID_IBAN = "SA0380000000608010167519"


def types_in(text: str) -> list[PIIType]:
    return [m.type for m in detect_pii(text)]


def test_saudi_id_checksum():
    assert is_valid_saudi_id(VALID_ID)
    assert is_valid_saudi_id(VALID_IQAMA)
    assert not is_valid_saudi_id("1000000009")
    assert not is_valid_saudi_id("3000000004")


def test_national_id_and_iqama_are_distinguished():
    assert types_in(f"هويتي {VALID_ID}") == [PIIType.NATIONAL_ID]
    assert types_in(f"my iqama is {VALID_IQAMA}") == [PIIType.IQAMA]


def test_order_number_with_bad_checksum_is_ignored():
    assert types_in("رقم الطلب 1234567890") == []


def test_arabic_indic_digits():
    matches = detect_pii("رقم الهوية ١٠٠٠٠٠٠٠٠٨")
    assert [m.type for m in matches] == [PIIType.NATIONAL_ID]
    assert matches[0].value == "١٠٠٠٠٠٠٠٠٨"


def test_iban_with_spaces_and_checksum():
    assert is_valid_iban(VALID_IBAN)
    assert not is_valid_iban("SA0480000000608010167519")
    assert types_in("IBAN: SA03 8000 0000 6080 1016 7519") == [PIIType.IBAN]
    assert types_in("IBAN: sa0480000000608010167519") == []


@pytest.mark.parametrize("phone", ["0551234567", "+966551234567", "00966 55 123 4567", "+966-55-123-4567", "551234567"])
def test_mobile_formats(phone):
    assert types_in(f"جوالي {phone} الله يعافيك") == [PIIType.MOBILE]


def test_landline_is_not_mobile():
    assert types_in("هاتف المكتب 0112345678") == []


def test_mask_replaces_every_item_and_keeps_other_text():
    text = f"هويتي {VALID_ID} وايميلي a.b@example.com وجوالي 0551234567"
    masked, matches = mask_pii(text)
    assert masked == "هويتي [NATIONAL_ID] وايميلي [EMAIL] وجوالي [MOBILE]"
    assert len(matches) == 3


def test_mobile_inside_id_is_not_double_counted():
    assert types_in(f"{VALID_ID}") == [PIIType.NATIONAL_ID]


def test_seed_eval_set_scores_perfectly():
    # Guards against regressions; the eval set itself should keep growing with harder cases.
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evals"))
    from pii_detector_eval import evaluate, load_cases

    results = evaluate(load_cases())
    assert results["failures"] == {}
