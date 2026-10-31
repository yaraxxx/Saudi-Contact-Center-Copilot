"""
Deterministic detection and masking of Saudi personal data.

Runs before anything is sent to an LLM, so national IDs, IBANs and phone numbers
never leave the service in clear text (a PDPL data-minimisation measure).
Regexes find candidates; checksums reject look-alikes such as order numbers.
"""

import re
from dataclasses import dataclass
from enum import StrEnum

# Arabic-Indic (U+0660..) and Extended Arabic-Indic (U+06F0..) digits -> ASCII.
# One-to-one character mapping keeps string offsets identical to the original text.
_DIGIT_MAP = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


class PIIType(StrEnum):
    NATIONAL_ID = "NATIONAL_ID"  # Saudi citizen ID, starts with 1
    IQAMA = "IQAMA"  # Resident ID, starts with 2
    IBAN = "IBAN"
    MOBILE = "MOBILE"
    EMAIL = "EMAIL"


@dataclass(frozen=True)
class PIIMatch:
    type: PIIType
    start: int
    end: int
    value: str


def normalize_digits(text: str) -> str:
    return text.translate(_DIGIT_MAP)


def is_valid_saudi_id(number: str) -> bool:
    """Luhn checksum used by Saudi national ID and Iqama numbers."""
    if len(number) != 10 or not number.isdigit() or number[0] not in "12":
        return False
    total = 0
    for i, ch in enumerate(number):
        d = int(ch)
        if i % 2 == 0:
            d *= 2
            total += d // 10 + d % 10
        else:
            total += d
    return total % 10 == 0


def is_valid_iban(iban: str) -> bool:
    """ISO 13616 mod-97 check."""
    compact = re.sub(r"\s", "", iban).upper()
    if len(compact) != 24 or not compact.startswith("SA"):
        return False
    rearranged = compact[4:] + compact[:4]
    numeric = "".join(str(int(ch, 36)) for ch in rearranged)
    return int(numeric) % 97 == 1


# Order matters: earlier detectors win when spans overlap.
_PATTERNS: list[tuple[PIIType | None, re.Pattern[str]]] = [
    (PIIType.EMAIL, re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    (PIIType.IBAN, re.compile(r"\bSA\d{2}(?:\s?[0-9A-Z]{4}){5}\b", re.IGNORECASE)),
    # National ID or Iqama, decided by the first digit.
    (None, re.compile(r"(?<!\d)[12]\d{9}(?!\d)")),
    (PIIType.MOBILE, re.compile(r"(?<![\d+])(?:(?:\+|00)966[\s-]?|0)?5\d(?:[\s-]?\d){7}(?!\d)")),
]


def detect_pii(text: str) -> list[PIIMatch]:
    normalized = normalize_digits(text)
    matches: list[PIIMatch] = []
    taken: list[tuple[int, int]] = []

    for pii_type, pattern in _PATTERNS:
        for m in pattern.finditer(normalized):
            start, end = m.span()
            if any(start < t_end and t_start < end for t_start, t_end in taken):
                continue
            candidate = m.group()
            if pii_type is PIIType.IBAN and not is_valid_iban(candidate):
                continue
            if pii_type is None:
                if not is_valid_saudi_id(candidate):
                    continue
                pii_type_resolved = PIIType.NATIONAL_ID if candidate[0] == "1" else PIIType.IQAMA
            else:
                pii_type_resolved = pii_type
            matches.append(PIIMatch(pii_type_resolved, start, end, text[start:end]))
            taken.append((start, end))

    return sorted(matches, key=lambda m: m.start)


def mask_pii(text: str) -> tuple[str, list[PIIMatch]]:
    """Replace every detected item with a placeholder such as [NATIONAL_ID]."""
    matches = detect_pii(text)
    masked = text
    for m in reversed(matches):
        masked = masked[: m.start] + f"[{m.type}]" + masked[m.end :]
    return masked, matches
