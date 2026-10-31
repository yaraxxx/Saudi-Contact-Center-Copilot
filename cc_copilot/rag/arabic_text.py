"""Light Arabic normalisation so dialect spellings match policy text."""

import re

_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
_TATWEEL = "ـ"
_CHAR_MAP = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه", "ؤ": "و", "ئ": "ي"})
_TOKEN = re.compile(r"\w+")

STOPWORDS = {
    # Arabic (normalised forms)
    "في",
    "من",
    "علي",
    "الي",
    "عن",
    "ان",
    "او",
    "و",
    "ما",
    "لا",
    "هل",
    "هذا",
    "هذه",
    "ذلك",
    "مع",
    "كل",
    "قد",
    "تم",
    # English
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "for",
    "is",
    "are",
    "be",
    "with",
    "my",
    "i",
    "it",
    "can",
}


def normalize_arabic(text: str) -> str:
    text = _DIACRITICS.sub("", text).replace(_TATWEEL, "")
    return text.translate(_CHAR_MAP).lower()


def _strip_prefix(token: str) -> str:
    # Drop the definite article and common attached conjunctions/prepositions (وال، بال، لل، ال).
    for prefix in ("وال", "بال", "فال", "كال", "لل", "ال"):
        if token.startswith(prefix) and len(token) - len(prefix) >= 3:
            return token[len(prefix) :]
    # A single attached letter (بتعويض -> تعويض); only on longer words to avoid eating roots.
    if token[0] in "وبلفك" and len(token) >= 5:
        return token[1:]
    return token


def tokenize(text: str) -> list[str]:
    tokens = (_strip_prefix(t) for t in _TOKEN.findall(normalize_arabic(text)))
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]
