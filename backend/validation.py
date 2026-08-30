"""
backend/validation.py
=====================
Structural validation and priority arbitration over NER output.

The model recovers nearly every entity but over-assigns labels: the same span
comes back typed several ways at once, because a 12-digit run satisfies the
surface form of both an Aadhaar number and a bank account number. Ranking those
collisions by model confidence does not help — confidence measures how sure the
model is that the span is *an entity*, not which label is correct.

So this module does two things the model cannot:

1. **Validate.** Each label declares the structure its values must have. A
   candidate that fails its own label's check is dropped outright, not surfaced
   with a lower score. Aadhaar goes further than a regex: the last digit is a
   Verhoeff checksum, so a 12-digit string that is not checksum-valid is not an
   Aadhaar number.

2. **Arbitrate.** When surviving candidates overlap in the source text, the more
   structurally specific label wins. A string matching PAN's five-letter /
   four-digit / one-letter shape is a PAN; "any 9-18 digit run" is the weakest
   claim any label can make, so Account Number loses every collision it enters.

Both steps are deterministic and independent of the model, which is what makes
the precision they buy reproducible.
"""

from __future__ import annotations

import re
from typing import Callable, Iterable

# --------------------------------------------------------------------------
# Verhoeff checksum — the algorithm UIDAI uses for the Aadhaar check digit.
# --------------------------------------------------------------------------

_D = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6),
    (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8),
    (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2),
    (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4),
    (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)

_P = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2),
    (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0),
    (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5),
    (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)


def verhoeff_valid(digits: str) -> bool:
    """True if `digits` carries a valid trailing Verhoeff check digit."""
    if not digits.isdigit():
        return False
    checksum = 0
    for i, ch in enumerate(reversed(digits)):
        checksum = _D[checksum][_P[i % 8][int(ch)]]
    return checksum == 0


# --------------------------------------------------------------------------
# Per-label structure
# --------------------------------------------------------------------------

# Used both to find candidates in raw text and to validate model output.
REGEX_PATTERNS: dict[str, str] = {
    "Aadhaar Number": r"\b[2-9]\d{3}[\s\-]?\d{4}[\s\-]?\d{4}\b",
    "PAN Number": r"\b[A-Z]{5}\d{4}[A-Z]\b",
    "IFSC Code": r"\b[A-Z]{4}0[A-Z0-9]{6}\b",
    # A VPA handle is not an email address: the domain half carries no dot.
    # Requiring that keeps "name@example.com" from being reported as a VPA.
    "VPA": r"\b[\w.\-]{2,}@[a-zA-Z][a-zA-Z0-9]{1,}\b(?!\.[a-zA-Z])",
    "Driving Licence": r"\b[A-Z]{2}[\s\-]?\d{2}[\s\-]?\d{4}[\s\-]?\d{7}\b",
    "Account Number": r"\b\d{9,18}\b",
}

# Lower number wins a collision. This encodes structural specificity, not
# importance: a label whose pattern accepts a wider set of strings must lose to
# one whose pattern accepts a narrower set, or the specific label never survives.
LABEL_PRIORITY: dict[str, int] = {
    "PAN Number": 10,
    "IFSC Code": 10,
    "Driving Licence": 20,
    "Aadhaar Number": 30,
    "VPA": 40,
    "Transaction ID": 50,
    "Bank Name": 60,
    "Beneficiary Name": 60,
    "Account Number": 90,  # "any 9-18 digit run" — the weakest possible claim
}

_DEFAULT_PRIORITY = 70

# The 4th character of a PAN encodes holder type. Anything else is malformed
# even though it satisfies the five-letter/four-digit/one-letter shape.
_PAN_HOLDER_TYPES = set("ABCFGHJLPT")


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text)


def _valid_aadhaar(text: str) -> bool:
    d = _digits(text)
    # UIDAI never issues a number beginning 0 or 1.
    return len(d) == 12 and d[0] not in "01" and verhoeff_valid(d)


def _valid_pan(text: str) -> bool:
    t = text.strip().upper()
    if not re.fullmatch(r"[A-Z]{5}\d{4}[A-Z]", t):
        return False
    return t[3] in _PAN_HOLDER_TYPES


def _valid_ifsc(text: str) -> bool:
    return bool(re.fullmatch(r"[A-Z]{4}0[A-Z0-9]{6}", text.strip().upper()))


def _valid_vpa(text: str) -> bool:
    t = text.strip()
    if "@" not in t:
        return False
    handle, _, domain = t.partition("@")
    # A dot in the domain half means this is an email address, not a VPA.
    return bool(handle) and bool(domain) and "." not in domain


def _valid_dl(text: str) -> bool:
    t = re.sub(r"[\s\-]", "", text.strip().upper())
    return bool(re.fullmatch(r"[A-Z]{2}\d{13}", t))


def _valid_account(text: str) -> bool:
    return 9 <= len(_digits(text)) <= 18


# Labels absent from this map carry no structural requirement (free-text names,
# bank names) and are accepted as the model reports them.
VALIDATORS: dict[str, Callable[[str], bool]] = {
    "Aadhaar Number": _valid_aadhaar,
    "PAN Number": _valid_pan,
    "IFSC Code": _valid_ifsc,
    "VPA": _valid_vpa,
    "Driving Licence": _valid_dl,
    "Account Number": _valid_account,
}


def is_structurally_valid(label: str, text: str) -> bool:
    """True if `text` satisfies the structure `label` requires (or has none)."""
    check = VALIDATORS.get(label)
    return True if check is None else check(text)


def priority(label: str) -> int:
    return LABEL_PRIORITY.get(label, _DEFAULT_PRIORITY)


# --------------------------------------------------------------------------
# Arbitration
# --------------------------------------------------------------------------


def _overlaps(a: dict, b: dict) -> bool:
    if a["start"] < 0 or b["start"] < 0:
        # No offsets recovered for one of them; fall back to exact text identity
        # so we never merge two genuinely different spans.
        return a["text"] == b["text"]
    return a["start"] < b["end"] and b["start"] < a["end"]


def _rank(entity: dict) -> tuple:
    # Most specific label first, then the longer span, then model confidence.
    # Confidence is the last resort precisely because it is the signal that was
    # misleading us: the model is confident about spans, not about labels.
    return (
        priority(entity["label"]),
        -(entity["end"] - entity["start"]),
        -float(entity.get("confidence") or 0.0),
    )


def arbitrate(candidates: Iterable[dict]) -> list[dict]:
    """Resolve overlapping candidates, keeping the most specific label for each span."""
    ordered = sorted(candidates, key=_rank)
    kept: list[dict] = []
    for cand in ordered:
        if not any(_overlaps(cand, k) for k in kept):
            kept.append(cand)
    # Present in reading order rather than arbitration order.
    return sorted(kept, key=lambda e: (e["start"] if e["start"] >= 0 else 1 << 30))


def validate_and_arbitrate(candidates: Iterable[dict]) -> tuple[list[dict], list[dict]]:
    """Drop structurally invalid candidates, then resolve overlaps.

    Returns (kept, rejected). Rejected entries carry a `reason` so the caller can
    report why a span the model proposed did not survive.
    """
    survivors: list[dict] = []
    rejected: list[dict] = []

    for cand in candidates:
        if is_structurally_valid(cand["label"], cand["text"]):
            survivors.append(cand)
        else:
            rejected.append({**cand, "reason": f"failed {cand['label']} structure check"})

    kept = arbitrate(survivors)
    kept_ids = {id(e) for e in kept}
    for cand in survivors:
        if id(cand) not in kept_ids:
            rejected.append({**cand, "reason": "lost span arbitration to a more specific label"})

    return kept, rejected
