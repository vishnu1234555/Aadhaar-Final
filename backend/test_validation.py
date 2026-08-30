"""
backend/test_validation.py
==========================
Tests for the structural validator and priority arbitration.

These run without the model, without weights and without a GPU — the whole point
of the validation layer is that it is deterministic and independent of the NER
output, so its behaviour is testable on its own.

    python -m pytest backend/test_validation.py -q
    python backend/test_validation.py          # no pytest needed
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from validation import (  # noqa: E402
    arbitrate,
    is_structurally_valid,
    priority,
    validate_and_arbitrate,
    verhoeff_valid,
)


def _aadhaar_with_valid_checksum(base11: str) -> str:
    """Append the check digit that makes `base11` Verhoeff-valid."""
    for d in "0123456789":
        if verhoeff_valid(base11 + d):
            return base11 + d
    raise AssertionError("no valid check digit exists")


VALID_AADHAAR = _aadhaar_with_valid_checksum("23456789012")


def _span(label, text, start, end, conf=90.0, source="model"):
    return {"label": label, "text": text, "start": start, "end": end,
            "confidence": conf, "source": source}


# --------------------------------------------------------------------------


def test_verhoeff_accepts_valid_and_rejects_mutation():
    assert verhoeff_valid(VALID_AADHAAR)
    # Changing any single digit must break the checksum.
    mutated = VALID_AADHAAR[:5] + str((int(VALID_AADHAAR[5]) + 1) % 10) + VALID_AADHAAR[6:]
    assert not verhoeff_valid(mutated)


def test_aadhaar_rejects_bad_checksum():
    assert is_structurally_valid("Aadhaar Number", VALID_AADHAAR)
    # Right length, right leading digit, wrong checksum.
    assert not is_structurally_valid("Aadhaar Number", "234567890123")


def test_aadhaar_rejects_reserved_leading_digits():
    for bad in ("0234 5678 9012", "1234 5678 9012"):
        assert not is_structurally_valid("Aadhaar Number", bad)


def test_aadhaar_accepts_spaced_and_hyphenated_forms():
    d = VALID_AADHAAR
    spaced = f"{d[0:4]} {d[4:8]} {d[8:12]}"
    hyphened = f"{d[0:4]}-{d[4:8]}-{d[8:12]}"
    assert is_structurally_valid("Aadhaar Number", spaced)
    assert is_structurally_valid("Aadhaar Number", hyphened)


def test_pan_holder_type_character():
    assert is_structurally_valid("PAN Number", "ABCPE1234F")      # P = individual
    assert not is_structurally_valid("PAN Number", "ABCZE1234F")  # Z is not a holder type


def test_ifsc_requires_zero_in_fifth_position():
    assert is_structurally_valid("IFSC Code", "HDFC0001234")
    assert not is_structurally_valid("IFSC Code", "HDFC1001234")


def test_vpa_is_not_an_email_address():
    assert is_structurally_valid("VPA", "someone@okhdfcbank")
    assert not is_structurally_valid("VPA", "someone@example.com")


def test_account_number_is_the_weakest_claim():
    assert priority("Account Number") > priority("Aadhaar Number")
    assert priority("Account Number") > priority("PAN Number")


def test_aadhaar_beats_account_number_on_the_same_span():
    """The collision this layer exists to resolve."""
    cands = [
        _span("Account Number", VALID_AADHAAR, 0, 12, conf=99.0),
        _span("Aadhaar Number", VALID_AADHAAR, 0, 12, conf=55.0),
    ]
    kept = arbitrate(cands)
    assert len(kept) == 1
    # Won on structural specificity despite far lower model confidence.
    assert kept[0]["label"] == "Aadhaar Number"


def test_non_overlapping_spans_all_survive():
    cands = [
        _span("PAN Number", "ABCPE1234F", 0, 10),
        _span("IFSC Code", "HDFC0001234", 20, 31),
    ]
    assert len(arbitrate(cands)) == 2


def test_invalid_candidates_are_dropped_with_a_reason():
    cands = [
        _span("Aadhaar Number", "234567890123", 0, 12),   # bad checksum
        _span("PAN Number", "ABCPE1234F", 20, 30),
    ]
    kept, rejected = validate_and_arbitrate(cands)
    assert [e["label"] for e in kept] == ["PAN Number"]
    assert len(rejected) == 1
    assert "structure check" in rejected[0]["reason"]


def test_arbitration_loss_is_reported_separately_from_invalidity():
    cands = [
        _span("Account Number", VALID_AADHAAR, 0, 12),
        _span("Aadhaar Number", VALID_AADHAAR, 0, 12),
    ]
    kept, rejected = validate_and_arbitrate(cands)
    assert len(kept) == 1
    assert len(rejected) == 1
    assert "arbitration" in rejected[0]["reason"]


def test_results_come_back_in_reading_order():
    cands = [
        _span("IFSC Code", "HDFC0001234", 40, 51),
        _span("PAN Number", "ABCPE1234F", 5, 15),
    ]
    assert [e["start"] for e in arbitrate(cands)] == [5, 40]


# --------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {name}  {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed.")
    sys.exit(1 if failed else 0)
