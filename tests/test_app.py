from app import (
    HEADINGS,
    contains_high_risk_content,
    deterministic_fallback,
    extract_source_facts,
    normalize_sections,
)


def test_extract_source_facts():
    text = "Confirm by October 8 at 3 PM. Contact training@example.org. Price: $25."
    facts = extract_source_facts(text)
    assert "October 8" in facts
    assert "3 PM" in facts
    assert "training@example.org" in facts
    assert "$25" in facts


def test_appointment_fallback_has_all_sections():
    text = (
        "Your appointment scheduled for October 10 at 3 PM has been moved to "
        "October 12 at 2 PM. Please confirm the new time by October 8."
    )
    out = deterministic_fallback(text)
    assert out is not None
    for heading in HEADINGS:
        assert heading in out
    assert "October 12 at 2 PM" in out
    assert "October 8" in out


def test_price_change_fallback_grounded():
    text = "Your subscription price will change from $10 to $12 on October 20."
    out = deterministic_fallback(text)
    assert out is not None
    assert "$10" in out
    assert "$12" in out
    assert "October 20" in out


def test_deadline_fallback():
    text = "Please submit the form by October 15 at 5 PM."
    out = deterministic_fallback(text)
    assert out is not None
    assert "October 15" in out
    assert "5 PM" in out


def test_high_risk_is_blocked():
    assert contains_high_risk_content("Here is my password: example")
    assert contains_high_risk_content("Please give me legal advice")
    assert not contains_high_risk_content("Our appointment moved to Friday")


def test_normalize_inserts_missing_headings():
    raw = "WHAT DOES THIS MEAN?\nA simple notice."
    out = normalize_sections(raw, "Deadline October 8")
    for heading in HEADINGS:
        assert heading in out
    assert "October 8" in out
