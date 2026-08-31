"""Unit tests for the rule-based baseline, one behavior per test."""

from extraction import baseline


def _status_of(conditions, name):
    for c in conditions:
        if c.name == name:
            return c.status
    return None


def test_negation_is_detected():
    conditions = baseline.extract_conditions("Denies chest pain.")
    assert _status_of(conditions, "chest pain") == "negated"


def test_affirmed_condition_is_active():
    conditions = baseline.extract_conditions("Reports chest pain over the past week.")
    assert _status_of(conditions, "chest pain") == "active"


def test_history_of_is_historical():
    conditions = baseline.extract_conditions("History of pneumonia in 2015, resolved.")
    assert _status_of(conditions, "pneumonia") == "historical"


def test_abbreviation_maps_to_canonical():
    conditions = baseline.extract_conditions("Past Medical History: HTN; T2DM.")
    assert _status_of(conditions, "hypertension") == "active"
    assert _status_of(conditions, "type 2 diabetes") == "active"


def test_family_history_is_excluded():
    conditions = baseline.extract_conditions("Family History: Mother with type 2 diabetes.")
    assert conditions == []


def test_short_abbreviation_needs_word_boundary():
    """'AF' must not be found inside the word 'after'."""
    conditions = baseline.extract_conditions("Feeling well after the visit.")
    assert _status_of(conditions, "atrial fibrillation") is None


def test_medication_list_parsing():
    text = "Medications:\n- lisinopril 20 mg daily\n- metformin 500mg BID\n"
    meds = {m.name: m for m in baseline.extract_medications(text)}
    assert meds["lisinopril"].dose == 20.0
    assert meds["lisinopril"].unit == "mg"
    assert meds["metformin"].dose == 500.0


def test_narrative_medications():
    text = "Clinic note. Current medications are amlodipine 10 mg daily; sertraline 50 mg daily."
    meds = {m.name for m in baseline.extract_medications(text)}
    assert meds == {"amlodipine", "sertraline"}


def test_discontinued_medication_is_missed_by_design():
    """Documents the known limitation: prose discontinuations are invisible
    to the rules. If this test ever fails, the baseline got smarter and the
    README claims about it need updating."""
    text = "We discontinued metformin 500 mg today because of GI upset."
    assert baseline.extract_medications(text) == []


def test_smoking_variants():
    assert baseline.extract_smoking("Denies any tobacco use.") == "never"
    assert baseline.extract_smoking("Former smoker, quit in 2015.") == "former"
    assert baseline.extract_smoking("Smokes one pack per day.") == "current"
    assert baseline.extract_smoking("Lives with family.") == "unknown"


def test_follow_up_unit_conversion():
    assert baseline.extract_follow_up("Follow up in 14 days.") == 14
    assert baseline.extract_follow_up("Return to clinic in 2 weeks.") == 14
    assert baseline.extract_follow_up("Return in 3 months for routine follow-up.") == 90
    assert baseline.extract_follow_up("RTC in 6 weeks.") == 42
    assert baseline.extract_follow_up("No plan stated.") is None


def test_a1c():
    assert baseline.extract_a1c("Most recent A1c 8.2%.") == 8.2
    assert baseline.extract_a1c("No labs today.") is None
