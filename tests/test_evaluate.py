"""Tests for the scorer, using tiny hand-built cases where the right
precision/recall/F1 can be worked out on paper."""

from extraction.evaluate import evaluate, normalize_name
from extraction.schema import Condition, Medication, NoteExtraction


def _gold(note_id="n0", **overrides):
    gold = {
        "note_id": note_id,
        "age": 70,
        "sex": "F",
        "conditions": [
            {"name": "hypertension", "status": "active"},
            {"name": "chest pain", "status": "negated"},
        ],
        "medications": [
            {"name": "lisinopril", "dose": 20, "unit": "mg", "status": "active"},
        ],
        "smoking_status": "never",
        "follow_up_days": 30,
        "a1c": None,
        "phenomena": ["negation"],
        "readmitted_30d": False,
    }
    gold.update(overrides)
    return gold


def test_normalize_name():
    assert normalize_name("HTN") == "hypertension"
    assert normalize_name("Hypertension") == "hypertension"
    assert normalize_name("T2DM") == "type 2 diabetes"
    assert normalize_name("something new") == "something new"


def test_perfect_extraction_scores_one():
    pred = NoteExtraction(
        conditions=[
            Condition(name="hypertension", status="active"),
            Condition(name="chest pain", status="negated"),
        ],
        medications=[
            Medication(name="lisinopril", dose=20, unit="mg", status="active"),
        ],
        smoking_status="never",
        follow_up_days=30,
        a1c=None,
    )
    m = evaluate([_gold()], {"n0": pred})
    assert m["conditions"]["f1"] == 1.0
    assert m["conditions"]["status_accuracy"] == 1.0
    assert m["medications"]["f1"] == 1.0
    assert m["medications"]["dose_accuracy"] == 1.0
    assert m["smoking_accuracy"] == 1.0
    assert m["follow_up_accuracy"] == 1.0
    assert m["recall_by_phenomenon"]["negation"]["recall"] == 1.0


def test_abbreviated_prediction_still_matches():
    """A prediction of 'HTN' should count as finding 'hypertension'."""
    pred = NoteExtraction(
        conditions=[Condition(name="HTN", status="active")],
        medications=[],
        smoking_status="unknown",
        follow_up_days=None,
        a1c=None,
    )
    m = evaluate([_gold()], {"n0": pred})
    assert m["conditions"]["tp"] == 1  # hypertension matched via HTN
    assert m["conditions"]["fn"] == 1  # chest pain missed
    assert m["conditions"]["fp"] == 0


def test_false_positive_counted():
    pred = NoteExtraction(
        conditions=[
            Condition(name="hypertension", status="active"),
            Condition(name="chest pain", status="negated"),
            Condition(name="gout", status="active"),  # not in the gold
        ],
        medications=[],
        smoking_status="never",
        follow_up_days=30,
        a1c=None,
    )
    m = evaluate([_gold()], {"n0": pred})
    assert m["conditions"]["fp"] == 1
    # precision 2/3, recall 2/2
    assert m["conditions"]["precision"] == round(2 / 3, 3)
    assert m["conditions"]["recall"] == 1.0


def test_wrong_status_hits_name_but_not_status():
    """Finding the entity with the wrong status: counts as a name match,
    but drags down status accuracy and phenomenon recall."""
    pred = NoteExtraction(
        conditions=[
            Condition(name="hypertension", status="active"),
            Condition(name="chest pain", status="active"),  # should be negated
        ],
        medications=[],
        smoking_status="never",
        follow_up_days=30,
        a1c=None,
    )
    m = evaluate([_gold()], {"n0": pred})
    assert m["conditions"]["tp"] == 2
    assert m["conditions"]["status_accuracy"] == 0.5
    assert m["recall_by_phenomenon"]["negation"]["recall"] == 0.0


def test_missing_note_counts_as_all_false_negatives():
    m = evaluate([_gold()], {})
    assert m["conditions"]["fn"] == 2
    assert m["conditions"]["recall"] == 0.0
    assert m["smoking_accuracy"] == 0.0
