"""Tests for the synthetic note generator.

The property that makes the whole project trustworthy is checked here:
everything in the gold labels must actually appear in the note text.
"""

import random

from notesynth import vocab
from notesynth.generator import generate_corpus, generate_note


def test_same_seed_same_corpus():
    notes_a, golds_a = generate_corpus(20, seed=7)
    notes_b, golds_b = generate_corpus(20, seed=7)
    assert notes_a == notes_b
    assert golds_a == golds_b


def test_different_seeds_differ():
    notes_a, _ = generate_corpus(5, seed=1)
    notes_b, _ = generate_corpus(5, seed=2)
    assert notes_a != notes_b


def _all_surface_forms(canonical):
    """Every string that could represent this entity in the text. The
    normalization map already knows every surface -> canonical pair (that is
    what evaluation uses), so invert it here rather than rebuilding it."""
    forms = [
        surface
        for surface, canon in vocab.build_normalization_map().items()
        if canon == canonical
    ]
    forms.append(canonical)
    if canonical in vocab.MISSPELLINGS:
        forms.append(vocab.MISSPELLINGS[canonical])
    return forms


def test_gold_entities_appear_in_text():
    """Every gold condition and medication must be findable in the note."""
    rng = random.Random(0)
    for i in range(200):
        text, gold = generate_note(f"n{i}", rng)
        low = text.lower()
        for condition in gold["conditions"]:
            assert any(
                form.lower() in low
                for form in _all_surface_forms(condition["name"])
            ), f"{condition['name']} not found in note:\n{text}"
        for med in gold["medications"]:
            name = med["name"]
            forms = [name, vocab.MISSPELLINGS.get(name, name)]
            assert any(f.lower() in low for f in forms), (
                f"{name} not found in note:\n{text}"
            )


def test_gold_statuses_are_valid():
    _, golds = generate_corpus(50, seed=3)
    for gold in golds:
        for c in gold["conditions"]:
            assert c["status"] in {"active", "historical", "negated", "suspected"}
        for m in gold["medications"]:
            assert m["status"] in {"active", "discontinued"}
        assert gold["smoking_status"] in {"current", "former", "never", "unknown"}
        assert gold["follow_up_days"] > 0


def test_no_duplicate_condition_names_in_gold():
    """Each condition name should appear once per note, so evaluation can
    match by name without ambiguity."""
    _, golds = generate_corpus(100, seed=4)
    for gold in golds:
        names = [c["name"] for c in gold["conditions"]]
        assert len(names) == len(set(names)), names


def test_readmission_rate_is_reasonable():
    _, golds = generate_corpus(300, seed=5)
    rate = sum(g["readmitted_30d"] for g in golds) / len(golds)
    assert 0.10 < rate < 0.60
