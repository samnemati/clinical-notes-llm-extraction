"""Seeded generator of synthetic outpatient progress notes with gold labels.

How it works, in one paragraph: for each note we randomly draw a patient
(age, sex), a few chronic conditions, medications that plausibly treat those
conditions, some symptoms that are affirmed or denied, a smoking status, a
lab value, and a follow-up plan. We then render all of that into note text
(in one of two formats), and we ALSO write the same facts into a "gold"
dictionary. Because the text and the labels come from the same draws, the
ground truth is exact by construction; nobody has to hand-label anything.
No real patient data is involved at any point.

Everything uses one seeded random number generator, so the same seed always
produces the same corpus.
"""

import json
import random

from . import vocab


def _surface(rng, canonical, surfaces, allow_misspelling, phenomena):
    """Pick how a condition is written in the text: its full name, an
    abbreviation, or (rarely) a misspelling. Records which trick was used
    in `phenomena`, so evaluation can later ask "how does the extractor do
    on notes containing abbreviations?"."""
    form = rng.choice(surfaces)
    if form != surfaces[0]:
        # surfaces[0] is always the full name; anything else is an
        # abbreviation or variant.
        phenomena.add("abbreviation")
    if allow_misspelling and canonical in vocab.MISSPELLINGS and rng.random() < 0.5:
        phenomena.add("misspelling")
        return vocab.MISSPELLINGS[canonical]
    return form


def generate_note(note_id, rng):
    """Build one synthetic note. Returns (text, gold) where `text` is the
    note a human or model would read, and `gold` is the dictionary of
    correct answers for that note."""
    phenomena = set()  # which "hard" language features this note contains
    age = rng.randint(35, 88)
    sex = rng.choice(["M", "F"])
    # In about 15% of notes we allow words to be misspelled, the way real
    # notes typed in a hurry are.
    misspell_note = rng.random() < 0.15

    # Chronic conditions (active) --------------------------------------------
    n_cond = rng.randint(2, 5)
    active = rng.sample(list(vocab.CONDITIONS), n_cond)
    conditions = [{"name": c, "status": "active"} for c in active]

    pmh_lines = []
    for c in active:
        pmh_lines.append(
            _surface(rng, c, vocab.CONDITIONS[c], misspell_note, phenomena)
        )

    # Historical items --------------------------------------------------------
    if rng.random() < 0.45:
        h = rng.choice([k for k in vocab.HISTORICAL if k not in active])
        year = rng.randint(2005, 2020)
        pmh_lines.append(vocab.HISTORICAL[h].format(year=year).rstrip("."))
        conditions.append({"name": h, "status": "historical"})
        phenomena.add("historical")

    # Medications -------------------------------------------------------------
    # Only draw medications that treat one of this patient's conditions (or
    # general-purpose ones), so the notes stay clinically coherent.
    medications = []
    med_pool = [
        m for m, (_, _, _, treats) in vocab.MEDICATIONS.items()
        if treats is None or treats in active
    ]
    rng.shuffle(med_pool)
    chosen = med_pool[: rng.randint(2, min(6, len(med_pool)))]

    # Sometimes one medication is stopped during the visit. It then appears
    # ONLY in a plan sentence ("We discontinued X today because of Y."),
    # never in the medication list. This is deliberately hard for the
    # rule-based baseline, which only reads medication lists.
    discontinued_line = None
    candidates = [m for m in chosen if m in vocab.DISCONTINUATION_REASONS]
    if candidates and rng.random() < 0.5:
        stopped = rng.choice(candidates)
        chosen.remove(stopped)
        doses, unit, _, _ = vocab.MEDICATIONS[stopped]
        dose = rng.choice(doses)
        name_in_text = stopped
        if misspell_note and stopped in vocab.MISSPELLINGS and rng.random() < 0.5:
            name_in_text = vocab.MISSPELLINGS[stopped]
            phenomena.add("misspelling")
        discontinued_line = (
            f"We discontinued {name_in_text} {dose} {unit} today because of "
            f"{vocab.DISCONTINUATION_REASONS[stopped]}."
        )
        medications.append(
            {"name": stopped, "dose": dose, "unit": unit, "status": "discontinued"}
        )
        phenomena.add("discontinued_med")

    med_lines = []
    for m in chosen:
        doses, unit, freqs, _ = vocab.MEDICATIONS[m]
        dose = rng.choice(doses)
        freq = rng.choice(freqs)
        name_in_text = m
        if misspell_note and m in vocab.MISSPELLINGS and rng.random() < 0.5:
            name_in_text = vocab.MISSPELLINGS[m]
            phenomena.add("misspelling")
        # Vary the formatting: "500mg" vs "500 mg", the way real lists do.
        if rng.random() < 0.3:
            med_lines.append(f"{name_in_text} {dose}{unit} {freq}")
        else:
            med_lines.append(f"{name_in_text} {dose} {unit} {freq}")
        medications.append({"name": m, "dose": dose, "unit": unit, "status": "active"})

    # HPI: affirmed and negated findings --------------------------------------
    hpi = [f"{age} year old {'man' if sex == 'M' else 'woman'} presenting for follow-up."]
    findings_pool = [
        f for f in vocab.FINDINGS if f not in {c["name"] for c in conditions}
    ]
    rng.shuffle(findings_pool)
    n_aff = rng.randint(0, 2)
    n_neg = rng.randint(1, 3)
    for f in findings_pool[:n_aff]:
        surf = _surface(rng, f, vocab.FINDINGS[f], False, phenomena)
        hpi.append(rng.choice(vocab.AFFIRMED_TEMPLATES).format(x=surf))
        conditions.append({"name": f, "status": "active"})
    negated = findings_pool[n_aff : n_aff + n_neg]
    if negated:
        phenomena.add("negation")
    for f in negated:
        surf = _surface(rng, f, vocab.FINDINGS[f], False, phenomena)
        hpi.append(rng.choice(vocab.NEGATION_TEMPLATES).format(x=surf))
        conditions.append({"name": f, "status": "negated"})

    # Hedged new diagnosis ----------------------------------------------------
    plan_extra = []
    if rng.random() < 0.35:
        options = [
            k for k in vocab.HEDGED
            if k not in {c["name"] for c in conditions}
        ]
        if options:
            hedge = rng.choice(options)
            plan_extra.append(vocab.HEDGED[hedge])
            conditions.append({"name": hedge, "status": "suspected"})
            phenomena.add("hedged_dx")

    # Social history ----------------------------------------------------------
    smoking = rng.choice(["current", "former", "never", "unknown"])
    if smoking == "unknown":
        social = "Lives with family." if rng.random() < 0.5 else "Retired, lives alone."
        phenomena.add("no_smoking_mention")
    else:
        social = rng.choice(vocab.SMOKING_PHRASES[smoking]).format(
            year=rng.randint(2005, 2020), n=rng.randint(2, 20)
        )

    # Labs --------------------------------------------------------------------
    a1c = None
    labs = None
    if "type 2 diabetes" in active or rng.random() < 0.2:
        a1c = round(rng.uniform(5.4, 11.5), 1)
        labs = f"Most recent A1c {a1c}%."

    # Follow-up ---------------------------------------------------------------
    unit_choice = rng.choice(["days", "weeks", "months"])
    if unit_choice == "days":
        n = rng.choice([7, 10, 14, 30])
        follow_up_days = n
        fu_line = f"Follow up in {n} days."
    elif unit_choice == "weeks":
        n = rng.choice([1, 2, 4, 6, 8])
        follow_up_days = n * 7
        fu_line = rng.choice(
            [f"Return to clinic in {n} week{'s' if n > 1 else ''}.",
             f"RTC in {n} week{'s' if n > 1 else ''}."]
        )
    else:
        n = rng.choice([1, 2, 3, 6])
        follow_up_days = n * 30
        fu_line = f"Return in {n} month{'s' if n > 1 else ''} for routine follow-up."

    fam = rng.choice(vocab.FAMILY_HISTORY)
    phenomena.add("family_history_distractor")
    allergy = rng.choice(vocab.ALLERGIES)

    # Assemble ----------------------------------------------------------------
    plan_lines = plan_extra + ([discontinued_line] if discontinued_line else []) + [fu_line]
    if rng.random() < 0.5:
        text = "\n".join(
            [
                "PROGRESS NOTE",
                "",
                "Subjective: " + " ".join(hpi),
                "",
                "Past Medical History: " + "; ".join(pmh_lines) + ".",
                "Family History: " + fam,
                "Allergies: " + allergy + ".",
                "Social History: " + social,
                "",
                "Medications:",
            ]
            + [f"- {line}" for line in med_lines]
            + ([""] if labs else [])
            + ([f"Labs: {labs}"] if labs else [])
            + ["", "Assessment and Plan: " + " ".join(plan_lines)]
        )
    else:
        body = (
            " ".join(hpi)
            + " Past medical history includes "
            + ", ".join(pmh_lines)
            + f". Family history: {fam} Allergies: {allergy}. {social} "
            + "Current medications are "
            + "; ".join(med_lines)
            + ". "
            + (f"{labs} " if labs else "")
            + " ".join(plan_lines)
        )
        text = f"Clinic note. {body}"

    gold = {
        "note_id": note_id,
        "age": age,
        "sex": sex,
        "conditions": sorted(conditions, key=lambda c: c["name"]),
        "medications": sorted(medications, key=lambda m: m["name"]),
        "smoking_status": smoking,
        "follow_up_days": follow_up_days,
        "a1c": a1c,
        "phenomena": sorted(phenomena),
    }
    return text, gold


def assign_readmission(gold, rng):
    """Give each synthetic patient a 30-day readmission outcome (True/False).

    The probability of readmission is a logistic function of the TRUE
    features: sicker patients (more conditions, heart failure, many
    medications, high A1c, older) are more likely to be readmitted, plus
    random noise so the outcome is not perfectly predictable. This exists
    so the downstream modeling stage has a real target to predict."""
    n_active = sum(1 for c in gold["conditions"] if c["status"] == "active")
    has_heart_failure = any(
        c["name"] == "heart failure" and c["status"] == "active"
        for c in gold["conditions"]
    )
    n_meds = sum(1 for m in gold["medications"] if m["status"] == "active")
    a1c_high = 1.0 if (gold["a1c"] or 0) > 9 else 0.0
    # A medication stopped at this visit signals clinical instability, and
    # it is also the feature the rule-based extractor systematically misses,
    # so it is where extraction quality shows up downstream.
    med_stopped = any(m["status"] == "discontinued" for m in gold["medications"])

    # z is a weighted sum of risk factors; the sigmoid turns it into a
    # probability between 0 and 1. The intercept sets the base rate.
    z = (
        -2.9
        + 0.35 * (n_active - 3)
        + 1.1 * has_heart_failure
        + 0.18 * (n_meds - 3)
        + 0.9 * a1c_high
        + 1.0 * med_stopped
        + 0.025 * (gold["age"] - 60)
        + rng.gauss(0, 0.6)  # noise
    )
    p = 1.0 / (1.0 + 2.718281828 ** (-z))  # sigmoid
    return rng.random() < p


def generate_corpus(n_notes, seed):
    rng = random.Random(seed)
    notes, golds = [], []
    for i in range(n_notes):
        text, gold = generate_note(f"note_{i:04d}", rng)
        gold["readmitted_30d"] = assign_readmission(gold, rng)
        notes.append({"note_id": gold["note_id"], "text": text})
        golds.append(gold)
    return notes, golds


def write_corpus(notes, golds, notes_path, gold_path):
    with open(notes_path, "w") as f:
        json.dump(notes, f, indent=1)
    with open(gold_path, "w") as f:
        json.dump(golds, f, indent=1)
