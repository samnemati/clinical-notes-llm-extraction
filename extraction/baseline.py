"""Rule-based baseline extractor.

This is the "traditional NLP" comparison point for the LLM: a dictionary of
known condition and medication names, plus a small negation rule modeled on
NegEx (Chapman et al., 2001). The idea of NegEx is simple: if a phrase like
"denies" or "no" appears shortly BEFORE a condition name, that condition is
negated ("denies chest pain" means the patient does NOT have chest pain).

The baseline is written to be fair, not a strawman. It handles the patterns
it was designed for (negation, "history of", medication lists) and fails
where dictionary methods genuinely fail: misspelled words, medications
discontinued in prose ("we stopped X because..."), and hedged diagnoses
("possible early COPD").
"""

import re

from notesynth import vocab
from .schema import Condition, Medication, NoteExtraction

# If one of these phrases appears just before a condition name,
# the condition is negated (the patient does not have it).
NEGATION_TRIGGERS = ["denies", "no ", "reports no", "negative for", "without"]

# If one of these appears just before a condition name, it is a past,
# resolved problem rather than a current one.
HISTORICAL_TRIGGERS = ["history of", "hospitalized for", "h/o", "s/p"]

# Build one flat list of every surface form we know, paired with its
# canonical name. Example entries: ("htn", "hypertension"),
# ("type 2 diabetes", "type 2 diabetes").
_SURFACES = []
for canonical, forms in {**vocab.CONDITIONS, **vocab.FINDINGS}.items():
    for form in forms:
        _SURFACES.append((form.lower(), canonical))
for canonical in vocab.HISTORICAL:
    _SURFACES.append((canonical.lower(), canonical))
# Sort longest-first so "type 2 diabetes" is matched before just "diabetes".
_SURFACES.sort(key=lambda pair: -len(pair[0]))


def _sentences(text):
    """Split the note into sentence-like chunks (at '.', ';', or newlines).
    Negation rules work per sentence so a 'denies' in one sentence cannot
    negate a condition in the next one."""
    return re.split(r"(?<=[.;])\s+|\n", text)


def extract_conditions(text):
    """Find known condition names and decide their status from nearby words."""
    found = {}
    for sentence in _sentences(text):
        low = sentence.lower()

        # Skip family history lines entirely: "mother with diabetes" is
        # about a relative, not the patient.
        if "family history" in low:
            continue

        for surface, canonical in _SURFACES:
            if canonical in found:
                continue  # keep the first (longest-match) occurrence only

            # \b...\b means "whole words only", so the surface form "af"
            # (atrial fibrillation) does not match inside the word "after".
            match = re.search(rf"\b{re.escape(surface)}\b", low)
            if not match:
                continue

            # Look at the 40 characters BEFORE the condition name for a
            # negation or historical trigger. This window is the NegEx idea.
            window = low[max(0, match.start() - 40): match.start()]
            if any(trigger in window for trigger in NEGATION_TRIGGERS):
                status = "negated"
            elif any(trigger in window for trigger in HISTORICAL_TRIGGERS):
                status = "historical"
            else:
                status = "active"

            found[canonical] = Condition(name=canonical, status=status)
    return list(found.values())


# Matches one line of a medication list, e.g. "- lisinopril 20 mg daily"
# or "metformin 500mg BID". Pieces: a name (letters and spaces), a number,
# then a unit (mg, mcg, or units).
_MED_LINE = re.compile(
    r"^[-\s]*(?P<name>[a-z][a-z ]+?)\s+(?P<dose>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>mg|mcg|units)\b",
    re.I,
)


def extract_medications(text):
    """Read medications from a 'Medications:' list or from the narrative
    sentence 'Current medications are X; Y; Z.'"""
    meds = {}

    # Case 1: a structured list under a "Medications:" header.
    in_meds_section = False
    for line in text.split("\n"):
        stripped = line.strip()
        if re.match(r"medications?\s*:", stripped, re.I):
            in_meds_section = True  # the following lines are the med list
            continue
        if in_meds_section:
            match = _MED_LINE.match(stripped)
            if match:
                name = match.group("name").strip().lower()
                meds[name] = Medication(
                    name=name,
                    dose=float(match.group("dose")),
                    unit=match.group("unit").lower(),
                    status="active",
                )
            elif stripped:
                in_meds_section = False  # a non-matching line ends the list

    # Case 2: a narrative sentence listing meds separated by semicolons.
    narrative = re.search(r"current medications are (.+?)\.(?:\s|$)", text, re.I | re.S)
    if narrative:
        for part in narrative.group(1).split(";"):
            match = _MED_LINE.match(part.strip())
            if match:
                name = match.group("name").strip().lower()
                meds[name] = Medication(
                    name=name,
                    dose=float(match.group("dose")),
                    unit=match.group("unit").lower(),
                    status="active",
                )

    # Note what is missing: nothing here reads "we discontinued X today...",
    # so discontinued medications are systematically missed. That is one of
    # the failure modes the evaluation is designed to surface.
    return list(meds.values())


def extract_smoking(text):
    """Map smoking phrases to current / former / never / unknown.
    Order matters: 'denies any tobacco' must be checked before the word
    'smokes' so 'never' phrasings are not misread as 'current'."""
    low = text.lower()
    if re.search(r"denies any tobacco|never smoker|no history of smoking", low):
        return "never"
    if re.search(r"former smoker|ex-smoker|quit smoking|no tobacco since", low):
        return "former"
    if re.search(r"smokes|current smoker|active tobacco|ppd", low):
        return "current"
    return "unknown"


def extract_follow_up(text):
    """Find 'follow up in N days/weeks/months' and convert to days."""
    match = re.search(
        r"(?:follow up|return(?: to clinic)?|rtc)\s+in\s+(\d+)\s*"
        r"(day|week|month)s?",
        text,
        re.I,
    )
    if not match:
        return None
    n = int(match.group(1))
    unit = match.group(2).lower()
    days_per_unit = {"day": 1, "week": 7, "month": 30}
    return n * days_per_unit[unit]


def extract_a1c(text):
    """Find a value like 'A1c 8.2%'."""
    match = re.search(r"a1c\s+(\d+(?:\.\d+)?)\s*%", text, re.I)
    return float(match.group(1)) if match else None


def extract(text):
    """Run all the rules on one note and return a NoteExtraction object,
    the same shape the LLM extractor returns."""
    return NoteExtraction(
        conditions=extract_conditions(text),
        medications=extract_medications(text),
        smoking_status=extract_smoking(text),
        follow_up_days=extract_follow_up(text),
        a1c=extract_a1c(text),
    )
