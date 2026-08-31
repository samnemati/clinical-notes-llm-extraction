"""LLM extractor: sends each note to the Claude API and gets back a
NoteExtraction object.

Two things make this reliable rather than "ask a chatbot and hope":

1. Structured output. We pass the NoteExtraction schema to
   `client.messages.parse(...)`, so the API is constrained to return JSON
   matching that exact shape, and the SDK validates it into a Pydantic
   object. No string parsing, no malformed JSON.

2. Prompt caching. The instructions are the same for every note, so they
   are marked cacheable and only the note text changes between calls.

Every raw result is written to a JSON file. That file is committed, so the
evaluation and downstream stages are reproducible by anyone without an API
key; only re-running the extraction itself needs one.
"""

import json

import anthropic

from .schema import NoteExtraction

DEFAULT_MODEL = "claude-opus-5"

# The instructions sent with every note. They describe the task and the
# judgment calls (what counts as negated, how to convert follow-up time).
# The output SHAPE is not described here; the schema enforces it.
SYSTEM_PROMPT = """You extract structured information from clinical notes.

Read the note and report:
- conditions: every condition, symptom, or finding attributed to the patient,
  with a status. "active" for current problems and affirmed symptoms;
  "historical" for resolved past events; "negated" for findings the note
  explicitly denies; "suspected" for hedged or provisional diagnoses.
  Use full lowercase names (write "hypertension", not "HTN"). Do not include
  conditions that belong to family members, and do not include medication
  allergies as conditions.
- medications: every medication the patient takes or that was changed at this
  visit, with dose, unit, and status "active" or "discontinued". Correct
  obvious misspellings. Allergies are not medications.
- smoking_status, follow_up_days (convert: 1 week = 7 days, 1 month = 30
  days), and a1c if reported.

Extract only what the note states. Use null when a value is not given."""


def extract_note(client, note_text, model=DEFAULT_MODEL):
    """Send one note to the API and return a validated NoteExtraction."""
    response = client.messages.parse(
        model=model,
        max_tokens=16000,
        # cache_control makes the API cache the system prompt, so repeated
        # calls only pay full price for the part that changes (the note).
        system=[{
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{"role": "user", "content": note_text}],
        output_format=NoteExtraction,
    )
    return response.parsed_output


def extract_corpus(client, notes, model=DEFAULT_MODEL, on_progress=None):
    """Extract every note in the corpus, one API call per note.

    Returns a dict: note_id -> NoteExtraction. Notes that fail after the
    SDK's built-in retries are recorded in the errors dict instead of
    crashing the whole run."""
    results = {}
    errors = {}
    for i, note in enumerate(notes):
        try:
            results[note["note_id"]] = extract_note(client, note["text"], model)
        except anthropic.APIError as exc:
            errors[note["note_id"]] = str(exc)
        if on_progress:
            on_progress(i + 1, len(notes))
    return results, errors


def save_extractions(results, path, model):
    """Write all extractions to one JSON file (this file gets committed)."""
    payload = {
        "model": model,
        "extractions": {
            note_id: extraction.model_dump()
            for note_id, extraction in sorted(results.items())
        },
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=1)


def load_extractions(path):
    """Read a saved extractions file back into validated objects."""
    with open(path) as f:
        payload = json.load(f)
    return {
        note_id: NoteExtraction.model_validate(data)
        for note_id, data in payload["extractions"].items()
    }
