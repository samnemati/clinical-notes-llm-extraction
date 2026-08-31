"""Tests for the LLM extraction module.

No API calls happen here. A fake client stands in for the real one, which
lets us test everything AROUND the API: that the right arguments are sent,
that results round-trip through the save/load files, and that one failed
note does not sink the whole run.
"""

import anthropic
import pytest

from extraction import llm
from extraction.schema import Condition, NoteExtraction


class FakeResponse:
    def __init__(self, parsed_output):
        self.parsed_output = parsed_output


class FakeMessages:
    def __init__(self, outputs, fail_on=None):
        self.outputs = outputs
        self.fail_on = fail_on or set()
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        note_text = kwargs["messages"][0]["content"]
        if note_text in self.fail_on:
            raise anthropic.APIConnectionError(request=None)
        return FakeResponse(self.outputs[note_text])


class FakeClient:
    def __init__(self, outputs, fail_on=None):
        self.messages = FakeMessages(outputs, fail_on)


EXTRACTION = NoteExtraction(
    conditions=[Condition(name="hypertension", status="active")],
    medications=[],
    smoking_status="unknown",
    follow_up_days=14,
    a1c=None,
)


def test_extract_note_sends_schema_and_cached_system_prompt():
    client = FakeClient({"the note": EXTRACTION})
    result = llm.extract_note(client, "the note")
    assert result == EXTRACTION

    call = client.messages.calls[0]
    assert call["output_format"] is NoteExtraction
    assert call["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert call["model"] == llm.DEFAULT_MODEL


def test_extract_corpus_survives_one_failure():
    notes = [
        {"note_id": "a", "text": "note a"},
        {"note_id": "b", "text": "note b"},
    ]
    client = FakeClient(
        {"note a": EXTRACTION, "note b": EXTRACTION}, fail_on={"note b"}
    )
    results, errors = llm.extract_corpus(client, notes)
    assert set(results) == {"a"}
    assert set(errors) == {"b"}


def test_save_and_load_round_trip(tmp_path):
    path = tmp_path / "extractions.json"
    llm.save_extractions({"a": EXTRACTION}, path, model="test-model")
    loaded = llm.load_extractions(path)
    assert loaded["a"] == EXTRACTION


def test_progress_callback_called_per_note():
    notes = [{"note_id": "a", "text": "note a"}]
    client = FakeClient({"note a": EXTRACTION})
    seen = []
    llm.extract_corpus(client, notes, on_progress=lambda d, t: seen.append((d, t)))
    assert seen == [(1, 1)]
