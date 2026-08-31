"""Step 2: run the rule-based baseline over every note.

Writes outputs/baseline_extractions.json in the same format the LLM step
uses, so evaluation treats both extractors identically.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from extraction import baseline

ROOT = Path(__file__).resolve().parents[1]

with open(ROOT / "data" / "notes.json") as f:
    notes = json.load(f)

extractions = {}
for note in notes:
    extractions[note["note_id"]] = baseline.extract(note["text"]).model_dump()

out = {"model": "rule-based baseline", "extractions": extractions}
with open(ROOT / "outputs" / "baseline_extractions.json", "w") as f:
    json.dump(out, f, indent=1)

print(f"Baseline extracted {len(extractions)} notes -> outputs/baseline_extractions.json")
