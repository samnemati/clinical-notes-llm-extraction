"""Step 3: run the LLM extractor over every note (needs API credentials).

This is the only step that calls the Claude API. Its output,
outputs/llm_extractions.json, is committed, so steps 4 and 5 work without
credentials. Re-run this step only to regenerate the extractions.

Usage:
    python scripts/30_run_llm.py            # default model
    python scripts/30_run_llm.py --model claude-opus-5
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import anthropic

from extraction import llm

ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser()
parser.add_argument("--model", default=llm.DEFAULT_MODEL)
args = parser.parse_args()

with open(ROOT / "data" / "notes.json") as f:
    notes = json.load(f)

# The client finds credentials on its own (ANTHROPIC_API_KEY, or a profile
# from `ant auth login`). Never hardcode a key.
client = anthropic.Anthropic()


def report(done, total):
    if done % 10 == 0 or done == total:
        print(f"  {done}/{total} notes extracted")


results, errors = llm.extract_corpus(client, notes, args.model, on_progress=report)
llm.save_extractions(results, ROOT / "outputs" / "llm_extractions.json", args.model)

print(f"LLM ({args.model}) extracted {len(results)} notes -> outputs/llm_extractions.json")
if errors:
    print(f"WARNING: {len(errors)} notes failed:")
    for note_id, message in errors.items():
        print(f"  {note_id}: {message}")
