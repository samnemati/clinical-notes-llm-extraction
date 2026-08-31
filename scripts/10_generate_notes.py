""" 1: generate the synthetic corpus.

Writes data/notes.json (the note texts) and data/gold.json (the correct
answers for every note). Both files are committed to the repo, so every
later step starts from the exact same corpus.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from notesynth.generator import generate_corpus, write_corpus

N_NOTES = 150
SEED = 20260831  # fixed seed: the corpus is fully reproducible

ROOT = Path(__file__).resolve().parents[1]

notes, golds = generate_corpus(N_NOTES, SEED)
write_corpus(notes, golds, ROOT / "data" / "notes.json", ROOT / "data" / "gold.json")

n_readmit = sum(1 for g in golds if g["readmitted_30d"])
print(f"Wrote {len(notes)} notes to data/")
print(f"Readmission rate: {n_readmit}/{len(golds)} = {n_readmit / len(golds):.0%}")
