"""Step 4: score both extractors against the gold labels.

Writes outputs/metrics.json and prints a comparison table. Runs entirely
offline from the committed extraction files.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from extraction.evaluate import evaluate
from extraction.llm import load_extractions

ROOT = Path(__file__).resolve().parents[1]

with open(ROOT / "data" / "gold.json") as f:
    golds = json.load(f)

metrics = {}
for name, path in [
    ("baseline", ROOT / "outputs" / "baseline_extractions.json"),
    ("llm", ROOT / "outputs" / "llm_extractions.json"),
]:
    if not path.exists():
        print(f"({name}: {path.name} not found, skipping)")
        continue
    metrics[name] = evaluate(golds, load_extractions(path))

with open(ROOT / "outputs" / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=1)

# Print a side-by-side summary ------------------------------------------------
systems = list(metrics)


def row(label, getter):
    cells = "".join(f"{getter(metrics[s]) if getter(metrics[s]) is not None else '-':>12}" for s in systems)
    print(f"{label:<34}{cells}")


print(f"{'metric':<34}" + "".join(f"{s:>12}" for s in systems))
row("conditions F1", lambda m: m["conditions"]["f1"])
row("conditions status accuracy", lambda m: m["conditions"]["status_accuracy"])
row("medications F1", lambda m: m["medications"]["f1"])
row("medications status accuracy", lambda m: m["medications"]["status_accuracy"])
row("medication dose accuracy", lambda m: m["medications"]["dose_accuracy"])
row("smoking accuracy", lambda m: m["smoking_accuracy"])
row("follow-up accuracy", lambda m: m["follow_up_accuracy"])
row("A1c accuracy", lambda m: m["a1c_accuracy"])
print()
print("Recall on hard phenomena (found with the correct status):")
phens = sorted({p for m in metrics.values() for p in m["recall_by_phenomenon"]})
for phen in phens:
    row(
        f"  {phen}",
        lambda m, p=phen: m["recall_by_phenomenon"].get(p, {}).get("recall"),
    )
print("\nWrote outputs/metrics.json")
