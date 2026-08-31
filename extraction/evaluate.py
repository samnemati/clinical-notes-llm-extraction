"""Score an extractor's output against the gold labels.

The logic is ordinary information-extraction scoring:

- Conditions and medications are matched BY NAME between prediction and
  gold (after normalizing abbreviations to canonical names). A matched name
  is a true positive; a predicted name with no gold partner is a false
  positive; a gold name nobody predicted is a false negative. Precision,
  recall, and F1 come from those counts.
- For the matched pairs we then ask a second question: was the STATUS
  (active / negated / historical / suspected, or active / discontinued)
  also right? That is reported as status accuracy.
- Scalar fields (smoking, follow-up days, A1c) are simple accuracy.

Because the generator recorded which linguistic tricks each note contains
(negation, abbreviations, a discontinued medication, ...), we can also report
recall per phenomenon: "of the gold negated conditions, how many did the
extractor find AND label negated?" and so on.
"""

from notesynth.vocab import build_normalization_map

_NORM = build_normalization_map()


def normalize_name(name):
    """Map a predicted name to its canonical form ("HTN" -> "hypertension").
    Unknown names are just lowercased and kept, counting as false positives
    unless the gold really contains them."""
    return _NORM.get(name.strip().lower(), name.strip().lower())


def _prf(tp, fp, fn):
    """Precision, recall, F1 from raw counts (0 when undefined)."""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {"precision": round(precision, 3), "recall": round(recall, 3),
            "f1": round(f1, 3), "tp": tp, "fp": fp, "fn": fn}


def score_entities(gold_items, pred_items):
    """Match one note's gold and predicted entity lists by canonical name.

    Returns the TP/FP/FN counts plus, for each matched pair, whether the
    status agreed."""
    gold_by_name = {normalize_name(g["name"]): g for g in gold_items}
    pred_by_name = {normalize_name(p.name): p for p in pred_items}

    matched = set(gold_by_name) & set(pred_by_name)
    tp = len(matched)
    fp = len(set(pred_by_name) - matched)
    fn = len(set(gold_by_name) - matched)

    status_pairs = [
        (gold_by_name[name]["status"], pred_by_name[name].status)
        for name in matched
    ]
    return tp, fp, fn, status_pairs


def evaluate(golds, extractions):
    """Score a whole corpus. `extractions` maps note_id -> NoteExtraction.
    Notes with no extraction (an API error, for example) count as missing
    everything, which is the honest way to score a partial run."""
    counts = {
        "conditions": [0, 0, 0], "medications": [0, 0, 0],
    }
    status_ok = {"conditions": [0, 0], "medications": [0, 0]}  # correct, total
    smoking = [0, 0]
    follow_up = [0, 0]
    a1c = [0, 0]
    dose_ok = [0, 0]

    # recall per phenomenon: phenomenon -> [found_and_correct, gold_total]
    by_phenomenon = {}

    for gold in golds:
        pred = extractions.get(gold["note_id"])
        pred_conditions = pred.conditions if pred else []
        pred_medications = pred.medications if pred else []

        for field, gold_items, pred_items in [
            ("conditions", gold["conditions"], pred_conditions),
            ("medications", gold["medications"], pred_medications),
        ]:
            tp, fp, fn, status_pairs = score_entities(gold_items, pred_items)
            counts[field][0] += tp
            counts[field][1] += fp
            counts[field][2] += fn
            status_ok[field][0] += sum(1 for g, p in status_pairs if g == p)
            status_ok[field][1] += len(status_pairs)

        # Dose accuracy on matched medications.
        pred_by_name = {normalize_name(m.name): m for m in pred_medications}
        for g in gold["medications"]:
            p = pred_by_name.get(normalize_name(g["name"]))
            if p is not None:
                dose_ok[1] += 1
                if p.dose == g["dose"] and (p.unit or "").lower() == g["unit"]:
                    dose_ok[0] += 1

        # Scalar fields.
        smoking[1] += 1
        if pred and pred.smoking_status == gold["smoking_status"]:
            smoking[0] += 1
        follow_up[1] += 1
        if pred and pred.follow_up_days == gold["follow_up_days"]:
            follow_up[0] += 1
        a1c[1] += 1
        if pred and pred.a1c == gold["a1c"]:
            a1c[0] += 1

        # Per-phenomenon recall: was each "hard" gold item found with the
        # right status?
        pred_cond = {normalize_name(c.name): c for c in pred_conditions}
        for g in gold["conditions"]:
            for phen, wanted_status in [
                ("negation", "negated"),
                ("historical", "historical"),
                ("hedged_dx", "suspected"),
            ]:
                if phen in gold["phenomena"] and g["status"] == wanted_status:
                    hit = pred_cond.get(normalize_name(g["name"]))
                    tally = by_phenomenon.setdefault(phen, [0, 0])
                    tally[1] += 1
                    if hit is not None and hit.status == wanted_status:
                        tally[0] += 1
        if "discontinued_med" in gold["phenomena"]:
            for g in gold["medications"]:
                if g["status"] == "discontinued":
                    hit = pred_by_name.get(normalize_name(g["name"]))
                    tally = by_phenomenon.setdefault("discontinued_med", [0, 0])
                    tally[1] += 1
                    if hit is not None and hit.status == "discontinued":
                        tally[0] += 1
        if "misspelling" in gold["phenomena"]:
            # Recall over every entity in a note that contains misspellings.
            for g in gold["conditions"] + gold["medications"]:
                name = normalize_name(g["name"])
                found = name in pred_cond or name in pred_by_name
                tally = by_phenomenon.setdefault("misspelling", [0, 0])
                tally[1] += 1
                if found:
                    tally[0] += 1

    def ratio(pair):
        return round(pair[0] / pair[1], 3) if pair[1] else None

    return {
        "n_notes": len(golds),
        "n_extracted": len(extractions),
        "conditions": {
            **_prf(*counts["conditions"]),
            "status_accuracy": ratio(status_ok["conditions"]),
        },
        "medications": {
            **_prf(*counts["medications"]),
            "status_accuracy": ratio(status_ok["medications"]),
            "dose_accuracy": ratio(dose_ok),
        },
        "smoking_accuracy": ratio(smoking),
        "follow_up_accuracy": ratio(follow_up),
        "a1c_accuracy": ratio(a1c),
        "recall_by_phenomenon": {
            phen: {"recall": ratio(tally), "n": tally[1]}
            for phen, tally in sorted(by_phenomenon.items())
        },
    }
