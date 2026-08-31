"""Step 5: use the extracted information for analysis, the way a real
project would.

Two things happen here:

1. Descriptive statistics of the cohort, computed from the LLM extractions
   and compared against the truth (condition prevalence, medication counts,
   smoking).

2. A predictive model. We fit the same logistic regression to predict
   30-day readmission three times, changing only WHERE the features come
   from: the gold labels, the LLM extractions, or the baseline extractions.
   The gap between the gold AUC and the others measures how much extraction
   errors cost downstream. This matters because extraction is rarely the end
   product; the analysis built on top of it is.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from extraction.evaluate import normalize_name
from extraction.llm import load_extractions

ROOT = Path(__file__).resolve().parents[1]

with open(ROOT / "data" / "gold.json") as f:
    golds = json.load(f)


def features_from_gold(gold):
    """Model features computed from the TRUE labels."""
    active = [c for c in gold["conditions"] if c["status"] == "active"]
    names = {c["name"] for c in active}
    return {
        "n_active_conditions": len(active),
        "heart_failure": int("heart failure" in names),
        "diabetes": int("type 2 diabetes" in names),
        "copd": int("COPD" in names),
        "n_active_meds": sum(1 for m in gold["medications"] if m["status"] == "active"),
        "med_stopped": int(any(m["status"] == "discontinued" for m in gold["medications"])),
        "a1c": gold["a1c"],
        "current_smoker": int(gold["smoking_status"] == "current"),
        "age": gold["age"],
    }


def features_from_extraction(extraction, age):
    """The same features, computed from an extractor's output instead.
    Age comes from the note header either way."""
    active = [c for c in extraction.conditions if c.status == "active"]
    names = {normalize_name(c.name) for c in active}
    return {
        "n_active_conditions": len(active),
        "heart_failure": int("heart failure" in names),
        "diabetes": int("type 2 diabetes" in names),
        "copd": int("COPD" in names),
        "n_active_meds": sum(1 for m in extraction.medications if m.status == "active"),
        "med_stopped": int(any(m.status == "discontinued" for m in extraction.medications)),
        "a1c": extraction.a1c,
        "current_smoker": int(extraction.smoking_status == "current"),
        "age": age,
    }


def auc_for(frame, outcome):
    """5-fold cross-validated AUC of a logistic regression. Missing A1c is
    filled with the column median inside the pipeline-ready frame."""
    X = frame.copy()
    X["a1c"] = X["a1c"].fillna(X["a1c"].median())
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_score(model, X, outcome, cv=cv, scoring="roc_auc")
    return scores.mean(), scores.std()


outcome = np.array([g["readmitted_30d"] for g in golds], dtype=int)
ages = {g["note_id"]: g["age"] for g in golds}

sources = {"gold": pd.DataFrame([features_from_gold(g) for g in golds])}
for name, path in [
    ("llm", ROOT / "outputs" / "llm_extractions.json"),
    ("baseline", ROOT / "outputs" / "baseline_extractions.json"),
]:
    if not path.exists():
        continue
    extractions = load_extractions(path)
    rows = []
    for g in golds:
        ext = extractions.get(g["note_id"])
        if ext is None:
            rows.append({k: np.nan for k in sources["gold"].columns})
        else:
            rows.append(features_from_extraction(ext, ages[g["note_id"]]))
    sources[name] = pd.DataFrame(rows)

# Descriptive statistics: does the extracted cohort look like the true one? --
description = {}
for name, frame in sources.items():
    description[name] = {
        "heart_failure_prevalence": round(frame["heart_failure"].mean(), 3),
        "diabetes_prevalence": round(frame["diabetes"].mean(), 3),
        "current_smoker_rate": round(frame["current_smoker"].mean(), 3),
        "mean_active_meds": round(frame["n_active_meds"].mean(), 2),
        "med_stopped_rate": round(frame["med_stopped"].mean(), 3),
        "mean_a1c_when_present": round(frame["a1c"].dropna().mean(), 2),
    }

# Predictive model, one row per feature source ------------------------------
model_results = {}
for name, frame in sources.items():
    mean_auc, sd_auc = auc_for(frame, outcome)
    model_results[name] = {"auc_mean": round(mean_auc, 3), "auc_sd": round(sd_auc, 3)}
    print(f"readmission AUC using {name:>8} features: "
          f"{mean_auc:.3f} (sd {sd_auc:.3f} across folds)")

with open(ROOT / "outputs" / "downstream.json", "w") as f:
    json.dump({"descriptives": description, "readmission_model": model_results}, f, indent=1)

print("\nCohort descriptives by feature source:")
print(pd.DataFrame(description).to_string())
print("\nWrote outputs/downstream.json")
