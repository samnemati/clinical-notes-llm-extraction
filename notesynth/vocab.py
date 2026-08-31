"""Vocabulary for the synthetic note generator.

Canonical names are the keys; surface forms are what may appear in the note text.
The gold labels always use canonical names, so evaluation is a normalization
problem for the extractors, not for the generator.
"""

# Chronic conditions: canonical -> surface forms that can appear in a PMH list
# or narrative. The first surface form is the full name.
CONDITIONS = {
    "hypertension": ["hypertension", "HTN"],
    "type 2 diabetes": ["type 2 diabetes", "T2DM", "diabetes mellitus type 2"],
    "COPD": ["chronic obstructive pulmonary disease", "COPD"],
    "atrial fibrillation": ["atrial fibrillation", "afib", "AF"],
    "heart failure": ["congestive heart failure", "CHF", "heart failure"],
    "hyperlipidemia": ["hyperlipidemia", "HLD"],
    "asthma": ["asthma"],
    "chronic kidney disease": ["chronic kidney disease", "CKD stage 3"],
    "GERD": ["gastroesophageal reflux disease", "GERD"],
    "depression": ["depression"],
    "hypothyroidism": ["hypothyroidism"],
    "osteoarthritis": ["osteoarthritis", "OA"],
}

# Findings that appear in the HPI and can be affirmed or negated.
FINDINGS = {
    "chest pain": ["chest pain"],
    "shortness of breath": ["shortness of breath", "dyspnea"],
    "fever": ["fever", "fevers"],
    "pneumonia": ["pneumonia"],
    "lower extremity edema": ["lower extremity edema", "leg swelling"],
    "palpitations": ["palpitations"],
}

NEGATION_TEMPLATES = [
    "Denies {x}.",
    "No {x}.",
    "Reports no {x}.",
    "Negative for {x}.",
]

AFFIRMED_TEMPLATES = [
    "Reports {x} over the past week.",
    "Endorses intermittent {x}.",
    "Presents with {x}.",
]

# Hedged new diagnoses: canonical -> phrase (status "suspected" in gold).
HEDGED = {
    "obstructive sleep apnea": "Symptoms are suspicious for obstructive sleep apnea; sleep study ordered.",
    "COPD": "Spirometry pending; possible early COPD.",
    "GERD": "Postprandial burning is most likely GERD; will trial empiric therapy.",
    "peripheral neuropathy": "Exam findings suggest possible peripheral neuropathy.",
}

# Historical, resolved items: canonical -> phrase (status "historical" in gold).
HISTORICAL = {
    "myocardial infarction": "History of myocardial infarction in {year}, stented.",
    "deep vein thrombosis": "History of DVT in {year}, completed anticoagulation.",
    "pneumonia": "Hospitalized for pneumonia in {year}, resolved.",
    "breast cancer": "History of breast cancer in {year}, in remission.",
}

# Medications: name -> (doses, unit, frequencies, condition treated or None).
MEDICATIONS = {
    "metformin": ([500, 850, 1000], "mg", ["BID", "twice daily"], "type 2 diabetes"),
    "insulin glargine": ([12, 18, 24], "units", ["at bedtime", "nightly"], "type 2 diabetes"),
    "lisinopril": ([10, 20, 40], "mg", ["daily", "once daily"], "hypertension"),
    "amlodipine": ([5, 10], "mg", ["daily"], "hypertension"),
    "metoprolol succinate": ([25, 50, 100], "mg", ["daily"], "atrial fibrillation"),
    "apixaban": ([5], "mg", ["BID", "twice daily"], "atrial fibrillation"),
    "furosemide": ([20, 40], "mg", ["daily", "each morning"], "heart failure"),
    "atorvastatin": ([20, 40, 80], "mg", ["at bedtime", "daily"], "hyperlipidemia"),
    "tiotropium": ([18], "mcg", ["daily inhaled"], "COPD"),
    "albuterol": ([90], "mcg", ["2 puffs as needed"], "asthma"),
    "omeprazole": ([20, 40], "mg", ["daily", "before breakfast"], "GERD"),
    "sertraline": ([50, 100], "mg", ["daily"], "depression"),
    "levothyroxine": ([50, 75, 100], "mcg", ["daily"], "hypothyroidism"),
    "gabapentin": ([300, 600], "mg", ["TID", "three times daily"], None),
    "acetaminophen": ([500], "mg", ["as needed"], "osteoarthritis"),
}

DISCONTINUATION_REASONS = {
    "metformin": "GI upset",
    "lisinopril": "a persistent dry cough",
    "amlodipine": "ankle swelling",
    "atorvastatin": "myalgias",
    "sertraline": "poor tolerability",
    "gabapentin": "daytime sedation",
    "omeprazole": "resolution of symptoms",
}

# Occasional misspellings the generator may inject (canonical -> misspelled).
MISSPELLINGS = {
    "hypertension": "hypertenion",
    "metformin": "metfromin",
    "hyperlipidemia": "hyperlipdemia",
    "levothyroxine": "levothroxine",
}

FAMILY_HISTORY = [
    "Mother with type 2 diabetes.",
    "Father had a myocardial infarction at age 60.",
    "Sister with breast cancer.",
    "Mother with hypertension and stroke.",
    "Father with colon cancer.",
]

ALLERGIES = [
    "penicillin (rash)",
    "sulfa drugs (hives)",
    "codeine (nausea)",
    "no known drug allergies",
]

SMOKING_PHRASES = {
    "current": [
        "Smokes one pack per day.",
        "Current smoker, about half a pack daily.",
        "Active tobacco use, 1 ppd for 20 years.",
    ],
    "former": [
        "Former smoker, quit in {year}.",
        "Quit smoking {n} years ago after a 15 pack-year history.",
        "Ex-smoker, no tobacco since {year}.",
    ],
    "never": [
        "Denies any tobacco use.",
        "Never smoker.",
        "No history of smoking.",
    ],
}

# Surface-form -> canonical map used by evaluation to normalize predictions.
def build_normalization_map():
    norm = {}
    for canonical, surfaces in {**CONDITIONS, **FINDINGS}.items():
        norm[canonical.lower()] = canonical
        for s in surfaces:
            norm[s.lower()] = canonical
    for canonical in list(HEDGED) + list(HISTORICAL):
        norm.setdefault(canonical.lower(), canonical)
    # A few common variants extractors are likely to emit.
    extra = {
        "dvt": "deep vein thrombosis",
        "mi": "myocardial infarction",
        "osa": "obstructive sleep apnea",
        "diabetes": "type 2 diabetes",
        "type 2 diabetes mellitus": "type 2 diabetes",
        "dyspnea on exertion": "shortness of breath",
        "edema": "lower extremity edema",
        "reflux": "GERD",
        "ckd": "chronic kidney disease",
    }
    for k, v in extra.items():
        norm.setdefault(k, v)
    for canonical, wrong in MISSPELLINGS.items():
        if canonical in CONDITIONS:
            norm.setdefault(wrong.lower(), canonical)
    return norm
