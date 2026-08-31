"""Pydantic schema shared by the LLM extractor and the evaluator.

The same model class drives the API's structured-output constraint and the
validation of anything read back from disk, so there is exactly one definition
of what an extraction looks like.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

ConditionStatus = Literal["active", "historical", "negated", "suspected"]
MedicationStatus = Literal["active", "discontinued"]
SmokingStatus = Literal["current", "former", "never", "unknown"]


class Condition(BaseModel):
    name: str = Field(description=(
        "The condition, symptom, or finding, in lowercase. Use the full name, "
        "not an abbreviation (e.g. 'hypertension', not 'HTN'). Include only "
        "problems attributed to the patient; exclude family history."
    ))
    status: ConditionStatus = Field(description=(
        "'active' for current problems and affirmed symptoms, 'historical' for "
        "resolved past events, 'negated' for explicitly denied or absent "
        "findings, 'suspected' for hedged or provisional diagnoses."
    ))


class Medication(BaseModel):
    name: str = Field(description=(
        "Medication name in lowercase, spelling corrected if obviously "
        "misspelled. Exclude allergies."
    ))
    dose: Optional[float] = Field(default=None, description="Numeric dose if stated.")
    unit: Optional[str] = Field(default=None, description="Dose unit, e.g. 'mg', 'mcg', 'units'.")
    status: MedicationStatus = Field(description=(
        "'discontinued' only if the note says the medication was stopped."
    ))


class NoteExtraction(BaseModel):
    conditions: List[Condition]
    medications: List[Medication]
    smoking_status: SmokingStatus = Field(description=(
        "'unknown' if the note does not mention tobacco use."
    ))
    follow_up_days: Optional[int] = Field(default=None, description=(
        "Planned follow-up interval converted to days (1 week = 7 days, "
        "1 month = 30 days). Null if no interval is given."
    ))
    a1c: Optional[float] = Field(default=None, description=(
        "Most recent hemoglobin A1c percentage if reported, else null."
    ))
