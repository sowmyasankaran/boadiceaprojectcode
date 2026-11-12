
from dataclasses import dataclass, field as dc_field
from typing import Optional, List, Dict, Any, Callable


@dataclass
class Field:
    name: str
    description: str                       # used in the extraction tool schema
    required: bool = True
    json_type: str = "string"              # json-schema type for extraction
    enum: Optional[List[str]] = None
    plausibility_key: Optional[str] = None
    items: Optional[Dict[str, Any]] = None
    coerce: Optional[Callable] = None


class Section:
    """Base class. Subclass and set `name` and `fields`, and implement
    `to_canrisk()`. Everything else (checklist gate, extraction schema,
    teach-back) is inherited."""

    name: str = "section"
    fields: List[Field] = []

    def __init__(self):
        # collected values, keyed by field name
        self.values: Dict[str, Any] = {}
        self.unconfirmed: set = set()

    def mark_confirmed(self, name: str) -> None:
        """Promote a field out of 'provisional, may still change' status.
        Called from agent.apply_extraction once a field has gone an entire
        turn without being re-touched by extraction or the correction scan
        -- see the docstring on `unconfirmed` above for why this exists."""
        self.unconfirmed.discard(name)

    def confirmed_values(self) -> Dict[str, Any]:
        """Values.EXCLUDING anything still provisional. This is what
        extract_turn's PRIMARY extraction call shows the model as "already
        recorded" context -- deliberately NOT section.values, so a
        not-yet-corrected wrong value can't condition how the model reads
        OTHER fields in the same conversation. The dedicated correction-scan
        pass (scan_for_corrections) is exempt from this and still sees FULL
        section.values: its entire job is finding a restatement of
        something already on record, confirmed or not, so hiding a
        provisional field from it would make it structurally unable to
        catch a correction to something that hasn't survived a turn yet --
        exactly the common case (the correction usually arrives one or two
        turns after the wrong value, not after it's already "settled")."""
        return {k: v for k, v in self.values.items() if k not in self.unconfirmed}

    def required_fields(self) -> List[str]:
        return [f.name for f in self.fields if f.required]

    def missing_required(self) -> List[str]:
        return [n for n in self.required_fields() if self.values.get(n) is None]

    def is_complete(self) -> bool:
        return len(self.missing_required()) == 0

    def set_value(self, name: str, value: Any) -> bool:
        field_obj = next((f for f in self.fields if f.name == name), None)
        if field_obj is None:
            return False
        if value is None:
            return False
        if field_obj.coerce is not None:
            try:
                value = field_obj.coerce(value)
            except Exception:
                return False  # reject malformed input; field stays missing
        self.values[name] = value
        self.unconfirmed.add(name)
        return True

    def extraction_tool_schema(self) -> dict:
        props = {}
        for f in self.fields:
            prop: Dict[str, Any] = {"type": f.json_type, "description": f.description}
            if f.enum:
                prop["enum"] = f.enum
            if f.json_type == "array" and f.items:
                prop["items"] = f.items
            props[f.name] = prop
        # always allow the model to flag ambiguity rather than guessing
        props["ambiguous_fields"] = {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "REQUIRED whenever the user's answer for a field contains a "
                "hedge: 'around', 'about', 'roughly', 'approximately', "
                "'maybe', 'ish', 'I think', 'somewhere between', or a range "
                "like '40-45'. Flag the field name here so the agent can "
                "follow up for a precise value, rather than silently "
                "extracting a guessed point value for it."
            ),
        }
        return {
            "name": f"record_{self.name}_fields",
            "description": (
                f"Record any {self.name} fields confidently extractable from "
                f"the user's message. Only include a field if the message "
                f"actually specifies it; never guess or fill defaults."
            ),
            "input_schema": {"type": "object", "properties": props},
        }

    def goals_text(self) -> str:
        lines = [f"- {f.name}: {f.description}" + ("" if f.required else " (optional)")
                 for f in self.fields]
        return f"Section '{self.name}' needs the following information:\n" + "\n".join(lines)

    def to_canrisk(self) -> dict:
        raise NotImplementedError



from datetime import date
from conversions import (
    compute_bmi, compute_age_yob, alcohol_table_to_grams_per_day,
)


VALID_DRINK_TYPES = {"wine", "pint", "bottle_beer", "alcopop", "spirit_shot"}
VALID_DRINK_FREQ = {"daily", "weekly", "monthly", "never"}


def _validate_iso_date(value) -> str:
    s = str(value).strip()
    date.fromisoformat(s)  # raises ValueError on bad format
    return s


def _to_float(value) -> float:
    return float(value)


def _to_age(value) -> int:

    a = int(round(float(value)))
    if a < 0 or a > 130:
        raise ValueError(f"implausible age: {value}")
    return a


def _validate_drinks(value) -> list:

    cleaned = []
    for d in (value or []):
        if not isinstance(d, dict):
            continue
        t, f, a = d.get("type"), d.get("frequency"), d.get("amount")
        if t in VALID_DRINK_TYPES and f in VALID_DRINK_FREQ:
            try:
                cleaned.append({"type": t, "frequency": f, "amount": float(a)})
            except (TypeError, ValueError):
                continue
    return cleaned


class PersonalLifestyleSection(Section):
    name = "personal_lifestyle"

    ask_one_at_a_time = True
    grouped_fields = [["sex", "country"]]

    fields = [
        Field("sex", "Patient's sex: 'M' or 'F'", required=True,
              json_type="string", enum=["M", "F"]),
        Field("country", "Country of residence (drives cancer incidence rates), e.g. 'UK'",
              required=True, plausibility_key="country"),
        Field("ethnic_group",
              "Ethnic group (UK ONS category). Drives baseline incidence. "
              "NOTE: distinct from PRS genetic ancestry.",
              required=False),
        Field("dob_iso",
              "Date of birth as YYYY-MM-DD. ONLY record this if the patient "
              "gives an actual birth date. If they give an age but not a date, "
              "do NOT invent one -- use the 'age' field instead.",
              required=False, coerce=_validate_iso_date),
        Field("age",
              "Patient's current age in years. Record this if the patient states "
              "their age (e.g. 'I'm 34'). Either age or a birth date is needed, "
              "not both -- never fabricate a birth date from an age.",
              required=False, json_type="integer", plausibility_key="current_age",
              coerce=_to_age),
        Field("height_cm", "Height converted to centimeters", required=True,
              json_type="number", plausibility_key="height_cm", coerce=_to_float),
        Field("weight_kg", "Weight converted to kilograms", required=True,
              json_type="number", plausibility_key="weight_kg", coerce=_to_float),
        Field("drinks",
              "Alcohol: list of {type in [wine,pint,bottle_beer,alcopop,spirit_shot], "
              "frequency in [daily,weekly,monthly,never], amount:number}. "
              "Empty list if the patient does not drink.",
              required=False, json_type="array", coerce=_validate_drinks,
              items={
                  "type": "object",
                  "properties": {
                      "type": {"type": "string", "enum": sorted(VALID_DRINK_TYPES)},
                      "frequency": {"type": "string", "enum": sorted(VALID_DRINK_FREQ)},
                      "amount": {"type": "number"},
                  },
                  "required": ["type", "frequency", "amount"],
              }),
    ]

    def missing_required(self) -> list:
        missing = super().missing_required()
        if self.values.get("age") is None and self.values.get("dob_iso") is None:
            missing.append("date_of_birth_or_age")
        return missing

    def bmi(self) -> Optional[float]:
        h, w = self.values.get("height_cm"), self.values.get("weight_kg")
        return compute_bmi(w, h) if (h and w) else None

    def age_yob(self) -> Optional[tuple]:
        dob_iso = self.values.get("dob_iso")
        if dob_iso:
            return compute_age_yob(date.fromisoformat(dob_iso))
        age = self.values.get("age")
        if age is not None:
            return (int(age), date.today().year - int(age))
        return None

    def alcohol_grams_per_day(self) -> float:
        drinks = self.values.get("drinks") or []
        return alcohol_table_to_grams_per_day(drinks) if drinks else 0.0

    def plausibility_values(self) -> dict:
        ay = self.age_yob()
        return {
            "height_cm": self.values.get("height_cm"),
            "weight_kg": self.values.get("weight_kg"),
            "bmi": self.bmi(),
            "current_age": ay[0] if ay else None,
        }

    def to_canrisk(self) -> dict:
        if not self.is_complete():
            raise ValueError(f"Section incomplete: missing {self.missing_required()}")
        age, yob = self.age_yob()
        country = (self.values.get("country") or "UK").upper()
        params = {"UK": {"mut_freq": "UK", "cancer_rates": "UK"}}.get(
            country, {"mut_freq": "UK", "cancer_rates": "UK"}
        )
        return {
            "header_lines": {
                "height": round(self.values["height_cm"], 1),
                "BMI": self.bmi(),
                "alcohol": self.alcohol_grams_per_day(),
            },
            "proband_overrides": {"sex": self.values["sex"], "age": age, "yob": yob},
            "request_params": params,
            "unresolved": {
                "ethnic_group": self.values.get("ethnic_group"),
            },
        }

    def teach_back(self) -> str:
        lines = []
        v = self.values
        if v.get("sex"):
            lines.append(f"- Sex: {'Female' if v['sex']=='F' else 'Male'}")
        if v.get("country"):
            lines.append(f"- Country: {v['country']}")
        if v.get("ethnic_group"):
            lines.append(f"- Ethnic group: {v['ethnic_group']}")
        ay = self.age_yob()
        if v.get("dob_iso"):
            lines.append(f"- Date of birth: {v['dob_iso']} (age {ay[0]}, born {ay[1]})")
        elif ay:
            lines.append(f"- Age: {ay[0]} (born ~{ay[1]})")
        if v.get("height_cm") and v.get("weight_kg"):
            lines.append(f"- Height {v['height_cm']}cm, weight {v['weight_kg']}kg (BMI {self.bmi()})")
        if v.get("drinks"):
            lines.append(f"- Alcohol: {self.alcohol_grams_per_day()}g/day")
        else:
            lines.append("- Alcohol: none reported")
        return "Here's what I have for personal details and lifestyle:\n" + "\n".join(lines)