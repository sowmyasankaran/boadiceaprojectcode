
from dataclasses import dataclass
from typing import Optional, List
import warnings

try:
    import pycountry
except ImportError:
    pycountry = None
    warnings.warn(
        "pycountry is not installed -- country plausibility checking is "
        "DISABLED. Fake or invalid countries (e.g. 'Wakanda') will NOT be "
        "flagged. Install with: pip install pycountry --break-system-packages",
        RuntimeWarning,
        stacklevel=2,
    )


@dataclass
class Bounds:
    hard_min: float
    hard_max: float
    soft_min: float
    soft_max: float
    unit: str = ""



HEIGHT_CM = Bounds(hard_min=50, hard_max=270, soft_min=140, soft_max=210, unit="cm")

WEIGHT_KG = Bounds(hard_min=20, hard_max=500, soft_min=35, soft_max=250, unit="kg")

BMI = Bounds(hard_min=10, hard_max=100, soft_min=14, soft_max=70, unit="kg/m^2")


MENARCHE_AGE = Bounds(hard_min=5, hard_max=20, soft_min=9, soft_max=17, unit="years")

MENOPAUSE_AGE = Bounds(hard_min=20, hard_max=65, soft_min=38, soft_max=60, unit="years")

FIRST_BIRTH_AGE = Bounds(hard_min=9, hard_max=60, soft_min=14, soft_max=50, unit="years")

CURRENT_AGE = Bounds(hard_min=18, hard_max=110, soft_min=18, soft_max=95, unit="years")


@dataclass
class PlausibilityIssue:
    field: str
    severity: str          # "hard" (must fix) or "soft" (confirm)
    value: float
    message: str


def _check_bounds(field: str, value: Optional[float], b: Bounds) -> Optional[PlausibilityIssue]:
    if value is None:
        return None
    if value < b.hard_min or value > b.hard_max:
        return PlausibilityIssue(
            field, "hard", value,
            f"{field}={value}{b.unit} is outside the physically plausible range "
            f"({b.hard_min}-{b.hard_max}{b.unit}). This looks like an error and "
            f"can't be submitted as-is."
        )
    if value < b.soft_min or value > b.soft_max:
        return PlausibilityIssue(
            field, "soft", value,
            f"{field}={value}{b.unit} is possible but unusual (typical range "
            f"{b.soft_min}-{b.soft_max}{b.unit}). Please confirm this is correct."
        )
    return None


# Map profile attribute names to their bounds for single-field checks.
FIELD_BOUNDS = {
    "height_cm": HEIGHT_CM,
    "weight_kg": WEIGHT_KG,
    "menarche_age": MENARCHE_AGE,
    "menopause_age": MENOPAUSE_AGE,
    "first_live_birth_age": FIRST_BIRTH_AGE,
    "current_age": CURRENT_AGE,
}


COUNTRY_ALIASES = {
    "uk": "United Kingdom", "u.k.": "United Kingdom", "great britain": "United Kingdom",
    "usa": "United States", "u.s.a.": "United States", "u.s.": "United States",
    "america": "United States",
    "south korea": "Korea, Republic of", "north korea": "Korea, Democratic People's Republic of",
    "ivory coast": "Côte d'Ivoire", "russia": "Russian Federation",
    "iran": "Iran, Islamic Republic of", "syria": "Syrian Arab Republic",
    "vietnam": "Viet Nam", "laos": "Lao People's Democratic Republic",
    "bolivia": "Bolivia, Plurinational State of",
    "venezuela": "Venezuela, Bolivarian Republic of",
    "tanzania": "Tanzania, United Republic of", "moldova": "Moldova, Republic of",
}


def check_country_plausibility(value: Optional[str]) -> Optional[PlausibilityIssue]:
    if not value:
        return None
    if pycountry is None:
        return None
    v = value.strip()
    if v.lower() in COUNTRY_ALIASES:
        return None
    try:
        pycountry.countries.lookup(v)
        return None
    except LookupError:
        pass

    suggestion = ""
    try:
        hit = pycountry.countries.search_fuzzy(v)[0]
        suggestion = f" Did you mean '{hit.name}'?"
    except LookupError:
        pass

    return PlausibilityIssue(
        "country", "soft", value,
        f"country='{value}' doesn't match any recognized country name.{suggestion} "
        f"Please confirm this is correct, or provide the correct country."
    )


def check_field_plausibility(field: str, value) -> Optional[PlausibilityIssue]:
    if field == "country":
        return check_country_plausibility(value)
    b = FIELD_BOUNDS.get(field)
    if b is None:
        return None
    return _check_bounds(field, value, b)


def check_missing_age(current_age: Optional[float]) -> Optional[PlausibilityIssue]:
    if current_age is not None:
        return None
    return PlausibilityIssue(
        "dob", "soft", None,
        "No age or date of birth is recorded. BOADICEA needs this to bin "
        "risk correctly -- the result will be less precise without it. "
        "Confirm this is intentional, or provide an age."
    )



def check_cross_field(values: dict) -> List[PlausibilityIssue]:

    issues: List[PlausibilityIssue] = []

    height = values.get("height_cm")
    weight = values.get("weight_kg")
    bmi = values.get("bmi")
    age = values.get("current_age")
    menarche = values.get("menarche_age")
    menopause = values.get("menopause_age")
    first_birth = values.get("first_live_birth_age")
    num_children = values.get("num_children")


    if height and weight and bmi:
        computed = weight / ((height / 100.0) ** 2)
        if abs(computed - bmi) > 1.0:  # allow rounding/binning slack
            issues.append(PlausibilityIssue(
                "bmi", "soft", bmi,
                f"Stated BMI ({bmi}) doesn't match BMI computed from height and "
                f"weight ({computed:.1f}). One of the three may be wrong."
            ))

    if height and weight:
        computed_bmi = weight / ((height / 100.0) ** 2)
        bmi_bounds_issue = _check_bounds("bmi", computed_bmi, BMI)
        if bmi_bounds_issue:
            issues.append(PlausibilityIssue(
                "bmi", bmi_bounds_issue.severity, round(computed_bmi, 1),
                f"Height ({height}cm) and weight ({weight}kg) together give a "
                f"BMI of {computed_bmi:.1f}, which is "
                + ("outside the physically plausible range "
                   f"({BMI.hard_min}-{BMI.hard_max}{BMI.unit})."
                   if bmi_bounds_issue.severity == "hard" else
                   "possible but unusual "
                   f"(typical range {BMI.soft_min}-{BMI.soft_max}{BMI.unit}).")
                + " Please double-check these two values."
            ))

    if age is not None and age >= 18 and height is not None and height < HEIGHT_CM.soft_min:
        issues.append(PlausibilityIssue(
            "height_cm", "soft", height,
            f"Height ({height}cm) is unusually short for an adult (age {age}). "
            f"Please double-check -- this is a common data-entry error."
        ))

    if menarche is not None and first_birth is not None and first_birth <= menarche:
        issues.append(PlausibilityIssue(
            "first_live_birth_age", "soft", first_birth,
            f"Age at first birth ({first_birth}) is not after age at first "
            f"period ({menarche}), which is biologically unexpected."
        ))
    if menarche is not None and menopause is not None and menopause <= menarche:
        issues.append(PlausibilityIssue(
            "menopause_age", "hard", menopause,
            f"Age at menopause ({menopause}) is not after age at first period "
            f"({menarche}). One of these is wrong."
        ))
    if first_birth is not None and age is not None and first_birth > age:
        issues.append(PlausibilityIssue(
            "first_live_birth_age", "hard", first_birth,
            f"Age at first birth ({first_birth}) is greater than current age "
            f"({age}), which is impossible."
        ))
    if menopause is not None and age is not None and menopause > age:
        issues.append(PlausibilityIssue(
            "menopause_age", "hard", menopause,
            f"Age at menopause ({menopause}) is greater than current age ({age})."
        ))

    if num_children is not None and num_children > 0 and first_birth is None:
        issues.append(PlausibilityIssue(
            "first_live_birth_age", "soft", 0,
            "Children were reported but no age at first birth was captured -- "
            "worth asking, since it affects the calculation."
        ))

    return issues


def summarize_issues(issues: List[PlausibilityIssue]) -> dict:
    return {
        "hard": [i for i in issues if i.severity == "hard"],
        "soft": [i for i in issues if i.severity == "soft"],
        "has_blocking": any(i.severity == "hard" for i in issues),
    }