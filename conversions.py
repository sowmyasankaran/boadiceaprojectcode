
from datetime import date
from dataclasses import dataclass, field
from typing import Optional, List



LB_TO_KG = 0.45359237   # exact, international avoirdupois pound
IN_TO_CM = 2.54         # exact, international inch


def lbs_to_kg(lbs: float) -> float:
    return round(lbs * LB_TO_KG, 1)


def kg_to_lbs(kg: float) -> float:
    return round(kg / LB_TO_KG, 1)


def feet_inches_to_cm(feet: float, inches: float = 0.0) -> float:
    return round((feet * 12 + inches) * IN_TO_CM, 1)


def cm_to_feet_inches(cm: float) -> tuple:
    total_inches = cm / IN_TO_CM
    feet = int(total_inches // 12)
    inches = round(total_inches - feet * 12, 1)
    return feet, inches

def compute_bmi(weight_kg: float, height_cm: float) -> float:
    """Standard BMI = kg / m^2. Matches the CanRisk frontend's own calculation."""
    if weight_kg <= 0 or height_cm <= 0:
        raise ValueError(f"Invalid weight/height: {weight_kg}kg, {height_cm}cm")
    height_m = height_cm / 100.0
    return round(weight_kg / (height_m ** 2), 1)

def compute_age_yob(dob: date, as_of: Optional[date] = None) -> tuple:
    """Returns (age, year_of_birth) as of a given date (defaults to today)."""
    as_of = as_of or date.today()
    age = as_of.year - dob.year - ((as_of.month, as_of.day) < (dob.month, dob.day))
    return age, dob.year


DRINK_TYPES = {
    "wine":    {"volume_ml": 175, "abv": 0.12},   # Glass of Wine (175ml)
    "pint":    {"volume_ml": 568, "abv": 0.04},   # Pint of Beer/Lager/Cider (568ml)
    "bottle_beer": {"volume_ml": 330, "abv": 0.05},  # Bottle of Beer (330ml)
    "alcopop": {"volume_ml": 275, "abv": 0.05},   # Alcoholic Pop Drink (275ml)
    "spirit_shot": {"volume_ml": 25, "abv": 0.40},  # Shot of spirits (25ml)
}


FREQUENCY_PER_DAY = {
    "daily": 1.0,
    "weekly": 1.0 / 7.0,
    "monthly": 1.0 / 30.0,
    "never": 0.0,
}

ETHANOL_DENSITY_G_PER_ML = 0.789


def drink_grams_per_unit(drink_type: str) -> float:
    if drink_type not in DRINK_TYPES:
        raise ValueError(f"Unknown drink type: {drink_type}. Valid: {list(DRINK_TYPES)}")
    d = DRINK_TYPES[drink_type]
    return d["volume_ml"] * d["abv"] * ETHANOL_DENSITY_G_PER_ML


def alcohol_table_to_grams_per_day(drinks: List[dict]) -> float:
    """
    drinks: list of {"type": "wine"|"pint"|"bottle_beer"|"alcopop"|"spirit_shot",
                      "frequency": "daily"|"weekly"|"monthly"|"never",
                      "amount": float}  # number of drinks per occurrence

    Returns total grams of alcohol per day, matching the frontend's own
    "grams of alcohol per day" figure (the ##alcohol= header value).

    NOTE: the exact unit CanRisk's ##alcohol= header expects is inferred from
    the docs' own example (##alcohol=22.4) and our own test call's round-trip
    (we sent alcohol=5 and got back the "5-<15" bin) -- both are consistent
    with grams/day, but this hasn't been independently confirmed against the
    v4 spec's explicit unit statement. Worth a quick check before relying on
    it for real cases.
    """
    total = 0.0
    for drink in drinks:
        freq = drink["frequency"].lower()
        if freq not in FREQUENCY_PER_DAY:
            raise ValueError(f"Unknown frequency: {freq}. Valid: {list(FREQUENCY_PER_DAY)}")
        grams_per_unit = drink_grams_per_unit(drink["type"])
        total += grams_per_unit * drink["amount"] * FREQUENCY_PER_DAY[freq]
    return round(total, 1)


@dataclass
class PersonalLifestyleProfile:
    sex: Optional[str] = None                # "M" or "F"
    country: Optional[str] = None            # e.g. "UK" -- also drives mut_freq/cancer_rates
    ethnic_group: Optional[str] = None       # UK ONS category; TAG SYNTAX UNCONFIRMED (see note below)
    dob: Optional[date] = None
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    drinks: List[dict] = field(default_factory=list)  # raw drink-table entries

    # --- derived fields, computed once inputs are present ---
    def bmi(self) -> Optional[float]:
        if self.height_cm and self.weight_kg:
            return compute_bmi(self.weight_kg, self.height_cm)
        return None

    def age_yob(self) -> Optional[tuple]:
        if self.dob:
            return compute_age_yob(self.dob)
        return None

    def alcohol_grams_per_day(self) -> Optional[float]:
        if self.drinks:
            return alcohol_table_to_grams_per_day(self.drinks)
        return None

    REQUIRED_FIELDS = ["sex", "country", "dob", "height_cm", "weight_kg"]

    def missing_required_fields(self) -> List[str]:
        return [f for f in self.REQUIRED_FIELDS if getattr(self, f) is None]

    def is_complete(self) -> bool:
        return len(self.missing_required_fields()) == 0


def profile_to_header_lines(profile: PersonalLifestyleProfile) -> dict:
    """
    Returns the CanRisk header lines this slice is responsible for, plus the
    proband row overrides (sex, age, yob) and the request-level params
    (mut_freq/cancer_rates) that depend on country.

    ethnic_group is NOT included here -- confirmed as a real, model-affecting
    field (referenced directly in BOADICEA's own web-service source), but the
    exact "##" header tag syntax isn't confirmed from public docs. Stored on
    the profile for now; wire it in once the tag name is confirmed (check the
    full v4 spec page, or ask your mentor/Aryaman).
    """
    if not profile.is_complete():
        raise ValueError(f"Profile incomplete: missing {profile.missing_required_fields()}")

    age, yob = profile.age_yob()
    bmi = profile.bmi()
    alcohol = profile.alcohol_grams_per_day()
    if alcohol is None:
        alcohol = 0.0  # no drinks reported -> treat as "never"

    country_to_params = {
        "UK": {"mut_freq": "UK", "cancer_rates": "UK"},
        # extend as needed for other countries CanRisk supports
    }
    request_params = country_to_params.get(
        profile.country.upper(), {"mut_freq": "UK", "cancer_rates": "UK"}
    )

    header_lines = {
        "height": round(profile.height_cm, 1),
        "BMI": bmi,
        "alcohol": alcohol,
    }

    proband_overrides = {
        "sex": profile.sex,
        "age": age,
        "yob": yob,
    }

    return {
        "header_lines": header_lines,
        "proband_overrides": proband_overrides,
        "request_params": request_params,
        "unresolved": {
            "ethnic_group": profile.ethnic_group,  # not yet wired into header syntax
        },
    }


def teach_back_summary(profile: PersonalLifestyleProfile) -> str:
    """Plain-language restatement of what was collected, for user confirmation
    before submission -- required per the project's safety design (see the
    project plan's Week 3 teach-back step)."""
    lines = []
    if profile.sex:
        lines.append(f"- Sex: {'Female' if profile.sex == 'F' else 'Male'}")
    if profile.country:
        lines.append(f"- Country of residence: {profile.country}")
    if profile.ethnic_group:
        lines.append(f"- Ethnic group: {profile.ethnic_group}")
    if profile.dob:
        age, yob = profile.age_yob()
        lines.append(f"- Date of birth: {profile.dob.isoformat()} (age {age}, born {yob})")
    if profile.height_cm and profile.weight_kg:
        lines.append(
            f"- Height: {profile.height_cm}cm, Weight: {profile.weight_kg}kg "
            f"(BMI: {profile.bmi()})"
        )
    if profile.drinks:
        lines.append(f"- Alcohol: {profile.alcohol_grams_per_day()}g/day (from {len(profile.drinks)} drink type(s) reported)")
    else:
        lines.append("- Alcohol: none reported")
    return "Here's what I have so far:\n" + "\n".join(lines)