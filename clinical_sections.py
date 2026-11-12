
from typing import Optional, List
from schema import Section, Field, _validate_drinks



YNU = ["Y", "N", "U"]   # Yes / No / Unknown (matches the form's radio options)
YN = ["Y", "N"]         # Yes / No toggle


def _to_age(value) -> int:
    """Coerce a diagnosis/event age to a plausible integer; raise on nonsense
    so it's rejected and re-asked rather than stored (same pattern as
    schema._to_age)."""
    a = int(round(float(value)))
    if a < 0 or a > 130:
        raise ValueError(f"implausible age: {value}")
    return a


def _to_count(value) -> int:
    """Coerce a count (e.g. number of children) to a non-negative int; reject
    absurd values so the checklist re-asks."""
    n = int(round(float(value)))
    if n < 0 or n > 30:
        raise ValueError(f"implausible count: {value}")
    return n


def _validate_children(value) -> list:
    """Keep only well-formed child entries: {year_of_birth:int, sex:'M'|'F'}.
    Drops malformed rows so downstream math never hits bad types. An empty
    list is valid (means none captured yet)."""
    cleaned = []
    for c in (value or []):
        if not isinstance(c, dict):
            continue
        yob, sex = c.get("year_of_birth"), c.get("sex")
        if sex not in ("M", "F"):
            continue
        try:
            cleaned.append({"year_of_birth": int(yob), "sex": sex})
        except (TypeError, ValueError):
            continue
    return cleaned


class WomensHealthSection(Section):
    name = "womens_health"

    ask_one_at_a_time = True

    fields = [

        Field("menarche_age",
              "Age at first period (menarche), if the patient states a "
              "specific age. If above 18 or below 8, record the closest age.",
              required=False, json_type="integer",
              plausibility_key="menarche_age", coerce=_to_age),
        Field("menarche_unknown",
              "True ONLY if the patient explicitly says they don't know, "
              "aren't sure, or prefer not to say their age at first period. "
              "Do NOT set this if they give an age.",
              required=False, json_type="boolean"),

        # Oral contraceptive pill
        Field("oc_use", "Has the patient ever taken the oral contraceptive "
              "pill? Y/N/U.", required=True, enum=YNU),
        Field("oc_years", "Total time on the oral contraceptive pill, as one "
              "of the form's bands.", required=False,
              enum=["<1", "1-4", "5-9", "10-14", "15+"]),
        Field("oc_last_two_years",
              "Has the patient taken the pill in the last two years? Y/N.",
              required=False, enum=YN),

        # Menopause
        Field("menopause", "Have the patient's periods stopped completely "
              "(>=6 months, not pregnant / not on hormonal contraception)? "
              "Y/N/U.", required=True, enum=YNU),
        Field("menopause_age_band",
              "Age band when periods stopped completely.", required=False,
              enum=["<40", "40-44", "45-49", "50-54", "54+"]),

        # HRT / MHT
        Field("hrt_use", "Has the patient ever used hormone replacement "
              "therapy (HRT/MHT) for menopause? Y/N/U.", required=True,
              enum=YNU),
        Field("hrt_years", "Total years of HRT use as an integer (0 if less "
              "than a year in total).", required=False, json_type="integer",
              coerce=_to_count),
        Field("hrt_last_five_years",
              "Has the patient used HRT in the last five years? Y/N.",
              required=False, enum=YN),
        Field("hrt_type", "Type of HRT used, if known.", required=False,
              enum=["oestrogen_only", "combined", "other", "unknown"]),
    ]

    def missing_required(self) -> list:
        missing = super().missing_required()

        if self.values.get("menarche_age") is None and not self.values.get("menarche_unknown"):
            missing.append("age_at_first_period")


        if self.values.get("oc_use") == "Y":
            if self.values.get("oc_years") is None:
                missing.append("oc_years")
            if self.values.get("oc_last_two_years") is None:
                missing.append("oc_last_two_years")
        if self.values.get("menopause") == "Y":
            if self.values.get("menopause_age_band") is None:
                missing.append("menopause_age_band")
        if self.values.get("hrt_use") == "Y":
            if self.values.get("hrt_years") is None:
                missing.append("hrt_years")
            if self.values.get("hrt_last_five_years") is None:
                missing.append("hrt_last_five_years")
            if self.values.get("hrt_type") is None:
                missing.append("hrt_type")
        return missing


    synthetic_field_descriptions = {
        "age_at_first_period": (
            "The age at which the patient had their first period "
            "(menarche). If they don't know or prefer not to say, that's a "
            "valid answer too -- record menarche_unknown=true in that case, "
            "do NOT ask them to say a specific word like 'unknown'."
        ),
    }

    def plausibility_values(self) -> dict:

        return {"menarche_age": self.values.get("menarche_age")}

    def to_canrisk(self) -> dict:
        if not self.is_complete():
            raise ValueError(f"Section incomplete: missing {self.missing_required()}")
        return {
            "header_lines": {},   # TODO(canrisk-tag): wire once tags confirmed
            "unresolved": {
                "menarche_age": self.values.get("menarche_age"),
                "oc_use": self.values.get("oc_use"),
                "oc_years": self.values.get("oc_years"),
                "oc_last_two_years": self.values.get("oc_last_two_years"),
                "menopause": self.values.get("menopause"),
                "menopause_age_band": self.values.get("menopause_age_band"),
                "hrt_use": self.values.get("hrt_use"),
                "hrt_years": self.values.get("hrt_years"),
                "hrt_last_five_years": self.values.get("hrt_last_five_years"),
                "hrt_type": self.values.get("hrt_type"),
            },
        }

    def teach_back(self) -> str:
        v = self.values
        lines = []
        if v.get("menarche_age") is not None:
            lines.append(f"- Age at first period: {v['menarche_age']}")
        elif v.get("menarche_unknown"):
            lines.append("- Age at first period: not known")
        if v.get("oc_use"):
            oc = {"Y": "Yes", "N": "No", "U": "Unknown"}[v["oc_use"]]
            extra = f" ({v['oc_years']} yrs)" if v.get("oc_years") else ""
            lines.append(f"- Ever taken oral contraceptive pill: {oc}{extra}")
        if v.get("menopause"):
            mp = {"Y": "Yes", "N": "No", "U": "Unknown"}[v["menopause"]]
            extra = f" (stopped {v['menopause_age_band']})" if v.get("menopause_age_band") else ""
            lines.append(f"- Periods stopped completely: {mp}{extra}")
        if v.get("hrt_use"):
            hrt = {"Y": "Yes", "N": "No", "U": "Unknown"}[v["hrt_use"]]
            lines.append(f"- Ever used HRT: {hrt}")
        return "Here's what I have for women's health:\n" + "\n".join(lines)

class ChildrenSection(Section):
    name = "children"

    fields = [
        Field("has_children", "Has the patient ever had any children? Y/N/U.",
              required=True, enum=YNU),
        Field("num_children", "How many children the patient has had.",
              required=False, json_type="integer", coerce=_to_count),
        Field("children",
              "One entry per child: {year_of_birth: 4-digit year, sex: 'M' or "
              "'F'}. Used to derive age at first birth (a model input).",
              required=False, json_type="array", coerce=_validate_children,
              items={
                  "type": "object",
                  "properties": {
                      "year_of_birth": {"type": "integer"},
                      "sex": {"type": "string", "enum": ["M", "F"]},
                  },
                  "required": ["year_of_birth", "sex"],
              }),
    ]

    def missing_required(self) -> list:
        missing = super().missing_required()

        if self.values.get("has_children") == "Y":
            n = self.values.get("num_children")
            if n is None:
                missing.append("num_children")
            else:
                kids = self.values.get("children") or []
                if len(kids) < n:
                    missing.append("children_details")
        return missing

    def age_at_first_birth(self, mother_dob_year: Optional[int]) -> Optional[int]:

        kids = self.values.get("children") or []
        if not kids or mother_dob_year is None:
            return None
        earliest = min(c["year_of_birth"] for c in kids)
        return earliest - mother_dob_year

    def plausibility_values(self) -> dict:

        return {}

    def to_canrisk(self) -> dict:
        if not self.is_complete():
            raise ValueError(f"Section incomplete: missing {self.missing_required()}")
        return {
            "header_lines": {},   # parity/first-birth handled at pedigree assembly
            "unresolved": {
                "has_children": self.values.get("has_children"),
                "num_children": self.values.get("num_children"),
                "children": self.values.get("children"),
            },
        }

    def teach_back(self) -> str:
        v = self.values
        if v.get("has_children") != "Y":
            state = {"N": "No children", "U": "Unknown"}.get(v.get("has_children"), "Not answered")
            return f"Here's what I have for children:\n- {state}"
        kids = v.get("children") or []
        lines = [f"- Number of children: {v.get('num_children')}"]
        for i, c in enumerate(kids, 1):
            lines.append(f"  - Child {i}: born {c['year_of_birth']}, "
                         f"{'female' if c['sex']=='F' else 'male'}")
        return "Here's what I have for children:\n" + "\n".join(lines)


class BreastScreeningSection(Section):
    name = "breast_screening"
    ask_one_at_a_time = True

    fields = [
        Field("had_mammogram", "Has the patient ever had a mammogram? Y/N/U.",
              required=True, enum=YNU),
        Field("told_density", "Was the patient told about their breast "
              "density? Y/N.", required=False, enum=YN),
        Field("density_method", "How breast density was measured.",
              required=False, enum=["BI-RADS", "Volpara", "Stratus", "Unknown"]),
        Field("birads_category", "BI-RADS density category (a/b/c/d), if the "
              "measure was BI-RADS.", required=False,
              enum=["a", "b", "c", "d"]),
    ]

    def to_canrisk(self) -> dict:
        if not self.is_complete():
            raise ValueError(f"Section incomplete: missing {self.missing_required()}")
        return {
            "header_lines": {},   # TODO(canrisk-tag): mammographic density tag
            "unresolved": {
                "had_mammogram": self.values.get("had_mammogram"),
                "told_density": self.values.get("told_density"),
                "density_method": self.values.get("density_method"),
                "birads_category": self.values.get("birads_category"),
            },
        }

    def teach_back(self) -> str:
        v = self.values
        mg = {"Y": "Yes", "N": "No", "U": "Unknown"}.get(v.get("had_mammogram"), "Not answered")
        lines = [f"- Ever had a mammogram: {mg}"]
        if v.get("told_density") == "Y":
            method = v.get("density_method") or "unknown method"
            cat = f", category {v['birads_category']}" if v.get("birads_category") else ""
            lines.append(f"- Breast density: measured by {method}{cat}")
        return "Here's what I have for breast screening:\n" + "\n".join(lines)


class MedicalHistorySection(Section):
    name = "medical_history"

    ask_one_at_a_time = True

    fields = [
        Field("endometriosis", "Ever diagnosed with endometriosis? Y/N/U.",
              required=True, enum=YNU),
        Field("tubal_ligation", "Has the patient had their tubes tied (tubal "
              "ligation)? Y/N/U.", required=True, enum=YNU),

        Field("oophorectomy", "Both ovaries removed (bilateral "
              "oophorectomy)? Y/N.", required=True, enum=YN),
        Field("mastectomy", "Both breasts removed (bilateral mastectomy)? "
              "Y/N.", required=True, enum=YN),


        Field("breast_cancer", "Has the patient had breast cancer in ONE "
              "breast (exclude DCIS)? Y/N.", required=True, enum=YN),
        Field("breast_cancer_age", "Age at first breast cancer diagnosis.",
              required=False, json_type="integer", coerce=_to_age),
        Field("contralateral_breast_cancer",
              "Has the patient had cancer in the SECOND breast "
              "(contralateral, its own primary; exclude DCIS)? Y/N.",
              required=False, enum=YN),
        Field("contralateral_breast_cancer_age",
              "Age at contralateral breast cancer diagnosis.",
              required=False, json_type="integer", coerce=_to_age),
        Field("ovarian_cancer", "Has the patient had ovarian cancer? Y/N.",
              required=True, enum=YN),
        Field("ovarian_cancer_age", "Age at ovarian cancer diagnosis.",
              required=False, json_type="integer", coerce=_to_age),
        Field("pancreatic_cancer", "Has the patient had pancreatic cancer? "
              "Y/N.", required=True, enum=YN),
        Field("pancreatic_cancer_age", "Age at pancreatic cancer diagnosis.",
              required=False, json_type="integer", coerce=_to_age),
    ]

    def plausibility_values(self) -> dict:

        return {
            "breast_cancer_age": self.values.get("breast_cancer_age"),
            "ovarian_cancer_age": self.values.get("ovarian_cancer_age"),
        }

    def to_canrisk(self) -> dict:
        if not self.is_complete():
            raise ValueError(f"Section incomplete: missing {self.missing_required()}")
        return {
            "header_lines": {},   # proband diagnoses map onto the pedigree row
            "unresolved": {
                "endometriosis": self.values.get("endometriosis"),
                "tubal_ligation": self.values.get("tubal_ligation"),
                "oophorectomy": self.values.get("oophorectomy"),
                "mastectomy": self.values.get("mastectomy"),
                "breast_cancer": self.values.get("breast_cancer"),
                "breast_cancer_age": self.values.get("breast_cancer_age"),
                "contralateral_breast_cancer": self.values.get("contralateral_breast_cancer"),
                "contralateral_breast_cancer_age": self.values.get("contralateral_breast_cancer_age"),
                "ovarian_cancer": self.values.get("ovarian_cancer"),
                "ovarian_cancer_age": self.values.get("ovarian_cancer_age"),
                "pancreatic_cancer": self.values.get("pancreatic_cancer"),
                "pancreatic_cancer_age": self.values.get("pancreatic_cancer_age"),
            },
        }

    def teach_back(self) -> str:
        v = self.values
        lines = []
        def ynu(k):
            return {"Y": "Yes", "N": "No", "U": "Unknown"}.get(v.get(k), "?")
        lines.append(f"- Endometriosis: {ynu('endometriosis')}")
        lines.append(f"- Tubal ligation: {ynu('tubal_ligation')}")
        lines.append(f"- Both ovaries removed: {ynu('oophorectomy')}")
        lines.append(f"- Both breasts removed: {ynu('mastectomy')}")
        for label, key, agekey in [
            ("Breast cancer", "breast_cancer", "breast_cancer_age"),
            ("Contralateral breast cancer", "contralateral_breast_cancer",
             "contralateral_breast_cancer_age"),
            ("Ovarian cancer", "ovarian_cancer", "ovarian_cancer_age"),
            ("Pancreatic cancer", "pancreatic_cancer", "pancreatic_cancer_age"),
        ]:
            if v.get(key) == "Y":
                age = v.get(agekey)
                lines.append(f"- {label}: yes" + (f" (age {age})" if age else ""))
        return "Here's what I have for medical history:\n" + "\n".join(lines)




class AlcoholSection(Section):
    name = "alcohol"
    ask_one_at_a_time = True

    fields = [
        Field("drinks_alcohol", "Does the patient drink alcohol? Y/N/U.",
              required=True, enum=YNU),
        Field("drinks",
              "The drink table: list of {type in [wine, pint, bottle_beer, "
              "alcopop, spirit_shot], frequency in [daily, weekly, monthly, "
              "never], amount: number of drinks per occasion}. Map what the "
              "patient describes onto the closest drink type: a glass of wine "
              "-> wine; a pint of beer/lager/cider -> pint; a bottle/can of "
              "beer -> bottle_beer; an alcopop/hard seltzer -> alcopop; a shot "
              "of gin/rum/vodka/whisky/tequila -> spirit_shot.",
              required=False, json_type="array", coerce=_validate_drinks,
              items={
                  "type": "object",
                  "properties": {
                      "type": {"type": "string",
                               "enum": ["alcopop", "bottle_beer", "pint",
                                        "spirit_shot", "wine"]},
                      "frequency": {"type": "string",
                                    "enum": ["daily", "monthly", "never", "weekly"]},
                      "amount": {"type": "number"},
                  },
                  "required": ["type", "frequency", "amount"],
              }),
    ]

    def missing_required(self) -> list:
        missing = super().missing_required()
        # If they drink, we need the table -- otherwise there's nothing to
        # convert to grams/day and the risk factor is lost.
        if self.values.get("drinks_alcohol") == "Y" and not self.values.get("drinks"):
            missing.append("drink_details")
        return missing

    def alcohol_grams_per_day(self) -> float:
        from conversions import alcohol_table_to_grams_per_day
        drinks = self.values.get("drinks") or []
        return alcohol_table_to_grams_per_day(drinks) if drinks else 0.0

    def to_canrisk(self) -> dict:
        if not self.is_complete():
            raise ValueError(f"Section incomplete: missing {self.missing_required()}")
        return {"header_lines": {}, "unresolved": {
            "drinks_alcohol": self.values.get("drinks_alcohol"),
            "drinks": self.values.get("drinks"),
        }}

    def teach_back(self) -> str:
        v = self.values
        if v.get("drinks_alcohol") == "N":
            return "Here's what I have for alcohol:\n- Does not drink alcohol"
        if v.get("drinks_alcohol") == "U":
            return "Here's what I have for alcohol:\n- Alcohol use: unknown"
        lines = []
        for d in (v.get("drinks") or []):
            label = {"wine": "glasses of wine", "pint": "pints of beer/lager/cider",
                     "bottle_beer": "bottles of beer", "alcopop": "alcopops",
                     "spirit_shot": "shots of spirits"}[d["type"]]
            lines.append(f"- {d['amount']:g} {label}, {d['frequency']}")
        lines.append(f"- Total: {self.alcohol_grams_per_day()}g of alcohol per day")
        return "Here's what I have for alcohol:\n" + "\n".join(lines)