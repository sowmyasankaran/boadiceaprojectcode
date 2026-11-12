import os
from typing import Optional, List
from schema import Section, Field


YN = ["Y", "N"]
YNU = ["Y", "N", "U"]

GENETIC_ANCESTRY = [
    "European", "East Asian", "South Asian", "African", "Hispanic",
    "Ashkenazi Jewish", "Other", "Unknown",
]

BREAST_PRS_REFERENCE_FILES = {
    "BCAC 313": "BCAC_313_PRS.prs",
    "BRIDGES 306": "BRIDGES_306_PRS.prs",
    "PERSPECTIVE 295": "PERSPECTIVE_295_PRS.prs",
    "PRISM 289": "PRISM_289_PRS.prs",
}
OVARIAN_PRS_REFERENCE_FILES = {
    "OCAC 36": "OCAC_36_PRS.prs",
}


def _to_float(value) -> float:
    return float(value)


def _existing_path(value) -> str:

    p = str(value).strip().strip(chr(34)).strip(chr(39))
    if not os.path.isfile(p):
        raise ValueError(f"no file at {p}")
    return p


class PRSSection(Section):
    name = "prs"
    ask_one_at_a_time = True

    fields = [
        Field("prs_run", "Has a SNP array / Polygenic Risk Score (PRS) been "
              "run for this patient? Y/N/U.", required=True, enum=YNU),
        Field("genetic_ancestry", "Genetic ancestry selected for PRS "
              "calibration -- distinct from the ethnic_group / country "
              "fields elsewhere.", required=False, enum=GENETIC_ANCESTRY),
        Field("input_mode",
              "How the PRS will be provided. Ask the patient directly: do "
              "they already have PRS alpha/z-score numbers ('values'), or do "
              "they have a VCF genotype file ('vcf')?",
              required=False, enum=["vcf", "values"]),

        Field("breast_prs_alpha", "Breast-cancer PRS alpha value.",
              required=False, json_type="number", coerce=_to_float),
        Field("breast_prs_zscore", "Breast-cancer PRS z-score.",
              required=False, json_type="number", coerce=_to_float),
        Field("ovarian_prs_alpha", "Ovarian-cancer PRS alpha value.",
              required=False, json_type="number", coerce=_to_float),
        Field("ovarian_prs_zscore", "Ovarian-cancer PRS z-score.",
              required=False, json_type="number", coerce=_to_float),

        Field("breast_panel", "Reference panel for the breast-cancer PRS.",
              required=False, enum=list(BREAST_PRS_REFERENCE_FILES)),
        Field("vcf_file_path", "Local file path to the patient's VCF file.",
              required=False, coerce=_existing_path),
        Field("vcf_sample_name", "Sample name within the VCF file.",
              required=False),
    ]

    def missing_required(self) -> list:
        missing = super().missing_required()
        if self.values.get("prs_run") != "Y":
            return missing
        v = self.values
        if v.get("genetic_ancestry") is None:
            missing.append("genetic_ancestry")
        mode = v.get("input_mode")
        if mode is None:
            missing.append("prs_input_mode")
        elif mode == "values":
            if v.get("breast_prs_zscore") is None and v.get("ovarian_prs_zscore") is None:
                missing.append("prs_values")
        elif mode == "vcf":
            if v.get("breast_panel") is None:
                missing.append("breast_panel")
            if v.get("vcf_file_path") is None:
                missing.append("vcf_file_path")
            if v.get("vcf_sample_name") is None:
                missing.append("vcf_sample_name")
        return missing

    synthetic_field_descriptions = {
        "genetic_ancestry": "The genetic ancestry group to use for PRS "
            "calibration. Options: " + ", ".join(GENETIC_ANCESTRY) + ".",
        "prs_input_mode": "Whether the patient already has PRS alpha/"
            "z-score NUMBERS to give directly, or has a VCF genotype FILE "
            "instead. Ask which one, plainly.",
        "prs_values": "The breast-cancer PRS alpha and z-score values "
            "(ovarian too, if available).",
        "breast_panel": "Which reference panel was used for the breast PRS "
            "(options: " + ", ".join(BREAST_PRS_REFERENCE_FILES) + ").",
        "vcf_file_path": "The local file path to the patient's VCF file.",
        "vcf_sample_name": "The sample name inside the VCF file.",
    }

    def to_canrisk(self) -> dict:
        if not self.is_complete():
            raise ValueError(f"Section incomplete: missing {self.missing_required()}")
        v = self.values
        unresolved = {"prs_run": v.get("prs_run"),
                      "genetic_ancestry": v.get("genetic_ancestry"),
                      "input_mode": v.get("input_mode")}
        if v.get("input_mode") == "values":
            unresolved["prs_values"] = {
                "breast_alpha": v.get("breast_prs_alpha"),
                "breast_zscore": v.get("breast_prs_zscore"),
                "ovarian_alpha": v.get("ovarian_prs_alpha"),
                "ovarian_zscore": v.get("ovarian_prs_zscore"),
            }
        elif v.get("input_mode") == "vcf":
            unresolved["vcf"] = {
                "breast_panel": v.get("breast_panel"),
                "reference_file": BREAST_PRS_REFERENCE_FILES.get(v.get("breast_panel")),
                "sample_name": v.get("vcf_sample_name"),
                "file_path": v.get("vcf_file_path"),
            }
        return {"header_lines": {}, "unresolved": unresolved}

    def teach_back(self) -> str:
        v = self.values
        if v.get("prs_run") != "Y":
            state = {"N": "No", "U": "Unknown"}.get(v.get("prs_run"), "Not answered")
            return "Here's what I have for polygenic risk score:\n- PRS run: " + state
        lines = ["- PRS run: Yes",
                 "- Genetic ancestry: " + (v.get("genetic_ancestry") or "not given")]
        if v.get("input_mode") == "values":
            lines.append(f"- Breast PRS: alpha={v.get('breast_prs_alpha')}, "
                         f"z-score={v.get('breast_prs_zscore')}")
            if v.get("ovarian_prs_zscore") is not None:
                lines.append(f"- Ovarian PRS: alpha={v.get('ovarian_prs_alpha')}, "
                             f"z-score={v.get('ovarian_prs_zscore')}")
        elif v.get("input_mode") == "vcf":
            lines.append(f"- Breast reference panel: {v.get('breast_panel')}")
            lines.append(f"- VCF file: {v.get('vcf_file_path')} "
                         f"(sample: {v.get('vcf_sample_name')})")
        return "Here's what I have for polygenic risk score:\n" + "\n".join(lines)


class AdvancedOptionsSection(Section):
    name = "advanced_options"

    fields = [
        # Not patient-facing questions -- configuration keyed off population.
        Field("population_preset", "Population preset driving pathogenic-"
              "variant frequencies, genetic-testing sensitivities, and cancer "
              "incidence rates (e.g. 'UK'). Defaults to match the patient's "
              "country; rarely changed in a patient intake.",
              required=True, enum=["UK", "Other"]),
        Field("use_custom_frequencies",
              "Whether custom mutation frequencies/sensitivities were supplied "
              "instead of the preset defaults. Y/N.", required=False, enum=YN),
    ]

    SUPPORTED_POPULATIONS = {"UK"}

    @classmethod
    def with_country_default(cls, country: Optional[str]) -> "AdvancedOptionsSection":
        """Build with the population preset pre-set from the Personal section's
        country, so this never needs to be asked in the common case."""
        s = cls()
        c = (country or "").upper()
        preset = "UK" if c in ("UK", "GB", "UNITED KINGDOM", "GREAT BRITAIN") else "UK"

        s.set_value("population_preset", preset)
        s.set_value("use_custom_frequencies", "N")
        return s

    def to_canrisk(self) -> dict:
        if not self.is_complete():
            raise ValueError(f"Section incomplete: missing {self.missing_required()}")
        return {
            "header_lines": {},
            "request_params": {
                "mut_freq": self.values.get("population_preset", "UK"),
                "cancer_rates": self.values.get("population_preset", "UK"),
            },
            "unresolved": {
                "use_custom_frequencies": self.values.get("use_custom_frequencies"),
            },
        }

    def teach_back(self) -> str:
        v = self.values
        return ("Here's the model configuration:\n"
                f"- Population preset (variant freqs & incidence): "
                f"{v.get('population_preset')}\n"
                f"- Custom frequencies: {v.get('use_custom_frequencies') or 'No'}")