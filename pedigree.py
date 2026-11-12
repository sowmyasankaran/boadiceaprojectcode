

from dataclasses import dataclass, field as dc_field
from typing import Optional, List, Dict, Any


RELATIONSHIPS = [
    "self", "mother", "father",
    "sister", "brother",
    "daughter", "son",
    "maternal_grandmother", "maternal_grandfather",
    "paternal_grandmother", "paternal_grandfather",
    "maternal_aunt", "maternal_uncle",
    "paternal_aunt", "paternal_uncle",
    "half_sister", "half_brother",
    "niece", "nephew", "cousin", "other",
]


_SEX_BY_REL = {
    "mother": "F", "father": "M", "sister": "F", "brother": "M",
    "daughter": "F", "son": "M",
    "maternal_grandmother": "F", "maternal_grandfather": "M",
    "paternal_grandmother": "F", "paternal_grandfather": "M",
    "maternal_aunt": "F", "maternal_uncle": "M",
    "paternal_aunt": "F", "paternal_uncle": "M",
    "half_sister": "F", "half_brother": "M",
    "niece": "F", "nephew": "M",
}

CANCER_TYPES = ["breast", "contralateral_breast", "ovarian", "pancreatic"]

GENE_PANEL = ["BRCA1", "BRCA2", "PALB2", "CHEK2", "ATM", "BARD1",
              "RAD51C", "RAD51D", "BRIP1", "HOXB13"]


@dataclass
class Diagnosis:
    cancer_type: str          # one of CANCER_TYPES
    age: Optional[int] = None


@dataclass
class Pathology:

    er: Optional[str] = None     # "P"/"N", or None for unspecified
    pr: Optional[str] = None
    her2: Optional[str] = None
    ck14: Optional[str] = None
    ck56: Optional[str] = None

    def to_column(self) -> str:
        vals = [self.er, self.pr, self.her2, self.ck14, self.ck56]
        return ":".join(v if v in ("P", "N") else "0" for v in vals)

    def is_any_known(self) -> bool:
        return any(v in ("P", "N") for v in
                   (self.er, self.pr, self.her2, self.ck14, self.ck56))


@dataclass
class GeneTest:
    gene: str                 # one of GENE_PANEL
    result: str               # see above -- codes depend on `gene`


@dataclass
class Individual:
    id: str
    rel_to_proband: str
    sex: Optional[str] = None                    # "M"/"F"
    name: Optional[str] = None
    alive: Optional[bool] = None
    age: Optional[int] = None                    # current age if alive
    age_at_death: Optional[int] = None           # if deceased
    year_of_birth: Optional[int] = None
    is_proband: bool = False
    diagnoses: List[Diagnosis] = dc_field(default_factory=list)
    gene_tests: List[GeneTest] = dc_field(default_factory=list)
    pathology: Optional[Pathology] = None
    # graph edges -- filled during structural build
    father_id: Optional[str] = None
    mother_id: Optional[str] = None
    linked_via_id: Optional[str] = None      # niece/nephew/cousin: the id of
                                              # the sibling (niece/nephew) or
                                              # aunt/uncle (cousin) who is
                                              # this person's parent
    shared_parent_side: Optional[str] = None  # half_sister/half_brother:
                                               # "mother" or "father" -- which
                                               # of the proband's parents is
                                               # shared
    is_placeholder: bool = False

    def __post_init__(self):
        if self.sex is None:
            self.sex = _SEX_BY_REL.get(self.rel_to_proband)

    def has_any_cancer(self) -> bool:
        return len(self.diagnoses) > 0

    def resolved_year_of_birth(self) -> Optional[int]:
        if self.year_of_birth is not None:
            return self.year_of_birth
        if self.age is not None:
            from datetime import date
            return date.today().year - self.age
        return None

    def summary_line(self) -> str:
        who = self.name or self.rel_to_proband.replace("_", " ")
        bits = [who]
        if self.sex:
            bits.append({"M": "male", "F": "female"}[self.sex])
        if self.alive is False:
            bits.append(f"deceased" + (f" at {self.age_at_death}" if self.age_at_death else ""))
        elif self.age is not None:
            bits.append(f"age {self.age}")
        if self.diagnoses:
            dxs = ", ".join(
                d.cancer_type.replace("_", " ") + (f" at {d.age}" if d.age else "")
                for d in self.diagnoses
            )
            bits.append(f"cancers: {dxs}")
        pos_genes = [g.gene for g in self.gene_tests if g.result == "P"]
        if pos_genes:
            bits.append("pathogenic: " + ", ".join(pos_genes))
        if self.pathology and self.pathology.is_any_known():
            p = self.pathology
            markers = [(name, val) for name, val in
                      [("ER", p.er), ("PR", p.pr), ("HER2", p.her2),
                       ("CK14", p.ck14), ("CK56", p.ck56)] if val in ("P", "N")]
            bits.append("pathology: " + ", ".join(f"{n}{'+' if v=='P' else '-'}"
                                                   for n, v in markers))
        return " -- ".join(bits)


class Pedigree:
    def __init__(self, ashkenazi: Optional[bool] = None):
        self.individuals: Dict[str, Individual] = {}
        self.ashkenazi = ashkenazi         # from the AJ-ancestry question
        self._counter = 0
        # the proband is always individual 0
        self.proband = self._add(Individual(id="I0", rel_to_proband="self",
                                             is_proband=True))

    def _new_id(self) -> str:
        self._counter += 1
        return f"I{self._counter}"

    def _add(self, ind: Individual) -> Individual:
        self.individuals[ind.id] = ind
        return ind

    UNIQUE_RELATIONSHIPS = {
        "self", "mother", "father",
        "maternal_grandmother", "maternal_grandfather",
        "paternal_grandmother", "paternal_grandfather",
    }

    def find_relative(self, rel_to_proband: str,
                       name: Optional[str] = None) -> Optional[Individual]:

        matches = self.get_by_rel(rel_to_proband)
        if not matches:
            return None
        if rel_to_proband in self.UNIQUE_RELATIONSHIPS:
            return matches[0]
        if name:
            for m in matches:
                if m.name and m.name.strip().lower() == name.strip().lower():
                    return m
        return None

    def upsert_relative(self, rel_to_proband: str, **attrs) -> Individual:

        if rel_to_proband not in RELATIONSHIPS:
            rel_to_proband = "other"
        existing = self.find_relative(rel_to_proband, attrs.get("name"))
        if existing is None:
            return self.add_relative(rel_to_proband, **attrs)
        for k, v in attrs.items():
            if v is not None:
                setattr(existing, k, v)
        if existing.sex is None:
            existing.sex = _SEX_BY_REL.get(rel_to_proband)
        return existing

    def connected_to_proband(self) -> set:

        children_of: Dict[str, List[str]] = {}
        for ind in self.individuals.values():
            for pid in (ind.father_id, ind.mother_id):
                if pid:
                    children_of.setdefault(pid, []).append(ind.id)

        seen = {self.proband.id}
        stack = [self.proband.id]
        while stack:
            cur = self.individuals[stack.pop()]
            neighbours = [cur.father_id, cur.mother_id] + children_of.get(cur.id, [])
            for nid in neighbours:
                if nid and nid not in seen:
                    seen.add(nid)
                    stack.append(nid)
        return seen

    def prune_disconnected(self) -> List[str]:

        keep = self.connected_to_proband()
        dropped = []
        for ind in list(self.individuals.values()):
            if ind.id not in keep:
                dropped.append(ind.name or ind.rel_to_proband.replace("_", " "))
                del self.individuals[ind.id]
        return dropped


    _GENERATION_OFFSET = {
        "mother": 25, "father": 25,
        "maternal_grandmother": 50, "maternal_grandfather": 50,
        "paternal_grandmother": 50, "paternal_grandfather": 50,
        "maternal_aunt": 25, "maternal_uncle": 25,
        "paternal_aunt": 25, "paternal_uncle": 25,
        "sister": 0, "brother": 0, "half_sister": 0, "half_brother": 0,
        "daughter": -25, "son": -25, "niece": -25, "nephew": -25,
        "cousin": 0,
    }

    def estimate_missing_years(self, proband_yob: Optional[int]) -> List[str]:
        if proband_yob is None:
            return []
        estimated = []
        for ind in self.individuals.values():
            if ind.is_proband or not ind.has_any_cancer():
                continue
            if ind.resolved_year_of_birth() is not None:
                continue
            offset = self._GENERATION_OFFSET.get(ind.rel_to_proband, 25)
            ind.year_of_birth = proband_yob - offset
            who = ind.name or ind.rel_to_proband.replace("_", " ")
            estimated.append(f"{who} (~{ind.year_of_birth})")
        return estimated

    def add_relative(self, rel_to_proband: str, **attrs) -> Individual:
        if rel_to_proband not in RELATIONSHIPS:
            rel_to_proband = "other"
        ind = Individual(id=self._new_id(), rel_to_proband=rel_to_proband, **attrs)
        return self._add(ind)

    def get_by_rel(self, rel: str) -> List[Individual]:
        return [i for i in self.individuals.values() if i.rel_to_proband == rel]

    def build_structure(self) -> List[str]:

        unplaceable = []

        def one(rel):
            got = self.get_by_rel(rel)
            return got[0] if got else None

        mother = one("mother")
        father = one("father")

        # proband's parents
        if mother:
            self.proband.mother_id = mother.id
        if father:
            self.proband.father_id = father.id

        # siblings share the proband's parents
        for sib in self.get_by_rel("sister") + self.get_by_rel("brother"):
            if mother:
                sib.mother_id = mother.id
            if father:
                sib.father_id = father.id

        # proband's children -- other parent unknown/unspecified in this intake
        for child in self.get_by_rel("daughter") + self.get_by_rel("son"):
            self.proband.__dict__  # proband is a parent; sex decides which slot
            if self.proband.sex == "F":
                child.mother_id = self.proband.id
            else:
                child.father_id = self.proband.id

        # maternal grandparents -> mother's parents
        mgm, mgf = one("maternal_grandmother"), one("maternal_grandfather")
        if mother:
            if mgm:
                mother.mother_id = mgm.id
            if mgf:
                mother.father_id = mgf.id
        elif mgm or mgf:
            unplaceable.append("maternal grandparents (no mother in pedigree)")

        # paternal grandparents -> father's parents
        pgm, pgf = one("paternal_grandmother"), one("paternal_grandfather")
        if father:
            if pgm:
                father.mother_id = pgm.id
            if pgf:
                father.father_id = pgf.id
        elif pgm or pgf:
            unplaceable.append("paternal grandparents (no father in pedigree)")
        synthesized = []
        for rel, gm, gf, parent, parent_rel, label in [
            ("maternal_aunt", mgm, mgf, mother, "mother", "maternal"),
            ("maternal_uncle", mgm, mgf, mother, "mother", "maternal"),
            ("paternal_aunt", pgm, pgf, father, "father", "paternal"),
            ("paternal_uncle", pgm, pgf, father, "father", "paternal"),
        ]:
            relatives = self.get_by_rel(rel)
            if not relatives:
                continue
            if parent is None:
                parent = self.add_relative(
                    parent_rel, sex=("F" if parent_rel == "mother" else "M"),
                    name=f"unknown {parent_rel}", is_placeholder=True)
                if parent_rel == "mother":
                    mother = parent
                    self.proband.mother_id = parent.id
                else:
                    father = parent
                    self.proband.father_id = parent.id
                synthesized.append(f"{label} parent")

            if gm is None:
                gm = self.add_relative("other", sex="F",
                                       name=f"unknown {label} grandmother",
                                       is_placeholder=True)
                if label == "maternal":
                    mgm = gm
                else:
                    pgm = gm
                synthesized.append(f"{label} grandmother")
            if gf is None:
                gf = self.add_relative("other", sex="M",
                                       name=f"unknown {label} grandfather",
                                       is_placeholder=True)
                if label == "maternal":
                    mgf = gf
                else:
                    pgf = gf
                synthesized.append(f"{label} grandfather")

            parent.mother_id, parent.father_id = gm.id, gf.id
            for au in relatives:
                au.mother_id = gm.id
                au.father_id = gf.id
        if synthesized:
            unplaceable.append(
                "(connected your " + ", ".join(sorted(set(synthesized)))
                + " via unnamed placeholder individuals so the family tree "
                "stays whole -- their own details are simply unknown, "
                "which is fine)")

        half_unplaced = []
        for hs in self.get_by_rel("half_sister") + self.get_by_rel("half_brother"):
            side = hs.shared_parent_side
            if side not in ("mother", "father"):
                if mother and not father:
                    side = "mother"
                elif father and not mother:
                    side = "father"
            if side == "mother" and mother:
                hs.mother_id = mother.id
            elif side == "father" and father:
                hs.father_id = father.id
            else:
                half_unplaced.append(hs)


        niece_nephew_unplaced = []
        siblings = self.get_by_rel("sister") + self.get_by_rel("brother")
        for nn in self.get_by_rel("niece") + self.get_by_rel("nephew"):
            parent = None
            if nn.linked_via_id and nn.linked_via_id in self.individuals:
                candidate = self.individuals[nn.linked_via_id]
                if candidate in siblings:
                    parent = candidate
            elif len(siblings) == 1:
                parent = siblings[0]
            if parent is not None:
                if parent.sex == "F":
                    nn.mother_id = parent.id
                else:
                    nn.father_id = parent.id
            else:
                niece_nephew_unplaced.append(nn)

        cousin_unplaced = []
        aunts_uncles = (self.get_by_rel("maternal_aunt") + self.get_by_rel("maternal_uncle")
                        + self.get_by_rel("paternal_aunt") + self.get_by_rel("paternal_uncle"))
        for c in self.get_by_rel("cousin"):
            parent = None
            if c.linked_via_id and c.linked_via_id in self.individuals:
                candidate = self.individuals[c.linked_via_id]
                if candidate in aunts_uncles:
                    parent = candidate
            elif len(aunts_uncles) == 1:
                parent = aunts_uncles[0]
            if parent is not None:
                if parent.sex == "F":
                    c.mother_id = parent.id
                else:
                    c.father_id = parent.id
            else:
                cousin_unplaced.append(c)

        if half_unplaced:
            unplaceable.append(
                f"{len(half_unplaced)} half-sibling(s) (which parent is shared "
                "wasn't captured, and both of your parents are in the pedigree "
                "so it's ambiguous -- needs manual placement)")
        if niece_nephew_unplaced:
            unplaceable.append(
                f"{len(niece_nephew_unplaced)} niece/nephew(s) (which sibling "
                "is their parent wasn't captured, and you have more than one "
                "sibling recorded so it's ambiguous -- needs manual placement)")
        if cousin_unplaced:
            unplaceable.append(
                f"{len(cousin_unplaced)} cousin(s) (which aunt/uncle is their "
                "parent wasn't captured, and you have more than one aunt/uncle "
                "recorded so it's ambiguous -- needs manual placement)")

        # relationships we still can't structurally resolve at all
        for rel in ["other"]:
            if self.get_by_rel(rel):
                unplaceable.append(rel.replace("_", " ") + " (needs manual placement)")

        return unplaceable

    def is_sufficient(self) -> bool:

        p = self.proband
        if p.sex is None or (p.age is None and p.year_of_birth is None):
            return False
        for ind in self.individuals.values():
            if ind.is_proband:
                continue
            if ind.sex is None:
                return False
            if ind.alive is None and ind.age is None and ind.year_of_birth is None:
                return False
        return True

    def missing_detail(self) -> List[str]:
        """Human-readable list of what's still thin, for the agent to ask about."""
        gaps = []
        p = self.proband
        if p.sex is None:
            gaps.append("your sex")
        if p.age is None and p.year_of_birth is None:
            gaps.append("your age or year of birth")
        for ind in self.individuals.values():
            if ind.is_proband:
                continue
            who = ind.name or ind.rel_to_proband.replace("_", " ")
            if ind.sex is None:
                gaps.append(f"{who}: sex")
            if ind.alive is None and ind.age is None and ind.year_of_birth is None:
                gaps.append(f"{who}: whether alive, and age")
            elif ind.has_any_cancer() and ind.resolved_year_of_birth() is None:
                gaps.append(f"{who}: year of birth (needed since they had "
                           f"cancer -- an estimate is fine if you're not sure)")
        return gaps

    def to_pedigree_rows(self) -> Dict[str, Any]:
        rows = []
        for ind in self.individuals.values():
            rows.append({
                "id": ind.id,
                "rel_to_proband": ind.rel_to_proband,
                "is_proband": ind.is_proband,
                "sex": ind.sex,
                "alive": ind.alive,
                "age": ind.age,
                "age_at_death": ind.age_at_death,
                "year_of_birth": ind.year_of_birth,
                "father_id": ind.father_id,
                "mother_id": ind.mother_id,
                "diagnoses": [{"type": d.cancer_type, "age": d.age} for d in ind.diagnoses],
                "gene_tests": [{"gene": g.gene, "result": g.result} for g in ind.gene_tests],
            })
        return {
            "ashkenazi": self.ashkenazi,
            "rows": rows,
            "unresolved": {
                "canrisk_pedigree_columns": "TODO(canrisk-pedigree): confirm "
                "BOADICEA v4 tab-delimited column order/encoding before submit",
            },
        }

    def teach_back(self) -> str:
        lines = [f"- {self.proband.summary_line()}"]
        for ind in self.individuals.values():
            if not ind.is_proband:
                lines.append(f"- {ind.summary_line()}")
        aj = {True: "Yes", False: "No/Unknown", None: "not asked"}[self.ashkenazi]
        lines.append(f"- Ashkenazi Jewish ancestry: {aj}")
        return "Here's the family history I have:\n" + "\n".join(lines)


def seed_structure(pedigree: Pedigree, preset: str) -> None:
    if preset == "nuclear":
        pedigree.add_relative("mother")
        pedigree.add_relative("father")
    elif preset == "first_degree":
        pedigree.add_relative("mother")
        pedigree.add_relative("father")
        pedigree.add_relative("sister")
        pedigree.add_relative("brother")
    elif preset == "second_degree":
        pedigree.add_relative("mother")
        pedigree.add_relative("father")
        pedigree.add_relative("sister")
        pedigree.add_relative("brother")
        for rel in ["maternal_grandmother", "maternal_grandfather",
                    "paternal_grandmother", "paternal_grandfather",
                    "maternal_aunt", "maternal_uncle",
                    "paternal_aunt", "paternal_uncle"]:
            pedigree.add_relative(rel)