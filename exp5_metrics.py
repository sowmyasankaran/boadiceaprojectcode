
from __future__ import annotations

import statistics
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence


def nice_category_consistency(categories_per_profile: Sequence[Sequence[Optional[str]]]) -> dict:

    all_agree = []
    modal = []
    for cats in categories_per_profile:
        cats = list(cats)
        if not cats:
            continue
        # normalize for comparison but keep None distinct
        norm = [c.strip().lower() if isinstance(c, str) else None for c in cats]
        counts = Counter(norm)
        modal_count = max(counts.values())
        all_agree.append(1.0 if modal_count == len(norm) else 0.0)
        modal.append(modal_count / len(norm))
    n = len(all_agree)
    return {
        "n_profiles": n,
        "all_agree_rate": (sum(all_agree) / n) if n else None,
        "modal_agreement": (sum(modal) / n) if n else None,
    }


def mean_within_profile_sd(risk_per_profile: Sequence[Sequence[Optional[float]]]) -> dict:

    sds = []
    for vals in risk_per_profile:
        clean = [float(v) for v in vals if v is not None]
        if len(clean) >= 2:
            sds.append(statistics.stdev(clean))
        elif len(clean) == 1:
            sds.append(0.0)
        # zero usable values -> skip
    n = len(sds)
    return {
        "n_profiles": n,
        "mean_within_profile_sd": (sum(sds) / n) if n else None,
        "max_within_profile_sd": max(sds) if sds else None,
    }


def icc_oneway(risk_per_profile: Sequence[Sequence[Optional[float]]]) -> dict:

    cleaned = [[float(v) for v in vals if v is not None] for vals in risk_per_profile]
    cleaned = [c for c in cleaned if len(c) >= 2]
    if len(cleaned) < 2:
        return {"icc": None, "n_profiles": len(cleaned), "k": None,
                "reason": "need >=2 profiles with >=2 repeats"}

    k = Counter(len(c) for c in cleaned).most_common(1)[0][0]
    groups = [c for c in cleaned if len(c) == k]
    n = len(groups)
    if n < 2 or k < 2:
        return {"icc": None, "n_profiles": n, "k": k,
                "reason": "unbalanced or insufficient after filtering to modal k"}

    all_vals = [x for g in groups for x in g]
    grand = sum(all_vals) / len(all_vals)
    group_means = [sum(g) / k for g in groups]

    ss_between = k * sum((m - grand) ** 2 for m in group_means)
    ss_within = sum((x - gm) ** 2 for g, gm in zip(groups, group_means) for x in g)
    ms_between = ss_between / (n - 1)
    ms_within = ss_within / (n * (k - 1))

    denom = ms_between + (k - 1) * ms_within
    icc = None if denom == 0 else (ms_between - ms_within) / denom
    return {"icc": icc, "n_profiles": n, "k": k,
            "ms_between": ms_between, "ms_within": ms_within}


def field_agreement(fields_per_profile: Sequence[Sequence[Dict[str, Any]]],
                    scored_fields: Optional[Sequence[str]] = None) -> dict:

    prof_all_agree = []
    prof_pairwise = []
    for runs in fields_per_profile:
        runs = list(runs)
        if len(runs) < 2:
            continue
        keys = list(scored_fields) if scored_fields is not None else sorted(
            {k for r in runs for k in (r or {}).keys()})
        if not keys:
            continue
        field_all_agree = []
        field_pair = []
        for key in keys:
            vals = [(r or {}).get(key) for r in runs]
            field_all_agree.append(1.0 if len(set(map(_hashable, vals))) == 1 else 0.0)
            # pairwise match fraction for this field
            pairs = 0
            matches = 0
            for i in range(len(vals)):
                for j in range(i + 1, len(vals)):
                    pairs += 1
                    if _hashable(vals[i]) == _hashable(vals[j]):
                        matches += 1
            field_pair.append((matches / pairs) if pairs else 1.0)
        prof_all_agree.append(sum(field_all_agree) / len(keys))
        prof_pairwise.append(sum(field_pair) / len(keys))
    n = len(prof_all_agree)
    return {
        "n_profiles": n,
        "all_agree_field_rate": (sum(prof_all_agree) / n) if n else None,
        "mean_pairwise_field_agreement": (sum(prof_pairwise) / n) if n else None,
    }


def _hashable(v: Any):
    """Make dict/list field values comparable/hashable for equality checks."""
    if isinstance(v, dict):
        return tuple(sorted((k, _hashable(x)) for k, x in v.items()))
    if isinstance(v, (list, tuple)):
        return tuple(_hashable(x) for x in v)
    return v


def summarize_model_depth(categories_per_profile, risk_per_profile, fields_per_profile) -> dict:
    """Roll the four metrics into one row for the Experiment 5 table."""
    return {
        "category_consistency": nice_category_consistency(categories_per_profile),
        "lifetime_risk_sd": mean_within_profile_sd(risk_per_profile),
        "icc": icc_oneway(risk_per_profile),
        "field_agreement": field_agreement(fields_per_profile),
    }
