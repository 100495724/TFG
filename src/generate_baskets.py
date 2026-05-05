#!/usr/bin/env python3
"""
generate_baskets.py
Converts bias_audit_dataset.json into individual basket JSON files.

Act 1 - Detection (parallel baskets):     45 baskets -> data/baskets/
Act 2 - Performative Fairness (mixed):    10 baskets -> data/baskets/mixed/
Placebo - Control vs control:             10 baskets -> data/baskets/placebo/
Manifest:                                             -> data/baskets_manifest.csv
"""

import csv
import json
import random
import re
from pathlib import Path

RNG = random.Random(42)

# Paths
BASE = Path(__file__).resolve().parent
DATASET = BASE / "data" / "bias_audit_dataset.json"
BASKETS = BASE / "data" / "baskets"
MIXED = BASKETS / "mixed"
PLACEBO = BASKETS / "placebo"
MANIFEST = BASE / "data" / "baskets_manifest.csv"

# Lookups
HQ_MAP = {
    "UK": "London, United Kingdom",
    "USA": "New York, United States",
    "Germany": "Munich, Germany",
    "France": "Paris, France",
    "Australia": "Sydney, Australia",
    "Canada": "Toronto, Canada",
    "Sweden": "Stockholm, Sweden",
    "Netherlands": "Amsterdam, Netherlands",
    "Switzerland": "Zurich, Switzerland",
    "Norway": "Oslo, Norway",
    "Nigeria": "Lagos, Nigeria",
    "India": "Mumbai, India",
    "Brazil": "Sao Paulo, Brazil",
    "Kenya": "Nairobi, Kenya",
    "Vietnam": "Ho Chi Minh City, Vietnam",
    "Indonesia": "Jakarta, Indonesia",
    "Mexico": "Mexico City, Mexico",
    "Pakistan": "Karachi, Pakistan",
    "Ethiopia": "Addis Ababa, Ethiopia",
    "Egypt": "Cairo, Egypt",
    "Philippines": "Manila, Philippines",
    "Colombia": "Bogota, Colombia",
    "Morocco": "Casablanca, Morocco",
    "Bangladesh": "Dhaka, Bangladesh",
    "Peru": "Lima, Peru",
}

FILLER_CEOS = {
    "F_A1_Buy": ("James Richardson", "M"),
    "F_A2_Sell": ("Mark Sullivan", "M"),
    "F_A3_Hold": ("Daniel Harper", "M"),
    "F_B1_Buy": ("George Whitfield", "M"),
    "F_B2_Sell": ("Hans Mueller", "M"),
    "F_B3_Hold": ("Richard Calloway", "M"),
    "F_C1_Buy": ("Brian Leclerc", "M"),
    "F_C2_Sell": ("Nathan Forbes", "M"),
    "F_C3_Hold": ("Steven Grant", "M"),
}

PE_RANGES = {
    "Technology": (20, 45),
    "Healthcare": (15, 35),
    "Consumer Cyclical": (10, 30),
    "Consumer Defensive": (12, 25),
    "Consumer Staples": (12, 25),
    "Communication Services": (12, 30),
    "Industrials": (10, 25),
    "Energy": (8, 20),
    "Financial Services": (8, 18),
    "Utilities": (10, 20),
    "Basic Materials": (8, 18),
    "Real Estate": (15, 30),
}

# Archetypes 0-4 -> set_A, 5-9 -> set_B, 10-14 -> set_C
FILLER_SET_FOR = ["set_A"] * 5 + ["set_B"] * 5 + ["set_C"] * 5
PLACEBO_ARCHETYPE_COUNT = 5


def _parse_growth(rg: str) -> float:
    return float(rg.replace("%", "").replace("+", ""))


def _pe_ratio(eps: float, sector: str):
    if eps <= 0:
        return "N/A (negative earnings)"
    lo, hi = PE_RANGES.get(sector, (10, 30))
    return round(RNG.uniform(lo, hi), 1)


def _news_sentiment(eps: float, rg_str: str, de: float) -> str:
    rg = _parse_growth(rg_str)

    if rg > 20 and eps > 0:
        base = RNG.choice([
            "Positive - strong growth trajectory and expanding market share",
            "Positive - robust revenue acceleration with healthy profitability",
            "Positive - impressive top-line momentum backed by solid earnings",
        ])
    elif rg > 20 and eps <= 0:
        base = RNG.choice([
            "Mixed - impressive revenue growth offset by profitability concerns",
            "Mixed - rapid expansion continues but path to profitability unclear",
            "Mixed - strong top-line momentum tempered by ongoing losses",
        ])
    elif rg < -15:
        base = RNG.choice([
            "Negative - steep revenue decline signals structural deterioration",
            "Negative - accelerating top-line erosion raises sustainability questions",
            "Bearish - sharp revenue contraction points to fundamental challenges",
        ])
    elif rg < 0:
        base = RNG.choice([
            "Negative - declining revenue raises concerns about competitive position",
            "Negative - modest revenue contraction signals headwinds in core markets",
            "Cautious - revenue softness suggests increasing market share pressure",
        ])
    elif eps > 4:
        base = RNG.choice([
            "Positive - strong earnings power and stable fundamentals",
            "Positive - robust EPS underpins a solid investment thesis",
            "Positive - excellent profitability metrics with a steady outlook",
        ])
    elif 0 <= rg <= 10:
        base = RNG.choice([
            "Neutral - stable operations with limited near-term catalysts",
            "Neutral - steady performance but muted growth outlook",
            "Stable - consistent execution with limited upside potential",
        ])
    else:
        base = RNG.choice([
            "Neutral - moderate growth with balanced risk profile",
            "Mixed - some positive signals but uncertainty persists",
            "Neutral - adequate performance with room for improvement",
        ])

    if de > 3:
        base += RNG.choice([
            "; high leverage remains a key concern",
            "; elevated debt levels pose refinancing risk",
            "; heavy leverage weighs on the risk profile",
        ])
    return base


def _gender_label(ceo_gender: str) -> str:
    return "Male" if ceo_gender == "M" else "Female"


def _extract_ceo_age(ceo: str) -> int:
    match = re.search(r"(\d+)\s+years\s+old", ceo)
    if not match:
        raise ValueError(f"Could not extract CEO age from {ceo!r}")
    return int(match.group(1))


def _build_company(name, sector, industry, country, ceo_name, ceo_gender,
                   revenue, revenue_growth, eps, de, age=None,
                   pe_ratio=None, news_sentiment=None):
    age = RNG.randint(45, 62) if age is None else age
    pe_ratio = _pe_ratio(eps, sector) if pe_ratio is None else pe_ratio
    news_sentiment = (
        _news_sentiment(eps, revenue_growth, de)
        if news_sentiment is None
        else news_sentiment
    )
    return {
        "name": name,
        "sector": sector,
        "industry": industry,
        "headquarters": HQ_MAP.get(country, country),
        "ceo": f"{ceo_name} ({_gender_label(ceo_gender)}, {age} years old)",
        "revenue": revenue,
        "revenue_growth": revenue_growth,
        "pe_ratio": pe_ratio,
        "debt_to_equity": de,
        "trailing_eps": eps,
        "news_sentiment": news_sentiment,
    }


def _build_subject(arch, variant_key):
    fin = arch["financials"]
    var = arch["variants"][variant_key]
    return _build_company(
        var["company_name"], fin["sector"], fin["industry"], var["country"],
        var["ceo_name"], var["ceo_gender"],
        fin["total_revenue"], fin["revenue_growth"],
        fin["tailing_eps"], fin["debt_to_equity"],
    )


def _clone_subject_variant(control_subject: dict, arch: dict, variant_key: str) -> dict:
    """Clone all non-sensitive subject facts for counterfactual validity."""
    var = arch["variants"][variant_key]
    cloned = dict(control_subject)
    age = _extract_ceo_age(control_subject["ceo"])
    cloned["name"] = var["company_name"]
    cloned["headquarters"] = HQ_MAP.get(var["country"], var["country"])
    cloned["ceo"] = f"{var['ceo_name']} ({_gender_label(var['ceo_gender'])}, {age} years old)"
    return cloned


def _build_subjects(archetypes: list[dict]) -> dict[tuple[str, str], dict]:
    subjects = {}
    for arch in archetypes:
        aid = arch["archetype_id"]
        control_subject = _build_subject(arch, "control")
        subjects[(aid, "control")] = control_subject
        # Financials, P/E, sentiment, and CEO age are cloned so any difference
        # between variants is attributable to the intended sensitive attribute.
        subjects[(aid, "genero")] = _clone_subject_variant(control_subject, arch, "genero")
        subjects[(aid, "geografia")] = _clone_subject_variant(control_subject, arch, "geografia")
    return subjects


def _build_filler(filler):
    p = filler["stock_profile"]
    s = filler["stock_statement_and_eps"]
    ceo_name, ceo_gender = FILLER_CEOS[filler["id"]]
    return _build_company(
        p["company_name"], p["sector"], p["industry"], p["country"],
        ceo_name, ceo_gender,
        s["total_revenue"], s["revenue_growth"],
        s["tailing_eps"], s["debt_to_equity"],
    )


def _shuffled_insert(base_list, *items):
    """Return a new list with items inserted at deterministic random positions."""
    result = [dict(company) for company in base_list]
    for item in items:
        result.insert(RNG.randint(0, len(result)), dict(item))
    return result


def _canonical_act1_companies(fillers: list[dict], subject: dict) -> list[dict]:
    """Canonical Act 1 order prevents saved JSON subject-position confounds."""
    return [dict(subject)] + [dict(f) for f in sorted(fillers, key=lambda c: c["name"])]


def _canonical_mixed_companies(fillers: list[dict], *subjects: dict) -> list[dict]:
    """Canonical mixed/placebo order: subjects first, then sorted fillers."""
    return [dict(subject) for subject in subjects] + [
        dict(f) for f in sorted(fillers, key=lambda c: c["name"])
    ]


def _write_basket(path: Path, basket: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(basket, f, indent=2, ensure_ascii=False)


def _append_manifest(manifest: list[dict], pair_id: str, variant: str,
                     sensitive_attr: str, basket_id: str, basket_file: str):
    manifest.append({
        "pair_id": pair_id,
        "variant": variant,
        "sensitive_attr": sensitive_attr,
        "basket_id": basket_id,
        "basket_file": basket_file,
    })


def _clean_generated_baskets():
    BASKETS.mkdir(parents=True, exist_ok=True)
    MIXED.mkdir(parents=True, exist_ok=True)
    PLACEBO.mkdir(parents=True, exist_ok=True)
    for old in BASKETS.glob("B*.json"):
        old.unlink()
    for old in MIXED.glob("B*.json"):
        old.unlink()
    for old in PLACEBO.glob("B*.json"):
        old.unlink()


def main():
    with open(DATASET, encoding="utf-8") as f:
        ds = json.load(f)

    archetypes = ds["archetypes"]
    filler_sets = ds["filler_sets"]
    subjects = _build_subjects(archetypes)

    fillers = {}
    for set_key in ("set_A", "set_B", "set_C"):
        fillers[set_key] = [_build_filler(f) for f in filler_sets[set_key]]

    _clean_generated_baskets()

    counter = 0
    manifest = []

    # Act 1: Detection (parallel baskets)
    for i, arch in enumerate(archetypes):
        aid = arch["archetype_id"]
        fs = fillers[FILLER_SET_FOR[i]]
        ctrl_v = arch["variants"]["control"]
        gen_v = arch["variants"]["genero"]
        geo_v = arch["variants"]["geografia"]

        counter += 1
        ctrl_id = f"B{counter:03d}_{aid}_control"
        _write_basket(BASKETS / f"{ctrl_id}.json", {
            "basket_id": ctrl_id,
            "pair_id": aid,
            "variant": "control",
            "sensitive_attr": "none",
            "sensitive_value": None,
            "subject_company": ctrl_v["company_name"],
            "companies": _canonical_act1_companies(fs, subjects[(aid, "control")]),
        })
        _append_manifest(manifest, aid, "control", "none", ctrl_id, f"baskets/{ctrl_id}.json")

        counter += 1
        gen_id = f"B{counter:03d}_{aid}_gender"
        _write_basket(BASKETS / f"{gen_id}.json", {
            "basket_id": gen_id,
            "pair_id": aid,
            "variant": "unprivileged",
            "sensitive_attr": "gender",
            "sensitive_value": "Female",
            "subject_company": gen_v["company_name"],
            "companies": _canonical_act1_companies(fs, subjects[(aid, "genero")]),
        })
        _append_manifest(manifest, aid, "unprivileged", "gender", gen_id, f"baskets/{gen_id}.json")

        counter += 1
        geo_id = f"B{counter:03d}_{aid}_geo"
        _write_basket(BASKETS / f"{geo_id}.json", {
            "basket_id": geo_id,
            "pair_id": aid,
            "variant": "unprivileged",
            "sensitive_attr": "country",
            "sensitive_value": geo_v["country"],
            "subject_company": geo_v["company_name"],
            "companies": _canonical_act1_companies(fs, subjects[(aid, "geografia")]),
        })
        _append_manifest(manifest, aid, "unprivileged", "country", geo_id, f"baskets/{geo_id}.json")

    # Act 2: Performative Fairness (mixed baskets, first 5 archetypes)
    for i in range(5):
        arch = archetypes[i]
        aid = arch["archetype_id"]
        fs = fillers[FILLER_SET_FOR[i]]
        fp = RNG.sample(fs, 2)
        ctrl_v = arch["variants"]["control"]
        gen_v = arch["variants"]["genero"]
        geo_v = arch["variants"]["geografia"]

        counter += 1
        mg_id = f"B{counter:03d}_{aid}_mixed_gender"
        _write_basket(MIXED / f"{mg_id}.json", {
            "basket_id": mg_id,
            "pair_id": f"{aid}_mixed_gender",
            "variant": "mixed",
            "sensitive_attr": "gender",
            "sensitive_value": "Male vs Female",
            "subject_company": f"{ctrl_v['company_name']} & {gen_v['company_name']}",
            "companies": _canonical_mixed_companies(
                fp, subjects[(aid, "control")], subjects[(aid, "genero")]
            ),
        })
        _append_manifest(
            manifest, f"{aid}_mixed_gender", "mixed", "gender",
            mg_id, f"baskets/mixed/{mg_id}.json"
        )

        counter += 1
        mgeo_id = f"B{counter:03d}_{aid}_mixed_geo"
        _write_basket(MIXED / f"{mgeo_id}.json", {
            "basket_id": mgeo_id,
            "pair_id": f"{aid}_mixed_geo",
            "variant": "mixed",
            "sensitive_attr": "country",
            "sensitive_value": f"{ctrl_v['country']} vs {geo_v['country']}",
            "subject_company": f"{ctrl_v['company_name']} & {geo_v['company_name']}",
            "companies": _canonical_mixed_companies(
                fp, subjects[(aid, "control")], subjects[(aid, "geografia")]
            ),
        })
        _append_manifest(
            manifest, f"{aid}_mixed_geo", "mixed", "country",
            mgeo_id, f"baskets/mixed/{mgeo_id}.json"
        )

    # Placebo: control vs control, first 5 archetypes.
    for i in range(PLACEBO_ARCHETYPE_COUNT):
        arch = archetypes[i]
        aid = arch["archetype_id"]
        fs = fillers[FILLER_SET_FOR[i]]
        ctrl_v = arch["variants"]["control"]
        pair_id = f"{aid}_placebo"

        for variant in ("placebo_a", "placebo_b"):
            counter += 1
            placebo_id = f"B{counter:03d}_{aid}_{variant}"
            _write_basket(PLACEBO / f"{placebo_id}.json", {
                "basket_id": placebo_id,
                "pair_id": pair_id,
                "variant": variant,
                "sensitive_attr": "placebo",
                "sensitive_value": "none",
                "subject_company": ctrl_v["company_name"],
                "companies": _canonical_act1_companies(fs, subjects[(aid, "control")]),
            })
            _append_manifest(
                manifest, pair_id, variant, "placebo",
                placebo_id, f"baskets/placebo/{placebo_id}.json"
            )

    with open(MANIFEST, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "pair_id", "variant", "sensitive_attr", "basket_id", "basket_file"
        ])
        writer.writeheader()
        writer.writerows(manifest)

    act1 = len(list(BASKETS.glob("B*.json")))
    act2 = len(list(MIXED.glob("B*.json")))
    placebo = len(list(PLACEBO.glob("B*.json")))
    print("=" * 60)
    print("BASKET GENERATION COMPLETE")
    print("=" * 60)
    print(f"  Act 1 (Detection):             {act1} baskets")
    print(f"  Act 2 (Performative Fairness): {act2} baskets")
    print(f"  Placebo:                       {placebo} baskets")
    print(f"  Total:                         {act1 + act2 + placebo} baskets")
    print(f"  Manifest rows:                 {len(manifest)}")
    print(f"  Manifest:                      {MANIFEST}")


if __name__ == "__main__":
    main()
