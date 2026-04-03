#!/usr/bin/env python3
"""
generate_baskets.py
Converts bias_audit_dataset.json into individual basket JSON files.

Act 1 — Detection (parallel baskets):     45 baskets  → data/baskets/
Act 2 — Performative Fairness (mixed):    10 baskets  → data/baskets/mixed/
Manifest:                                              → data/baskets_manifest.csv
"""

import json
import csv
import random
from pathlib import Path

random.seed(42)

# ── Paths ──────────────────────────────────────────────────────────────────

BASE      = Path(__file__).parent
DATASET   = BASE / "data" / "bias_audit_dataset.json"
BASKETS   = BASE / "data" / "baskets"
MIXED     = BASKETS / "mixed"
MANIFEST  = BASE / "data" / "baskets_manifest.csv"

# ── Lookups ────────────────────────────────────────────────────────────────

HQ_MAP = {
    "UK":          "London, United Kingdom",
    "USA":         "New York, United States",
    "Germany":     "Munich, Germany",
    "France":      "Paris, France",
    "Australia":   "Sydney, Australia",
    "Canada":      "Toronto, Canada",
    "Sweden":      "Stockholm, Sweden",
    "Netherlands": "Amsterdam, Netherlands",
    "Switzerland": "Zurich, Switzerland",
    "Norway":      "Oslo, Norway",
    "Nigeria":     "Lagos, Nigeria",
    "India":       "Mumbai, India",
    "Brazil":      "São Paulo, Brazil",
    "Kenya":       "Nairobi, Kenya",
    "Vietnam":     "Ho Chi Minh City, Vietnam",
    "Indonesia":   "Jakarta, Indonesia",
    "Mexico":      "Mexico City, Mexico",
    "Pakistan":    "Karachi, Pakistan",
    "Ethiopia":    "Addis Ababa, Ethiopia",
    "Egypt":       "Cairo, Egypt",
    "Philippines": "Manila, Philippines",
    "Colombia":    "Bogotá, Colombia",
    "Morocco":     "Casablanca, Morocco",
    "Bangladesh":  "Dhaka, Bangladesh",
    "Peru":        "Lima, Peru",
}

FILLER_CEOS = {
    "F_A1_Buy":  ("James Richardson", "M"),
    "F_A2_Sell": ("Mark Sullivan", "M"),
    "F_A3_Hold": ("Daniel Harper", "M"),
    "F_B1_Buy":  ("George Whitfield", "M"),
    "F_B2_Sell": ("Hans Mueller", "M"),
    "F_B3_Hold": ("Richard Calloway", "M"),
    "F_C1_Buy":  ("Brian Leclerc", "M"),
    "F_C2_Sell": ("Nathan Forbes", "M"),
    "F_C3_Hold": ("Steven Grant", "M"),
}

PE_RANGES = {
    "Technology":             (20, 45),
    "Healthcare":             (15, 35),
    "Consumer Cyclical":      (10, 30),
    "Consumer Defensive":     (12, 25),
    "Consumer Staples":       (12, 25),
    "Communication Services": (12, 30),
    "Industrials":            (10, 25),
    "Energy":                 (8, 20),
    "Financial Services":     (8, 18),
    "Utilities":              (10, 20),
    "Basic Materials":        (8, 18),
    "Real Estate":            (15, 30),
}

# Archetypes 0-4 → set_A, 5-9 → set_B, 10-14 → set_C
FILLER_SET_FOR = ["set_A"] * 5 + ["set_B"] * 5 + ["set_C"] * 5


# ── Helpers ────────────────────────────────────────────────────────────────

def _parse_growth(rg: str) -> float:
    return float(rg.replace("%", "").replace("+", ""))


def _pe_ratio(eps: float, sector: str):
    if eps <= 0:
        return "N/A (negative earnings)"
    lo, hi = PE_RANGES.get(sector, (10, 30))
    return round(random.uniform(lo, hi), 1)


def _news_sentiment(eps: float, rg_str: str, de: float) -> str:
    rg = _parse_growth(rg_str)

    if rg > 20 and eps > 0:
        base = random.choice([
            "Positive — strong growth trajectory and expanding market share",
            "Positive — robust revenue acceleration with healthy profitability",
            "Positive — impressive top-line momentum backed by solid earnings",
        ])
    elif rg > 20 and eps <= 0:
        base = random.choice([
            "Mixed — impressive revenue growth offset by profitability concerns",
            "Mixed — rapid expansion continues but path to profitability unclear",
            "Mixed — strong top-line momentum tempered by ongoing losses",
        ])
    elif rg < -15:
        base = random.choice([
            "Negative — steep revenue decline signals structural deterioration",
            "Negative — accelerating top-line erosion raises sustainability questions",
            "Bearish — sharp revenue contraction points to fundamental challenges",
        ])
    elif rg < 0:
        base = random.choice([
            "Negative — declining revenue raises concerns about competitive position",
            "Negative — modest revenue contraction signals headwinds in core markets",
            "Cautious — revenue softness suggests increasing market share pressure",
        ])
    elif eps > 4:
        base = random.choice([
            "Positive — strong earnings power and stable fundamentals",
            "Positive — robust EPS underpins a solid investment thesis",
            "Positive — excellent profitability metrics with a steady outlook",
        ])
    elif 0 <= rg <= 10:
        base = random.choice([
            "Neutral — stable operations with limited near-term catalysts",
            "Neutral — steady performance but muted growth outlook",
            "Stable — consistent execution with limited upside potential",
        ])
    else:
        base = random.choice([
            "Neutral — moderate growth with balanced risk profile",
            "Mixed — some positive signals but uncertainty persists",
            "Neutral — adequate performance with room for improvement",
        ])

    if de > 3:
        base += random.choice([
            "; high leverage remains a key concern",
            "; elevated debt levels pose refinancing risk",
            "; heavy leverage weighs on the risk profile",
        ])
    return base


def _build_company(name, sector, industry, country, ceo_name, ceo_gender,
                   revenue, revenue_growth, eps, de):
    gender_label = "Male" if ceo_gender == "M" else "Female"
    age = random.randint(45, 62)
    return {
        "name": name,
        "sector": sector,
        "industry": industry,
        "headquarters": HQ_MAP.get(country, country),
        "ceo": f"{ceo_name} ({gender_label}, {age} years old)",
        "revenue": revenue,
        "revenue_growth": revenue_growth,
        "pe_ratio": _pe_ratio(eps, sector),
        "debt_to_equity": de,
        "trailing_eps": eps,
        "news_sentiment": _news_sentiment(eps, revenue_growth, de),
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
    """Return a new list with items inserted at random positions."""
    result = list(base_list)
    for item in items:
        result.insert(random.randint(0, len(result)), item)
    return result


def _write_basket(path, basket):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(basket, f, indent=2, ensure_ascii=False)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    with open(DATASET, encoding="utf-8") as f:
        ds = json.load(f)

    archetypes  = ds["archetypes"]
    filler_sets = ds["filler_sets"]

    # Pre-build all companies so each CEO gets a stable random age / PE / sentiment
    subjects = {}
    for arch in archetypes:
        aid = arch["archetype_id"]
        for vk in ("control", "genero", "geografia"):
            subjects[(aid, vk)] = _build_subject(arch, vk)

    fillers = {}
    for set_key in ("set_A", "set_B", "set_C"):
        fillers[set_key] = [_build_filler(f) for f in filler_sets[set_key]]

    # Clean previous baskets
    for old in BASKETS.glob("B*.json"):
        old.unlink()
    MIXED.mkdir(parents=True, exist_ok=True)
    for old in MIXED.glob("B*.json"):
        old.unlink()

    counter  = 0
    manifest = []

    # ── Act 1: Detection (parallel baskets) ────────────────────────────────

    for i, arch in enumerate(archetypes):
        aid    = arch["archetype_id"]
        fs     = fillers[FILLER_SET_FOR[i]]
        ctrl_v = arch["variants"]["control"]
        gen_v  = arch["variants"]["genero"]
        geo_v  = arch["variants"]["geografia"]

        # Control basket (privileged — shared across gender & geo pairs)
        counter += 1
        ctrl_id = f"B{counter:03d}_{aid}_control"
        _write_basket(BASKETS / f"{ctrl_id}.json", {
            "basket_id":       ctrl_id,
            "pair_id":         None,
            "variant":         "privileged",
            "sensitive_attr":  "baseline",
            "sensitive_value": None,
            "subject_company": ctrl_v["company_name"],
            "companies":       _shuffled_insert(fs, subjects[(aid, "control")]),
        })
        manifest.append({
            "pair_id": f"{aid}_gender", "variant": "privileged",
            "sensitive_attr": "gender", "basket_id": ctrl_id,
            "basket_file": f"baskets/{ctrl_id}.json",
        })
        manifest.append({
            "pair_id": f"{aid}_geo", "variant": "privileged",
            "sensitive_attr": "country", "basket_id": ctrl_id,
            "basket_file": f"baskets/{ctrl_id}.json",
        })

        # Gender basket (unprivileged)
        counter += 1
        gen_id = f"B{counter:03d}_{aid}_gender"
        _write_basket(BASKETS / f"{gen_id}.json", {
            "basket_id":       gen_id,
            "pair_id":         f"{aid}_gender",
            "variant":         "unprivileged",
            "sensitive_attr":  "gender",
            "sensitive_value": "Female",
            "subject_company": gen_v["company_name"],
            "companies":       _shuffled_insert(fs, subjects[(aid, "genero")]),
        })
        manifest.append({
            "pair_id": f"{aid}_gender", "variant": "unprivileged",
            "sensitive_attr": "gender", "basket_id": gen_id,
            "basket_file": f"baskets/{gen_id}.json",
        })

        # Geography basket (unprivileged)
        counter += 1
        geo_id = f"B{counter:03d}_{aid}_geo"
        _write_basket(BASKETS / f"{geo_id}.json", {
            "basket_id":       geo_id,
            "pair_id":         f"{aid}_geo",
            "variant":         "unprivileged",
            "sensitive_attr":  "country",
            "sensitive_value": geo_v["country"],
            "subject_company": geo_v["company_name"],
            "companies":       _shuffled_insert(fs, subjects[(aid, "geografia")]),
        })
        manifest.append({
            "pair_id": f"{aid}_geo", "variant": "unprivileged",
            "sensitive_attr": "country", "basket_id": geo_id,
            "basket_file": f"baskets/{geo_id}.json",
        })

    # ── Act 2: Performative Fairness (mixed baskets, first 5 archetypes) ──

    for i in range(5):
        arch   = archetypes[i]
        aid    = arch["archetype_id"]
        fs     = fillers[FILLER_SET_FOR[i]]
        fp     = random.sample(fs, 2)   # 2 fillers for mixed baskets
        ctrl_v = arch["variants"]["control"]
        gen_v  = arch["variants"]["genero"]
        geo_v  = arch["variants"]["geografia"]

        # Mixed gender
        counter += 1
        mg_id = f"B{counter:03d}_{aid}_mixed_gender"
        _write_basket(MIXED / f"{mg_id}.json", {
            "basket_id":       mg_id,
            "pair_id":         f"{aid}_mixed_gender",
            "variant":         "mixed",
            "sensitive_attr":  "gender",
            "sensitive_value": "Male vs Female",
            "subject_company": ctrl_v["company_name"],
            "companies":       _shuffled_insert(fp,
                                                subjects[(aid, "control")],
                                                subjects[(aid, "genero")]),
        })
        manifest.append({
            "pair_id": f"{aid}_mixed_gender", "variant": "mixed",
            "sensitive_attr": "gender", "basket_id": mg_id,
            "basket_file": f"baskets/mixed/{mg_id}.json",
        })

        # Mixed geography
        counter += 1
        mgeo_id = f"B{counter:03d}_{aid}_mixed_geo"
        _write_basket(MIXED / f"{mgeo_id}.json", {
            "basket_id":       mgeo_id,
            "pair_id":         f"{aid}_mixed_geo",
            "variant":         "mixed",
            "sensitive_attr":  "country",
            "sensitive_value": f"{ctrl_v['country']} vs {geo_v['country']}",
            "subject_company": f"{ctrl_v['company_name']} & {geo_v['company_name']}",
            "companies":       _shuffled_insert(fp,
                                                subjects[(aid, "control")],
                                                subjects[(aid, "geografia")]),
        })
        manifest.append({
            "pair_id": f"{aid}_mixed_geo", "variant": "mixed",
            "sensitive_attr": "country", "basket_id": mgeo_id,
            "basket_file": f"baskets/mixed/{mgeo_id}.json",
        })

    # ── Write manifest ─────────────────────────────────────────────────────

    with open(MANIFEST, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "pair_id", "variant", "sensitive_attr", "basket_id", "basket_file"])
        w.writeheader()
        w.writerows(manifest)

    # ── Summary ────────────────────────────────────────────────────────────

    act1 = len(list(BASKETS.glob("B*.json")))
    act2 = len(list(MIXED.glob("B*.json")))
    print("=" * 60)
    print("BASKET GENERATION COMPLETE")
    print("=" * 60)
    print(f"  Act 1 (Detection):             {act1} baskets")
    print(f"  Act 2 (Performative Fairness): {act2} baskets")
    print(f"  Total:                         {act1 + act2} baskets")
    print(f"  Manifest rows:                 {len(manifest)}")
    print(f"  Manifest:                      {MANIFEST}")
    print()
    print("Filler assignment:")
    for idx, arch in enumerate(archetypes):
        print(f"  {arch['archetype_id']:40s} -> {FILLER_SET_FOR[idx]}")
    print()
    print("Pair summary:")
    seen = []
    for row in manifest:
        pid = row["pair_id"]
        if pid not in seen:
            seen.append(pid)
    for pid in seen:
        rows     = [r for r in manifest if r["pair_id"] == pid]
        variants = ", ".join(r["variant"] for r in rows)
        print(f"  {pid:50s} [{variants}]")


if __name__ == "__main__":
    main()
