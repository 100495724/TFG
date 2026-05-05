#!/usr/bin/env python3
"""
validate_baskets.py

Cheap pre-flight validation for generated investment baskets. This never calls
an LLM; it checks that Act 1 counterfactual variants clone the subject's
financial and derived fields before expensive experiments are run.
"""

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from config import BASKETS_DIR, BALANCE_SUBJECT_NAMES


COUNTERFACTUAL_FIELDS = [
    "sector",
    "industry",
    "revenue",
    "revenue_growth",
    "pe_ratio",
    "debt_to_equity",
    "trailing_eps",
    "news_sentiment",
]
REQUIRED_ATTRS = {"none", "gender", "country"}


def _stable_hash(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)


def _normalize_company_name(name: str) -> str:
    return " ".join(str(name).strip().casefold().split())


def _subject_name_set(subject_field: str) -> set[str]:
    return {
        _normalize_company_name(part)
        for part in str(subject_field).split(" & ")
        if part.strip()
    }


def _canonicalize_companies_for_shuffle(
    companies: list[dict],
    subject_field: str,
) -> list[dict]:
    subject_names = _subject_name_set(subject_field)
    subjects = []
    fillers = []
    for company in companies:
        company_copy = dict(company)
        if _normalize_company_name(company_copy["name"]) in subject_names:
            subjects.append(company_copy)
        else:
            fillers.append(company_copy)

    subjects = sorted(subjects, key=lambda c: _normalize_company_name(c["name"]))
    fillers = sorted(fillers, key=lambda c: _normalize_company_name(c["name"]))
    return subjects + fillers


def _ceo_age(ceo: str):
    match = re.search(r"(\d+)\s+years\s+old", str(ceo))
    return int(match.group(1)) if match else None


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["_path"] = str(path)
    return data


def _basket_files(baskets_dir: Path, include_mixed: bool, include_placebo: bool) -> list[Path]:
    files = sorted(baskets_dir.glob("*.json"))
    if include_mixed:
        files.extend(sorted((baskets_dir / "mixed").glob("*.json")))
    if include_placebo:
        files.extend(sorted((baskets_dir / "placebo").glob("*.json")))
    return files


def _find_subject(basket: dict):
    subject_name = basket.get("subject_company", "")
    companies = basket.get("companies", [])
    for company in companies:
        if company.get("name") == subject_name:
            return company
    return None


def _declared_subject_parts(basket: dict) -> list[str]:
    return [
        part.strip()
        for part in str(basket.get("subject_company", "")).split(" & ")
        if part.strip()
    ]


def _declared_subjects_exist(basket: dict) -> bool:
    subject_name = str(basket.get("subject_company", ""))
    company_counts = Counter(company.get("name") for company in basket.get("companies", []))
    if " & " in subject_name:
        declared_counts = Counter(_declared_subject_parts(basket))
        return all(company_counts[name] >= count for name, count in declared_counts.items())
    return company_counts[subject_name] > 0


def _format_values(values_by_attr: dict) -> str:
    return ", ".join(
        f"{attr}={values_by_attr.get(attr)!r}"
        for attr in sorted(values_by_attr)
    )


def _paired_shuffle_subject_position(basket: dict, seed: int) -> tuple[int, list[str]]:
    shuffle_key = basket.get("pair_id") or basket.get("basket_id")
    shuffle_seed = int(seed) + _stable_hash(str(shuffle_key))
    canonical_companies = _canonicalize_companies_for_shuffle(
        basket.get("companies", []),
        basket.get("subject_company", ""),
    )
    rng = random.Random(shuffle_seed)
    shuffled = list(canonical_companies)
    rng.shuffle(shuffled)

    subject_names = _subject_name_set(basket.get("subject_company", ""))
    subject_position = next(
        (
            i for i, company in enumerate(shuffled)
            if _normalize_company_name(company.get("name", "")) in subject_names
        ),
        -1,
    )
    return subject_position, [company.get("name", "") for company in shuffled]


def validate_paired_subject_positions(baskets: list[dict], seeds: list[int]) -> list[str]:
    """Simulate orchestrator shuffle and require paired Act 1 subject positions."""
    errors = []
    act1_baskets = [
        b for b in baskets
        if b.get("sensitive_attr") in REQUIRED_ATTRS and b.get("variant") != "mixed"
    ]
    grouped = defaultdict(list)
    for basket in act1_baskets:
        grouped[basket.get("pair_id", "")].append(basket)

    for pair_id, pair_baskets in sorted(grouped.items()):
        by_attr = defaultdict(list)
        for basket in pair_baskets:
            by_attr[basket.get("sensitive_attr")].append(basket)
        if set(by_attr) != REQUIRED_ATTRS or any(len(by_attr[attr]) != 1 for attr in REQUIRED_ATTRS):
            continue

        for seed in seeds:
            positions = {}
            orders = {}
            subject_fields = {}
            variants = {}
            for attr in sorted(REQUIRED_ATTRS):
                basket = by_attr[attr][0]
                position, order = _paired_shuffle_subject_position(basket, seed)
                positions[attr] = position
                orders[attr] = order
                subject_fields[attr] = basket.get("subject_company", "")
                variants[attr] = basket.get("variant", "")

            if len(set(positions.values())) != 1:
                detail = "; ".join(
                    f"{attr}/{variants[attr]}: subject_company={subject_fields[attr]!r}, "
                    f"pos={positions[attr]}, order={orders[attr]!r}"
                    for attr in sorted(REQUIRED_ATTRS)
                )
                errors.append(
                    f"pair_id={pair_id}, seed={seed}: paired subject positions differ: {detail}"
                )

    return errors


def validate_baskets(baskets: list[dict], require_subject_names: bool) -> list[str]:
    errors = []

    for basket in baskets:
        basket_id = basket.get("basket_id", "<missing>")
        companies = basket.get("companies", [])
        if len(companies) != 4:
            errors.append(
                f"basket_id={basket_id}: expected exactly 4 companies, found {len(companies)}"
            )
        if not _declared_subjects_exist(basket):
            errors.append(
                f"basket_id={basket_id}: subject_company {basket.get('subject_company')!r} "
                "is not present in companies"
            )
        if basket.get("variant") == "mixed":
            subject_field = str(basket.get("subject_company", ""))
            subject_parts = _declared_subject_parts(basket)
            company_counts = Counter(company.get("name") for company in companies)
            if " & " not in subject_field:
                errors.append(
                    f"basket_id={basket_id}: mixed basket subject_company must declare "
                    "two subjects separated by ' & '"
                )
            elif len(subject_parts) != 2:
                errors.append(
                    f"basket_id={basket_id}: mixed basket expected exactly two declared "
                    f"subjects, found {len(subject_parts)}: {subject_parts!r}"
                )
            else:
                declared_counts = Counter(subject_parts)
                missing_subjects = [
                    name for name, count in declared_counts.items()
                    if company_counts[name] < count
                ]
                if missing_subjects:
                    errors.append(
                        f"basket_id={basket_id}: mixed basket declared subjects missing "
                        f"from companies: {missing_subjects!r}"
                    )

    act1_baskets = [
        b for b in baskets
        if b.get("sensitive_attr") in REQUIRED_ATTRS and b.get("variant") != "mixed"
    ]
    grouped = defaultdict(list)
    for basket in act1_baskets:
        grouped[basket.get("pair_id", "")].append(basket)

    for pair_id, pair_baskets in sorted(grouped.items()):
        by_attr = defaultdict(list)
        for basket in pair_baskets:
            by_attr[basket.get("sensitive_attr")].append(basket)

        attrs_found = set(by_attr)
        if attrs_found != REQUIRED_ATTRS:
            errors.append(
                f"pair_id={pair_id}: expected sensitive_attr variants "
                f"{sorted(REQUIRED_ATTRS)}, found {sorted(attrs_found)}"
            )
            continue

        duplicate_attrs = [attr for attr, rows in by_attr.items() if len(rows) != 1]
        if duplicate_attrs:
            errors.append(
                f"pair_id={pair_id}: expected one basket per attr, duplicates/count issues for "
                f"{duplicate_attrs}"
            )
            continue

        subjects = {}
        for attr in sorted(REQUIRED_ATTRS):
            basket = by_attr[attr][0]
            subject = _find_subject(basket)
            if subject is None:
                errors.append(
                    f"pair_id={pair_id}, attr={attr}: declared subject "
                    f"{basket.get('subject_company')!r} not found"
                )
            else:
                subjects[attr] = subject

        if set(subjects) != REQUIRED_ATTRS:
            continue

        for field in COUNTERFACTUAL_FIELDS:
            values = {attr: subjects[attr].get(field) for attr in REQUIRED_ATTRS}
            if len({json.dumps(v, sort_keys=True) for v in values.values()}) != 1:
                errors.append(
                    f"pair_id={pair_id}: field {field!r} differs across variants: "
                    f"{_format_values(values)}"
                )

        ages = {attr: _ceo_age(subjects[attr].get("ceo", "")) for attr in REQUIRED_ATTRS}
        if None in ages.values() or len(set(ages.values())) != 1:
            errors.append(
                f"pair_id={pair_id}: CEO age differs or is missing across variants: "
                f"{_format_values(ages)}"
            )

        if require_subject_names:
            names = {attr: subjects[attr].get("name") for attr in REQUIRED_ATTRS}
            if len(set(names.values())) != 1:
                errors.append(
                    f"pair_id={pair_id}: subject company names differ across variants: "
                    f"{_format_values(names)}"
                )

    return errors


def main():
    parser = argparse.ArgumentParser(description="Validate generated basket assumptions.")
    parser.add_argument("--baskets-dir", default=BASKETS_DIR,
                        help="Directory containing basket JSON files")
    parser.add_argument("--include-mixed", action="store_true",
                        help="Also load data/baskets/mixed")
    parser.add_argument("--include-placebo", action="store_true",
                        help="Also load data/baskets/placebo")
    parser.add_argument("--require-subject-names-identical", action="store_true",
                        help="Require subject company names to match across variants")
    parser.add_argument("--check-paired-subject-position", action="store_true",
                        help="Simulate orchestrator shuffle and verify paired Act 1 subject positions")
    parser.add_argument("--seeds", default="42",
                        help="Comma-separated seeds for paired subject position validation")
    args = parser.parse_args()

    baskets_dir = Path(args.baskets_dir)
    files = _basket_files(baskets_dir, args.include_mixed, args.include_placebo)
    if not files:
        print(f"FAIL: no basket JSON files found in {baskets_dir}")
        sys.exit(1)

    baskets = [_load_json(path) for path in files]
    require_names = BALANCE_SUBJECT_NAMES or args.require_subject_names_identical
    errors = validate_baskets(baskets, require_subject_names=require_names)
    if args.check_paired_subject_position:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
        errors.extend(validate_paired_subject_positions(baskets, seeds))

    print("=" * 60)
    print("BASKET VALIDATION REPORT")
    print("=" * 60)
    print(f"Loaded basket files: {len(files)}")
    print(f"Subject names identical required: {require_names}")
    print(f"Paired subject position check: {args.check_paired_subject_position}")
    if args.check_paired_subject_position:
        print(f"Paired subject position seeds: {args.seeds}")

    if errors:
        print("\nFAIL")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)

    pair_ids = sorted({
        b.get("pair_id") for b in baskets
        if b.get("sensitive_attr") in REQUIRED_ATTRS and b.get("variant") != "mixed"
    })
    print(f"Act 1 pair_ids validated: {len(pair_ids)}")
    print("\nPASS")


if __name__ == "__main__":
    main()
