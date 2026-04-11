#!/usr/bin/env python3
"""
fix_existing_csv.py - Patch result CSVs generated under the old pair_id schema.

Old schema (what's in results/act1_baseline_seed42.csv):
  control rows:  pair_id=""           variant="privileged"     sensitive_attr="baseline"
  gender rows:   pair_id="<A>_gender"  variant="unprivileged"   sensitive_attr="gender"
  geo rows:      pair_id="<A>_geo"     variant="unprivileged"   sensitive_attr="country"

New schema (what metrics.py expects):
  control rows:  pair_id="<A>"         variant="control"        sensitive_attr="none"
  gender rows:   pair_id="<A>"         variant="unprivileged"   sensitive_attr="gender"
  geo rows:      pair_id="<A>"         variant="unprivileged"   sensitive_attr="country"

Usage:
    python fix_existing_csv.py results/act1_baseline_seed42.csv
    python fix_existing_csv.py results/act1_baseline_seed42.csv -o results/act1_baseline_seed42_fixed.csv
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd


# basket_id looks like "B001_A1_HighGrowth_HighDebt_control"
# archetype is the middle slice: "A1_HighGrowth_HighDebt"
_BASKET_RE = re.compile(r"^B\d+_(.+?)_(control|gender|geo)$")


def extract_archetype_from_basket_id(basket_id: str) -> str | None:
    """Return archetype_id stripped of the B### prefix and variant suffix."""
    if not isinstance(basket_id, str):
        return None
    m = _BASKET_RE.match(basket_id)
    if not m:
        return None
    return m.group(1)


def fix_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the schema fix in place (returns the same df for chaining)."""
    df = df.copy()

    # Derive archetype from basket_id for every row
    archetypes = df["basket_id"].apply(extract_archetype_from_basket_id)
    unmatched = df[archetypes.isna()]
    if not unmatched.empty:
        print(f"WARNING: {len(unmatched)} rows have unrecognised basket_id format:")
        print(unmatched["basket_id"].unique()[:10])

    # Normalise columns that may have NaN / empty for control rows
    df["sensitive_attr"] = df["sensitive_attr"].fillna("").astype(str)
    df["variant"] = df["variant"].fillna("").astype(str)

    # --- Identify row type from the suffix of basket_id ---
    suffix = df["basket_id"].apply(
        lambda b: _BASKET_RE.match(b).group(2) if isinstance(b, str) and _BASKET_RE.match(b) else None
    )

    is_control = suffix == "control"
    is_gender = suffix == "gender"
    is_geo = suffix == "geo"

    # --- Apply new schema ---
    df.loc[is_control, "pair_id"] = archetypes[is_control]
    df.loc[is_control, "variant"] = "control"
    df.loc[is_control, "sensitive_attr"] = "none"

    df.loc[is_gender, "pair_id"] = archetypes[is_gender]
    df.loc[is_gender, "variant"] = "unprivileged"
    df.loc[is_gender, "sensitive_attr"] = "gender"

    df.loc[is_geo, "pair_id"] = archetypes[is_geo]
    df.loc[is_geo, "variant"] = "unprivileged"
    df.loc[is_geo, "sensitive_attr"] = "country"

    return df


def print_schema_summary(df: pd.DataFrame, title: str):
    print(f"\n--- {title} ---")
    print(f"Total rows: {len(df)}")
    print("\nUnique (variant, sensitive_attr) combinations:")
    combos = df.groupby(["variant", "sensitive_attr"]).size().reset_index(name="rows")
    print(combos.to_string(index=False))

    print("\nPair completeness (pair_id × sensitive_attr presence):")
    subject = df[df["is_subject"].astype(str).str.lower() == "true"] if "is_subject" in df.columns else df
    if subject.empty:
        print("  (no subject rows to inspect)")
        return
    pair_matrix = subject.groupby(["pair_id", "sensitive_attr"]).size().unstack(fill_value=0)
    print(pair_matrix.to_string())

    # Warn on incomplete pairs
    incomplete = []
    for pid, row in pair_matrix.iterrows():
        if not pid:
            continue
        has_control = row.get("none", 0) > 0
        has_gender = row.get("gender", 0) > 0
        has_country = row.get("country", 0) > 0
        missing = []
        if not has_control:
            missing.append("control/none")
        if not has_gender:
            missing.append("gender")
        if not has_country:
            missing.append("country")
        if missing:
            incomplete.append((pid, missing))

    if incomplete:
        print("\nWARNING: incomplete pairs:")
        for pid, missing in incomplete:
            print(f"  {pid}: missing {missing}")
    else:
        print("\nAll pairs complete (control/none + gender + country present).")


def main():
    ap = argparse.ArgumentParser(description="Patch old-schema result CSVs to the new pair_id schema.")
    ap.add_argument("csv", type=str, help="Input CSV path (old schema)")
    ap.add_argument("-o", "--output", type=str, default=None,
                    help="Output CSV path (default: <input>_fixed.csv)")
    ap.add_argument("--inplace", action="store_true",
                    help="Overwrite input CSV instead of creating *_fixed.csv")
    args = ap.parse_args()

    in_path = Path(args.csv)
    if not in_path.exists():
        print(f"ERROR: input CSV not found: {in_path}")
        sys.exit(1)

    df = pd.read_csv(in_path)
    print(f"Loaded {len(df)} rows from {in_path}")

    print_schema_summary(df, "BEFORE FIX")

    fixed = fix_dataframe(df)

    print_schema_summary(fixed, "AFTER FIX")

    if args.inplace:
        out_path = in_path
    elif args.output:
        out_path = Path(args.output)
    else:
        out_path = in_path.with_name(in_path.stem + "_fixed.csv")

    fixed.to_csv(out_path, index=False)
    print(f"\nWrote fixed CSV to: {out_path}")


if __name__ == "__main__":
    main()
