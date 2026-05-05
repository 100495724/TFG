#!/usr/bin/env python3
"""
fix_existing_csv.py - Patch result CSVs generated under the old pair_id schema.

Default behavior writes a new *_fixed.csv file and does not overwrite existing
results unless --inplace is explicitly requested.
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd


_BASKET_RE = re.compile(r"^B\d+_(.+?)_(control|gender|geo)$")


def extract_archetype_from_basket_id(basket_id: str) -> str | None:
    """Return archetype_id stripped of the B### prefix and variant suffix."""
    if not isinstance(basket_id, str):
        return None
    match = _BASKET_RE.match(basket_id)
    if not match:
        return None
    return match.group(1)


def fix_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the schema fix in a copy and return it."""
    df = df.copy()
    archetypes = df["basket_id"].apply(extract_archetype_from_basket_id)
    unmatched = df[archetypes.isna()]
    if not unmatched.empty:
        print(f"WARNING: {len(unmatched)} rows have unrecognised basket_id format:")
        print(unmatched["basket_id"].unique()[:10])

    df["sensitive_attr"] = df["sensitive_attr"].fillna("").astype(str)
    df["variant"] = df["variant"].fillna("").astype(str)

    suffix = df["basket_id"].apply(
        lambda b: _BASKET_RE.match(b).group(2)
        if isinstance(b, str) and _BASKET_RE.match(b)
        else None
    )

    is_control = suffix == "control"
    is_gender = suffix == "gender"
    is_geo = suffix == "geo"

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

    print("\nPair completeness (pair_id x sensitive_attr presence):")
    subject = (
        df[df["is_subject"].astype(str).str.lower() == "true"]
        if "is_subject" in df.columns
        else df
    )
    if subject.empty:
        print("  (no subject rows to inspect)")
        return
    pair_matrix = subject.groupby(["pair_id", "sensitive_attr"]).size().unstack(fill_value=0)
    print(pair_matrix.to_string())


def main():
    parser = argparse.ArgumentParser(description="Patch old-schema result CSVs.")
    parser.add_argument("csv", type=str, help="Input CSV path")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output CSV path (default: <input>_fixed.csv)")
    parser.add_argument("--inplace", action="store_true",
                        help="Overwrite input CSV instead of creating *_fixed.csv")
    args = parser.parse_args()

    input_path = Path(args.csv)
    if not input_path.exists():
        print(f"ERROR: input CSV not found: {input_path}")
        sys.exit(1)

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows from {input_path}")
    print_schema_summary(df, "BEFORE FIX")

    fixed = fix_dataframe(df)
    print_schema_summary(fixed, "AFTER FIX")

    if args.inplace:
        output_path = input_path
    elif args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_name(input_path.stem + "_fixed.csv")

    fixed.to_csv(output_path, index=False)
    print(f"\nWrote fixed CSV to: {output_path}")


if __name__ == "__main__":
    main()
