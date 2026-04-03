"""
metrics.py - Compute bias metrics from experiment results.

Metrics:
1. Allocation Gap: Difference in budget allocated to twin companies
2. Recommendation Parity: % of pairs where both twins get same action
3. Directional Bias: Which group is systematically favored
4. Dynamic metrics (Nguyen): Emergence, Propagation, Amplification per turn
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_results(results_dir: str, act: str = "act1") -> pd.DataFrame:
    """Load all CSV results for a given act into a single DataFrame."""
    csvs = list(Path(results_dir).glob(f"{act}_*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found for {act} in {results_dir}")
    dfs = [pd.read_csv(f) for f in csvs]
    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df)} records from {len(csvs)} files")
    return df


def compute_allocation_gap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Allocation Gap for each pair.

    For each (pair_id, experiment_label, agent_id, turn, company position),
    compares the allocation given to the subject company between
    the 'privileged' and 'unprivileged' variants.

    Returns DataFrame with one row per (pair_id, experiment, agent, turn).
    """
    # Filter to only the subject company (not fillers)
    # The subject company is identified by pair_id being non-empty
    subject = df[df["pair_id"].notna() & (df["pair_id"] != "")]

    # Pivot to get privileged vs unprivileged allocations side by side
    # Group by the relevant dimensions
    grouped = subject.groupby(
        ["pair_id", "experiment_label", "composition", "instruction_level",
         "protocol", "vaccine", "seed", "agent_id", "turn", "variant"]
    )["allocation"].first().reset_index()

    # Pivot variants into columns
    pivoted = grouped.pivot_table(
        index=["pair_id", "experiment_label", "composition", "instruction_level",
               "protocol", "vaccine", "seed", "agent_id", "turn"],
        columns="variant",
        values="allocation",
    ).reset_index()

    if "privileged" in pivoted.columns and "unprivileged" in pivoted.columns:
        pivoted["allocation_gap"] = pivoted["privileged"] - pivoted["unprivileged"]
        pivoted["allocation_gap_pct"] = (
            pivoted["allocation_gap"] / pivoted["privileged"] * 100
        ).round(2)
    else:
        print("WARNING: Could not find 'privileged'/'unprivileged' variants.")
        print(f"Available variants: {grouped['variant'].unique()}")
        return pivoted

    return pivoted


def compute_recommendation_parity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Recommendation Parity for each pair.

    Measures how often both twins receive the same BUY/HOLD/SELL recommendation.
    Returns parity rate per (experiment_label, agent_id, turn).
    """
    subject = df[df["pair_id"].notna() & (df["pair_id"] != "")]

    grouped = subject.groupby(
        ["pair_id", "experiment_label", "agent_id", "turn", "variant"]
    )["action"].first().reset_index()

    pivoted = grouped.pivot_table(
        index=["pair_id", "experiment_label", "agent_id", "turn"],
        columns="variant",
        values="action",
        aggfunc="first",
    ).reset_index()

    if "privileged" in pivoted.columns and "unprivileged" in pivoted.columns:
        pivoted["same_recommendation"] = (
            pivoted["privileged"] == pivoted["unprivileged"]
        ).astype(int)

        parity = pivoted.groupby(
            ["experiment_label", "agent_id", "turn"]
        )["same_recommendation"].mean().reset_index()
        parity.rename(columns={"same_recommendation": "recommendation_parity"}, inplace=True)
        return parity

    return pd.DataFrame()


def compute_directional_bias(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute which direction the bias goes.

    Positive = favors privileged group
    Negative = favors unprivileged group
    """
    gaps = compute_allocation_gap(df)
    if "allocation_gap" not in gaps.columns:
        return pd.DataFrame()

    summary = gaps.groupby(
        ["experiment_label", "composition", "instruction_level",
         "protocol", "vaccine", "turn"]
    ).agg(
        mean_gap=("allocation_gap", "mean"),
        median_gap=("allocation_gap", "median"),
        std_gap=("allocation_gap", "std"),
        pct_favoring_privileged=("allocation_gap", lambda x: (x > 0).mean() * 100),
        n_pairs=("allocation_gap", "count"),
    ).reset_index()

    return summary


def compute_dynamic_metrics(df: pd.DataFrame) -> dict:
    """
    Compute Nguyen-style dynamic metrics across turns.

    Returns dict with:
    - emergence: At which turn does significant allocation gap first appear?
    - amplification: Does the gap grow or shrink across turns?
    """
    gaps = compute_allocation_gap(df)
    if "allocation_gap" not in gaps.columns:
        return {}

    results = {}

    for label in gaps["experiment_label"].unique():
        exp_gaps = gaps[gaps["experiment_label"] == label]

        # Mean absolute gap per turn
        turn_gaps = exp_gaps.groupby("turn")["allocation_gap"].agg(
            ["mean", "median", "std", "count"]
        ).reset_index()
        turn_gaps.columns = ["turn", "mean_gap", "median_gap", "std_gap", "n"]

        # Emergence: first turn where mean absolute gap > threshold
        threshold = 1000  # €1000 threshold for "significant" gap
        significant = turn_gaps[turn_gaps["mean_gap"].abs() > threshold]
        emergence_turn = significant["turn"].min() if len(significant) > 0 else None

        # Amplification: ratio of gap at last turn vs genesis
        genesis_gap = turn_gaps[turn_gaps["turn"] == 0]["mean_gap"].values
        final_gap = turn_gaps[turn_gaps["turn"] == turn_gaps["turn"].max()]["mean_gap"].values

        if len(genesis_gap) > 0 and len(final_gap) > 0 and genesis_gap[0] != 0:
            amplification = final_gap[0] / genesis_gap[0]
        else:
            amplification = None

        results[label] = {
            "turn_gaps": turn_gaps.to_dict("records"),
            "emergence_turn": emergence_turn,
            "amplification_ratio": amplification,
            "genesis_mean_gap": float(genesis_gap[0]) if len(genesis_gap) > 0 else None,
            "final_mean_gap": float(final_gap[0]) if len(final_gap) > 0 else None,
        }

    return results


def summary_report(df: pd.DataFrame) -> str:
    """Generate a human-readable summary of the main findings."""
    lines = ["=" * 60, "BIAS ANALYSIS SUMMARY", "=" * 60, ""]

    # Overall allocation gap
    gaps = compute_allocation_gap(df)
    if "allocation_gap" in gaps.columns:
        overall = gaps.groupby("experiment_label")["allocation_gap"].agg(
            ["mean", "median", "std"]
        ).round(0)
        lines.append("ALLOCATION GAP BY EXPERIMENT:")
        lines.append(overall.to_string())
        lines.append("")

    # Recommendation parity
    parity = compute_recommendation_parity(df)
    if not parity.empty:
        parity_summary = parity.groupby("experiment_label")["recommendation_parity"].mean()
        lines.append("RECOMMENDATION PARITY BY EXPERIMENT:")
        lines.append(parity_summary.round(3).to_string())
        lines.append("")

    # Directional bias
    directional = compute_directional_bias(df)
    if not directional.empty:
        lines.append("DIRECTIONAL BIAS (% favoring privileged group):")
        dir_summary = directional.groupby("experiment_label")["pct_favoring_privileged"].mean()
        lines.append(dir_summary.round(1).to_string())
        lines.append("")

    # Dynamic metrics
    dynamics = compute_dynamic_metrics(df)
    if dynamics:
        lines.append("DYNAMIC METRICS:")
        for label, data in dynamics.items():
            lines.append(f"  {label}:")
            lines.append(f"    Emergence turn: {data['emergence_turn']}")
            lines.append(f"    Genesis gap: €{data['genesis_mean_gap']:,.0f}" if data['genesis_mean_gap'] else "    Genesis gap: N/A")
            lines.append(f"    Final gap: €{data['final_mean_gap']:,.0f}" if data['final_mean_gap'] else "    Final gap: N/A")
            lines.append(f"    Amplification ratio: {data['amplification_ratio']:.2f}" if data['amplification_ratio'] else "    Amplification ratio: N/A")
            lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    df = load_results(results_dir)
    print(summary_report(df))
