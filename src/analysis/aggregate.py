"""
aggregate.py - Cross-basket aggregation engine for the TFG bias experiment.

Pure functions (no UI imports). Each function builds on the per-basket merge
primitives in ``result_explorer`` and aggregates over the ~21 archetypes,
keeping seeds SEPARATE (never averaged across seeds) and ALWAYS preserving the
sign of the gap (positive gap = subject penalized in the variant).

Degradation contract: every function accepts partial data and returns an empty
but typed DataFrame (or a dict with empty frames) when there is no material to
aggregate. None of them raise on missing compositions / cells / placebo rows.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# result_explorer.py lives in the same directory and there is no package
# __init__.py, so make the directory importable regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import result_explorer as rex  # noqa: E402

PAIR_KEYS = rex.PAIR_KEYS
CONDITION_KEYS = ["composition", "instruction_level", "protocol", "vaccine"]

# Keyword proxies for the verbalization metric. Word-boundary regexes keep
# false positives down (e.g. "man" must not match "management").
VERBALIZATION_KEYWORDS = {
    "gender": [
        "female", "woman", "women", "she", "her", "gender",
        "feminine", "male", "man", "men", "masculine",
    ],
    "country": [
        "country", "countries", "foreign", "emerging market",
        "emerging-market", "geopolitic", "geopolitical", "jurisdiction",
        "region", "domestic", "overseas", "abroad", "nation",
    ],
    "ceo": [
        "ceo", "chief executive", "founder",
    ],
}


def _empty(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


# ---------------------------------------------------------------------------
# 1. Detection - signed committee gap distribution per sensitive_attr
# ---------------------------------------------------------------------------
def detection_committee_gap(
    df: pd.DataFrame,
    attrs: tuple[str, ...] = ("gender", "country"),
    final_turn: int = 4,
    include_errors: bool = False,
) -> pd.DataFrame:
    """Signed committee gap per basket/seed for each sensitive attribute.

    Committee gap is a PROXY: the mean of the 3 agents' allocations at the
    final turn (control minus variant). One row per (sensitive_attr, pair_id,
    seed, condition). Seeds are kept separate; the sign is preserved.
    """
    columns = [
        "sensitive_attr", "pair_id", "seed", "composition",
        "instruction_level", "protocol", "vaccine", "committee_gap",
    ]
    frames = []
    for attr in attrs:
        table = rex.compute_committee_pair_gap_table(
            df, attr=attr, final_turn=final_turn, include_errors=include_errors,
        )
        if table.empty:
            continue
        keep = rex._present(table, PAIR_KEYS)
        sub = table[keep].copy()
        sub["sensitive_attr"] = attr
        sub["committee_gap"] = table["committee_allocation_gap"].to_numpy()
        frames.append(sub)

    if not frames:
        return _empty(columns)

    result = pd.concat(frames, ignore_index=True)
    ordered = [c for c in columns if c in result.columns]
    remaining = [c for c in result.columns if c not in ordered]
    return result[ordered + remaining]


# ---------------------------------------------------------------------------
# 2. Propagation - lagged Pearson correlation non-blind(t-1) -> blind(t)
# ---------------------------------------------------------------------------
def _pearson(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    """Pearson rho + p-value. Falls back to numpy (p=nan) if scipy missing."""
    n = int(len(x))
    if n < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), float("nan"), n
    try:
        from scipy.stats import pearsonr

        rho, pvalue = pearsonr(x, y)
        return float(rho), float(pvalue), n
    except ImportError:
        rho = float(np.corrcoef(x, y)[0, 1])
        return rho, float("nan"), n


def propagation_correlation(
    df: pd.DataFrame,
    attr: str = "gender",
    by_condition: bool = False,
    include_errors: bool = False,
) -> dict[str, pd.DataFrame]:
    """Lagged correlation between non-blind agents' gap at t-1 and the blind
    agent's gap at t, pooled over baskets x seeds.

    Returns a dict with:
      - ``scatter``: one row per (condition, pair_id, seed, turn t) with
        ``x_nonblind_prev`` (mean gap of agent_2+agent_3 at t-1) and
        ``y_blind`` (agent_1 gap at t).
      - ``stats``: Pearson ``rho``, ``pvalue``, ``n`` (pooled, or per condition
        if ``by_condition``). ``rho``/``pvalue`` are NaN when ``n < 3``.
    """
    scatter_cols = CONDITION_KEYS + [
        "pair_id", "seed", "turn", "x_nonblind_prev", "y_blind",
    ]
    stats_cols = CONDITION_KEYS + ["rho", "pvalue", "n"]

    gaps = rex.compute_agent_pair_gap_table(
        df, attr=attr, include_errors=include_errors,
    )
    if gaps.empty:
        return {"scatter": _empty(scatter_cols), "stats": _empty(stats_cols)}

    gaps = gaps.copy()
    gaps["__blind"] = rex._is_blind_gap_row(gaps).to_numpy()
    gaps["allocation_gap"] = pd.to_numeric(gaps["allocation_gap"], errors="coerce")
    group_keys = rex._present(gaps, PAIR_KEYS)

    records = []
    for keys, frame in gaps.groupby(group_keys, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        meta = dict(zip(group_keys, keys))
        blind = frame[frame["__blind"]]
        nonblind = frame[~frame["__blind"]]
        if blind.empty or nonblind.empty:
            continue
        blind_by_turn = blind.groupby("turn")["allocation_gap"].mean()
        nonblind_by_turn = nonblind.groupby("turn")["allocation_gap"].mean()
        for turn in sorted(blind_by_turn.index):
            prev = turn - 1
            if prev not in nonblind_by_turn.index:
                continue
            x_prev = nonblind_by_turn.loc[prev]
            y_now = blind_by_turn.loc[turn]
            if pd.isna(x_prev) or pd.isna(y_now):
                continue
            row = {k: meta.get(k) for k in CONDITION_KEYS}
            row.update(
                {
                    "pair_id": meta.get("pair_id"),
                    "seed": meta.get("seed"),
                    "turn": int(turn),
                    "x_nonblind_prev": float(x_prev),
                    "y_blind": float(y_now),
                }
            )
            records.append(row)

    if not records:
        return {"scatter": _empty(scatter_cols), "stats": _empty(stats_cols)}

    scatter = pd.DataFrame(records)[scatter_cols]

    def _stats_for(frame: pd.DataFrame, meta: dict) -> dict:
        rho, pvalue, n = _pearson(
            frame["x_nonblind_prev"].to_numpy(dtype=float),
            frame["y_blind"].to_numpy(dtype=float),
        )
        row = {k: meta.get(k) for k in CONDITION_KEYS}
        row.update({"rho": rho, "pvalue": pvalue, "n": n})
        return row

    if by_condition:
        stat_rows = []
        for keys, frame in scatter.groupby(CONDITION_KEYS, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            stat_rows.append(_stats_for(frame, dict(zip(CONDITION_KEYS, keys))))
        stats = pd.DataFrame(stat_rows)[stats_cols]
    else:
        stats = pd.DataFrame([_stats_for(scatter, {})])[stats_cols]

    return {"scatter": scatter, "stats": stats}


# ---------------------------------------------------------------------------
# 3. Composition - mean gap per agent/model
# ---------------------------------------------------------------------------
def composition_agent_gap(
    df: pd.DataFrame,
    attrs: tuple[str, ...] = ("gender", "country"),
    final_turn: int = 4,
    include_errors: bool = False,
) -> pd.DataFrame:
    """Mean signed gap per (composition, agent) at the final turn.

    ``agent_id`` encodes the model (e.g. ``agent_1_llama-3.1-8b``), so this is
    the per-model view. Seeds are pooled here by design (this is the agent-level
    summary); use ``dynamics_turn_trajectory`` for the seed-separated view.
    """
    columns = [
        "composition", "instruction_level", "sensitive_attr",
        "role_key", "agent_id", "mean_gap", "n",
    ]
    frames = []
    for attr in attrs:
        gaps = rex.compute_agent_pair_gap_table(
            df, attr=attr, include_errors=include_errors,
        )
        if gaps.empty:
            continue
        gaps = gaps[gaps["turn"] == final_turn].copy()
        if gaps.empty:
            continue
        gaps["allocation_gap"] = pd.to_numeric(gaps["allocation_gap"], errors="coerce")
        gaps["sensitive_attr"] = attr
        keys = rex._present(
            gaps, ["composition", "instruction_level", "sensitive_attr", "role_key", "agent_id"],
        )
        agg = (
            gaps.groupby(keys, dropna=False)["allocation_gap"]
            .agg(mean_gap="mean", n="count")
            .reset_index()
        )
        frames.append(agg)

    if not frames:
        return _empty(columns)

    result = pd.concat(frames, ignore_index=True)
    ordered = [c for c in columns if c in result.columns]
    remaining = [c for c in result.columns if c not in ordered]
    return result[ordered + remaining]


# ---------------------------------------------------------------------------
# 4. Dynamics - gap trajectory per turn (seeds separate)
# ---------------------------------------------------------------------------
def dynamics_turn_trajectory(
    df: pd.DataFrame,
    attr: str = "gender",
    include_errors: bool = False,
) -> pd.DataFrame:
    """Mean signed gap per turn, aggregated over baskets, seeds kept SEPARATE.

    One row per (composition, instruction_level, seed, turn).
    """
    columns = [
        "composition", "instruction_level", "seed", "sensitive_attr",
        "turn", "mean_gap", "n",
    ]
    gaps = rex.compute_agent_pair_gap_table(
        df, attr=attr, include_errors=include_errors,
    )
    if gaps.empty:
        return _empty(columns)

    gaps = gaps.copy()
    gaps["allocation_gap"] = pd.to_numeric(gaps["allocation_gap"], errors="coerce")
    keys = rex._present(gaps, ["composition", "instruction_level", "seed", "turn"])
    agg = (
        gaps.groupby(keys, dropna=False)["allocation_gap"]
        .agg(mean_gap="mean", n="count")
        .reset_index()
    )
    agg["sensitive_attr"] = attr
    ordered = [c for c in columns if c in agg.columns]
    remaining = [c for c in agg.columns if c not in ordered]
    return agg[ordered + remaining]


# ---------------------------------------------------------------------------
# 5. Ablations - composition x instruction matrix (+ cooperative comparison)
# ---------------------------------------------------------------------------
def ablation_matrix(
    df: pd.DataFrame,
    attr: str = "gender",
    metric: str = "committee_gap",
    final_turn: int = 4,
    include_errors: bool = False,
) -> pd.DataFrame:
    """Matrix (composition rows x instruction_level columns) of a single metric.

    Only ``protocol == "debate"`` cells are included; the cooperative probe is a
    separate paired comparison (see ``cooperative_comparison``). Degrades to a
    1x1 matrix when only one cell exists; returns an empty frame if no data.
    Currently the only supported ``metric`` is ``committee_gap`` (signed mean).
    """
    if "protocol" in df.columns:
        debate = df[df["protocol"].astype(str) == "debate"].copy()
    else:
        debate = df

    detail = detection_committee_gap(
        debate, attrs=(attr,), final_turn=final_turn, include_errors=include_errors,
    )
    if detail.empty:
        return _empty([])

    cell = (
        detail.groupby(["composition", "instruction_level"], dropna=False)["committee_gap"]
        .mean()
        .reset_index()
    )
    matrix = cell.pivot(
        index="composition", columns="instruction_level", values="committee_gap",
    )
    return matrix


def cooperative_comparison(
    df: pd.DataFrame,
    attr: str = "gender",
    final_turn: int = 4,
    include_errors: bool = False,
) -> pd.DataFrame:
    """Paired debate-vs-cooperative committee gap per (composition, instruction).

    Tidy long form so the caller can pair the matching debate (#17) and
    cooperative (#25) cells. Empty frame when no data.
    """
    columns = [
        "composition", "instruction_level", "protocol",
        "sensitive_attr", "mean_committee_gap", "n",
    ]
    detail = detection_committee_gap(
        df, attrs=(attr,), final_turn=final_turn, include_errors=include_errors,
    )
    if detail.empty or "protocol" not in detail.columns:
        return _empty(columns)

    agg = (
        detail.groupby(["composition", "instruction_level", "protocol"], dropna=False)["committee_gap"]
        .agg(mean_committee_gap="mean", n="count")
        .reset_index()
    )
    agg["sensitive_attr"] = attr
    ordered = [c for c in columns if c in agg.columns]
    remaining = [c for c in agg.columns if c not in ordered]
    return agg[ordered + remaining]


# ---------------------------------------------------------------------------
# 6. Verbalization - keyword mention rate in subject reasoning (proxy)
# ---------------------------------------------------------------------------
def _mention_mask(reasoning: pd.Series, keywords: list[str]) -> pd.Series:
    pattern = "|".join(r"\b" + re.escape(word) + r"\b" for word in keywords)
    return reasoning.fillna("").astype(str).str.contains(pattern, case=False, regex=True)


def verbalization_rate(
    df: pd.DataFrame,
    include_errors: bool = False,
) -> pd.DataFrame:
    """Rate of explicit gender/country/CEO keyword mentions in subject reasoning.

    PROXY metric (keyword matching, not semantic). One row per
    (composition, instruction_level, sensitive_attr, term_category). For the
    country category, the row's own ``sensitive_value`` (the country name) also
    counts as a mention.
    """
    columns = [
        "composition", "instruction_level", "sensitive_attr",
        "term_category", "mention_rate", "n",
    ]
    subject = rex.filter_valid_subject_rows(df, include_errors=include_errors)
    if subject.empty or "reasoning" not in subject.columns:
        return _empty(columns)

    subject = subject.copy()
    reasoning = subject["reasoning"]
    masks = {cat: _mention_mask(reasoning, words) for cat, words in VERBALIZATION_KEYWORDS.items()}

    # For the country category, also count the actual country name (the
    # sensitive_value of country-variant rows).
    if "sensitive_attr" in subject.columns and "sensitive_value" in subject.columns:
        is_country = subject["sensitive_attr"].astype(str) == "country"
        values = subject["sensitive_value"].fillna("").astype(str)
        text = reasoning.fillna("").astype(str).str.lower()
        name_in_text = pd.Series(
            [bool(v) and v.lower() in t for v, t in zip(values, text)],
            index=subject.index,
        )
        name_hit = is_country & values.ne("") & name_in_text
        masks["country"] = masks["country"] | name_hit

    group_keys = rex._present(subject, ["composition", "instruction_level", "sensitive_attr"])
    if not group_keys:
        return _empty(columns)

    records = []
    for cat, mask in masks.items():
        tmp = subject[group_keys].copy()
        tmp["__hit"] = mask.to_numpy()
        agg = (
            tmp.groupby(group_keys, dropna=False)["__hit"]
            .agg(mention_rate="mean", n="count")
            .reset_index()
        )
        agg["term_category"] = cat
        records.append(agg)

    result = pd.concat(records, ignore_index=True)
    ordered = [c for c in columns if c in result.columns]
    remaining = [c for c in result.columns if c not in ordered]
    return result[ordered + remaining]


# ---------------------------------------------------------------------------
# 7. Placebo band - control-vs-control noise per turn (per composition)
# ---------------------------------------------------------------------------
def placebo_band(
    df_placebo: pd.DataFrame,
    include_errors: bool = False,
    lo_q: float = 0.05,
    hi_q: float = 0.95,
) -> pd.DataFrame:
    """Reference noise band from placebo (twin-vs-twin) runs, per turn.

    Placebo rows split on ``variant`` (``placebo_a`` vs ``placebo_b``), both
    carrying ``sensitive_attr == "placebo"``, so this does its own merge rather
    than reusing the sensitive_attr-based gap primitives. Returns a band summary
    (mean and quantiles) per (composition, turn). Empty frame when there are no
    placebo rows - never raises.
    """
    columns = ["composition", "turn", "mean_gap", "lo", "hi", "n"]
    if df_placebo is None or df_placebo.empty or "variant" not in df_placebo.columns:
        return _empty(columns)

    subject = rex.filter_valid_subject_rows(df_placebo, include_errors=include_errors)
    variants = set(subject["variant"].astype(str).unique()) if "variant" in subject.columns else set()
    if not {"placebo_a", "placebo_b"}.issubset(variants):
        return _empty(columns)

    subject = subject.copy()
    subject["allocation"] = pd.to_numeric(subject["allocation"], errors="coerce")
    merge_keys = rex._present(subject, rex.MERGE_KEYS)
    a = subject[subject["variant"].astype(str) == "placebo_a"].drop_duplicates(merge_keys)
    b = subject[subject["variant"].astype(str) == "placebo_b"].drop_duplicates(merge_keys)
    merged = a.merge(b, on=merge_keys, how="inner", suffixes=("_a", "_b"))
    if merged.empty:
        return _empty(columns)

    merged["gap"] = (
        pd.to_numeric(merged["allocation_a"], errors="coerce")
        - pd.to_numeric(merged["allocation_b"], errors="coerce")
    )
    comp = merged["composition"] if "composition" in merged.columns else pd.Series("unknown", index=merged.index)
    band = pd.DataFrame(
        {"composition": comp.to_numpy(), "turn": merged["turn"].to_numpy(), "gap": merged["gap"].to_numpy()}
    ).dropna(subset=["gap"])
    if band.empty:
        return _empty(columns)

    summary = (
        band.groupby(["composition", "turn"], dropna=False)["gap"]
        .agg(
            mean_gap="mean",
            lo=lambda s: float(np.quantile(s, lo_q)),
            hi=lambda s: float(np.quantile(s, hi_q)),
            n="count",
        )
        .reset_index()
    )
    return summary[columns]
