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
# 2. Debate coupling - lagged Pearson correlation non-blind(t-1) -> blind(t)
#
# This is NOT bias propagation: it is a generic debate-coupling channel. It
# must ALWAYS be reported next to its placebo reference (twin-vs-twin runs give
# essentially the same correlation), so ``debate_coupling`` embeds the placebo
# figures when a placebo frame is supplied and no consumer can show the
# treatment correlation without its null.
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


def _placebo_agent_gap_long(
    df_placebo: pd.DataFrame,
    include_errors: bool = False,
) -> pd.DataFrame:
    """Per-agent placebo gap (``placebo_b - placebo_a``) by pair/seed/agent/turn.

    Mirrors the shape of ``rex.compute_agent_pair_gap_table`` (same PAIR_KEYS,
    ``role_key``, ``is_blind``, ``turn``) but the gap is the twin-vs-twin
    difference, so it carries no sensitive attribute. Empty frame (never raises)
    when there are no ``placebo_a``/``placebo_b`` rows.
    """
    columns = rex.PAIR_KEYS + ["agent_id", "role_key", "is_blind", "turn", "gap"]
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

    out = pd.DataFrame()
    for column in rex._present(merged, rex.PAIR_KEYS + ["agent_id", "turn"]):
        out[column] = merged[column]
    out["role_key"] = merged.get("role_key_a", merged.get("role_key"))
    out["is_blind"] = merged.get("is_blind_a", merged.get("is_blind"))
    # placebo_b - placebo_a (same sign convention as variant - control).
    out["gap"] = (
        pd.to_numeric(merged["allocation_b"], errors="coerce")
        - pd.to_numeric(merged["allocation_a"], errors="coerce")
    )
    ordered = [c for c in columns if c in out.columns]
    remaining = [c for c in out.columns if c not in ordered]
    return out[ordered + remaining]


def _lagged_coupling(
    gaps: pd.DataFrame,
    value_col: str,
    by_condition: bool = False,
) -> dict[str, pd.DataFrame]:
    """Lagged non-blind(t-1) -> blind(t) correlation over a per-agent gap table.

    ``gaps`` must expose PAIR_KEYS, ``turn``, a blind marker (``is_blind`` or
    ``role_key``) and ``value_col``. Returns ``{"scatter", "stats"}``.
    """
    scatter_cols = CONDITION_KEYS + [
        "pair_id", "seed", "turn", "x_nonblind_prev", "y_blind",
    ]
    stats_cols = CONDITION_KEYS + ["rho", "pvalue", "n"]

    if gaps is None or gaps.empty or value_col not in gaps.columns:
        return {"scatter": _empty(scatter_cols), "stats": _empty(stats_cols)}

    gaps = gaps.copy()
    gaps["__blind"] = rex._is_blind_gap_row(gaps).to_numpy()
    gaps[value_col] = pd.to_numeric(gaps[value_col], errors="coerce")
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
        blind_by_turn = blind.groupby("turn")[value_col].mean()
        nonblind_by_turn = nonblind.groupby("turn")[value_col].mean()
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


def placebo_debate_coupling(
    df_placebo: pd.DataFrame,
    by_condition: bool = False,
    include_errors: bool = False,
) -> dict[str, pd.DataFrame]:
    """Lagged non-blind(t-1) -> blind(t) coupling on placebo twin-vs-twin gaps.

    Same computation as ``debate_coupling`` but on ``placebo_b - placebo_a``.
    This is the reference r that ALWAYS accompanies the treatment coupling.
    """
    gaps = _placebo_agent_gap_long(df_placebo, include_errors=include_errors)
    return _lagged_coupling(gaps, value_col="gap", by_condition=by_condition)


def debate_coupling(
    df: pd.DataFrame,
    attr: str = "gender",
    by_condition: bool = False,
    include_errors: bool = False,
    df_placebo: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Lagged correlation between non-blind agents' gap at t-1 and the blind
    agent's gap at t, pooled over baskets x seeds. Generic debate coupling, NOT
    bias propagation.

    Returns a dict with:
      - ``scatter``: one row per (condition, pair_id, seed, turn t) with
        ``x_nonblind_prev`` (mean gap of agent_2+agent_3 at t-1) and
        ``y_blind`` (agent_1 gap at t).
      - ``stats``: Pearson ``rho``, ``pvalue``, ``n`` (pooled, or per condition
        if ``by_condition``). ``rho``/``pvalue`` are NaN when ``n < 3``.
    When ``df_placebo`` is supplied, also returns ``placebo_scatter`` and
    ``placebo_stats`` so the treatment coupling can never be shown without its
    placebo reference.
    """
    gaps = rex.compute_agent_pair_gap_table(
        df, attr=attr, include_errors=include_errors,
    )
    out = _lagged_coupling(gaps, value_col="allocation_gap", by_condition=by_condition)
    if df_placebo is not None:
        placebo = placebo_debate_coupling(
            df_placebo, by_condition=by_condition, include_errors=include_errors,
        )
        out["placebo_scatter"] = placebo["scatter"]
        out["placebo_stats"] = placebo["stats"]
    return out


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
# 5. Ablations - composition x instruction matrix
# ---------------------------------------------------------------------------
def ablation_matrix(
    df: pd.DataFrame,
    attr: str = "gender",
    metric: str = "committee_gap",
    final_turn: int = 4,
    include_errors: bool = False,
) -> pd.DataFrame:
    """Matrix (composition rows x instruction_level columns) of a single metric.

    Only ``protocol == "debate"`` cells are included. Degrades to a 1x1 matrix
    when only one cell exists; returns an empty frame if no data. Currently the
    only supported ``metric`` is ``committee_gap`` (signed mean).
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


# ---------------------------------------------------------------------------
# 8. Genesis (t=0) statistics with the placebo as the empirical null
#
# Sign convention (CRITICAL): all functions below report the gap as
# ``variant - control`` (i.e. ``-allocation_gap`` from result_explorer, which
# returns ``control - test``) so a NEGATIVE gap means the subject was penalized
# in the variant. Placebo gap is ``placebo_b - placebo_a``. The bias endpoint
# is the genesis turn t=0; permutation / bootstrap use a FIXED seed (0).
# ---------------------------------------------------------------------------
GENESIS_STAT_COLS = [
    "sensitive_attr", "instruction_level", "mean_gap", "sd",
    "n_pairs", "p_signflip", "ci95_lo", "ci95_hi",
]


def _signflip_p(values, n_perm: int = 10000, seed: int = 0) -> float:
    """Two-sided sign-flip permutation p-value for a mean == 0 null."""
    v = np.asarray(list(values), dtype=float)
    v = v[~np.isnan(v)]
    n = v.size
    if n == 0:
        return float("nan")
    obs = abs(float(v.mean()))
    rng = np.random.default_rng(seed)
    signs = rng.integers(0, 2, size=(n_perm, n)) * 2 - 1
    perm_means = np.abs((signs * v).mean(axis=1))
    return float((perm_means >= obs).mean())


def _bootstrap_ci(
    values, n_boot: int = 10000, seed: int = 0, lo: float = 2.5, hi: float = 97.5,
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean (fixed seed)."""
    v = np.asarray(list(values), dtype=float)
    v = v[~np.isnan(v)]
    n = v.size
    if n == 0:
        return float("nan"), float("nan")
    if n == 1:
        return float(v[0]), float(v[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = v[idx].mean(axis=1)
    return float(np.percentile(boot_means, lo)), float(np.percentile(boot_means, hi))


def _treatment_agent_gap_long(
    df: pd.DataFrame, attr: str, include_errors: bool = False,
) -> pd.DataFrame:
    """Per-agent treatment gap in the ``variant - control`` convention."""
    gaps = rex.compute_agent_pair_gap_table(df, attr=attr, include_errors=include_errors)
    if gaps.empty:
        return gaps
    gaps = gaps.copy()
    gaps["gap"] = -pd.to_numeric(gaps["allocation_gap"], errors="coerce")
    return gaps


def _seed_avg_pair_gaps(
    long: pd.DataFrame, turn: int, visible_only: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate an agent-gap long frame to per-pair gaps.

    Returns ``(committee, seed_avg)`` where ``committee`` is one gap per
    (instruction_level, seed, pair_id) [pair x seed units] and ``seed_avg`` is
    one gap per (instruction_level, pair_id) [seeds averaged].
    """
    empty = _empty(["instruction_level", "seed", "pair_id", "gap"])
    if long is None or long.empty or "gap" not in long.columns:
        return empty, _empty(["instruction_level", "pair_id", "gap"])
    sub = long.copy()
    if "turn" in sub.columns:
        sub = sub[pd.to_numeric(sub["turn"], errors="coerce") == turn]
    if visible_only and "is_blind" in sub.columns:
        sub = sub[~rex._truthy(sub["is_blind"])]
    sub = sub.copy()
    sub["gap"] = pd.to_numeric(sub["gap"], errors="coerce")
    keys1 = rex._present(sub, ["instruction_level", "seed", "pair_id"])
    if not keys1:
        return empty, _empty(["instruction_level", "pair_id", "gap"])
    committee = sub.groupby(keys1, dropna=False)["gap"].mean().reset_index()
    keys2 = rex._present(committee, ["instruction_level", "pair_id"])
    seed_avg = committee.groupby(keys2, dropna=False)["gap"].mean().reset_index()
    return committee, seed_avg


def _stats_row(sensitive_attr: str, instruction_level: str, values) -> dict:
    v = np.asarray(list(values), dtype=float)
    v = v[~np.isnan(v)]
    n = int(v.size)
    if n == 0:
        return {
            "sensitive_attr": sensitive_attr, "instruction_level": instruction_level,
            "mean_gap": float("nan"), "sd": float("nan"), "n_pairs": 0,
            "p_signflip": float("nan"), "ci95_lo": float("nan"), "ci95_hi": float("nan"),
        }
    lo, hi = _bootstrap_ci(v)
    return {
        "sensitive_attr": sensitive_attr,
        "instruction_level": instruction_level,
        "mean_gap": float(v.mean()),
        "sd": float(v.std(ddof=1)) if n > 1 else float("nan"),
        "n_pairs": n,
        "p_signflip": _signflip_p(v),
        "ci95_lo": lo,
        "ci95_hi": hi,
    }


def _summarize_pair_gaps(seed_avg: pd.DataFrame, sensitive_attr: str) -> pd.DataFrame:
    """One stats row per instruction_level + a ``pooled`` row (levels grouped)."""
    if seed_avg is None or seed_avg.empty:
        return _empty(GENESIS_STAT_COLS)
    rows = []
    if "instruction_level" in seed_avg.columns:
        for lvl, frame in seed_avg.groupby("instruction_level", dropna=False):
            rows.append(_stats_row(sensitive_attr, str(lvl), frame["gap"].to_numpy()))
    rows.append(_stats_row(sensitive_attr, "pooled", seed_avg["gap"].to_numpy()))
    return pd.DataFrame(rows)[GENESIS_STAT_COLS]


def genesis_gap_stats(
    df: pd.DataFrame,
    attrs: tuple[str, ...] = ("gender", "country"),
    turn: int = 0,
    visible_only: bool = True,
    include_errors: bool = False,
) -> pd.DataFrame:
    """Genesis (t=0) gap statistics per (sensitive_attr, instruction_level).

    Gap is ``variant - control`` at the agent level (visible agents only when
    ``visible_only``), averaged first to committee-partial level by
    (instruction_level, seed, pair_id), then across seeds by
    (instruction_level, pair_id). Returns per-level rows plus a ``pooled`` row
    (levels grouped) with ``mean_gap, sd, n_pairs, p_signflip`` (sign-flip
    permutation, 10k, fixed seed) and ``ci95_lo/ci95_hi`` (mean bootstrap, 10k).
    """
    frames = []
    for attr in attrs:
        long = _treatment_agent_gap_long(df, attr, include_errors=include_errors)
        _, seed_avg = _seed_avg_pair_gaps(long, turn=turn, visible_only=visible_only)
        summ = _summarize_pair_gaps(seed_avg, attr)
        if not summ.empty:
            frames.append(summ)
    if not frames:
        return _empty(GENESIS_STAT_COLS)
    return pd.concat(frames, ignore_index=True)


def placebo_gap_stats(
    df_placebo: pd.DataFrame,
    turn: int = 0,
    visible_only: bool = True,
    include_errors: bool = False,
) -> pd.DataFrame:
    """Genesis-style stats on the placebo gap ``placebo_b - placebo_a``.

    This is the reference row that ALWAYS accompanies ``genesis_gap_stats``.
    Same aggregation and statistics. The placebo is a per-composition noise
    floor, so ``instruction_level`` is incidental (currently only level_1).
    """
    long = _placebo_agent_gap_long(df_placebo, include_errors=include_errors)
    _, seed_avg = _seed_avg_pair_gaps(long, turn=turn, visible_only=visible_only)
    return _summarize_pair_gaps(seed_avg, "placebo")


def permutation_vs_placebo(
    df: pd.DataFrame,
    df_placebo: pd.DataFrame,
    attr: str,
    turn: int = 0,
    visible_only: bool = True,
    n_perm: int = 10000,
) -> dict:
    """Two-sample permutation test: treatment pair gaps vs placebo pair gaps.

    Statistic = difference of means (treatment - placebo) over per-pair gaps
    (seeds averaged); labels treatment/placebo are permuted (fixed seed 0).
    Returns observed ``statistic``, two-sided ``pvalue``, and sample sizes.

    NOTE: with the current placebo (5 pairs) the power is minimal and this test
    is only informative after the placebo is expanded to all 21 archetypes.
    """
    tlong = _treatment_agent_gap_long(df, attr, include_errors=False)
    plong = _placebo_agent_gap_long(df_placebo, include_errors=False)
    _, t_seed = _seed_avg_pair_gaps(tlong, turn=turn, visible_only=visible_only)
    _, p_seed = _seed_avg_pair_gaps(plong, turn=turn, visible_only=visible_only)

    treat = t_seed["gap"].dropna().to_numpy(dtype=float) if not t_seed.empty else np.array([])
    plac = p_seed["gap"].dropna().to_numpy(dtype=float) if not p_seed.empty else np.array([])
    result = {
        "sensitive_attr": attr, "statistic": float("nan"), "pvalue": float("nan"),
        "n_treat": int(treat.size), "n_placebo": int(plac.size),
    }
    if treat.size == 0 or plac.size == 0:
        return result

    obs = float(treat.mean() - plac.mean())
    pooled = np.concatenate([treat, plac])
    n_treat = treat.size
    rng = np.random.default_rng(0)
    order = np.argsort(rng.random((n_perm, pooled.size)), axis=1)
    permuted = pooled[order]
    diffs = permuted[:, :n_treat].mean(axis=1) - permuted[:, n_treat:].mean(axis=1)
    result["statistic"] = obs
    result["pvalue"] = float((np.abs(diffs) >= abs(obs)).mean())
    return result


def trajectory_with_placebo(
    df: pd.DataFrame,
    df_placebo: pd.DataFrame,
    attr: str,
    include_errors: bool = False,
) -> pd.DataFrame:
    """Per-turn gap trajectory (mean +/- bootstrap CI, seeds grouped) with the
    placebo band overlaid, ready to plot.

    Columns: ``turn, mean_gap, ci_lo, ci_hi, placebo_mean, placebo_lo,
    placebo_hi`` per (composition, instruction_level, sensitive_attr).

    The placebo band is joined by (composition, turn) ONLY - never by
    instruction_level - so the single per-composition placebo (currently only
    level_1) is broadcast to every instruction_level of that composition.
    """
    columns = [
        "composition", "instruction_level", "sensitive_attr", "turn",
        "mean_gap", "ci_lo", "ci_hi", "placebo_mean", "placebo_lo", "placebo_hi",
    ]
    long = _treatment_agent_gap_long(df, attr, include_errors=include_errors)
    if long.empty:
        return _empty(columns)
    long = long.copy()
    long["gap"] = pd.to_numeric(long["gap"], errors="coerce")
    long["turn"] = pd.to_numeric(long["turn"], errors="coerce")

    ckeys = rex._present(long, ["composition", "instruction_level", "seed", "pair_id", "turn"])
    committee = long.groupby(ckeys, dropna=False)["gap"].mean().reset_index()
    gkeys = rex._present(committee, ["composition", "instruction_level", "turn"])
    rows = []
    for keys, frame in committee.groupby(gkeys, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        meta = dict(zip(gkeys, keys))
        vals = frame["gap"].dropna().to_numpy(dtype=float)
        lo, hi = _bootstrap_ci(vals)
        meta["mean_gap"] = float(np.mean(vals)) if vals.size else float("nan")
        meta["ci_lo"] = lo
        meta["ci_hi"] = hi
        rows.append(meta)
    traj = pd.DataFrame(rows)
    if traj.empty:
        return _empty(columns)
    traj["sensitive_attr"] = attr

    band = placebo_band(df_placebo, include_errors=include_errors) if df_placebo is not None else _empty([])
    if not band.empty:
        # placebo_band gap is (placebo_a - placebo_b) with lo=q05, hi=q95;
        # negate to (placebo_b - placebo_a): mean -> -mean, [lo,hi] -> [-hi,-lo].
        pb = pd.DataFrame({
            "composition": band["composition"].to_numpy(),
            "turn": pd.to_numeric(band["turn"], errors="coerce").to_numpy(),
            "placebo_mean": -band["mean_gap"].to_numpy(dtype=float),
            "placebo_lo": -band["hi"].to_numpy(dtype=float),
            "placebo_hi": -band["lo"].to_numpy(dtype=float),
        })
        join_keys = rex._present(traj, ["composition", "turn"])
        traj = traj.merge(pb, on=join_keys, how="left")
    else:
        traj["placebo_mean"] = np.nan
        traj["placebo_lo"] = np.nan
        traj["placebo_hi"] = np.nan

    ordered = [c for c in columns if c in traj.columns]
    remaining = [c for c in traj.columns if c not in ordered]
    return traj[ordered + remaining]


def _seed_consistency_from_long(
    long: pd.DataFrame, sensitive_attr: str, visible_only: bool,
) -> pd.DataFrame:
    """Between-seed Pearson reliability of per-pair gaps, per (level, turn)."""
    columns = [
        "sensitive_attr", "instruction_level", "turn",
        "r_42_123", "r_42_456", "r_123_456", "mean_r",
    ]
    if long is None or long.empty or "gap" not in long.columns:
        return _empty(columns)
    sub = long.copy()
    if visible_only and "is_blind" in sub.columns:
        sub = sub[~rex._truthy(sub["is_blind"])]
    sub = sub.copy()
    sub["gap"] = pd.to_numeric(sub["gap"], errors="coerce")
    sub["turn"] = pd.to_numeric(sub["turn"], errors="coerce")
    sub["seed"] = pd.to_numeric(sub["seed"], errors="coerce")
    keys = rex._present(sub, ["instruction_level", "seed", "pair_id", "turn"])
    if "seed" not in keys or "turn" not in keys:
        return _empty(columns)
    committee = sub.groupby(keys, dropna=False)["gap"].mean().reset_index()

    seed_pairs = [(42, 123), (42, 456), (123, 456)]
    group_keys = rex._present(committee, ["instruction_level", "turn"])
    rows = []
    for keys_val, frame in committee.groupby(group_keys, dropna=False):
        if not isinstance(keys_val, tuple):
            keys_val = (keys_val,)
        meta = dict(zip(group_keys, keys_val))
        r_values = {}
        for s1, s2 in seed_pairs:
            a = frame[frame["seed"] == s1][["pair_id", "gap"]]
            b = frame[frame["seed"] == s2][["pair_id", "gap"]]
            merged = a.merge(b, on="pair_id", suffixes=("_1", "_2"))
            rho, _, _ = _pearson(
                merged["gap_1"].to_numpy(dtype=float),
                merged["gap_2"].to_numpy(dtype=float),
            )
            r_values[(s1, s2)] = rho
        valid = [r for r in r_values.values() if pd.notna(r)]
        rows.append({
            "sensitive_attr": sensitive_attr,
            "instruction_level": meta.get("instruction_level"),
            "turn": int(meta["turn"]) if pd.notna(meta.get("turn")) else meta.get("turn"),
            "r_42_123": r_values[(42, 123)],
            "r_42_456": r_values[(42, 456)],
            "r_123_456": r_values[(123, 456)],
            "mean_r": float(np.mean(valid)) if valid else float("nan"),
        })
    if not rows:
        return _empty(columns)
    return pd.DataFrame(rows)[columns].sort_values(
        rex._present(pd.DataFrame(rows), ["instruction_level", "turn"])
    ).reset_index(drop=True)


def seed_consistency(
    df: pd.DataFrame,
    attr: str,
    visible_only: bool = True,
    include_errors: bool = False,
) -> pd.DataFrame:
    """Between-seed reliability of per-pair gaps by (sensitive_attr, level, turn).

    Pearson r between per-pair gaps for each seed pair (42-123, 42-456,
    123-456) plus their mean. Quantitative evidence of the dissolution: weakly
    positive reliability at t=0 that collapses to ~0 at t=4. Also applicable to
    the placebo via ``_seed_consistency_from_long``.
    """
    long = _treatment_agent_gap_long(df, attr, include_errors=include_errors)
    return _seed_consistency_from_long(long, attr, visible_only)
