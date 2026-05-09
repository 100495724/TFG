# %% Imports
"""Notebook-friendly Act 1 result explorer.

This module is intentionally offline-only: it reads existing CSV outputs,
filters to subject-company rows, and builds plots/tables for basket-level
inspection. It can be imported from a notebook or executed as a CLI script.
"""

from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import pandas as pd

if TYPE_CHECKING:
    from matplotlib.figure import Figure


# %% Configuration
NUMERIC_COLUMNS = ("turn", "seed", "allocation", "subject_position")
DEFAULT_VARIANT_SPECS = (
    ("control", "none"),
    ("gender", "gender"),
    ("geo", "country"),
)
SUPPORTED_TABLE_FIELDS = {
    "action",
    "allocation",
    "reasoning",
    "pre_normalize_total",
    "was_normalized",
    "validation_error",
}
MERGE_KEYS = [
    "pair_id",
    "seed",
    "experiment_label",
    "composition",
    "instruction_level",
    "protocol",
    "vaccine",
    "agent_id",
    "turn",
]
PAIR_KEYS = [
    "pair_id",
    "seed",
    "experiment_label",
    "composition",
    "instruction_level",
    "protocol",
    "vaccine",
]
ACTION_TIE_ORDER = {"BUY": 0, "HOLD": 1, "SELL": 2}
TRUE_VALUES = {"true", "1", "yes", "y", "t"}


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plotting. Install matplotlib to use "
            "--plots or plot_subject_allocation_evolution()."
        ) from exc
    return plt, FuncFormatter


def _present(df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    return [column for column in columns if column in df.columns]


def _truthy(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(str).str.strip().str.lower().isin(TRUE_VALUES)


def _natural_key(value: object) -> tuple:
    parts = re.split(r"(\d+)", str(value))
    return tuple(int(part) if part.isdigit() else part.lower() for part in parts)


def _pair_short_id(pair_id: str) -> str:
    text = str(pair_id)
    first = text.split("_", 1)[0]
    return first if re.fullmatch(r"[A-Za-z]+\d+", first) else text


def _safe_slug(value: object) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return slug or "unknown"


def _agent_label(frame: pd.DataFrame) -> pd.Series:
    if "role_key" in frame.columns:
        labels = frame["role_key"].where(frame["role_key"].notna(), "")
        labels = labels.astype(str).str.strip()
        if labels.ne("").any():
            if "agent_id" in frame.columns:
                fallback = frame["agent_id"].fillna("").astype(str)
                labels = labels.where(labels.ne(""), fallback)
            return labels
    if "agent_id" in frame.columns:
        return frame["agent_id"].fillna("").astype(str)
    return pd.Series(["agent"] * len(frame), index=frame.index)


def _format_company_names(values: pd.Series) -> str:
    names = [str(value) for value in values.dropna().unique()]
    if not names:
        return "unknown company"
    if len(names) == 1:
        return names[0]
    return f"{names[0]} (+{len(names) - 1} more)"


def _majority_action(actions: pd.Series) -> str | None:
    normalized = actions.dropna().astype(str).str.upper()
    normalized = normalized[normalized.isin(ACTION_TIE_ORDER)]
    if normalized.empty:
        return None
    counts = normalized.value_counts()
    ordered = sorted(
        counts.items(),
        key=lambda item: (-item[1], ACTION_TIE_ORDER.get(item[0], 99)),
    )
    return ordered[0][0]


def _upper_or_na(series: pd.Series) -> pd.Series:
    return series.where(series.notna(), pd.NA).astype("string").str.upper()


def _coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    for column in NUMERIC_COLUMNS:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    return cleaned


def _require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required column(s): {', '.join(missing)}")


def _resolve_pair_id(df: pd.DataFrame, pair_id: str) -> str:
    pairs = available_pairs(df)
    if pair_id in pairs:
        return pair_id

    matches = [
        value
        for value in pairs
        if value.startswith(f"{pair_id}_") or _pair_short_id(value) == pair_id
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        examples = ", ".join(pairs[:8])
        raise ValueError(
            f"No pair_id matched {pair_id!r}. Available examples: {examples}"
        )

    raise ValueError(
        f"pair_id {pair_id!r} is ambiguous. Matches: {', '.join(matches)}"
    )


def _expand_csv_paths(paths_or_globs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in paths_or_globs:
        matches = glob.glob(item) if glob.has_magic(item) else [item]
        for match in matches:
            path = Path(match)
            if path.exists() and path.suffix.lower() == ".csv":
                paths.append(path)

    unique: dict[Path, Path] = {}
    for path in paths:
        unique[path.resolve()] = path
    return [unique[key] for key in sorted(unique, key=lambda value: str(value).lower())]


# %% Loading helpers
def load_result_csvs(paths_or_globs: list[str]) -> pd.DataFrame:
    """
    Load multiple CSV files from explicit paths or glob patterns.

    Example:
        load_result_csvs(["src/results/act1_baseline_seed*.csv"])
    """
    csv_paths = _expand_csv_paths(paths_or_globs)
    if not csv_paths:
        raise FileNotFoundError(
            "No CSV files matched: " + ", ".join(map(str, paths_or_globs))
        )

    frames = []
    for path in csv_paths:
        frame = pd.read_csv(path)
        frame["source_file"] = str(path)
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)
    df = _coerce_numeric_columns(df)

    parse_error_rows = (
        int(_truthy(df["parse_error"]).sum()) if "parse_error" in df.columns else 0
    )
    subject_rows = int(_truthy(df["is_subject"]).sum()) if "is_subject" in df.columns else 0
    seeds = available_seeds(df)
    pair_count = df["pair_id"].nunique() if "pair_id" in df.columns else 0

    print(
        "\n".join(
            [
                f"Loaded {len(df):,} rows from {len(csv_paths)} CSV file(s).",
                "Source files: " + ", ".join(str(path) for path in csv_paths),
                f"Seeds: {seeds}",
                f"pair_id count: {pair_count}",
                f"parse_error rows: {parse_error_rows:,}",
                f"subject rows: {subject_rows:,}",
            ]
        )
    )
    return df


def filter_valid_subject_rows(
    df: pd.DataFrame,
    include_errors: bool = False,
) -> pd.DataFrame:
    """
    Keep only subject rows. By default remove parse_error rows and action == ERROR.
    """
    _require_columns(df, ["is_subject"])
    filtered = df[_truthy(df["is_subject"])].copy()

    if not include_errors:
        if "parse_error" in filtered.columns:
            filtered = filtered[~_truthy(filtered["parse_error"])]
        if "action" in filtered.columns:
            filtered = filtered[
                filtered["action"].fillna("").astype(str).str.upper() != "ERROR"
            ]

    return _coerce_numeric_columns(filtered)


def available_pairs(df: pd.DataFrame) -> list[str]:
    if "pair_id" not in df.columns:
        return []
    pairs = [str(value) for value in df["pair_id"].dropna().unique()]
    return sorted(pairs, key=_natural_key)


def available_seeds(df: pd.DataFrame) -> list[int]:
    if "seed" not in df.columns:
        return []
    seeds = pd.to_numeric(df["seed"], errors="coerce").dropna().unique()
    return sorted(int(seed) for seed in seeds)


# %% Plotting helpers
def plot_subject_allocation_evolution(
    df: pd.DataFrame,
    pair_id: str,
    seed: int | None = None,
    variant: str | None = None,
    sensitive_attr: str | None = None,
    output_path: str | Path | None = None,
    include_errors: bool = False,
    show: bool = True,
) -> Figure:
    """Plot subject-company allocation over turns for one basket/pair variant."""
    plt, FuncFormatter = _require_matplotlib()
    resolved_pair_id = _resolve_pair_id(df, pair_id)
    plot_df = filter_valid_subject_rows(df, include_errors=include_errors)
    plot_df = plot_df[plot_df["pair_id"].astype(str) == resolved_pair_id].copy()

    if seed is not None:
        plot_df = plot_df[plot_df["seed"] == seed]
    else:
        seeds = available_seeds(plot_df)
        if len(seeds) > 1:
            raise ValueError(
                f"pair_id {resolved_pair_id!r} has multiple seeds {seeds}; "
                "pass seed=... to avoid a cluttered plot."
            )

    if variant is not None and "variant" in plot_df.columns:
        plot_df = plot_df[plot_df["variant"].astype(str) == variant]
    if sensitive_attr is not None and "sensitive_attr" in plot_df.columns:
        plot_df = plot_df[plot_df["sensitive_attr"].astype(str) == sensitive_attr]

    _require_columns(plot_df, ["turn", "allocation"])
    plot_df = plot_df.dropna(subset=["turn", "allocation"]).copy()
    if plot_df.empty:
        raise ValueError(
            "No subject allocation rows found for "
            f"pair_id={resolved_pair_id!r}, seed={seed!r}, "
            f"variant={variant!r}, sensitive_attr={sensitive_attr!r}."
        )

    plot_df["_agent_label"] = _agent_label(plot_df)
    plot_df = plot_df.sort_values(["_agent_label", "turn"])

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, frame in plot_df.groupby("_agent_label", sort=False):
        line = (
            frame.groupby("turn", as_index=False)["allocation"]
            .mean()
            .sort_values("turn")
        )
        ax.plot(line["turn"], line["allocation"], marker="o", label=str(label))

    variant_label = variant or ", ".join(sorted(plot_df.get("variant", pd.Series()).dropna().astype(str).unique()))
    attr_label = sensitive_attr or ", ".join(sorted(plot_df.get("sensitive_attr", pd.Series()).dropna().astype(str).unique()))
    seed_label = seed if seed is not None else available_seeds(plot_df)[0]
    company_label = (
        _format_company_names(plot_df["company_name"])
        if "company_name" in plot_df.columns
        else "unknown company"
    )

    ax.set_title(
        " | ".join(
            [
                resolved_pair_id,
                f"seed {seed_label}",
                f"variant {variant_label}",
                f"sensitive_attr {attr_label}",
                company_label,
            ]
        )
    )
    ax.set_xlabel("Turn")
    ax.set_ylabel("Allocation (EUR)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"EUR {value:,.0f}"))
    ax.legend(title="Agent")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved plot: {path}")

    if show:
        plt.show()

    return fig


def plot_pair_all_variants(
    df: pd.DataFrame,
    pair_id: str,
    seed: int,
    output_dir: str | Path | None = None,
    include_errors: bool = False,
    show: bool = True,
) -> list[Figure]:
    """Generate control, gender, and country allocation plots for one pair."""
    resolved_pair_id = _resolve_pair_id(df, pair_id)
    pair_slug = _safe_slug(_pair_short_id(resolved_pair_id))
    figures: list[Figure] = []

    for variant, sensitive_attr in DEFAULT_VARIANT_SPECS:
        output_path = None
        if output_dir is not None:
            output_path = (
                Path(output_dir)
                / f"seed{seed}"
                / f"{pair_slug}_{_safe_slug(sensitive_attr)}.png"
            )
        figures.append(
            plot_subject_allocation_evolution(
                df,
                pair_id=resolved_pair_id,
                seed=seed,
                variant=variant,
                sensitive_attr=sensitive_attr,
                output_path=output_path,
                include_errors=include_errors,
                show=show,
            )
        )

    return figures


def plot_all_pairs_all_variants(
    df: pd.DataFrame,
    seed: int,
    output_dir: str | Path = "src/analysis/figures/allocation_evolution",
    include_errors: bool = False,
    show: bool = False,
) -> None:
    """Loop over all pair_ids and save control/gender/country allocation plots."""
    plt, _ = _require_matplotlib()
    for pair_id in available_pairs(df):
        figures = plot_pair_all_variants(
            df,
            pair_id=pair_id,
            seed=seed,
            output_dir=output_dir,
            include_errors=include_errors,
            show=show,
        )
        if not show:
            for figure in figures:
                plt.close(figure)


# %% Table helpers
def _display_object(obj: object) -> None:
    try:
        from IPython.display import display
    except ImportError:
        if isinstance(obj, pd.DataFrame):
            print(obj.to_string(index=False))
        else:
            print(obj)
    else:
        display(obj)


def _display_title(title: str) -> None:
    try:
        from IPython.display import Markdown, display
    except ImportError:
        print(f"\n{title}\n{'-' * len(title)}")
    else:
        display(Markdown(f"### {title}"))


def subject_decision_table(
    df: pd.DataFrame,
    pair_id: str,
    seed: int,
    sensitive_attr: str,
    field: str = "action",
    include_errors: bool = False,
) -> pd.DataFrame:
    """Return a turn-by-agent pivot table for one subject company."""
    if field not in SUPPORTED_TABLE_FIELDS:
        raise ValueError(
            f"Unsupported field {field!r}. Supported fields: "
            + ", ".join(sorted(SUPPORTED_TABLE_FIELDS))
        )
    _require_columns(df, ["pair_id", "seed", "sensitive_attr", "turn", field])

    resolved_pair_id = _resolve_pair_id(df, pair_id)
    rows = filter_valid_subject_rows(df, include_errors=include_errors)
    rows = rows[
        (rows["pair_id"].astype(str) == resolved_pair_id)
        & (rows["seed"] == seed)
        & (rows["sensitive_attr"].astype(str) == sensitive_attr)
    ].copy()

    if rows.empty:
        raise ValueError(
            "No subject decision rows found for "
            f"pair_id={resolved_pair_id!r}, seed={seed!r}, "
            f"sensitive_attr={sensitive_attr!r}."
        )

    rows["_agent_label"] = _agent_label(rows)
    rows = rows.sort_values(["turn", "_agent_label"])
    values = rows.groupby(["turn", "_agent_label"], dropna=False)[field].first()
    table = values.unstack("_agent_label").reset_index()

    agent_columns = sorted(
        [column for column in table.columns if column != "turn"],
        key=_natural_key,
    )
    table = table[["turn", *agent_columns]]

    if field == "allocation":
        for column in agent_columns:
            table[column] = pd.to_numeric(table[column], errors="coerce")

    company_name = (
        _format_company_names(rows["company_name"])
        if "company_name" in rows.columns
        else "unknown company"
    )
    table.insert(0, "company_name", company_name)
    table.insert(0, "sensitive_attr", sensitive_attr)
    table.insert(0, "seed", seed)
    table.insert(0, "pair_id", resolved_pair_id)
    return table


def display_subject_decision_table(
    df: pd.DataFrame,
    pair_id: str,
    seed: int,
    sensitive_attr: str,
    field: str = "action",
    include_errors: bool = False,
) -> pd.DataFrame:
    """Display and return one subject decision table in a notebook."""
    table = subject_decision_table(
        df,
        pair_id=pair_id,
        seed=seed,
        sensitive_attr=sensitive_attr,
        field=field,
        include_errors=include_errors,
    )
    _display_object(table)
    return table


def build_pair_tables_all_fields(
    df: pd.DataFrame,
    pair_id: str,
    seed: int,
    fields: tuple[str, ...] = ("action", "allocation", "reasoning"),
    include_errors: bool = False,
) -> dict[str, pd.DataFrame]:
    """Build subject decision tables without saving them."""
    resolved_pair_id = _resolve_pair_id(df, pair_id)
    tables: dict[str, pd.DataFrame] = {}

    for _, sensitive_attr in DEFAULT_VARIANT_SPECS:
        for field in fields:
            key = f"{sensitive_attr}_{field}"
            tables[key] = subject_decision_table(
                df,
                pair_id=resolved_pair_id,
                seed=seed,
                sensitive_attr=sensitive_attr,
                field=field,
                include_errors=include_errors,
            )

    return tables


def display_pair_tables_all_fields(
    df: pd.DataFrame,
    pair_id: str,
    seed: int,
    fields: tuple[str, ...] = ("action", "allocation", "reasoning"),
    include_errors: bool = False,
) -> dict[str, pd.DataFrame]:
    """Display control/gender/country subject tables in a notebook."""
    tables = build_pair_tables_all_fields(
        df,
        pair_id=pair_id,
        seed=seed,
        fields=fields,
        include_errors=include_errors,
    )
    for name, table in tables.items():
        _display_title(name)
        _display_object(table)
    return tables


def _dataframe_to_markdown(table: pd.DataFrame) -> str:
    headers = [str(column) for column in table.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in table.iterrows():
        values = [
            str(value).replace("\n", "<br>").replace("|", "\\|")
            if pd.notna(value)
            else ""
            for value in row
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def export_subject_decision_table(
    table: pd.DataFrame,
    output_path: str | Path,
    format: str = "html",
) -> None:
    """Export a subject decision table as csv, html, or markdown."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = format.lower().lstrip(".")

    if normalized == "csv":
        table.to_csv(path, index=False)
    elif normalized == "html":
        table.to_html(path, index=False, escape=True)
    elif normalized in {"md", "markdown"}:
        try:
            path.write_text(table.to_markdown(index=False) + "\n", encoding="utf-8")
        except ImportError:
            path.write_text(_dataframe_to_markdown(table), encoding="utf-8")
    else:
        raise ValueError("format must be one of: csv, html, md")

    print(f"Saved table: {path}")


def export_pair_tables_all_fields(
    df: pd.DataFrame,
    pair_id: str,
    seed: int,
    output_dir: str | Path = "src/analysis/tables/subject_decisions",
    fields: tuple[str, ...] = ("action", "allocation", "reasoning"),
    include_errors: bool = False,
) -> None:
    """Export subject decision tables for none/gender/country and selected fields."""
    resolved_pair_id = _resolve_pair_id(df, pair_id)
    pair_slug = _safe_slug(_pair_short_id(resolved_pair_id))
    base_dir = Path(output_dir) / f"seed{seed}"
    tables = build_pair_tables_all_fields(
        df,
        pair_id=resolved_pair_id,
        seed=seed,
        fields=fields,
        include_errors=include_errors,
    )

    for _, sensitive_attr in DEFAULT_VARIANT_SPECS:
        for field in fields:
            table = tables[f"{sensitive_attr}_{field}"]
            fmt = "html" if field == "reasoning" else "csv"
            output_path = base_dir / f"{pair_slug}_{sensitive_attr}_{field}.{fmt}"
            export_subject_decision_table(table, output_path, format=fmt)


# %% Per-basket metric helpers
def compute_agent_pair_gap_table(
    df: pd.DataFrame,
    attr: str = "gender",
    include_errors: bool = False,
) -> pd.DataFrame:
    """Compute subject allocation/action gaps by pair, seed, agent, and turn."""
    subject = filter_valid_subject_rows(df, include_errors=include_errors)
    _require_columns(
        subject,
        ["pair_id", "seed", "sensitive_attr", "agent_id", "turn", "allocation"],
    )
    subject = subject.copy()
    subject["allocation"] = pd.to_numeric(subject["allocation"], errors="coerce")

    control = subject[subject["sensitive_attr"].astype(str) == "none"].copy()
    test = subject[subject["sensitive_attr"].astype(str) == attr].copy()
    merge_keys = _present(subject, MERGE_KEYS)
    if not merge_keys:
        raise ValueError("No merge keys were available for agent pair gaps.")

    control = control.drop_duplicates(subset=merge_keys, keep="first")
    test = test.drop_duplicates(subset=merge_keys, keep="first")
    merged = control.merge(
        test,
        on=merge_keys,
        how="inner",
        suffixes=("_control", "_test"),
    )

    output_columns = [
        "pair_id",
        "seed",
        "experiment_label",
        "composition",
        "instruction_level",
        "protocol",
        "vaccine",
        "agent_id",
        "role_key",
        "is_blind",
        "turn",
        "allocation_control",
        "allocation_test",
        "allocation_gap",
        "action_control",
        "action_test",
        "same_action",
        "company_name_control",
        "company_name_test",
        "reasoning_control",
        "reasoning_test",
        "sensitive_attr_tested",
    ]
    if merged.empty:
        return pd.DataFrame(columns=output_columns)

    result = pd.DataFrame()
    for column in _present(merged, PAIR_KEYS + ["agent_id", "turn"]):
        result[column] = merged[column]

    result["role_key"] = merged.get("role_key_control", merged.get("role_key"))
    result["is_blind"] = merged.get("is_blind_control", merged.get("is_blind"))
    result["allocation_control"] = pd.to_numeric(
        merged["allocation_control"],
        errors="coerce",
    )
    result["allocation_test"] = pd.to_numeric(
        merged["allocation_test"],
        errors="coerce",
    )
    result["allocation_gap"] = (
        result["allocation_control"] - result["allocation_test"]
    )
    result["action_control"] = merged.get("action_control")
    result["action_test"] = merged.get("action_test")
    result["same_action"] = (
        _upper_or_na(result["action_control"]) == _upper_or_na(result["action_test"])
    )
    result["company_name_control"] = merged.get("company_name_control")
    result["company_name_test"] = merged.get("company_name_test")
    result["reasoning_control"] = merged.get("reasoning_control")
    result["reasoning_test"] = merged.get("reasoning_test")
    result["sensitive_attr_tested"] = attr

    ordered = [column for column in output_columns if column in result.columns]
    remaining = [column for column in result.columns if column not in ordered]
    return result[ordered + remaining].sort_values(
        _present(result, ["pair_id", "seed", "turn", "agent_id"])
    )


def _committee_snapshot(
    df: pd.DataFrame,
    final_turn: int,
    include_errors: bool = False,
) -> pd.DataFrame:
    subject = filter_valid_subject_rows(df, include_errors=include_errors)
    _require_columns(subject, ["pair_id", "seed", "sensitive_attr", "turn", "allocation"])
    subject = subject[subject["turn"] == final_turn].copy()
    subject["allocation"] = pd.to_numeric(subject["allocation"], errors="coerce")
    subject = subject.dropna(subset=["allocation"])
    if subject.empty:
        return pd.DataFrame()

    group_keys = _present(subject, PAIR_KEYS + ["sensitive_attr"])
    records = []
    for keys, frame in subject.groupby(group_keys, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = dict(zip(group_keys, keys))
        actions = frame["action"] if "action" in frame.columns else pd.Series(dtype=str)
        valid_actions = _upper_or_na(actions)
        valid_actions = valid_actions[valid_actions.isin(ACTION_TIE_ORDER)]
        allocations = frame["allocation"].astype(float)
        record.update(
            {
                "committee_allocation": float(allocations.mean()),
                "committee_action": _majority_action(actions),
                "n_valid_agents": int(frame["agent_id"].nunique())
                if "agent_id" in frame.columns
                else int(len(frame)),
                "disagreement_std": float(allocations.std(ddof=0))
                if len(allocations)
                else float("nan"),
                "consensus_reached": bool(len(set(valid_actions.dropna())) == 1)
                if len(valid_actions)
                else False,
            }
        )
        records.append(record)

    return pd.DataFrame(records)


def compute_committee_pair_gap_table(
    df: pd.DataFrame,
    attr: str = "gender",
    final_turn: int = 4,
    include_errors: bool = False,
) -> pd.DataFrame:
    """Compute final-turn committee control-vs-test gaps by pair and seed."""
    committee = _committee_snapshot(
        df,
        final_turn=final_turn,
        include_errors=include_errors,
    )
    output_columns = [
        "pair_id",
        "seed",
        "experiment_label",
        "composition",
        "instruction_level",
        "protocol",
        "vaccine",
        "committee_allocation_control",
        "committee_allocation_test",
        "committee_allocation_gap",
        "committee_action_control",
        "committee_action_test",
        "same_committee_action",
        "n_valid_agents_control",
        "n_valid_agents_test",
        "disagreement_std_control",
        "disagreement_std_test",
        "consensus_reached_control",
        "consensus_reached_test",
        "sensitive_attr_tested",
    ]
    if committee.empty:
        return pd.DataFrame(columns=output_columns)

    control = committee[committee["sensitive_attr"].astype(str) == "none"].copy()
    test = committee[committee["sensitive_attr"].astype(str) == attr].copy()
    merge_keys = _present(committee, PAIR_KEYS)
    merged = control.merge(
        test,
        on=merge_keys,
        how="inner",
        suffixes=("_control", "_test"),
    )
    if merged.empty:
        return pd.DataFrame(columns=output_columns)

    result = pd.DataFrame()
    for column in merge_keys:
        result[column] = merged[column]
    result["committee_allocation_control"] = merged["committee_allocation_control"]
    result["committee_allocation_test"] = merged["committee_allocation_test"]
    result["committee_allocation_gap"] = (
        result["committee_allocation_control"] - result["committee_allocation_test"]
    )
    result["committee_action_control"] = merged["committee_action_control"]
    result["committee_action_test"] = merged["committee_action_test"]
    result["same_committee_action"] = (
        _upper_or_na(result["committee_action_control"])
        == _upper_or_na(result["committee_action_test"])
    )
    result["n_valid_agents_control"] = merged["n_valid_agents_control"]
    result["n_valid_agents_test"] = merged["n_valid_agents_test"]
    result["disagreement_std_control"] = merged["disagreement_std_control"]
    result["disagreement_std_test"] = merged["disagreement_std_test"]
    result["consensus_reached_control"] = merged["consensus_reached_control"]
    result["consensus_reached_test"] = merged["consensus_reached_test"]
    result["sensitive_attr_tested"] = attr

    ordered = [column for column in output_columns if column in result.columns]
    remaining = [column for column in result.columns if column not in ordered]
    return result[ordered + remaining].sort_values(_present(result, ["pair_id", "seed"]))


def _turn_gap_mean(
    frame: pd.DataFrame,
    turn: int,
    value_col: str = "allocation_gap",
) -> float:
    values = frame[frame["turn"] == turn][value_col].dropna()
    return float(values.mean()) if not values.empty else float("nan")


def _is_blind_gap_row(gaps: pd.DataFrame) -> pd.Series:
    if "is_blind" in gaps.columns and _truthy(gaps["is_blind"]).any():
        return _truthy(gaps["is_blind"])
    if "role_key" in gaps.columns:
        return gaps["role_key"].fillna("").astype(str).str.lower() == "agent_1"
    return pd.Series(False, index=gaps.index)


def compute_dynamic_pair_summary(
    df: pd.DataFrame,
    attr: str = "gender",
    include_errors: bool = False,
) -> pd.DataFrame:
    """Compute turn-0 to turn-4 gap dynamics per pair and seed."""
    gaps = compute_agent_pair_gap_table(
        df,
        attr=attr,
        include_errors=include_errors,
    )
    output_columns = [
        "pair_id",
        "seed",
        "experiment_label",
        "composition",
        "instruction_level",
        "protocol",
        "vaccine",
        "sensitive_attr_tested",
        "gap_t0_agent_mean",
        "gap_t4_agent_mean",
        "delta_gap",
        "abs_delta_gap",
        "amplification_ratio",
        "blind_gap_t0",
        "blind_gap_t4",
        "blind_delta_gap",
    ]
    if gaps.empty:
        return pd.DataFrame(columns=output_columns)

    group_keys = _present(gaps, PAIR_KEYS)
    blind_mask = _is_blind_gap_row(gaps)
    records = []
    for keys, frame in gaps.groupby(group_keys, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = dict(zip(group_keys, keys))
        gap_t0 = _turn_gap_mean(frame, 0)
        gap_t4 = _turn_gap_mean(frame, 4)
        blind_frame = frame[blind_mask.loc[frame.index]]
        blind_t0 = _turn_gap_mean(blind_frame, 0)
        blind_t4 = _turn_gap_mean(blind_frame, 4)
        delta = gap_t4 - gap_t0 if pd.notna(gap_t0) and pd.notna(gap_t4) else float("nan")
        blind_delta = (
            blind_t4 - blind_t0
            if pd.notna(blind_t0) and pd.notna(blind_t4)
            else float("nan")
        )
        amplification = (
            gap_t4 / gap_t0
            if pd.notna(gap_t0) and pd.notna(gap_t4) and gap_t0 != 0
            else float("nan")
        )
        record.update(
            {
                "sensitive_attr_tested": attr,
                "gap_t0_agent_mean": gap_t0,
                "gap_t4_agent_mean": gap_t4,
                "delta_gap": delta,
                "abs_delta_gap": abs(delta) if pd.notna(delta) else float("nan"),
                "amplification_ratio": amplification,
                "blind_gap_t0": blind_t0,
                "blind_gap_t4": blind_t4,
                "blind_delta_gap": blind_delta,
            }
        )
        records.append(record)

    result = pd.DataFrame(records)
    ordered = [column for column in output_columns if column in result.columns]
    remaining = [column for column in result.columns if column not in ordered]
    return result[ordered + remaining].sort_values(_present(result, ["pair_id", "seed"]))


def _export_metric_table(table: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    print(f"Saved table: {output_path}")
    html_path = output_path.with_suffix(".html")
    table.to_html(html_path, index=False, escape=True)
    print(f"Saved table: {html_path}")


def build_pair_metric_report(
    df: pd.DataFrame,
    output_dir: str | Path | None = None,
    final_turn: int = 4,
    include_errors: bool = False,
    export: bool = False,
) -> dict[str, pd.DataFrame]:
    """Compute per-pair metric reports, optionally exporting them."""
    base_dir = Path(output_dir or "src/analysis/tables/pair_metrics")
    report: dict[str, pd.DataFrame] = {}

    for attr in ("gender", "country"):
        agent_key = f"agent_pair_gap_{attr}"
        committee_key = f"committee_pair_gap_{attr}"
        dynamic_key = f"dynamic_pair_summary_{attr}"

        report[agent_key] = compute_agent_pair_gap_table(
            df,
            attr=attr,
            include_errors=include_errors,
        )
        report[committee_key] = compute_committee_pair_gap_table(
            df,
            attr=attr,
            final_turn=final_turn,
            include_errors=include_errors,
        )
        report[dynamic_key] = compute_dynamic_pair_summary(
            df,
            attr=attr,
            include_errors=include_errors,
        )

        if export:
            _export_metric_table(report[agent_key], base_dir / f"{agent_key}.csv")
            _export_metric_table(report[committee_key], base_dir / f"{committee_key}.csv")
            _export_metric_table(report[dynamic_key], base_dir / f"{dynamic_key}.csv")

    return report


def display_pair_metric_report(
    df: pd.DataFrame,
    final_turn: int = 4,
    include_errors: bool = False,
) -> dict[str, pd.DataFrame]:
    """Display per-pair metric reports in a notebook without saving files."""
    report = build_pair_metric_report(
        df,
        final_turn=final_turn,
        include_errors=include_errors,
        export=False,
    )
    for name, table in report.items():
        _display_title(name)
        _display_object(table)
    return report


def display_pair_metrics_for_basket(
    df: pd.DataFrame,
    pair_id: str,
    seed: int | None = None,
    final_turn: int = 4,
    include_errors: bool = False,
) -> dict[str, pd.DataFrame]:
    """Display the three metric tables filtered to a single basket (pair_id).

    Convenience wrapper for notebook use: it computes the global metric
    tables once, then filters each one to the requested pair_id (and seed,
    if provided) before displaying. Returns the filtered tables in a dict
    keyed by ``agent_pair_gap_<attr>`` / ``committee_pair_gap_<attr>`` /
    ``dynamic_pair_summary_<attr>``.
    """
    resolved_pair_id = _resolve_pair_id(df, pair_id)
    report = build_pair_metric_report(
        df,
        final_turn=final_turn,
        include_errors=include_errors,
        export=False,
    )

    filtered: dict[str, pd.DataFrame] = {}
    for name, table in report.items():
        if "pair_id" not in table.columns:
            filtered[name] = table
            continue
        subset = table[table["pair_id"].astype(str) == resolved_pair_id].copy()
        if seed is not None and "seed" in subset.columns:
            subset = subset[subset["seed"] == seed]
        filtered[name] = subset

    seed_label = f", seed {seed}" if seed is not None else ""
    _display_title(f"Metrics for {resolved_pair_id}{seed_label}")
    for name, table in filtered.items():
        _display_title(name)
        _display_object(table)
    return filtered


# %% Optional reasoning inspection helpers
def reasoning_trace(
    df: pd.DataFrame,
    pair_id: str,
    seed: int,
    sensitive_attr: str,
    agent: str | None = None,
    include_errors: bool = False,
) -> pd.DataFrame:
    """Return long-form reasoning rows for manual inspection."""
    resolved_pair_id = _resolve_pair_id(df, pair_id)
    rows = filter_valid_subject_rows(df, include_errors=include_errors)
    rows = rows[
        (rows["pair_id"].astype(str) == resolved_pair_id)
        & (rows["seed"] == seed)
        & (rows["sensitive_attr"].astype(str) == sensitive_attr)
    ].copy()

    if agent is not None:
        agent_mask = pd.Series(False, index=rows.index)
        if "role_key" in rows.columns:
            agent_mask = agent_mask | (rows["role_key"].astype(str) == agent)
        if "agent_id" in rows.columns:
            agent_mask = agent_mask | (rows["agent_id"].astype(str) == agent)
        rows = rows[agent_mask].copy()

    columns = _present(
        rows,
        ["turn", "role_key", "agent_id", "action", "allocation", "reasoning"],
    )
    return rows.sort_values(_present(rows, ["turn", "role_key", "agent_id"]))[columns]


def export_reasoning_trace_html(trace: pd.DataFrame, output_path: str | Path) -> None:
    """Export a reasoning trace DataFrame to HTML."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    trace.to_html(path, index=False, escape=True)
    print(f"Saved table: {path}")


# %% CLI entry point / examples
def _field_was_supplied(argv: list[str]) -> bool:
    return any(arg == "--field" or arg.startswith("--field=") for arg in argv)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Explore Act 1 subject-company results from existing CSV files.",
    )
    parser.add_argument(
        "--results",
        nargs="+",
        required=True,
        help="One or more CSV paths/globs, e.g. src/results/act1_baseline_seed*.csv",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed to inspect.")
    parser.add_argument("--pair-id", default=None, help="Pair id, e.g. A2.")
    parser.add_argument("--plots", action="store_true", help="Generate allocation plots.")
    parser.add_argument("--tables", action="store_true", help="Generate subject tables.")
    parser.add_argument(
        "--pair-metrics",
        action="store_true",
        help="Generate per-pair metric reports.",
    )
    parser.add_argument(
        "--output-dir",
        default="src/analysis",
        help="Base output directory.",
    )
    parser.add_argument(
        "--include-errors",
        action="store_true",
        help="Include parse_error/action==ERROR rows.",
    )
    parser.add_argument(
        "--field",
        default="action",
        choices=sorted(SUPPORTED_TABLE_FIELDS),
        help="Field for single-table mode.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    field_supplied = _field_was_supplied(argv)

    df = load_result_csvs(args.results)
    output_dir = Path(args.output_dir)

    if args.plots:
        plt, _ = _require_matplotlib()
        plot_dir = output_dir / "figures" / "allocation_evolution"
        if args.pair_id:
            if args.seed is None:
                parser.error("--plots with --pair-id requires --seed")
            figures = plot_pair_all_variants(
                df,
                pair_id=args.pair_id,
                seed=args.seed,
                output_dir=plot_dir,
                include_errors=args.include_errors,
                show=False,
            )
            for figure in figures:
                plt.close(figure)
        else:
            if args.seed is None:
                parser.error("--plots without --pair-id requires --seed")
            plot_all_pairs_all_variants(
                df,
                seed=args.seed,
                output_dir=plot_dir,
                include_errors=args.include_errors,
                show=False,
            )

    if args.tables:
        if args.pair_id is None or args.seed is None:
            parser.error("--tables requires both --pair-id and --seed")
        table_dir = output_dir / "tables" / "subject_decisions"
        fields = (args.field,) if field_supplied else ("action", "allocation", "reasoning")
        export_pair_tables_all_fields(
            df,
            pair_id=args.pair_id,
            seed=args.seed,
            output_dir=table_dir,
            fields=fields,
            include_errors=args.include_errors,
        )

    if args.pair_metrics:
        build_pair_metric_report(
            df,
            output_dir=output_dir / "tables" / "pair_metrics",
            include_errors=args.include_errors,
            export=True,
        )

    if not (args.plots or args.tables or args.pair_metrics):
        print("No action requested. Use --plots, --tables, and/or --pair-metrics.")
        print(f"Available seeds: {available_seeds(df)}")
        print(f"Available pairs: {available_pairs(df)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
