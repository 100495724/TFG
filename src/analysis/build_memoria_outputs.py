"""build_memoria_outputs.py - deterministic outputs for the memoria (Brief v2).

Single source of truth for every gap-based table of the memoria. Nothing here
is interactive and nothing here has hidden filters: every excluded row is
counted and logged.

Scope
-----
- B.4 ``gap_por_pareja`` / ``promedio_entre_semillas``: the ONE definition of a
  per-pair gap. Every table must be built on it.
- B.5 ``build_t04``: primary table, visible/blind variance ratio at genesis.
- B.6 ``build_t05``: between-seed ICC, the evidence that justifies NOT doing a
  per-basket analysis.
- B.1-B.3: CLI (``main``), deterministic output layout, ``MANIFEST.json``.
- B.7-B.9: ``build_t01`` .. ``build_t11`` (data quality, placebo, means,
  variance-by-turn, transmission slopes, level contrast, composition
  comparison, hetero-vs-members, verbalization).
- B.10: figures F01-F05.
- C.4: ``build_t12`` (mitigation, variance-ratio endpoint).

Conventions (fixed, do not change silently)
-------------------------------------------
- Gap sign: ``gap = allocation(variant) - allocation(control)``. NEGATIVE means
  the subject company was penalised in the variant.
- Placebo gap sign: ``gap = placebo_b - placebo_a``. The a/b assignment is
  arbitrary, so this convention is asserted in the tests: an accidental flip
  would silently reverse the sign of every placebo table.
- Endpoint turn: genesis, ``turn == 0`` (mitigation, Tarea C, is genesis-only
  by construction).
- ``instruction_level`` keeps the value found in the CSVs
  (``level_0_neutral``, ``level_1_professional``, ``level_2_identity``); no
  relabelling happens anywhere in this module.
- Bootstrap and permutation seeds are fixed constants, so two runs over the
  same data produce identical files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import subprocess
import sys
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

# result_explorer.py / aggregate.py live in this directory and there is no
# package __init__.py, so make the directory importable regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import aggregate as agg  # noqa: E402
import result_explorer as rex  # noqa: E402

log = logging.getLogger("build_memoria_outputs")

# --- Fixed parameters (mirrored into MANIFEST.json) -------------------------
GENESIS_TURN = 0
ALL_TURNS = (0, 1, 2, 3, 4)
SENSITIVE_ATTRS = ("gender", "country")
N_BOOT = 10_000
BOOTSTRAP_SEED = 0
N_PERM = 10_000
PERMUTATION_SEED = 0
DEFAULT_FDR_Q = 0.05
# 6 compositions x 3 instruction levels x 2 sensitive attributes.
EXPECTED_T04_ROWS = 36
EXPECTED_T03_ROWS = 36

# The 6 compositions that have been run for detection (Brief v2 preamble). If a
# 7th composition (e.g. ``homo_claude``, present in config.py but never run)
# shows up in the input CSVs, that is exactly the "unexpected composition"
# case B.11 requires the pipeline to abort on rather than silently absorb.
KNOWN_COMPOSITIONS = (
    "hetero_api_gpt", "hetero_local",
    "homo_gpt", "homo_llama", "homo_mistral", "homo_qwen",
)
ALL_LEVELS = ("level_0_neutral", "level_1_professional", "level_2_identity")
LEVEL_SHORT = {
    "level_0_neutral": "level_0",
    "level_1_professional": "level_1",
    "level_2_identity": "level_2",
}

# Pre-declared multiple-testing families of T04 (declared BEFORE looking at the
# 5 remaining compositions). Written into every row so the file documents its
# own inference plan.
T04_FAMILIA = (
    "primaria=q_global (BH sobre los 36 tests) | "
    "secundaria=q_por_composicion (BH sobre los 6 tests de cada composicion)"
)

# Output filenames (B.2). T13 is not in the Brief v2 layout: it was added as
# the citable evidence that the blind agent is a valid internal null (see
# ``build_t13``).
TABLE_FILENAMES = {
    "T01": "T01_calidad_datos.csv",
    "T02": "T02_placebo_suelo_ruido.csv",
    "T03": "T03_genesis_medias_36tests_BH.csv",
    "T04": "T04_genesis_VARIANZAS_36tests_BH.csv",
    "T05": "T05_icc_entre_semillas.csv",
    "T06": "T06_varianza_por_turno.csv",
    "T07": "T07_pendientes_transmision_BH.csv",
    "T08": "T08_contraste_niveles.csv",
    "T09": "T09_comparacion_composiciones.csv",
    "T10": "T10_hetero_vs_miembros.csv",
    "T11": "T11_verbalizacion.csv",
    "T12": "T12_mitigacion.csv",
    "T13": "T13_validacion_null_interno.csv",
}
FIGURE_FILENAMES = {
    "F01": "F01_varianza_visible_vs_ciego.png",
    "F02": "F02_varianza_por_turno.png",
    "F03": "F03_genesis_forest_medias.png",
    "F04": "F04_icc_entre_semillas.png",
    # F05 is one file per (composition, sensitive_attr); see build_f05.
    "F05": "F05_trayectoria_gap_{composition}_{attr}.png",
}

GAP_COLS = [
    "composition", "instruction_level", "seed", "pair_id",
    "sensitive_attr", "gap", "n_agents",
]
SEED_AVG_COLS = [
    "composition", "instruction_level", "sensitive_attr", "pair_id",
    "gap", "n_seeds",
]
T04_COLS = [
    "composition", "instruction_level", "sensitive_attr",
    "n_pairs", "n_pairs_visible", "n_pairs_ciego",
    "n_pairs_semillas_incompletas", "k_seeds_max",
    "sd_visible", "sd_ciego", "sd_ciego_atributo", "F", "ci95_lo_F", "ci95_hi_F",
    "p_fisher", "p_levene",
    "F_homog_ciego", "p_homog_ciego", "ciego_identico_entre_atributos",
    "q_global", "q_por_composicion",
    "sig_BH_global", "sig_BH_composicion", "familia",
]
T05_COLS = [
    "composition", "instruction_level", "sensitive_attr",
    "n_pairs", "n_pairs_descartadas", "k_seeds",
    "icc", "F", "p_value", "q_value", "sig_BH",
    "sd_entre_parejas", "sd_dentro_pareja",
]
# T05 declares its own BH family: the 42 ICC tests (36 treatment cells + 6
# placebo references). The "no basket structure" warning fires on q, not on the
# uncorrected p.
T05_FAMILIA = "BH sobre los tests de ICC de T05 (36 celdas + 6 placebos)"


def _empty(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _require_scipy():
    try:
        from scipy import stats
    except ImportError as exc:  # pragma: no cover - environment issue
        raise ImportError(
            "scipy is required for the F / Levene / ICC tests of T04 and T05."
        ) from exc
    return stats


# ---------------------------------------------------------------------------
# B.4 - core function: per-pair gaps
# ---------------------------------------------------------------------------
def gap_por_pareja(
    df: pd.DataFrame,
    turn: int = GENESIS_TURN,
    blind: bool = False,
    placebo: bool = False,
    include_errors: bool = False,
) -> pd.DataFrame:
    """The single definition of a per-pair gap (Brief v2, B.4).

    Steps:
      1. keep ``is_subject`` rows of the requested ``turn`` and blindness,
         dropping ``parse_error`` rows (and ``action == ERROR`` rows, which the
         loader treats as unusable too - both counts are logged);
      2. average ``allocation`` by
         (composition, instruction_level, seed, pair_id, variant,
         sensitive_attr) -> partial-committee level (the mean of the 2 visible
         agents, or the single blind agent);
      3. merge the ``control`` arm onto the variant arms by
         (composition, instruction_level, seed, pair_id);
      4. ``gap = allocation(variant) - allocation(control)``.

    With ``placebo=True`` step 3/4 become ``gap = placebo_b - placebo_a``
    (the sign convention is fixed: b minus a).

    Step 5 of the brief - averaging across seeds - is deliberately a separate
    function, ``promedio_entre_semillas``, because the per-seed frame is what
    T05 needs. Comparing arms aggregated at different levels (e.g. a cell
    averaged over 3 seeds against a placebo averaged over 9 measurements)
    deflates the placebo sd by sqrt(3); always compare frames produced at the
    SAME aggregation level.

    One row per (composition, instruction_level, seed, pair_id,
    sensitive_attr). Returns an empty typed frame, never raises, when there is
    nothing to merge.
    """
    if df is None or df.empty:
        return _empty(GAP_COLS)

    subject_total = int(rex._truthy(df["is_subject"]).sum()) if "is_subject" in df.columns else 0
    subject = rex.filter_valid_subject_rows(df, include_errors=include_errors)
    if not include_errors and subject_total:
        parse_errors = (
            int((rex._truthy(df["is_subject"]) & rex._truthy(df["parse_error"])).sum())
            if "parse_error" in df.columns else 0
        )
        dropped = subject_total - len(subject)
        if dropped:
            log.info(
                "gap_por_pareja(turn=%s, blind=%s, placebo=%s): %d/%d subject rows "
                "excluded (%d parse_error, %d action==ERROR)",
                turn, blind, placebo, dropped, subject_total,
                parse_errors, dropped - parse_errors,
            )
    if subject.empty:
        return _empty(GAP_COLS)

    sub = subject.copy()
    sub["turn"] = pd.to_numeric(sub["turn"], errors="coerce")
    sub = sub[sub["turn"] == turn]
    if "is_blind" not in sub.columns:
        raise ValueError("gap_por_pareja needs an 'is_blind' column.")
    is_blind = rex._truthy(sub["is_blind"])
    sub = sub[is_blind] if blind else sub[~is_blind]
    if sub.empty:
        return _empty(GAP_COLS)

    sub = sub.copy()
    sub["allocation"] = pd.to_numeric(sub["allocation"], errors="coerce")
    n_before = len(sub)
    sub = sub.dropna(subset=["allocation"])
    if len(sub) != n_before:
        log.info(
            "gap_por_pareja(turn=%s, blind=%s): %d rows dropped for a "
            "non-numeric allocation", turn, blind, n_before - len(sub),
        )

    group_keys = [
        "composition", "instruction_level", "seed", "pair_id",
        "variant", "sensitive_attr",
    ]
    missing = [key for key in group_keys if key not in sub.columns]
    if missing:
        raise ValueError(f"gap_por_pareja: missing required columns {missing}")

    cell = (
        sub.groupby(group_keys, dropna=False)["allocation"]
        .agg(allocation="mean", n_agents="count")
        .reset_index()
    )

    base_variant = "placebo_a" if placebo else "control"
    variants = cell["variant"].astype(str)
    base = cell[variants == base_variant]
    test = cell[variants == "placebo_b"] if placebo else cell[variants != base_variant]
    if base.empty or test.empty:
        log.warning(
            "gap_por_pareja(placebo=%s): no rows for the '%s'/'%s' arms at turn %s",
            placebo, base_variant, "placebo_b" if placebo else "variant", turn,
        )
        return _empty(GAP_COLS)

    merge_keys = ["composition", "instruction_level", "seed", "pair_id"]
    merged = test.merge(
        base[merge_keys + ["allocation", "n_agents"]],
        on=merge_keys, how="inner", suffixes=("_var", "_base"),
    )
    if merged.empty:
        return _empty(GAP_COLS)

    asymmetric = int((merged["n_agents_var"] != merged["n_agents_base"]).sum())
    if asymmetric:
        # Declared vigilance point: unequal agent counts between arms mean the
        # two allocations being subtracted are averages over different agents.
        log.warning(
            "gap_por_pareja(turn=%s, blind=%s, placebo=%s): %d cells average a "
            "different number of agents in each arm (asymmetric row loss)",
            turn, blind, placebo, asymmetric,
        )

    out = merged[merge_keys + ["sensitive_attr"]].copy()
    out["gap"] = merged["allocation_var"].to_numpy(dtype=float) - merged[
        "allocation_base"
    ].to_numpy(dtype=float)
    out["n_agents"] = merged["n_agents_var"].to_numpy()
    return (
        out[GAP_COLS]
        .sort_values(["composition", "instruction_level", "sensitive_attr", "pair_id", "seed"])
        .reset_index(drop=True)
    )


def promedio_entre_semillas(gaps: pd.DataFrame) -> pd.DataFrame:
    """Step 5 of B.4: average the per-seed gaps -> 21 pairs per cell.

    One row per (composition, instruction_level, sensitive_attr, pair_id) with
    ``n_seeds`` kept, so an unbalanced cell is visible instead of silently
    weighting pairs differently.
    """
    if gaps is None or gaps.empty:
        return _empty(SEED_AVG_COLS)
    keys = ["composition", "instruction_level", "sensitive_attr", "pair_id"]
    out = (
        gaps.groupby(keys, dropna=False)["gap"]
        .agg(gap="mean", n_seeds="count")
        .reset_index()
    )
    return out[SEED_AVG_COLS].sort_values(keys).reset_index(drop=True)


def _cell_values(seed_avg: pd.DataFrame, composition, level, attr) -> pd.Series:
    """Per-pair gaps of one cell, indexed by ``pair_id``."""
    sel = seed_avg[
        (seed_avg["composition"] == composition)
        & (seed_avg["instruction_level"] == level)
        & (seed_avg["sensitive_attr"] == attr)
    ]
    return pd.Series(
        sel["gap"].to_numpy(dtype=float), index=sel["pair_id"].to_numpy()
    ).sort_index()


def _cell_seed_counts(seed_avg: pd.DataFrame, composition, level, attr) -> pd.Series:
    """``n_seeds`` behind each pair of one cell, indexed by ``pair_id``."""
    sel = seed_avg[
        (seed_avg["composition"] == composition)
        & (seed_avg["instruction_level"] == level)
        & (seed_avg["sensitive_attr"] == attr)
    ]
    return pd.Series(
        sel["n_seeds"].to_numpy(dtype=int), index=sel["pair_id"].to_numpy()
    ).sort_index()


# ---------------------------------------------------------------------------
# B.5 - T04: variance ratio visible / blind at genesis (PRIMARY TABLE)
# ---------------------------------------------------------------------------
def _f_test_var_ratio(visible: np.ndarray, blind: np.ndarray) -> tuple[float, float]:
    """Fisher F test of equal variances. Returns ``(F, two-sided p)``.

    ``F = var(visible) / var(blind)`` with (n_vis-1, n_blind-1) degrees of
    freedom - 20 and 20 with the current 21 pairs.
    """
    stats = _require_scipy()
    n1, n2 = visible.size, blind.size
    if n1 < 2 or n2 < 2:
        return float("nan"), float("nan")
    var1 = float(np.var(visible, ddof=1))
    var2 = float(np.var(blind, ddof=1))
    if var2 == 0:
        return float("inf") if var1 > 0 else float("nan"), float("nan")
    f_stat = var1 / var2
    cdf = float(stats.f.cdf(f_stat, n1 - 1, n2 - 1))
    return f_stat, float(min(1.0, 2 * min(cdf, 1.0 - cdf)))


def _levene_p(visible: np.ndarray, blind: np.ndarray) -> float:
    """Median-centred Levene test - robust cross-check of the F test."""
    stats = _require_scipy()
    if visible.size < 2 or blind.size < 2:
        return float("nan")
    try:
        return float(stats.levene(visible, blind, center="median").pvalue)
    except ValueError:
        return float("nan")


def _f_test_against_pooled(
    visible: np.ndarray, var_blind: float, df_blind: int,
) -> tuple[float, float]:
    """F test of the visible variance against a POOLED blind variance.

    Same as ``_f_test_var_ratio`` but the denominator is supplied as a variance
    with its own degrees of freedom, because the pooled blind variance is an
    average of the two per-attribute estimators rather than a single sample.
    """
    stats = _require_scipy()
    n1 = visible.size
    if n1 < 2 or df_blind < 1 or not np.isfinite(var_blind) or var_blind <= 0:
        return float("nan"), float("nan")
    f_stat = float(np.var(visible, ddof=1)) / float(var_blind)
    cdf = float(stats.f.cdf(f_stat, n1 - 1, df_blind))
    return f_stat, float(min(1.0, 2 * min(cdf, 1.0 - cdf)))


def _bootstrap_var_ratio_ci(
    visible: np.ndarray,
    blind_samples: list[np.ndarray],
    n_boot: int = N_BOOT,
    seed: int = BOOTSTRAP_SEED,
    lo: float = 2.5,
    hi: float = 97.5,
) -> tuple[float, float]:
    """Percentile bootstrap CI of the variance ratio, resampling PAIRS.

    Every arm is measured on the SAME pairs (same baskets, same runs), so the
    resample is paired: one draw of pair indices is applied to the visible arm
    AND to each blind sample, whose variances are then pooled exactly as in the
    point estimate. Fixed seed - the CI is reproducible byte for byte.
    """
    n = visible.size
    blind_samples = [b for b in blind_samples if b.size == n]
    if n < 2 or not blind_samples:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    var_v = visible[idx].var(axis=1, ddof=1)
    var_b = np.mean([b[idx].var(axis=1, ddof=1) for b in blind_samples], axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = var_v / var_b
    ratios = ratios[np.isfinite(ratios)]
    if ratios.size == 0:
        return float("nan"), float("nan")
    return float(np.percentile(ratios, lo)), float(np.percentile(ratios, hi))


def _pooled_blind_variance(
    ciego: pd.DataFrame, composition, level, attrs: tuple[str, ...],
) -> dict | None:
    """The single blind denominator of a (composition, instruction_level).

    The blind agent's prompt at genesis is IDENTICAL byte for byte in the
    control, gender and country arms - audited over the prompt logs, 1135 of
    1135 (basket, composition, level, seed) triples, alias map included. Its
    gap in the gender arm and in the country arm are therefore two measurements
    of the same quantity, and both cells of a level must divide by the same
    denominator.

    The pooling is done on the VARIANCES, ``var = mean(var_gender,
    var_country)``, not by concatenating the two samples into one of n=42.
    Concatenation would be wrong twice over:

    - in 26 of the 36 cells (every local-model composition: llama, qwen,
      mistral, plus the hetero levels served by a local blind agent) the two
      blind samples are IDENTICAL pair by pair, so concatenating duplicates
      each of the 21 values - one sample counted twice, not two samples;
    - it would raise the denominator's degrees of freedom from 20 to 41 and
      cut the p-values by an order of magnitude without a single new
      observation.

    The degrees of freedom stay at ``n_pairs - 1``: the independent units are
    the 21 baskets, whichever attribute they were measured under.

    Where the two samples are NOT identical - homo_gpt at the three levels and
    the hetero compositions at level_0, i.e. exactly the runs served by an API
    backend - the divergence is backend non-reproducibility between calls, not
    a leak of the attribute (the prompt diff rules that out). It is not a
    threat to validity either: the visible arm of those same runs is subject to
    the same non-determinism, in the same execution, so the ratio remains a
    fair comparison. It only means the two blind draws are genuinely distinct
    measurements of the same noise - which is precisely the case where pooling
    the variances buys information instead of duplicating it.

    Returns ``{"var", "df", "n_pairs", "samples"}`` or None if unusable.
    """
    samples = []
    index = None
    for attr in attrs:
        values = _cell_values(ciego, composition, level, attr)
        if values.empty:
            continue
        index = values.index if index is None else index.intersection(values.index)
        samples.append(values)
    if not samples or index is None or len(index) < 2:
        return None
    aligned = [s.loc[index].to_numpy(dtype=float) for s in samples]
    variances = [float(np.var(a, ddof=1)) for a in aligned]
    return {
        "var": float(np.mean(variances)),
        "df": int(len(index) - 1),
        "n_pairs": int(len(index)),
        "samples": aligned,
    }


def _add_blind_homogeneity(
    table: pd.DataFrame, ciego: pd.DataFrame, attrs: tuple[str, ...],
) -> pd.DataFrame:
    """Diagnostic: are the two blind estimators of a level the same quantity?

    The blind agent never sees the sensitive attribute - verified byte for byte
    on the prompt logs - so its gap in the gender arm and in the country arm
    are two measurements of the same noise. This adds, per (composition,
    level):

    - ``F_homog_ciego`` / ``p_homog_ciego``: F test of equal variances between
      the two blind estimators (df 20,20). This is a VALIDATION column, it
      never feeds the ratio.
    - ``ciego_identico_entre_atributos``: True when the two blind samples are
      numerically identical pair by pair. When True the homogeneity test is
      degenerate (F = 1, p = 1 by construction, not by evidence) and the two
      estimators are ONE sample counted twice, not two independent draws -
      which is why they must not be concatenated into a single n=42 sample.
    """
    table = table.copy()
    table["F_homog_ciego"] = np.nan
    table["p_homog_ciego"] = np.nan
    table["ciego_identico_entre_atributos"] = pd.NA
    if len(attrs) != 2:
        return table

    attr_a, attr_b = attrs
    for (composition, level), frame in table.groupby(
        ["composition", "instruction_level"], dropna=False
    ):
        sample_a = _cell_values(ciego, composition, level, attr_a)
        sample_b = _cell_values(ciego, composition, level, attr_b)
        common = sample_a.index.intersection(sample_b.index)
        if len(common) < 2:
            continue
        values_a = sample_a.loc[common].to_numpy(dtype=float)
        values_b = sample_b.loc[common].to_numpy(dtype=float)
        identical = bool(np.allclose(values_a, values_b, rtol=0, atol=1e-9))
        f_stat, p_value = _f_test_var_ratio(values_a, values_b)
        table.loc[frame.index, "F_homog_ciego"] = f_stat
        table.loc[frame.index, "p_homog_ciego"] = p_value
        table.loc[frame.index, "ciego_identico_entre_atributos"] = identical
        if identical:
            log.info(
                "T04 %s/%s: the blind gap is IDENTICAL pair by pair between "
                "%s and %s (%d/%d pairs) - one sample, not two",
                composition, level, attr_a, attr_b, len(common), len(common),
            )
        elif p_value < 0.05:
            log.warning(
                "T04 %s/%s: the two blind estimators differ in variance "
                "(F=%.2f, p=%.4f) although the blind prompt is identical "
                "across arms - sampling noise, not a leak, but they are not "
                "interchangeable.",
                composition, level, f_stat, p_value,
            )
    return table


def build_t04(
    df: pd.DataFrame,
    turn: int = GENESIS_TURN,
    attrs: tuple[str, ...] = SENSITIVE_ATTRS,
    fdr_q: float = DEFAULT_FDR_Q,
    expected_rows: int | None = EXPECTED_T04_ROWS,
    include_errors: bool = False,
) -> pd.DataFrame:
    """T04 - variance ratio of the visible vs the blind gap at genesis.

    The blind agent at t=0 is the internal null: its prompt is identical in
    control and variant (it omits the sensitive attribute) - verified byte for
    byte against the prompt logs, see ``build_t13`` - over the same baskets and
    the same pairs. Its gap therefore measures pure run-to-run noise, and
    ``F = sd_visible^2 / sd_ciego^2`` is the excess variance that the visible
    agents add.

    Both arms are aggregated identically (partial committee -> mean over
    seeds), so the ratio is not distorted by the aggregation level.

    ``sd_ciego`` is ONE denominator per (composition, instruction_level),
    shared by the two cells of that level: the pooled variance of the two
    per-attribute blind estimators, with df = n_pairs - 1. See
    ``_pooled_blind_variance`` for why they are pooled as variances and not
    concatenated into a single n=42 sample. The per-attribute estimator stays
    in the table as ``sd_ciego_atributo``, next to the ``F_homog_ciego`` /
    ``p_homog_ciego`` homogeneity test between the two, as validation columns
    that never feed the ratio.

    Two pre-declared BH families, both always reported (column ``familia``):
    ``q_global`` over all 36 tests (PRIMARY, for coherence with the means
    table) and ``q_por_composicion`` over the 6 tests of each composition
    (SECONDARY: "does this model destabilise?" is a per-model question).

    Raises ValueError if the resulting table does not have exactly
    ``expected_rows`` rows, listing the missing cells. Pass
    ``expected_rows=None`` to skip the check (e.g. single-composition runs) or
    the number of rows you expect for the subset you are building.
    """
    visible = promedio_entre_semillas(
        gap_por_pareja(df, turn=turn, blind=False, include_errors=include_errors)
    )
    ciego = promedio_entre_semillas(
        gap_por_pareja(df, turn=turn, blind=True, include_errors=include_errors)
    )
    if visible.empty or ciego.empty:
        raise ValueError(
            "build_t04: no genesis gaps available "
            f"(visible rows={len(visible)}, blind rows={len(ciego)})."
        )

    compositions = sorted(visible["composition"].dropna().unique().tolist())
    levels = sorted(visible["instruction_level"].dropna().unique().tolist())

    rows = []
    missing_cells = []
    for composition, level in product(compositions, levels):
        # ONE blind denominator per (composition, level), shared by the two
        # cells of that level: the pooled variance of the per-attribute blind
        # estimators, over the pairs measured in both.
        pooled = _pooled_blind_variance(ciego, composition, level, attrs)
        for attr in attrs:
            vis = _cell_values(visible, composition, level, attr)
            blind = _cell_values(ciego, composition, level, attr)
            common = vis.index.intersection(blind.index)
            if len(common) < 2 or pooled is None:
                missing_cells.append((composition, level, attr))
                continue
            if len(common) != len(vis) or len(common) != len(blind):
                log.warning(
                    "T04 %s/%s/%s: keeping the %d pairs present in both arms "
                    "(visible=%d, blind=%d)",
                    composition, level, attr, len(common), len(vis), len(blind),
                )
            v = vis.loc[common].to_numpy(dtype=float)
            b = blind.loc[common].to_numpy(dtype=float)

            # A pair measured in 2 of the 3 seeds still yields a seed average,
            # so it does NOT reduce n_pairs - it just rests on less data.
            # Report the count instead of letting it disappear into the mean.
            seeds_vis = _cell_seed_counts(visible, composition, level, attr).reindex(common)
            seeds_blind = _cell_seed_counts(ciego, composition, level, attr).reindex(common)
            k_max = int(max(seeds_vis.max(), seeds_blind.max()))
            incomplete = int(((seeds_vis < k_max) | (seeds_blind < k_max)).sum())
            if incomplete:
                log.warning(
                    "T04 %s/%s/%s: %d of the %d pairs are averaged over fewer "
                    "than %d seeds",
                    composition, level, attr, incomplete, len(common), k_max,
                )

            f_stat, p_fisher = _f_test_against_pooled(
                v, pooled["var"], pooled["df"],
            )
            ci_lo, ci_hi = _bootstrap_var_ratio_ci(v, pooled["samples"])
            rows.append({
                "composition": composition,
                "instruction_level": level,
                "sensitive_attr": attr,
                "n_pairs": int(len(common)),
                "n_pairs_visible": int(len(vis)),
                "n_pairs_ciego": int(pooled["n_pairs"]),
                "n_pairs_semillas_incompletas": incomplete,
                "k_seeds_max": k_max,
                "sd_visible": float(np.std(v, ddof=1)),
                "sd_ciego": float(np.sqrt(pooled["var"])),
                "sd_ciego_atributo": float(np.std(b, ddof=1)),
                "F": f_stat,
                "ci95_lo_F": ci_lo,
                "ci95_hi_F": ci_hi,
                "p_fisher": p_fisher,
                # Levene keeps the per-attribute blind sample as its second
                # group: it is a sample-based test and cannot take a pooled
                # variance, so it stays a 21-vs-21 robust cross-check.
                "p_levene": _levene_p(v, b),
                "familia": T04_FAMILIA,
            })

    if not rows:
        raise ValueError("build_t04: no cell could be computed.")
    table = pd.DataFrame(rows)

    table = _add_blind_homogeneity(table, ciego, attrs)

    q_global, sig_global = agg.benjamini_hochberg(table["p_fisher"].to_numpy(dtype=float), q=fdr_q)
    table["q_global"] = q_global
    table["sig_BH_global"] = sig_global
    table["q_por_composicion"] = np.nan
    table["sig_BH_composicion"] = False
    for composition, frame in table.groupby("composition", dropna=False):
        q_comp, sig_comp = agg.benjamini_hochberg(
            frame["p_fisher"].to_numpy(dtype=float), q=fdr_q
        )
        table.loc[frame.index, "q_por_composicion"] = q_comp
        table.loc[frame.index, "sig_BH_composicion"] = sig_comp

    table = (
        table[T04_COLS]
        .sort_values(["composition", "instruction_level", "sensitive_attr"])
        .reset_index(drop=True)
    )

    if expected_rows is not None and len(table) != expected_rows:
        detail = (
            "; missing cells: "
            + ", ".join(f"{c}/{lv}/{a}" for c, lv, a in missing_cells)
            if missing_cells else
            "; no cell failed to compute - the input data does not cover the "
            f"expected grid (compositions={compositions}, levels={levels})"
        )
        raise ValueError(
            f"build_t04 produced {len(table)} rows, expected {expected_rows}{detail}"
        )
    return table


# ---------------------------------------------------------------------------
# T13 - validation of the internal null
# ---------------------------------------------------------------------------
def build_t13(
    df: pd.DataFrame,
    turn: int = GENESIS_TURN,
    attrs: tuple[str, ...] = SENSITIVE_ATTRS,
    include_errors: bool = False,
) -> pd.DataFrame:
    """T13 - is the blind agent's gap really independent of the attribute?

    One row per (composition, instruction_level). The blind agent's genesis
    prompt is identical byte for byte across the control, gender and country
    arms, so its gap must not depend on the attribute. This table is the
    quantitative side of that claim and it is meant to be CITED, not left in a
    log: it counts, cell by cell, in how many pairs the blind gap of the gender
    arm and of the country arm coincide, and how correlated the two are.

    Two regimes appear, and both are consistent with the prompt audit:

    - ``gaps_identicos = True`` (local backends: llama, qwen, mistral, and the
      hetero levels served by a local blind agent): the model reproduces its
      own sample, so the two arms give the same number pair by pair and
      ``correlacion_entre_atributos`` is exactly 1. The two estimators are one
      sample counted twice - the reason T04 pools variances instead of
      concatenating samples.
    - ``gaps_identicos = False`` (homo_gpt and the hetero compositions at
      level_0, i.e. the API-served runs): the backend is not reproducible
      between calls, so the two arms are two distinct draws of the same noise.
      This is NOT a leak - the prompt diff rules that out - and it is not a
      threat to validity: the visible arm of those same runs carries the same
      non-determinism, within the same execution, so the variance ratio stays a
      fair comparison.

    Columns: ``composition, instruction_level, gaps_identicos,
    n_parejas_identicas, correlacion_entre_atributos`` plus ``n_parejas``,
    ``max_diferencia_abs`` and the two per-attribute sds as support.
    """
    columns = [
        "composition", "instruction_level", "gaps_identicos",
        "n_parejas_identicas", "correlacion_entre_atributos", "n_parejas",
        "max_diferencia_abs", "sd_ciego_" + attrs[0], "sd_ciego_" + attrs[1],
    ]
    if len(attrs) != 2:
        raise ValueError("build_t13 compares exactly 2 sensitive attributes")

    ciego = promedio_entre_semillas(
        gap_por_pareja(df, turn=turn, blind=True, include_errors=include_errors)
    )
    if ciego.empty:
        return _empty(columns)

    attr_a, attr_b = attrs
    rows = []
    for composition in sorted(ciego["composition"].dropna().unique().tolist()):
        for level in sorted(ciego["instruction_level"].dropna().unique().tolist()):
            sample_a = _cell_values(ciego, composition, level, attr_a)
            sample_b = _cell_values(ciego, composition, level, attr_b)
            common = sample_a.index.intersection(sample_b.index)
            if len(common) < 2:
                continue
            values_a = sample_a.loc[common].to_numpy(dtype=float)
            values_b = sample_b.loc[common].to_numpy(dtype=float)
            diff = np.abs(values_a - values_b)
            rho, _, _ = agg._pearson(values_a, values_b)
            rows.append({
                "composition": composition,
                "instruction_level": level,
                "gaps_identicos": bool(np.all(diff <= 1e-9)),
                "n_parejas_identicas": int(np.sum(diff <= 1e-9)),
                "correlacion_entre_atributos": rho,
                "n_parejas": int(len(common)),
                "max_diferencia_abs": float(diff.max()),
                "sd_ciego_" + attr_a: float(np.std(values_a, ddof=1)),
                "sd_ciego_" + attr_b: float(np.std(values_b, ddof=1)),
            })

    if not rows:
        return _empty(columns)
    table = (
        pd.DataFrame(rows)[columns]
        .sort_values(["composition", "instruction_level"])
        .reset_index(drop=True)
    )
    log.info(
        "T13: the blind gap is identical between %s and %s in %d of the %d "
        "(composition, level) cells",
        attr_a, attr_b, int(table["gaps_identicos"].sum()), len(table),
    )
    return table


# ---------------------------------------------------------------------------
# B.6 - T05: between-seed ICC
# ---------------------------------------------------------------------------
def _icc_one_way(matrix: np.ndarray) -> dict:
    """One-way random-effects ICC(1) over a ``n pairs x k seeds`` matrix.

    ``MSB = k * sum((pair_mean - grand_mean)^2) / (n-1)``,
    ``MSW = sum((x - pair_mean)^2) / (n*(k-1))``,
    ``F = MSB/MSW`` with (n-1, n(k-1)) df, ``ICC = (F-1)/(F+k-1)``.

    ``sd_entre_parejas`` is the between-pair variance component
    ``sqrt(max(MSB-MSW, 0)/k)`` (0 when MSB < MSW, i.e. negative ICC) and
    ``sd_dentro_pareja`` is ``sqrt(MSW)``, the run-to-run noise.
    """
    stats = _require_scipy()
    n, k = matrix.shape
    empty = {
        "icc": float("nan"), "F": float("nan"), "p_value": float("nan"),
        "sd_entre_parejas": float("nan"), "sd_dentro_pareja": float("nan"),
        "n_pairs": int(n), "k_seeds": int(k),
    }
    if n < 2 or k < 2 or not np.isfinite(matrix).all():
        return empty
    pair_mean = matrix.mean(axis=1)
    grand_mean = matrix.mean()
    ms_between = k * float(((pair_mean - grand_mean) ** 2).sum()) / (n - 1)
    ms_within = float(((matrix - pair_mean[:, None]) ** 2).sum()) / (n * (k - 1))
    if ms_within == 0:
        return empty
    f_stat = ms_between / ms_within
    return {
        "icc": float((f_stat - 1) / (f_stat + k - 1)),
        "F": float(f_stat),
        "p_value": float(stats.f.sf(f_stat, n - 1, n * (k - 1))),
        "sd_entre_parejas": float(np.sqrt(max(ms_between - ms_within, 0.0) / k)),
        "sd_dentro_pareja": float(np.sqrt(ms_within)),
        "n_pairs": int(n),
        "k_seeds": int(k),
    }


def _seed_matrix(gaps: pd.DataFrame, label: str) -> tuple[np.ndarray | None, int]:
    """``n pairs x k seeds`` matrix from per-seed gaps, and how many pairs were
    dropped for not being measured in every seed.

    The one-way ANOVA assumes a balanced design, so incomplete pairs cannot be
    used - but they are never imputed and never dropped silently: the count is
    logged AND returned, so it lands in the ``n_pairs_descartadas`` column of
    the table. Returns ``(None, dropped)`` when nothing usable is left.
    """
    if gaps.empty:
        return None, 0
    matrix = gaps.pivot_table(index="pair_id", columns="seed", values="gap", aggfunc="mean")
    matrix = matrix.sort_index(axis=0).sort_index(axis=1)
    complete = matrix.dropna(axis=0, how="any")
    dropped = len(matrix) - len(complete)
    if dropped:
        log.warning(
            "T05 %s: %d/%d pairs dropped - not present in all %d seeds",
            label, dropped, len(matrix), matrix.shape[1],
        )
    if complete.shape[0] < 2 or complete.shape[1] < 2:
        return None, dropped
    return complete.to_numpy(dtype=float), dropped


def build_t05(
    df: pd.DataFrame,
    df_placebo: pd.DataFrame | None = None,
    turn: int = GENESIS_TURN,
    attrs: tuple[str, ...] = SENSITIVE_ATTRS,
    alpha: float = 0.05,
    fdr_q: float = DEFAULT_FDR_Q,
    include_errors: bool = False,
) -> pd.DataFrame:
    """T05 - between-seed ICC of the genesis gap, per cell, plus the placebo.

    Built on the ``21 pairs x 3 seeds`` matrix of visible genesis gaps (per
    seed, NOT averaged). The ICC answers whether the gap is a property of the
    basket or of the run.

    This table is what documents the decision NOT to implement a per-basket
    analysis: with a non-significant ICC the effect lives in the execution, not
    in the basket.

    All the ICC tests of the table form ONE declared BH family (``q_value``,
    ``sig_BH``). The "there is basket structure here" warning fires only for
    cells that survive BH - an uncorrected p < ``alpha`` among 42 tests is
    expected by chance and is logged at INFO, not as a finding.

    Pairs missing in some seed cannot enter the balanced ANOVA; they are never
    imputed and the count is reported in ``n_pairs_descartadas`` so a cell with
    19 or 20 pairs is visible in the table itself.
    """
    gaps = gap_por_pareja(df, turn=turn, blind=False, include_errors=include_errors)
    rows = []
    if not gaps.empty:
        keys = ["composition", "instruction_level", "sensitive_attr"]
        for (composition, level, attr), frame in gaps.groupby(keys, dropna=False):
            if attr not in attrs:
                continue
            label = f"{composition}/{level}/{attr}"
            matrix, dropped = _seed_matrix(frame, label)
            if matrix is None:
                log.warning("T05 %s: not enough complete pairs/seeds, cell skipped", label)
                continue
            row = {
                "composition": composition,
                "instruction_level": level,
                "sensitive_attr": attr,
                "n_pairs_descartadas": dropped,
            }
            row.update(_icc_one_way(matrix))
            rows.append(row)

    if df_placebo is not None and not df_placebo.empty:
        placebo_gaps = gap_por_pareja(
            df_placebo, turn=turn, blind=False, placebo=True,
            include_errors=include_errors,
        )
        keys = ["composition", "instruction_level", "sensitive_attr"]
        for (composition, level, attr), frame in placebo_gaps.groupby(keys, dropna=False):
            label = f"{composition}/{level}/placebo"
            matrix, dropped = _seed_matrix(frame, label)
            if matrix is None:
                log.warning("T05 %s: not enough complete pairs/seeds, cell skipped", label)
                continue
            row = {
                "composition": composition,
                "instruction_level": level,
                "sensitive_attr": str(attr),
                "n_pairs_descartadas": dropped,
            }
            row.update(_icc_one_way(matrix))
            rows.append(row)

    if not rows:
        return _empty(T05_COLS)

    table = pd.DataFrame(rows).sort_values(
        ["composition", "instruction_level", "sensitive_attr"]
    ).reset_index(drop=True)

    # Declared family: every ICC test of the table (treatment cells + placebo
    # references) corrected together.
    q_value, sig = agg.benjamini_hochberg(
        table["p_value"].to_numpy(dtype=float), q=fdr_q
    )
    table["q_value"] = q_value
    table["sig_BH"] = sig
    table = table[T05_COLS]

    for _, row in table[table["sig_BH"]].iterrows():
        log.warning(
            "T05: ICC survives BH in %s/%s/%s (ICC=%.3f, p=%.4f, q=%.4f). "
            "There IS basket structure in this cell and it deserves "
            "exploration - that analysis is intentionally NOT implemented "
            "in this script.",
            row["composition"], row["instruction_level"], row["sensitive_attr"],
            row["icc"], row["p_value"], row["q_value"],
        )
    uncorrected = table[
        (pd.to_numeric(table["p_value"], errors="coerce") < alpha) & (~table["sig_BH"])
    ]
    if not uncorrected.empty:
        log.info(
            "T05: %d cells with uncorrected p < %.2f do NOT survive BH over "
            "the %d tests of the family (%s) - not reported as basket "
            "structure: %s",
            len(uncorrected), alpha, len(table), T05_FAMILIA,
            ", ".join(
                f"{r['composition']}/{r['instruction_level']}/{r['sensitive_attr']}"
                for _, r in uncorrected.iterrows()
            ),
        )
    return table


# ---------------------------------------------------------------------------
# B.9 - T01: data quality per composition
# ---------------------------------------------------------------------------
T01_COLS = [
    "stage", "composition", "n_rows_total", "n_runs_completed", "n_subject_rows",
    "n_parse_error", "rate_parse_error", "n_was_normalized", "rate_was_normalized",
    "n_rows_descartadas", "rate_descartadas",
    "n_parse_error_control", "n_control_rows",
    "n_parse_error_variant", "n_variant_rows",
    "p_binom_asimetria_parseo",
]


def build_t01(df: pd.DataFrame, stage: str = "detection") -> pd.DataFrame:
    """T01 - data quality per composition (B.9).

    Reports, per (``stage``, composition): completed runs (distinct
    ``basket_id``), the ``parse_error`` rate, the ``was_normalized`` rate, and
    rows discarded by the default filter
    (``filter_valid_subject_rows(include_errors=False)``). ``stage`` keeps
    detection and placebo rows from colliding when the caller builds both
    (same composition names, different arms) - call this once per stage and
    concatenate, do not mix ``df_detection``/``df_placebo`` in one call.

    Closes the pre-declared vigilance point: a binomial test of whether the
    parse-error PROPORTION differs between the variant arm(s) and the control
    arm (``placebo_a`` stands in for ``control`` in placebo-only frames, since
    neither literally has a ``control`` row). ``p_binom_asimetria_parseo`` is
    ``NaN`` when either arm has zero rows - not zero, which would misreport
    "no asymmetry" for data that was never compared.
    """
    if df is None or df.empty or "composition" not in df.columns:
        return _empty(T01_COLS)
    stats = _require_scipy()
    rows = []
    for composition, frame in df.groupby("composition", dropna=False):
        subject = rex.filter_valid_subject_rows(frame, include_errors=True)
        parse_error = (
            rex._truthy(subject["parse_error"])
            if "parse_error" in subject.columns
            else pd.Series(False, index=subject.index)
        )
        was_norm = (
            rex._truthy(subject["was_normalized"])
            if "was_normalized" in subject.columns
            else pd.Series(False, index=subject.index)
        )
        kept = rex.filter_valid_subject_rows(frame, include_errors=False)
        n_subject = len(subject)
        n_error = int(parse_error.sum())
        n_descartadas = n_subject - len(kept)
        n_runs = int(frame["basket_id"].nunique()) if "basket_id" in frame.columns else 0

        variant = (
            subject["variant"].astype(str) if "variant" in subject.columns
            else pd.Series("", index=subject.index)
        )
        is_control = variant == "control"
        if not is_control.any() and (variant == "placebo_a").any():
            is_control = variant == "placebo_a"
        is_variant = (~is_control) & variant.ne("")

        n_control = int(is_control.sum())
        n_variant = int(is_variant.sum())
        err_control = int((parse_error & is_control).sum())
        err_variant = int((parse_error & is_variant).sum())
        p_binom = float("nan")
        if n_control > 0 and n_variant > 0:
            p_control = err_control / n_control
            p_control = min(max(p_control, 1e-9), 1 - 1e-9)
            try:
                p_binom = float(stats.binomtest(err_variant, n_variant, p_control).pvalue)
            except (ValueError, ZeroDivisionError):
                p_binom = float("nan")

        rows.append({
            "stage": stage,
            "composition": composition,
            "n_rows_total": int(len(frame)),
            "n_runs_completed": n_runs,
            "n_subject_rows": n_subject,
            "n_parse_error": n_error,
            "rate_parse_error": n_error / n_subject if n_subject else float("nan"),
            "n_was_normalized": int(was_norm.sum()),
            "rate_was_normalized": float(was_norm.mean()) if n_subject else float("nan"),
            "n_rows_descartadas": n_descartadas,
            "rate_descartadas": n_descartadas / n_subject if n_subject else float("nan"),
            "n_parse_error_control": err_control,
            "n_control_rows": n_control,
            "n_parse_error_variant": err_variant,
            "n_variant_rows": n_variant,
            "p_binom_asimetria_parseo": p_binom,
        })
    if not rows:
        return _empty(T01_COLS)
    return pd.DataFrame(rows)[T01_COLS].sort_values(["stage", "composition"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# B.9 - T02: placebo noise floor, per composition and turn
# ---------------------------------------------------------------------------
T02_COLS = [
    "composition", "turn", "n_pairs", "mean_gap", "sd", "p_signflip",
    "ci95_lo", "ci95_hi", "band_lo_p5", "band_hi_p95",
]


def build_t02(df_placebo: pd.DataFrame, turns: tuple[int, ...] = ALL_TURNS) -> pd.DataFrame:
    """T02 - placebo (twin-vs-twin) noise floor per composition and turn.

    ``mean_gap``/``sd``/``ci95_*`` reuse the B.4 core function (placebo gap =
    ``placebo_b - placebo_a``, seeds averaged -> 21 pairs). ``band_lo_p5`` /
    ``band_hi_p95`` are the 5th/95th percentile band from ``placebo_band``
    (per pair x seed, NOT seed-averaged - a wider, more conservative band than
    the seed-averaged ``sd``). ``placebo_band`` reports the opposite sign
    convention (``placebo_a - placebo_b``), so its band is negated here to
    match this module's ``placebo_b - placebo_a`` convention - same negation
    ``trajectory_with_placebo`` already applies.
    """
    if df_placebo is None or df_placebo.empty or "composition" not in df_placebo.columns:
        return _empty(T02_COLS)
    band = agg.placebo_band(df_placebo)
    rows = []
    for composition in sorted(df_placebo["composition"].dropna().unique().tolist()):
        sub = df_placebo[df_placebo["composition"] == composition]
        for turn in turns:
            values = promedio_entre_semillas(
                gap_por_pareja(sub, turn=turn, blind=False, placebo=True)
            )["gap"].to_numpy(dtype=float)
            n = values.size
            ci_lo, ci_hi = _bootstrap_ci_mean(values)
            band_row = band[(band["composition"] == composition) & (band["turn"] == turn)]
            rows.append({
                "composition": composition,
                "turn": turn,
                "n_pairs": int(n),
                "mean_gap": float(values.mean()) if n else float("nan"),
                "sd": float(values.std(ddof=1)) if n > 1 else float("nan"),
                "p_signflip": agg._signflip_p(values) if n else float("nan"),
                "ci95_lo": ci_lo,
                "ci95_hi": ci_hi,
                "band_lo_p5": -float(band_row["hi"].iloc[0]) if not band_row.empty else float("nan"),
                "band_hi_p95": -float(band_row["lo"].iloc[0]) if not band_row.empty else float("nan"),
            })
    if not rows:
        return _empty(T02_COLS)
    return pd.DataFrame(rows)[T02_COLS].sort_values(["composition", "turn"]).reset_index(drop=True)


def _bootstrap_ci_mean(values: np.ndarray) -> tuple[float, float]:
    return agg._bootstrap_ci(values, n_boot=N_BOOT, seed=BOOTSTRAP_SEED)


# ---------------------------------------------------------------------------
# B.9 - T03: genesis means, 36 tests, BH (the null-result half of the story)
# ---------------------------------------------------------------------------
T03_COLS = [
    "composition", "instruction_level", "sensitive_attr",
    "n_pairs", "mean_gap", "sd", "p_signflip", "ci95_lo", "ci95_hi",
    "q_value", "sig_BH",
]


def build_t03(
    df: pd.DataFrame,
    fdr_q: float = DEFAULT_FDR_Q,
    expected_rows: int | None = EXPECTED_T03_ROWS,
) -> pd.DataFrame:
    """T03 - genesis mean gap, 36 tests, BH (B.9).

    Concatenates ``agg.genesis_gap_stats`` per composition (the pre-existing,
    UNTOUCHED function), drops the ``POOLED_INVALID_LABEL`` diagnostic row
    (Tarea A.1: pseudo-replication, never reported), and BH-corrects the 36
    ``p_signflip`` values as ONE family - the same 6x3x2 grid as T04, "for
    coherence with the means table" (B.5). This is the null-result half of the
    story: no cell should survive BH here, which is what justifies looking at
    the variance instead (T04).

    Raises ValueError if the result does not have exactly ``expected_rows``
    rows (mirrors T04's noisy-failure contract). Pass ``expected_rows=None``
    for a deliberate subset run (e.g. a single composition).
    """
    if df is None or df.empty or "composition" not in df.columns:
        raise ValueError("build_t03: no data to build the genesis means table.")

    frames = []
    compositions = sorted(df["composition"].dropna().unique().tolist())
    for composition in compositions:
        stats = agg.genesis_gap_stats(df[df["composition"] == composition])
        if stats.empty:
            continue
        stats = stats[stats["instruction_level"] != agg.POOLED_INVALID_LABEL].copy()
        stats.insert(0, "composition", composition)
        frames.append(stats)

    if not frames:
        raise ValueError("build_t03: no cell could be computed.")
    table = pd.concat(frames, ignore_index=True)

    q_value, sig = agg.benjamini_hochberg(table["p_signflip"].to_numpy(dtype=float), q=fdr_q)
    table["q_value"] = q_value
    table["sig_BH"] = sig
    table = (
        table[T03_COLS]
        .sort_values(["composition", "instruction_level", "sensitive_attr"])
        .reset_index(drop=True)
    )

    if expected_rows is not None and len(table) != expected_rows:
        present = set(zip(table["composition"], table["instruction_level"], table["sensitive_attr"]))
        expected = {
            (c, lv, a)
            for c in compositions for lv in ALL_LEVELS for a in SENSITIVE_ATTRS
        }
        missing = sorted(expected - present)
        raise ValueError(
            f"build_t03 produced {len(table)} rows, expected {expected_rows}; "
            f"missing cells: {', '.join(f'{c}/{lv}/{a}' for c, lv, a in missing)}"
        )
    return table


# ---------------------------------------------------------------------------
# B.7 - T06: variance ratio by turn (genesis to dissolution)
# ---------------------------------------------------------------------------
T06_COLS = [
    "composition", "instruction_level", "sensitive_attr", "turn",
    "n_pairs", "sd_visible", "sd_ciego", "sd_placebo", "F", "p_fisher", "p_levene",
]


def build_t06(
    df: pd.DataFrame,
    df_placebo: pd.DataFrame | None = None,
    turns: tuple[int, ...] = ALL_TURNS,
    attrs: tuple[str, ...] = SENSITIVE_ATTRS,
) -> pd.DataFrame:
    """T06 - the T04 variance ratio, at every turn (B.7).

    Same computation as ``build_t04`` (pooled blind denominator, F against the
    pooled variance, median-centred Levene as cross-check) but looped over
    turns 0-4 instead of fixed at genesis, with the placebo sd of the SAME
    turn and composition as a third reference (broadcast across levels: the
    placebo currently runs at one instruction level only, per composition -
    same broadcast ``trajectory_with_placebo`` uses).

    Reuses ``_pooled_blind_variance`` / ``_f_test_against_pooled`` /
    ``_levene_p`` from B.5 unchanged - it does NOT touch ``build_t04``. No BH
    correction here: this is the decay diagnostic (F04 is), not a family of
    pre-declared tests.

    Degrades to an empty typed frame (never raises) when a turn has no data -
    the contract Tarea C.3 requires for genesis-only CSVs.
    """
    if df is None or df.empty:
        return _empty(T06_COLS)

    placebo_sd_by_turn: dict[tuple[str, int], float] = {}
    if df_placebo is not None and not df_placebo.empty and "composition" in df_placebo.columns:
        for composition in sorted(df_placebo["composition"].dropna().unique().tolist()):
            sub_p = df_placebo[df_placebo["composition"] == composition]
            for turn in turns:
                vals = promedio_entre_semillas(
                    gap_por_pareja(sub_p, turn=turn, blind=False, placebo=True)
                )["gap"].to_numpy(dtype=float)
                placebo_sd_by_turn[(composition, turn)] = (
                    float(vals.std(ddof=1)) if vals.size > 1 else float("nan")
                )

    rows = []
    for turn in turns:
        visible = promedio_entre_semillas(gap_por_pareja(df, turn=turn, blind=False))
        ciego = promedio_entre_semillas(gap_por_pareja(df, turn=turn, blind=True))
        if visible.empty or ciego.empty:
            continue
        compositions = sorted(visible["composition"].dropna().unique().tolist())
        levels = sorted(visible["instruction_level"].dropna().unique().tolist())
        for composition, level in product(compositions, levels):
            pooled = _pooled_blind_variance(ciego, composition, level, attrs)
            if pooled is None:
                continue
            for attr in attrs:
                vis = _cell_values(visible, composition, level, attr)
                blind = _cell_values(ciego, composition, level, attr)
                common = vis.index.intersection(blind.index)
                if len(common) < 2:
                    continue
                v = vis.loc[common].to_numpy(dtype=float)
                b = blind.loc[common].to_numpy(dtype=float)
                f_stat, p_fisher = _f_test_against_pooled(v, pooled["var"], pooled["df"])
                rows.append({
                    "composition": composition, "instruction_level": level,
                    "sensitive_attr": attr, "turn": turn,
                    "n_pairs": int(len(common)),
                    "sd_visible": float(np.std(v, ddof=1)),
                    "sd_ciego": float(np.sqrt(pooled["var"])),
                    "sd_placebo": placebo_sd_by_turn.get((composition, turn), float("nan")),
                    "F": f_stat,
                    "p_fisher": p_fisher,
                    "p_levene": _levene_p(v, b),
                })
    if not rows:
        log.warning("build_t06: no (turn, composition, level, attr) cell could be computed.")
        return _empty(T06_COLS)
    return (
        pd.DataFrame(rows)[T06_COLS]
        .sort_values(["composition", "instruction_level", "sensitive_attr", "turn"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# B.8 - T07: transmission slopes, gap_ciego(t+1) ~ gap_visible(t)
# ---------------------------------------------------------------------------
T07_COLS = [
    "composition", "n_treatment", "slope_treatment", "se_treatment", "r_treatment",
    "n_placebo", "slope_placebo", "se_placebo", "r_placebo",
    "z_diff", "p_diff", "q_value", "sig_BH",
    "x_range_treatment", "x_range_placebo", "familia",
]
T07_FAMILIA = "BH sobre las 6 pendientes de transmision (una por composicion, tratamiento vs placebo)"


def _ols_slope(x: np.ndarray, y: np.ndarray) -> dict:
    """Simple OLS slope of y on x, with its standard error. n < 3 -> NaN."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = int(x.size)
    out = {"slope": float("nan"), "se": float("nan"), "n": n, "r": float("nan"), "x_range": float("nan")}
    if n < 3:
        return out
    x_mean, y_mean = x.mean(), y.mean()
    sxx = float(((x - x_mean) ** 2).sum())
    out["x_range"] = float(x.max() - x.min())
    if sxx == 0:
        return out
    sxy = float(((x - x_mean) * (y - y_mean)).sum())
    slope = sxy / sxx
    intercept = y_mean - slope * x_mean
    resid = y - (slope * x + intercept)
    sse = float((resid ** 2).sum())
    se = float(np.sqrt((sse / (n - 2)) / sxx)) if n > 2 else float("nan")
    rho, _, _ = agg._pearson(x, y)
    out.update({"slope": float(slope), "se": se, "r": rho})
    return out


def _lag_pairs_for_slope(df: pd.DataFrame, placebo: bool) -> pd.DataFrame:
    """Pool (x = visible gap at t, y = blind gap at t+1) over turns 0..3,
    levels, attributes, seeds and pairs, per composition.

    Built on ``gap_por_pareja`` (turn-by-turn, unaveraged over seeds), never
    on an aggregate that hides the pairing. With a single-turn CSV (Tarea C.3,
    e.g. genesis-only mitigation runs) every ``turn+1`` slice is empty, so the
    merge is empty for every ``t`` and this returns an empty typed frame - the
    caller (``build_t07``) turns that into a graceful empty table, not an
    exception.
    """
    columns = ["composition", "x", "y"]
    if df is None or df.empty:
        return _empty(columns)
    frames = []
    for t in range(0, 4):
        vis = gap_por_pareja(df, turn=t, blind=False, placebo=placebo)
        bl = gap_por_pareja(df, turn=t + 1, blind=True, placebo=placebo)
        if vis.empty or bl.empty:
            continue
        merge_keys = ["composition", "instruction_level", "seed", "pair_id", "sensitive_attr"]
        merged = vis.merge(bl, on=merge_keys, how="inner", suffixes=("_vis", "_bl"))
        if merged.empty:
            continue
        frames.append(merged[["composition", "gap_vis", "gap_bl"]].rename(
            columns={"gap_vis": "x", "gap_bl": "y"}
        ))
    if not frames:
        return _empty(columns)
    return pd.concat(frames, ignore_index=True)


def build_t07(
    df: pd.DataFrame,
    df_placebo: pd.DataFrame | None,
    fdr_q: float = DEFAULT_FDR_Q,
) -> pd.DataFrame:
    """T07 - transmission slope: does the visible gap at t predict the blind
    agent's gap at t+1, more than it does in the placebo (B.8)?

    Replaces the coupling correlation of the prior brief (subject to range
    restriction). Per composition: OLS slope of ``gap_ciego(t+1)`` on
    ``gap_visible(t)``, pooled over turns 0->1..3->4, levels, attributes,
    seeds and pairs; the same regression on the placebo
    (``placebo_b - placebo_a``, same lag structure). ``z_diff`` is the
    standard two-slope-difference z test, ``p_diff`` its two-sided p-value.
    Declared family: the 6 compositions, BH-corrected together.

    ``r_treatment`` / ``r_placebo`` and ``x_range_*`` are reported as
    diagnostics: the Pearson r of the same pairing, and the range of the
    predictor in each arm, so a restricted range in one arm relative to the
    other is visible directly in the table.

    Degrades to an empty typed frame (never raises) when no (t, t+1) pair is
    available in EITHER arm - genesis-only CSVs (Tarea C.3) have no t+1.
    """
    lag_treat = _lag_pairs_for_slope(df, placebo=False)
    lag_placebo = _lag_pairs_for_slope(df_placebo, placebo=True) if df_placebo is not None else _empty(["composition", "x", "y"])
    if lag_treat.empty:
        log.warning("build_t07: no (t, t+1) turn pairs in the treatment arm - "
                    "returning an empty table (genesis-only data?).")
        return _empty(T07_COLS)

    stats = _require_scipy()
    rows = []
    for composition in sorted(lag_treat["composition"].dropna().unique().tolist()):
        sub_t = lag_treat[lag_treat["composition"] == composition]
        fit_t = _ols_slope(sub_t["x"].to_numpy(), sub_t["y"].to_numpy())
        sub_p = lag_placebo[lag_placebo["composition"] == composition] if not lag_placebo.empty else lag_placebo
        fit_p = _ols_slope(sub_p["x"].to_numpy(), sub_p["y"].to_numpy()) if sub_p is not None and not sub_p.empty else _ols_slope(np.array([]), np.array([]))

        z_diff, p_diff = float("nan"), float("nan")
        if np.isfinite(fit_t["se"]) and np.isfinite(fit_p["se"]) and (fit_t["se"] ** 2 + fit_p["se"] ** 2) > 0:
            z_diff = (fit_t["slope"] - fit_p["slope"]) / np.sqrt(fit_t["se"] ** 2 + fit_p["se"] ** 2)
            p_diff = float(2 * stats.norm.sf(abs(z_diff)))

        rows.append({
            "composition": composition,
            "n_treatment": fit_t["n"], "slope_treatment": fit_t["slope"],
            "se_treatment": fit_t["se"], "r_treatment": fit_t["r"],
            "n_placebo": fit_p["n"], "slope_placebo": fit_p["slope"],
            "se_placebo": fit_p["se"], "r_placebo": fit_p["r"],
            "z_diff": z_diff, "p_diff": p_diff,
            "x_range_treatment": fit_t["x_range"], "x_range_placebo": fit_p["x_range"],
            "familia": T07_FAMILIA,
        })
    if not rows:
        return _empty(T07_COLS)
    table = pd.DataFrame(rows)
    q_value, sig = agg.benjamini_hochberg(table["p_diff"].to_numpy(dtype=float), q=fdr_q)
    table["q_value"] = q_value
    table["sig_BH"] = sig
    return table[T07_COLS].sort_values("composition").reset_index(drop=True)


# ---------------------------------------------------------------------------
# B.9 - T08: contrast between instruction levels
# ---------------------------------------------------------------------------
T08_LEVEL_PAIRS = (
    ("level_0_neutral", "level_1_professional"),
    ("level_1_professional", "level_2_identity"),
    ("level_0_neutral", "level_2_identity"),
)
T08_COLS = [
    "composition", "sensitive_attr", "level_a", "level_b",
    "n_pairs", "mean_diff", "p_signflip", "q_value", "sig_BH",
    "spearman_rho", "spearman_p", "spearman_n", "familia",
]
T08_FAMILIA = "BH sobre los 3 contrastes pareados entre niveles (sign-flip); Spearman es diagnostico, no forma parte de esta familia"


def build_t08(df: pd.DataFrame, fdr_q: float = DEFAULT_FDR_Q, attrs: tuple[str, ...] = SENSITIVE_ATTRS) -> pd.DataFrame:
    """T08 - paired contrast between instruction levels, per composition (B.9).

    For each (composition, sensitive_attr): paired sign-flip test of the gap
    difference between two levels, on the SAME 21 pairs (level_0 vs level_1,
    level_1 vs level_2, level_0 vs level_2) - one declared BH family of up to
    36 tests (6 compositions x 2 attrs x 3 level pairs; fewer where a level is
    missing, e.g. hetero_local/level_1).

    ``spearman_rho``/``spearman_p`` are a separate diagnostic: Spearman
    correlation between the instruction-level index (0/1/2) and the per-pair
    gap, pooled over the 3 levels (n up to 63) - NOT part of the BH family and
    NOT a replacement for the per-level tests, since pooling levels is exactly
    the pseudo-replication Tarea A.1 warns against. It answers only "is there
    a monotone trend", attached to every row of its (composition, attr) group.
    """
    stats = _require_scipy()
    visible = promedio_entre_semillas(gap_por_pareja(df, turn=GENESIS_TURN, blind=False))
    if visible.empty:
        return _empty(T08_COLS)

    rows = []
    for composition in sorted(visible["composition"].dropna().unique().tolist()):
        for attr in attrs:
            level_vals = {
                level: _cell_values(visible, composition, level, attr) for level in ALL_LEVELS
            }
            trend_x, trend_y = [], []
            for idx, level in enumerate(ALL_LEVELS):
                series = level_vals[level]
                trend_x.extend([idx] * len(series))
                trend_y.extend(series.to_numpy(dtype=float).tolist())
            if len(trend_x) >= 3 and len(set(trend_x)) > 1:
                rho, p_s = stats.spearmanr(trend_x, trend_y)
                rho, p_s = float(rho), float(p_s)
            else:
                rho, p_s = float("nan"), float("nan")
            n_trend = len(trend_x)

            for level_a, level_b in T08_LEVEL_PAIRS:
                a, b = level_vals.get(level_a), level_vals.get(level_b)
                if a is None or b is None or a.empty or b.empty:
                    continue
                common = a.index.intersection(b.index)
                if len(common) < 2:
                    continue
                diff = (b.loc[common] - a.loc[common]).to_numpy(dtype=float)
                rows.append({
                    "composition": composition, "sensitive_attr": attr,
                    "level_a": level_a, "level_b": level_b,
                    "n_pairs": int(len(common)), "mean_diff": float(diff.mean()),
                    "p_signflip": agg._signflip_p(diff),
                    "spearman_rho": rho, "spearman_p": p_s, "spearman_n": n_trend,
                    "familia": T08_FAMILIA,
                })
    if not rows:
        return _empty(T08_COLS)
    table = pd.DataFrame(rows)
    q_value, sig = agg.benjamini_hochberg(table["p_signflip"].to_numpy(dtype=float), q=fdr_q)
    table["q_value"] = q_value
    table["sig_BH"] = sig
    return (
        table[T08_COLS]
        .sort_values(["composition", "sensitive_attr", "level_a", "level_b"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# B.9 - T09: composition comparison, at a glance
# ---------------------------------------------------------------------------
T09_COLS = [
    "composition", "sensitive_attr",
    "mean_gap_level_0", "mean_gap_level_1", "mean_gap_level_2",
    "F_level_0", "F_level_1", "F_level_2", "F_max", "level_F_max",
]


def build_t09(t04: pd.DataFrame, t03: pd.DataFrame | None = None) -> pd.DataFrame:
    """T09 - composition comparison, ordered by variance-ratio magnitude (B.9).

    Reshapes T04 (and, if given, T03) wide by level - one row per
    (composition, sensitive_attr) with the 3 levels as columns - instead of
    averaging across levels, which would be the same pseudo-replication Tarea
    A.1 forbids. ``F_max`` / ``level_F_max`` name the strongest level, and the
    table is sorted by ``F_max`` descending: "does the pattern replicate" at a
    glance across the 6 compositions.
    """
    if t04 is None or t04.empty:
        return _empty(T09_COLS)

    f_wide = t04.assign(level_short=t04["instruction_level"].map(LEVEL_SHORT)).pivot_table(
        index=["composition", "sensitive_attr"], columns="level_short", values="F", aggfunc="first",
    )
    f_wide.columns = [f"F_{c}" for c in f_wide.columns]

    if t03 is not None and not t03.empty:
        m_wide = t03.assign(level_short=t03["instruction_level"].map(LEVEL_SHORT)).pivot_table(
            index=["composition", "sensitive_attr"], columns="level_short", values="mean_gap", aggfunc="first",
        )
        m_wide.columns = [f"mean_gap_{c}" for c in m_wide.columns]
    else:
        m_wide = pd.DataFrame()

    table = f_wide.join(m_wide, how="left").reset_index()
    for col in T09_COLS:
        if col not in table.columns:
            table[col] = np.nan

    f_cols = [f"F_{c}" for c in LEVEL_SHORT.values()]
    f_matrix = table[f_cols]
    table["F_max"] = f_matrix.max(axis=1, skipna=True)
    has_any = f_matrix.notna().any(axis=1)
    level_f_max = pd.Series(pd.NA, index=table.index, dtype="object")
    if has_any.any():
        level_f_max.loc[has_any] = (
            f_matrix.loc[has_any].idxmax(axis=1).str.replace("F_", "", regex=False)
        )
    table["level_F_max"] = level_f_max

    return table[T09_COLS].sort_values("F_max", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# B.9 - T10: heterogeneous committee vs. the mean of its homogeneous members
# ---------------------------------------------------------------------------
T10_MEMBERS = {
    "hetero_local": ("homo_llama", "homo_qwen", "homo_mistral"),
    "hetero_api_gpt": ("homo_llama", "homo_gpt", "homo_mistral"),
}
T10_COLS = [
    "hetero_composition", "instruction_level", "sensitive_attr",
    "n_pairs", "mean_diff", "ci95_lo", "ci95_hi", "p_signflip", "member_compositions",
]


def build_t10(df: pd.DataFrame, attrs: tuple[str, ...] = SENSITIVE_ATTRS) -> pd.DataFrame:
    """T10 - heterogeneous committee vs. the MEAN of its homogeneous member
    models (B.9), never vs. lone agents, on the same pairs.

    ``hetero_local`` -> mean of {homo_llama, homo_qwen, homo_mistral};
    ``hetero_api_gpt`` -> mean of {homo_llama, homo_gpt, homo_mistral}, both at
    genesis. ``mean_diff = hetero - mean(members)`` per pair, bootstrap CI and
    sign-flip p over the pairs common to the hetero cell and every available
    member. A missing member composition (or a missing level, e.g.
    hetero_local/level_1) is logged and the row is skipped, not silently
    dropped from the count.
    """
    visible = promedio_entre_semillas(gap_por_pareja(df, turn=GENESIS_TURN, blind=False))
    if visible.empty:
        return _empty(T10_COLS)

    rows = []
    for hetero, members in T10_MEMBERS.items():
        levels = sorted(
            visible[visible["composition"] == hetero]["instruction_level"].dropna().unique().tolist()
        )
        for level in levels:
            for attr in attrs:
                hetero_vals = _cell_values(visible, hetero, level, attr)
                if hetero_vals.empty:
                    continue
                member_series = []
                missing = []
                for member in members:
                    vals = _cell_values(visible, member, level, attr)
                    (member_series if not vals.empty else missing).append(vals if not vals.empty else member)
                if missing:
                    log.warning("T10 %s/%s/%s: missing member composition(s) %s",
                                hetero, level, attr, missing)
                if not member_series:
                    continue
                common = hetero_vals.index
                for s in member_series:
                    common = common.intersection(s.index)
                if len(common) < 2:
                    continue
                member_mean = pd.concat([s.loc[common] for s in member_series], axis=1).mean(axis=1)
                diff = (hetero_vals.loc[common] - member_mean).to_numpy(dtype=float)
                lo, hi = _bootstrap_ci_mean(diff)
                rows.append({
                    "hetero_composition": hetero, "instruction_level": level, "sensitive_attr": attr,
                    "n_pairs": int(len(common)), "mean_diff": float(diff.mean()),
                    "ci95_lo": lo, "ci95_hi": hi, "p_signflip": agg._signflip_p(diff),
                    "member_compositions": ",".join(members),
                })
    if not rows:
        return _empty(T10_COLS)
    return (
        pd.DataFrame(rows)[T10_COLS]
        .sort_values(["hetero_composition", "instruction_level", "sensitive_attr"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# B.9 - T11: verbalization rate
# ---------------------------------------------------------------------------
def build_t11(df: pd.DataFrame) -> pd.DataFrame:
    """T11 - keyword verbalization rate, per composition/level/condition (B.9).

    Thin wrapper: ``aggregate.verbalization_rate`` already groups by
    composition when the column is present, so this reuses it unchanged
    rather than duplicating the keyword logic.
    """
    return agg.verbalization_rate(df)


# ---------------------------------------------------------------------------
# C.4 - T12: mitigation, variance-ratio endpoint
# ---------------------------------------------------------------------------
T12_COLS = [
    "composition", "instruction_level", "sensitive_attr", "vaccine",
    "n_pairs", "sd_visible", "sd_ciego", "F", "ci95_lo_F", "ci95_hi_F",
    "sd_visible_sin_vacuna", "F_sin_vacuna", "F_ratio_vacuna_vs_sin",
    "p_diff_varianza", "mean_gap", "es_ancla_level2_sin_vacuna",
]
T12_ANCHOR_LEVEL = "level_2_identity"


def build_t12(
    df_mitigation: pd.DataFrame,
    df_detection: pd.DataFrame | None,
    attrs: tuple[str, ...] = SENSITIVE_ATTRS,
    turn: int = GENESIS_TURN,
) -> pd.DataFrame:
    """T12 - mitigation table, variance-ratio endpoint (Tarea C.4).

    The mitigation plan (C, brief v2) is genesis-only by construction, so the
    endpoint stays ``turn=0``. Each ``mitigation_*_<vaccine>_seed*.csv`` file
    carries a single ``vaccine`` value, so this filters ``df_mitigation`` to
    one vaccine BEFORE calling ``gap_por_pareja`` - that function's group keys
    do not include ``vaccine`` (see B.4), so mixing two vaccines in one call
    would silently average them into one row instead of raising.

    For every vaccinated cell: ``sd_visible``/``sd_ciego``/``F`` and its
    bootstrap CI (same machinery as T04/T06), the matched unvaccinated cell
    from ``df_detection`` (``sd_visible_sin_vacuna``, ``F_sin_vacuna``), an F
    test of ``sd_visible`` vaccinated vs. unvaccinated on the SAME pairs
    (``p_diff_varianza``: has the vaccine changed the visible variance?), and
    ``mean_gap`` as the secondary over-correction check the brief asks for.

    Explicitly includes the reference the brief calls out: for every
    composition present, the level_2/UNVACCINATED cell from ``df_detection``
    (F approx 0.9 in homo_llama), flagged ``es_ancla_level2_sin_vacuna=True`` -
    "stabilisation already achieved by role specification", obtained without
    running anything.
    """
    if df_mitigation is None or df_mitigation.empty or "vaccine" not in df_mitigation.columns:
        return _empty(T12_COLS)

    det_visible = _empty(SEED_AVG_COLS)
    det_ciego = _empty(SEED_AVG_COLS)
    if df_detection is not None and not df_detection.empty:
        det_visible = promedio_entre_semillas(gap_por_pareja(df_detection, turn=turn, blind=False))
        det_ciego = promedio_entre_semillas(gap_por_pareja(df_detection, turn=turn, blind=True))

    vaccines = sorted(
        v for v in df_mitigation["vaccine"].dropna().astype(str).unique().tolist()
        if v and v != "none"
    )
    if not vaccines:
        log.warning("build_t12: df_mitigation has no vaccine != 'none' rows.")
        return _empty(T12_COLS)

    rows = []
    anchors_done: set[str] = set()
    for vaccine in vaccines:
        sub = df_mitigation[df_mitigation["vaccine"].astype(str) == vaccine]
        vac_visible = promedio_entre_semillas(gap_por_pareja(sub, turn=turn, blind=False))
        vac_ciego = promedio_entre_semillas(gap_por_pareja(sub, turn=turn, blind=True))
        if vac_visible.empty:
            log.warning("build_t12: no genesis gaps for vaccine=%s", vaccine)
            continue

        for composition in sorted(vac_visible["composition"].dropna().unique().tolist()):
            levels = sorted(
                vac_visible[vac_visible["composition"] == composition]["instruction_level"]
                .dropna().unique().tolist()
            )
            for level in levels:
                pooled_vac = _pooled_blind_variance(vac_ciego, composition, level, attrs)
                pooled_det = (
                    _pooled_blind_variance(det_ciego, composition, level, attrs)
                    if not det_ciego.empty else None
                )
                if pooled_vac is None:
                    continue
                for attr in attrs:
                    vis_vac = _cell_values(vac_visible, composition, level, attr)
                    if vis_vac.empty:
                        continue
                    v = vis_vac.to_numpy(dtype=float)
                    f_stat, _ = _f_test_against_pooled(v, pooled_vac["var"], pooled_vac["df"])
                    ci_lo, ci_hi = _bootstrap_var_ratio_ci(v, pooled_vac["samples"])

                    vis_det = _cell_values(det_visible, composition, level, attr) if not det_visible.empty else pd.Series(dtype=float)
                    f_det, sd_det, p_diff = float("nan"), float("nan"), float("nan")
                    if pooled_det is not None and not vis_det.empty:
                        v_det = vis_det.to_numpy(dtype=float)
                        f_det, _ = _f_test_against_pooled(v_det, pooled_det["var"], pooled_det["df"])
                        sd_det = float(np.std(v_det, ddof=1))
                        common = vis_vac.index.intersection(vis_det.index)
                        if len(common) >= 2:
                            _, p_diff = _f_test_var_ratio(
                                vis_vac.loc[common].to_numpy(dtype=float),
                                vis_det.loc[common].to_numpy(dtype=float),
                            )

                    rows.append({
                        "composition": composition, "instruction_level": level,
                        "sensitive_attr": attr, "vaccine": vaccine,
                        "n_pairs": int(len(vis_vac)),
                        "sd_visible": float(np.std(v, ddof=1)),
                        "sd_ciego": float(np.sqrt(pooled_vac["var"])),
                        "F": f_stat, "ci95_lo_F": ci_lo, "ci95_hi_F": ci_hi,
                        "sd_visible_sin_vacuna": sd_det, "F_sin_vacuna": f_det,
                        "F_ratio_vacuna_vs_sin": (
                            f_stat / f_det if pd.notna(f_det) and f_det > 0 and pd.notna(f_stat) else float("nan")
                        ),
                        "p_diff_varianza": p_diff,
                        "mean_gap": float(vis_vac.mean()),
                        "es_ancla_level2_sin_vacuna": False,
                    })

            if composition not in anchors_done and not det_ciego.empty:
                pooled_anchor = _pooled_blind_variance(det_ciego, composition, T12_ANCHOR_LEVEL, attrs)
                if pooled_anchor is not None:
                    for anchor_attr in attrs:
                        anchor_vis = _cell_values(det_visible, composition, T12_ANCHOR_LEVEL, anchor_attr)
                        if anchor_vis.empty:
                            continue
                        a = anchor_vis.to_numpy(dtype=float)
                        f_anchor, _ = _f_test_against_pooled(a, pooled_anchor["var"], pooled_anchor["df"])
                        rows.append({
                            "composition": composition, "instruction_level": T12_ANCHOR_LEVEL,
                            "sensitive_attr": anchor_attr, "vaccine": "none",
                            "n_pairs": int(len(anchor_vis)),
                            "sd_visible": float(np.std(a, ddof=1)),
                            "sd_ciego": float(np.sqrt(pooled_anchor["var"])),
                            "F": f_anchor, "ci95_lo_F": float("nan"), "ci95_hi_F": float("nan"),
                            "sd_visible_sin_vacuna": float("nan"), "F_sin_vacuna": float("nan"),
                            "F_ratio_vacuna_vs_sin": float("nan"), "p_diff_varianza": float("nan"),
                            "mean_gap": float(anchor_vis.mean()),
                            "es_ancla_level2_sin_vacuna": True,
                        })
                anchors_done.add(composition)

    if not rows:
        return _empty(T12_COLS)
    table = pd.DataFrame(rows).drop_duplicates(
        subset=["composition", "instruction_level", "sensitive_attr", "vaccine"]
    )
    return (
        table[T12_COLS]
        .sort_values(["composition", "instruction_level", "sensitive_attr", "vaccine"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# B.10 - figures
# ---------------------------------------------------------------------------
def build_f01(t04: pd.DataFrame, output_path: Path) -> Path | None:
    """F01 (central) - sd(gap visible) vs sd(gap ciego) per composition, level
    and attribute, star-marked where the cell survives BH (global family)."""
    if t04 is None or t04.empty:
        log.warning("build_f01: T04 is empty, figure skipped.")
        return None
    plt, _ = rex._require_matplotlib()
    compositions = sorted(t04["composition"].dropna().unique().tolist())
    ncols = 3
    nrows = int(np.ceil(len(compositions) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.2 * nrows), squeeze=False)
    for idx, composition in enumerate(compositions):
        ax = axes[idx // ncols][idx % ncols]
        sub = t04[t04["composition"] == composition]
        labels, sd_vis, sd_ciego, sig = [], [], [], []
        for level in ALL_LEVELS:
            for attr in SENSITIVE_ATTRS:
                cell = sub[(sub["instruction_level"] == level) & (sub["sensitive_attr"] == attr)]
                if cell.empty:
                    continue
                row = cell.iloc[0]
                labels.append(f"{LEVEL_SHORT.get(level, level)}\n{attr}")
                sd_vis.append(float(row["sd_visible"]))
                sd_ciego.append(float(row["sd_ciego"]))
                sig.append(bool(row["sig_BH_global"]))
        x = np.arange(len(labels))
        width = 0.38
        ax.bar(x - width / 2, sd_vis, width, label="sd visible", color="#c0392b")
        ax.bar(x + width / 2, sd_ciego, width, label="sd ciego", color="#2c3e50")
        for xi, height, is_sig in zip(x, sd_vis, sig):
            if is_sig:
                ax.text(xi, height * 1.03, "*", ha="center", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(composition, fontsize=10)
        ax.set_ylabel("sd del gap (EUR)")
        ax.legend(fontsize=7)
    for idx in range(len(compositions), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
    fig.suptitle("F01 - sd visible vs sd ciego en genesis, por composicion (* = sig. BH global)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def build_f02(t06: pd.DataFrame, output_path: Path) -> Path | None:
    """F02 - ratio of variances by turn, decaying away from genesis."""
    if t06 is None or t06.empty:
        log.warning("build_f02: T06 is empty, figure skipped.")
        return None
    plt, _ = rex._require_matplotlib()
    compositions = sorted(t06["composition"].dropna().unique().tolist())
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("tab10")
    for i, composition in enumerate(compositions):
        sub = t06[t06["composition"] == composition]
        by_turn = sub.groupby("turn")["F"].mean().sort_index()
        ax.plot(by_turn.index, by_turn.to_numpy(), marker="o", label=composition, color=cmap(i % 10))
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=1, label="F=1 (sin exceso)")
    ax.set_xlabel("Turno (0 = genesis)")
    ax.set_ylabel("F = var(visible)/var(ciego), media sobre niveles y atributos")
    ax.set_title("F02 - decaimiento del exceso de varianza por turno")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def build_f03(t03: pd.DataFrame, output_path: Path) -> Path | None:
    """F03 - forest plot of the 36 genesis means, IC95 and zero line: the
    visual evidence of the null result T03 documents."""
    if t03 is None or t03.empty:
        log.warning("build_f03: T03 is empty, figure skipped.")
        return None
    plt, _ = rex._require_matplotlib()
    table = t03.sort_values(["sensitive_attr", "composition", "instruction_level"]).reset_index(drop=True)
    y = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(8, max(4, 0.28 * len(table))))
    colors = ["#c0392b" if s else "#7f8c8d" for s in table["sig_BH"]]
    # matplotlib's errorbar takes ONE ecolor, not a per-point list, so the
    # whiskers are drawn neutral and the BH significance is carried by the
    # marker colour (scatter) instead.
    ax.errorbar(
        table["mean_gap"], y,
        xerr=[table["mean_gap"] - table["ci95_lo"], table["ci95_hi"] - table["mean_gap"]],
        fmt="none", ecolor="#bdbdbd", elinewidth=1.5, capsize=2, zorder=1,
    )
    ax.scatter(table["mean_gap"], y, c=colors, s=28, zorder=2)
    ax.axvline(0, color="grey", linestyle="--", linewidth=1)
    labels = [
        f"{r['composition']} / {LEVEL_SHORT.get(r['instruction_level'], r['instruction_level'])} / {r['sensitive_attr']}"
        for _, r in table.iterrows()
    ]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("Media del gap (EUR), IC 95% bootstrap")
    ax.set_title("F03 - forest plot de las 36 medias de genesis (resultado nulo)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _icc_bootstrap_ci(matrix: np.ndarray, n_boot: int = 2000, seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    """Percentile bootstrap CI of the ICC, resampling PAIR ROWS. Figure-only:
    T05 itself reports F/p as its uncertainty measure and is not touched."""
    n = matrix.shape[0]
    if n < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    values = [
        res["icc"] for res in (_icc_one_way(matrix[row]) for row in idx)
        if np.isfinite(res["icc"])
    ]
    if not values:
        return float("nan"), float("nan")
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def build_f04(
    df: pd.DataFrame, df_placebo: pd.DataFrame | None, t05: pd.DataFrame, output_path: Path,
) -> Path | None:
    """F04 - ICC per cell with its bootstrap CI, and the placebo reference."""
    if t05 is None or t05.empty:
        log.warning("build_f04: T05 is empty, figure skipped.")
        return None
    plt, _ = rex._require_matplotlib()

    treatment = t05[t05["sensitive_attr"] != "placebo"].copy()
    placebo_rows = t05[t05["sensitive_attr"] == "placebo"]
    if treatment.empty:
        log.warning("build_f04: T05 has no treatment cells, figure skipped.")
        return None

    gaps = gap_por_pareja(df, turn=GENESIS_TURN, blind=False)
    ci_lo, ci_hi = [], []
    for _, row in treatment.iterrows():
        cell = gaps[
            (gaps["composition"] == row["composition"])
            & (gaps["instruction_level"] == row["instruction_level"])
            & (gaps["sensitive_attr"] == row["sensitive_attr"])
        ]
        label = f"{row['composition']}/{row['instruction_level']}/{row['sensitive_attr']}"
        matrix, _ = _seed_matrix(cell, label)
        if matrix is None:
            ci_lo.append(float("nan"))
            ci_hi.append(float("nan"))
            continue
        lo, hi = _icc_bootstrap_ci(matrix)
        ci_lo.append(lo)
        ci_hi.append(hi)
    treatment = treatment.assign(ci95_lo=ci_lo, ci95_hi=ci_hi).sort_values(
        ["composition", "instruction_level", "sensitive_attr"]
    ).reset_index(drop=True)

    y = np.arange(len(treatment))
    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(treatment))))
    xerr_lo = (treatment["icc"] - treatment["ci95_lo"]).clip(lower=0).fillna(0.0)
    xerr_hi = (treatment["ci95_hi"] - treatment["icc"]).clip(lower=0).fillna(0.0)
    colors = ["#c0392b" if s else "#7f8c8d" for s in treatment["sig_BH"]]
    ax.errorbar(
        treatment["icc"], y, xerr=[xerr_lo, xerr_hi],
        fmt="none", ecolor="#bdbdbd", elinewidth=1.5, capsize=2, zorder=1,
    )
    ax.scatter(treatment["icc"], y, c=colors, s=28, zorder=2)
    ax.axvline(0, color="grey", linestyle="--", linewidth=1)
    if not placebo_rows.empty:
        placebo_icc = float(placebo_rows["icc"].mean())
        ax.axvline(placebo_icc, color="#2980b9", linestyle=":", linewidth=1.5, label=f"placebo ICC={placebo_icc:.3f}")
        ax.legend(fontsize=8)
    labels = [
        f"{r['composition']} / {LEVEL_SHORT.get(r['instruction_level'], r['instruction_level'])} / {r['sensitive_attr']}"
        for _, r in treatment.iterrows()
    ]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("ICC entre semillas, IC 95% bootstrap (resampleo de parejas, seed fija)")
    ax.set_title("F04 - ICC entre semillas por celda, con referencia placebo")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def build_f05(df: pd.DataFrame, df_placebo: pd.DataFrame | None, output_dir: Path) -> list[Path]:
    """F05 - gap trajectory with the placebo band, one file per
    (composition, sensitive_attr): ``trajectory_with_placebo`` reused as-is."""
    written: list[Path] = []
    if df is None or df.empty or "composition" not in df.columns:
        return written
    plt, _ = rex._require_matplotlib()
    for composition in sorted(df["composition"].dropna().unique().tolist()):
        sub = df[df["composition"] == composition]
        sub_placebo = None
        if df_placebo is not None and not df_placebo.empty and "composition" in df_placebo.columns:
            candidate = df_placebo[df_placebo["composition"] == composition]
            sub_placebo = candidate if not candidate.empty else None
        for attr in SENSITIVE_ATTRS:
            traj = agg.trajectory_with_placebo(sub, sub_placebo, attr)
            if traj.empty:
                continue
            fig, ax = plt.subplots(figsize=(7, 4.5))
            for level, frame in traj.groupby("instruction_level"):
                frame = frame.sort_values("turn")
                ax.plot(frame["turn"], frame["mean_gap"], marker="o", label=LEVEL_SHORT.get(level, level))
                ax.fill_between(frame["turn"], frame["ci_lo"], frame["ci_hi"], alpha=0.15)
            placebo_frame = traj.dropna(subset=["placebo_mean"]).drop_duplicates("turn").sort_values("turn")
            if not placebo_frame.empty:
                ax.plot(placebo_frame["turn"], placebo_frame["placebo_mean"], color="grey", linestyle="--", label="placebo")
                ax.fill_between(
                    placebo_frame["turn"], placebo_frame["placebo_lo"], placebo_frame["placebo_hi"],
                    color="grey", alpha=0.12,
                )
            ax.axhline(0, color="black", linewidth=0.7)
            ax.set_xlabel("Turno")
            ax.set_ylabel("Gap medio (EUR)")
            ax.set_title(f"F05 - trayectoria del gap, {composition} / {attr}")
            ax.legend(fontsize=7)
            fig.tight_layout()
            path = output_dir / FIGURE_FILENAMES["F05"].format(composition=composition, attr=attr)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=150)
            plt.close(fig)
            written.append(path)
    return written


# ---------------------------------------------------------------------------
# B.3 - MANIFEST.json
# ---------------------------------------------------------------------------
def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_info(repo_root: Path) -> dict:
    def _run(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(
                args, cwd=str(repo_root), stderr=subprocess.DEVNULL, text=True,
            ).strip()
        except Exception:
            return None

    commit = _run(["git", "rev-parse", "HEAD"])
    status = _run(["git", "status", "--porcelain"])
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


def _versions() -> dict:
    import matplotlib
    import scipy

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
    }


def _scope_of(df: pd.DataFrame | None) -> dict:
    """(compositions, instruction_levels, seeds) present in a source frame -
    the "subconjunto de datos" B.3 requires recorded per output file."""
    if df is None or df.empty:
        return {"compositions": [], "instruction_levels": [], "seeds": []}
    return {
        "compositions": (
            sorted(df["composition"].dropna().unique().tolist()) if "composition" in df.columns else []
        ),
        "instruction_levels": (
            sorted(df["instruction_level"].dropna().unique().tolist()) if "instruction_level" in df.columns else []
        ),
        "seeds": (
            sorted(int(s) for s in df["seed"].dropna().unique().tolist()) if "seed" in df.columns else []
        ),
    }


class _Manifest:
    """Accumulates the B.3 MANIFEST.json contents as tables/figures are built.

    ``input_files``/``outputs`` are sorted before being written so the JSON is
    byte-identical across runs over the same data (B.11 determinism) except
    for the ``timestamp`` field, which records the run itself by design.
    """

    def __init__(self, fdr_q: float, repo_root: Path):
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.fdr_q = fdr_q
        self.repo_root = repo_root
        self.input_files: list[dict] = []
        self.outputs: list[dict] = []

    def add_input_files(self, paths: list[Path]) -> None:
        for path in paths:
            self.input_files.append({
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_of(path),
            })

    def add_output(self, filename: str, function: str, **sources: pd.DataFrame | None) -> None:
        self.outputs.append({
            "file": filename,
            "function": function,
            "sources": {name: _scope_of(source) for name, source in sources.items()},
        })

    def write(self, path: Path) -> None:
        payload = {
            "timestamp": self.started_at,
            "git": _git_info(self.repo_root),
            "input_files": sorted(self.input_files, key=lambda item: item["path"]),
            "params": {
                "n_perm": N_PERM,
                "permutation_seed": PERMUTATION_SEED,
                "n_boot": N_BOOT,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "fdr_q": self.fdr_q,
                "endpoint_turn": GENESIS_TURN,
                "exclusion_criteria": (
                    "Subject rows with parse_error==True or action=='ERROR' are "
                    "dropped by default (include_errors=False) before any gap is "
                    "computed; see T01 for the per-composition counts."
                ),
            },
            "outputs": sorted(self.outputs, key=lambda item: item["file"]),
            "versions": _versions(),
        }
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# B.1-B.2 - CLI and deterministic output layout
# ---------------------------------------------------------------------------
def _write_table(table: pd.DataFrame, out_dir: Path, key: str) -> Path:
    path = out_dir / "tables" / TABLE_FILENAMES[key]
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    path.with_suffix(".md").write_text(rex._dataframe_to_markdown(table), encoding="utf-8")
    return path


def _load_stage_csvs(results_dir: Path, prefix: str) -> tuple[pd.DataFrame, list[Path]]:
    paths = sorted(results_dir.glob(f"{prefix}_*.csv"))
    if not paths:
        return _empty([]), []
    frames = [pd.read_csv(path) for path in paths]
    df = pd.concat(frames, ignore_index=True)
    df = rex._coerce_numeric_columns(df)
    df = rex._normalize_instruction_level(df)
    log.info("Loaded %d %s_*.csv file(s), %d rows total.", len(paths), prefix, len(df))
    return df, paths


def _check_known_compositions(df: pd.DataFrame, label: str) -> None:
    """B.11 noisy failure: an unrecognised composition must abort, not silently
    join the pipeline and inflate the 36-row grid to 37+."""
    if df is None or df.empty or "composition" not in df.columns:
        return
    unexpected = sorted(set(df["composition"].dropna().unique().tolist()) - set(KNOWN_COMPOSITIONS))
    if unexpected:
        raise ValueError(
            f"{label}: unexpected composition(s) {unexpected}, not in "
            f"KNOWN_COMPOSITIONS {KNOWN_COMPOSITIONS}. If this is a deliberate "
            f"new run, update KNOWN_COMPOSITIONS in build_memoria_outputs.py "
            f"first - do not let it pass silently."
        )


def _configure_file_log(out_dir: Path) -> None:
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_dir / "build.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    root.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    root.addHandler(stream_handler)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build every deterministic table and figure of the memoria "
            "(Brief v2, Tarea B/C). No interactive state, no implicit filters."
        ),
    )
    parser.add_argument(
        "--results-dir", required=True,
        help="Directory with detection_*.csv / placebo_*.csv / mitigation_*.csv.",
    )
    parser.add_argument("--out-dir", required=True, help="Output directory (created if missing).")
    parser.add_argument(
        "--stage", choices=["detection", "mitigation", "both"], default="detection",
        help="Which stage to build outputs for (default: detection).",
    )
    parser.add_argument("--fdr-q", type=float, default=DEFAULT_FDR_Q, help="BH FDR threshold (default: 0.05).")
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_arg_parser().parse_args(argv)

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    _configure_file_log(out_dir)

    manifest = _Manifest(fdr_q=args.fdr_q, repo_root=Path(__file__).resolve().parents[2])
    input_paths: list[Path] = []
    df_detection, df_placebo, df_mitigation = _empty([]), _empty([]), _empty([])

    if args.stage in ("detection", "both"):
        df_detection, paths_d = _load_stage_csvs(results_dir, "detection")
        if df_detection.empty:
            raise ValueError(f"No detection_*.csv files found in {results_dir}")
        df_placebo, paths_p = _load_stage_csvs(results_dir, "placebo")
        input_paths += paths_d + paths_p
        _check_known_compositions(df_detection, "detection")
        _check_known_compositions(df_placebo, "placebo")

    if args.stage in ("mitigation", "both"):
        df_mitigation, paths_m = _load_stage_csvs(results_dir, "mitigation")
        if df_mitigation.empty:
            raise ValueError(f"No mitigation_*.csv files found in {results_dir}")
        input_paths += paths_m
        _check_known_compositions(df_mitigation, "mitigation")
        if df_detection.empty:
            # T12 needs the unvaccinated comparator and the level_2 anchor even
            # in a mitigation-only run.
            df_detection, paths_d = _load_stage_csvs(results_dir, "detection")
            input_paths += paths_d
            _check_known_compositions(df_detection, "detection")

    manifest.add_input_files(input_paths)

    if args.stage in ("detection", "both"):
        log.info("Building T01 (data quality)...")
        t01 = pd.concat(
            [build_t01(df_detection, stage="detection"), build_t01(df_placebo, stage="placebo")],
            ignore_index=True,
        ) if not df_placebo.empty else build_t01(df_detection, stage="detection")
        _write_table(t01, out_dir, "T01")
        manifest.add_output(TABLE_FILENAMES["T01"], "build_t01", detection=df_detection, placebo=df_placebo)

        log.info("Building T02 (placebo noise floor)...")
        t02 = build_t02(df_placebo)
        _write_table(t02, out_dir, "T02")
        manifest.add_output(TABLE_FILENAMES["T02"], "build_t02", placebo=df_placebo)

        log.info("Building T03 (genesis means, 36 tests, BH)...")
        t03 = build_t03(df_detection, fdr_q=args.fdr_q)
        _write_table(t03, out_dir, "T03")
        manifest.add_output(TABLE_FILENAMES["T03"], "build_t03", detection=df_detection)

        log.info("Building T04 (PRIMARY: genesis variance ratio, 36 tests, BH)...")
        t04 = build_t04(df_detection, fdr_q=args.fdr_q)
        _write_table(t04, out_dir, "T04")
        manifest.add_output(TABLE_FILENAMES["T04"], "build_t04", detection=df_detection)

        log.info("Building T05 (between-seed ICC)...")
        t05 = build_t05(df_detection, df_placebo, fdr_q=args.fdr_q)
        _write_table(t05, out_dir, "T05")
        manifest.add_output(TABLE_FILENAMES["T05"], "build_t05", detection=df_detection, placebo=df_placebo)

        log.info("Building T06 (variance ratio by turn)...")
        t06 = build_t06(df_detection, df_placebo)
        _write_table(t06, out_dir, "T06")
        manifest.add_output(TABLE_FILENAMES["T06"], "build_t06", detection=df_detection, placebo=df_placebo)

        log.info("Building T07 (transmission slopes)...")
        t07 = build_t07(df_detection, df_placebo, fdr_q=args.fdr_q)
        _write_table(t07, out_dir, "T07")
        manifest.add_output(TABLE_FILENAMES["T07"], "build_t07", detection=df_detection, placebo=df_placebo)

        log.info("Building T08 (level contrast)...")
        t08 = build_t08(df_detection, fdr_q=args.fdr_q)
        _write_table(t08, out_dir, "T08")
        manifest.add_output(TABLE_FILENAMES["T08"], "build_t08", detection=df_detection)

        log.info("Building T09 (composition comparison)...")
        t09 = build_t09(t04, t03)
        _write_table(t09, out_dir, "T09")
        manifest.add_output(TABLE_FILENAMES["T09"], "build_t09", detection=df_detection)

        log.info("Building T10 (hetero vs. members)...")
        t10 = build_t10(df_detection)
        _write_table(t10, out_dir, "T10")
        manifest.add_output(TABLE_FILENAMES["T10"], "build_t10", detection=df_detection)

        log.info("Building T11 (verbalization)...")
        t11 = build_t11(df_detection)
        _write_table(t11, out_dir, "T11")
        manifest.add_output(TABLE_FILENAMES["T11"], "build_t11", detection=df_detection)

        log.info("Building T13 (internal null validation)...")
        t13 = build_t13(df_detection)
        _write_table(t13, out_dir, "T13")
        manifest.add_output(TABLE_FILENAMES["T13"], "build_t13", detection=df_detection)

        log.info("Building figures F01-F05...")
        if build_f01(t04, out_dir / "figures" / FIGURE_FILENAMES["F01"]) is not None:
            manifest.add_output(FIGURE_FILENAMES["F01"], "build_f01", detection=df_detection)
        if build_f02(t06, out_dir / "figures" / FIGURE_FILENAMES["F02"]) is not None:
            manifest.add_output(FIGURE_FILENAMES["F02"], "build_f02", detection=df_detection)
        if build_f03(t03, out_dir / "figures" / FIGURE_FILENAMES["F03"]) is not None:
            manifest.add_output(FIGURE_FILENAMES["F03"], "build_f03", detection=df_detection)
        if build_f04(df_detection, df_placebo, t05, out_dir / "figures" / FIGURE_FILENAMES["F04"]) is not None:
            manifest.add_output(FIGURE_FILENAMES["F04"], "build_f04", detection=df_detection, placebo=df_placebo)
        for figure_path in build_f05(df_detection, df_placebo, out_dir / "figures"):
            manifest.add_output(figure_path.name, "build_f05", detection=df_detection, placebo=df_placebo)

    if args.stage in ("mitigation", "both"):
        log.info("Building T12 (mitigation, variance-ratio endpoint)...")
        t12 = build_t12(df_mitigation, df_detection)
        _write_table(t12, out_dir, "T12")
        manifest.add_output(TABLE_FILENAMES["T12"], "build_t12", mitigation=df_mitigation, detection=df_detection)

    manifest.write(out_dir / "MANIFEST.json")
    log.info("Done. Outputs written to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())