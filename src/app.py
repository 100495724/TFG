"""
app.py - Streamlit explorer for the TFG implicit-bias experiment.

Thin UI layer over the analysis library: it orchestrates calls to
``analysis/result_explorer.py`` (per-basket inspection) and
``analysis/aggregate.py`` (cross-basket metrics) and renders the results.
It contains NO computation of its own - every number comes from those modules.

Run with:
    streamlit run src/app.py

The app degrades gracefully: with only one composition / one ablation cell on
disk, the cross-cell views show an "insufficient data" notice instead of
raising.
"""
from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# The analysis modules live in src/analysis/ and there is no package __init__,
# so make that directory importable regardless of the launch cwd.
_SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC_DIR / "analysis"))

import aggregate as agg  # noqa: E402
import result_explorer as rex  # noqa: E402

DEFAULT_RESULTS_DIR = "results"
INSUFFICIENT = "Datos insuficientes para esta vista todavía."

# Structural filters (control + test rows share these, so pre-filtering the
# DataFrame with them is safe for the control-vs-test merges downstream).
STRUCTURAL_COLS = ["composition", "instruction_level", "protocol", "vaccine"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _resolve_dir(results_dir: str) -> Path:
    path = Path(results_dir)
    return path if path.is_absolute() else (_SRC_DIR / path)


def _dir_signature(results_dir: str) -> tuple:
    """Sorted (name, mtime) of result CSVs, so the cache busts when files change."""
    base = _resolve_dir(results_dir)
    if not base.is_dir():
        return ()
    sig = []
    for name in sorted(os.listdir(base)):
        if name.endswith(".csv") and (name.startswith("act1_") or name.startswith("placebo_")):
            sig.append((name, os.path.getmtime(base / name)))
    return tuple(sig)


@st.cache_data(show_spinner=False)
def load_data(results_dir: str, signature: tuple) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load act1 and placebo CSVs from a directory. ``signature`` is only used
    as part of the cache key (file list + mtimes)."""
    base = _resolve_dir(results_dir)

    def _load(pattern: str) -> pd.DataFrame:
        paths = sorted(glob.glob(str(base / pattern)))
        if not paths:
            return pd.DataFrame()
        return rex.load_result_csvs(paths)

    return _load("act1_*.csv"), _load("placebo_*.csv")


# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
def _options(df: pd.DataFrame, column: str) -> list:
    if column not in df.columns:
        return []
    return sorted(str(v) for v in df[column].dropna().unique())


def structural_scope(df: pd.DataFrame, selections: dict) -> pd.DataFrame:
    """Apply only the structural filters (never sensitive_attr/agent/turn)."""
    scope = df
    for col in STRUCTURAL_COLS:
        chosen = selections.get(col)
        if chosen and col in scope.columns:
            scope = scope[scope[col].astype(str).isin(chosen)]
    seeds = selections.get("seed")
    if seeds and "seed" in scope.columns:
        scope = scope[scope["seed"].isin(seeds)]
    return scope


def build_sidebar(df: pd.DataFrame) -> dict:
    st.sidebar.header("Filtros globales")
    sel: dict = {}

    sel["results_dir"] = st.session_state.get("results_dir", DEFAULT_RESULTS_DIR)

    st.sidebar.subheader("Estructurales")
    for col, label in [
        ("composition", "Composición"),
        ("instruction_level", "Nivel de instrucción"),
        ("protocol", "Protocolo"),
        ("vaccine", "Vacuna"),
    ]:
        opts = _options(df, col)
        sel[col] = st.sidebar.multiselect(label, opts, default=opts, key=f"f_{col}")

    seed_opts = rex.available_seeds(df)
    sel["seed"] = st.sidebar.multiselect("Seeds", seed_opts, default=seed_opts, key="f_seed")
    sel["include_errors"] = st.sidebar.toggle("Incluir filas con error", value=False, key="f_errors")

    st.sidebar.subheader("Selección (no filtra cálculos de gap)")
    attr_opts = [a for a in ("gender", "country") if a in _options(df, "sensitive_attr")] or ["gender", "country"]
    sel["sensitive_attr"] = st.sidebar.radio(
        "Atributo sensible", attr_opts, index=0, key="f_attr",
        help="Selecciona qué contrafactual analizar en las vistas agregadas.",
    )
    role_opts = _options(df, "role_key")
    sel["agent"] = st.sidebar.selectbox(
        "Agente (display)", ["(todos)"] + role_opts, index=0, key="f_agent",
    )
    return sel


# ---------------------------------------------------------------------------
# Plot helpers (matplotlib for document-quality figures)
# ---------------------------------------------------------------------------
SEED_CMAP = plt.get_cmap("tab10")


def _seed_colors(seeds) -> dict:
    return {seed: SEED_CMAP(i % 10) for i, seed in enumerate(sorted(seeds))}


def _strip_by_seed(ax, data: pd.DataFrame, value_col: str, title: str) -> None:
    """Horizontal strip plot of a signed value, colored by seed."""
    colors = _seed_colors(data["seed"].unique())
    rng = np.random.default_rng(0)
    for seed, frame in data.groupby("seed"):
        y = rng.uniform(-0.18, 0.18, size=len(frame))
        ax.scatter(
            frame[value_col], y + 0, s=28, alpha=0.75,
            color=colors[seed], label=f"seed {seed}", edgecolors="none",
        )
    mean = data[value_col].mean()
    ax.axvline(0, color="0.4", lw=1)
    ax.axvline(mean, color="crimson", lw=1.5, ls="--", label=f"media {mean:,.0f}")
    ax.set_yticks([])
    ax.set_xlabel("Gap de comité (€)   +penaliza / −favorece")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="best")


def _overlay_placebo_band(ax, band: pd.DataFrame, final_turn: int = 4) -> None:
    if band.empty:
        return
    row = band[band["turn"] == band["turn"].max()] if "turn" in band.columns else band
    if row.empty:
        return
    lo = float(row["lo"].mean())
    hi = float(row["hi"].mean())
    ax.axvspan(lo, hi, color="green", alpha=0.10, label="banda placebo (ruido)")
    ax.legend(fontsize=7, loc="best")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
def tab_detection(df_scope, df_placebo, sel):
    st.subheader("Detección · gap neto de comité (con signo)")
    st.caption("Gap de comité = media de los 3 agentes en el turno final (proxy, no hay voto). "
               "Positivo = subject penalizado en la variante.")
    det = agg.detection_committee_gap(df_scope, include_errors=sel["include_errors"])
    if det.empty:
        st.info(INSUFFICIENT)
    else:
        band = agg.placebo_band(df_placebo, include_errors=sel["include_errors"]) if not df_placebo.empty else pd.DataFrame()
        attrs = [a for a in ("gender", "country") if a in det["sensitive_attr"].unique()]
        fig, axes = plt.subplots(len(attrs), 1, figsize=(8, 2.6 * len(attrs)), squeeze=False)
        for ax, attr in zip(axes[:, 0], attrs):
            sub = det[det["sensitive_attr"] == attr]
            _strip_by_seed(ax, sub, "committee_gap", f"sensitive_attr = {attr}  (n={len(sub)} baskets×seeds)")
            _overlay_placebo_band(ax, band)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        if df_placebo.empty:
            st.caption("Banda placebo no disponible (aún sin datos placebo).")

    st.markdown("---")
    st.subheader("Verbalización (proxy por keywords)")
    st.caption("Tasa de mención explícita de términos de género/país/CEO en el `reasoning` del subject.")
    verb = agg.verbalization_rate(df_scope, include_errors=sel["include_errors"])
    if verb.empty:
        st.info(INSUFFICIENT)
    else:
        pivot = verb.pivot_table(
            index="sensitive_attr", columns="term_category", values="mention_rate", aggfunc="mean",
        )
        fig, ax = plt.subplots(figsize=(7, 3))
        pivot.plot(kind="bar", ax=ax)
        ax.set_ylabel("tasa de mención")
        ax.set_xlabel("sensitive_attr de la fila")
        ax.legend(title="categoría", fontsize=7)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.dataframe(verb)


def tab_propagation(df_scope, sel):
    st.subheader("Propagación · no-ciego (t−1) → ciego (t)")
    st.caption("Correlación de Pearson retardada entre el gap medio de los agentes NO ciegos en t−1 "
               "y el gap del agente ciego (agent_1) en t. Pooled sobre baskets × seeds.")
    by_cond = st.checkbox("Desglosar por condición", value=False, key="prop_bycond")
    out = agg.propagation_correlation(
        df_scope, attr=sel["sensitive_attr"], by_condition=by_cond,
        include_errors=sel["include_errors"],
    )
    scatter, stats = out["scatter"], out["stats"]
    if scatter.empty or stats.empty:
        st.info(INSUFFICIENT)
        return

    if not by_cond:
        row = stats.iloc[0]
        n = int(row["n"])
        c1, c2, c3 = st.columns(3)
        c1.metric("ρ (Pearson)", "—" if pd.isna(row["rho"]) else f"{row['rho']:.3f}")
        c2.metric("p-valor", "—" if pd.isna(row["pvalue"]) else f"{row['pvalue']:.2g}")
        c3.metric("n (pares)", n)
        if n < 3:
            st.info("n < 3: muestra insuficiente para una correlación fiable.")
    else:
        st.dataframe(stats)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    sc = ax.scatter(
        scatter["x_nonblind_prev"], scatter["y_blind"],
        c=scatter["turn"], cmap="viridis", s=30, alpha=0.8,
    )
    if len(scatter) >= 2 and scatter["x_nonblind_prev"].std() > 0:
        m, b = np.polyfit(scatter["x_nonblind_prev"], scatter["y_blind"], 1)
        xs = np.linspace(scatter["x_nonblind_prev"].min(), scatter["x_nonblind_prev"].max(), 50)
        ax.plot(xs, m * xs + b, color="crimson", lw=1.5, label=f"ajuste (pend={m:.2f})")
        ax.legend(fontsize=8)
    ax.axhline(0, color="0.7", lw=0.8)
    ax.axvline(0, color="0.7", lw=0.8)
    ax.set_xlabel("gap medio no-ciego en t−1 (€)")
    ax.set_ylabel("gap ciego (agent_1) en t (€)")
    fig.colorbar(sc, ax=ax, label="turno t")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def tab_composition(df_scope, df_placebo, sel):
    st.subheader("Composición · gap medio por agente/modelo")
    comp = agg.composition_agent_gap(df_scope, include_errors=sel["include_errors"])
    if comp.empty:
        st.info(INSUFFICIENT)
    else:
        sub = comp[comp["sensitive_attr"] == sel["sensitive_attr"]]
        if sub.empty:
            st.info(INSUFFICIENT)
        else:
            fig, ax = plt.subplots(figsize=(8, 3.5))
            for comp_name, frame in sub.groupby("composition"):
                frame = frame.sort_values("role_key")
                ax.bar(
                    [f"{comp_name}\n{r}" for r in frame["role_key"]],
                    frame["mean_gap"], alpha=0.8,
                )
            ax.axhline(0, color="0.4", lw=1)
            ax.set_ylabel(f"gap medio en t_final ({sel['sensitive_attr']}) €")
            ax.tick_params(axis="x", labelsize=7)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        if df_scope["composition"].nunique() < 2:
            st.info("Vista por-modelo entre composiciones: " + INSUFFICIENT
                    + " (solo hay 1 composición en el conjunto filtrado).")

    st.markdown("---")
    st.subheader("Dinámica · trayectoria del gap por turno (seeds separados)")
    dyn = agg.dynamics_turn_trajectory(df_scope, attr=sel["sensitive_attr"], include_errors=sel["include_errors"])
    if dyn.empty:
        st.info(INSUFFICIENT)
        return
    band = agg.placebo_band(df_placebo, include_errors=sel["include_errors"]) if not df_placebo.empty else pd.DataFrame()
    colors = _seed_colors(dyn["seed"].unique())
    comps = sorted(dyn["composition"].unique())
    styles = ["-", "--", ":", "-."]
    fig, ax = plt.subplots(figsize=(8, 4))
    for ci, comp_name in enumerate(comps):
        for seed, frame in dyn[dyn["composition"] == comp_name].groupby("seed"):
            frame = frame.sort_values("turn")
            ax.plot(
                frame["turn"], frame["mean_gap"], marker="o",
                color=colors[seed], ls=styles[ci % len(styles)],
                label=f"{comp_name} · seed {seed}",
            )
    if not band.empty:
        b = band.groupby("turn")[["lo", "hi"]].mean().sort_index()
        ax.fill_between(b.index, b["lo"], b["hi"], color="green", alpha=0.10, label="banda placebo")
    ax.axhline(0, color="0.4", lw=1)
    ax.set_xlabel("turno")
    ax.set_ylabel(f"gap medio ({sel['sensitive_attr']}) €")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _heatmap(ax, matrix: pd.DataFrame, title: str) -> None:
    vals = matrix.to_numpy(dtype=float)
    lim = np.nanmax(np.abs(vals)) if np.isfinite(vals).any() else 1.0
    lim = lim or 1.0
    im = ax.imshow(vals, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, rotation=30, ha="right", fontsize=7)
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels(matrix.index, fontsize=7)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = vals[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:,.0f}", ha="center", va="center", fontsize=7, color="black")
    ax.set_title(title, fontsize=9)
    return im


def tab_ablations(df_scope, sel):
    st.subheader("Ablaciones · heatmap composición × instrucción")
    st.caption("Gap medio de comité por celda (solo protocolo debate). Cooperative va aparte.")
    matrix = agg.ablation_matrix(df_scope, attr=sel["sensitive_attr"], include_errors=sel["include_errors"])
    n_cells = int(np.isfinite(matrix.to_numpy(dtype=float)).sum()) if not matrix.empty else 0
    if n_cells < 2:
        st.info("Se necesitan ≥2 celdas para el heatmap. " + INSUFFICIENT
                + f" (celdas con datos: {n_cells}).")
        if n_cells == 1:
            st.dataframe(matrix)
    else:
        homo = matrix.loc[[i for i in matrix.index if str(i).startswith("homo_")]]
        hetero = matrix.loc[[i for i in matrix.index if str(i).startswith("hetero_")]]
        groups = [("Homogéneas", homo), ("Heterogéneas", hetero)]
        groups = [(t, m) for t, m in groups if not m.empty]
        fig, axes = plt.subplots(len(groups), 1, figsize=(7, 2 + 1.6 * len(matrix)), squeeze=False)
        im = None
        for ax, (title, m) in zip(axes[:, 0], groups):
            im = _heatmap(ax, m, title)
        if im is not None:
            fig.colorbar(im, ax=axes[:, 0].tolist(), label=f"gap medio ({sel['sensitive_attr']}) €")
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")
    st.subheader("Probe cooperativo · debate vs cooperative (#17 vs #25)")
    coop = agg.cooperative_comparison(df_scope, attr=sel["sensitive_attr"], include_errors=sel["include_errors"])
    if coop.empty:
        st.info(INSUFFICIENT)
        return
    protocols = sorted(coop["protocol"].unique())
    if len(protocols) < 2:
        st.caption(f"Solo hay protocolo `{protocols[0] if protocols else '—'}` en los datos filtrados; "
                   "la comparación pareada requiere debate y cooperative.")
    st.dataframe(coop)


def tab_explorer(df_scope, sel):
    st.subheader("Explorador de baskets")
    pairs = rex.available_pairs(df_scope)
    seeds = rex.available_seeds(df_scope)
    if not pairs or not seeds:
        st.info(INSUFFICIENT)
        return
    c1, c2, c3 = st.columns(3)
    pair = c1.selectbox("pair_id", pairs, key="exp_pair")
    seed = c2.selectbox("seed", seeds, key="exp_seed")
    attr = c3.radio("variante", ["none", "gender", "country"], index=0, horizontal=True, key="exp_attr")

    try:
        fig = rex.plot_subject_allocation_evolution(
            df_scope, pair_id=pair, seed=seed, sensitive_attr=attr,
            include_errors=sel["include_errors"], show=False,
        )
        st.pyplot(fig)
        plt.close(fig)
    except (ValueError, KeyError) as exc:
        st.info(f"Sin datos para esta combinación ({exc}).")

    st.markdown("**Decisiones del subject por turno**")
    for field in ("action", "allocation", "reasoning"):
        try:
            table = rex.subject_decision_table(
                df_scope, pair_id=pair, seed=seed, sensitive_attr=attr,
                field=field, include_errors=sel["include_errors"],
            )
            st.caption(field)
            st.dataframe(table)
        except (ValueError, KeyError):
            st.caption(f"{field}: sin datos para esta combinación.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="TFG · Sesgo implícito en MAS", layout="wide")
    st.title("TFG · Explorador de sesgo implícito en sistemas multi-agente")

    results_dir = st.sidebar.text_input(
        "Directorio de resultados", value=st.session_state.get("results_dir", DEFAULT_RESULTS_DIR),
        key="results_dir",
        help="Relativo a src/. Usa 'results_without_dataset_yahoo' para ver los datos de ejemplo.",
    )
    if st.sidebar.button("🔄 Recargar datos"):
        st.cache_data.clear()

    df_act1, df_placebo = load_data(results_dir, _dir_signature(results_dir))

    if df_act1.empty:
        st.warning(
            f"No hay CSV `act1_*.csv` en `{_resolve_dir(results_dir)}`. "
            "Ejecuta el experimento o apunta el directorio a `results_without_dataset_yahoo`."
        )
        st.stop()

    sel = build_sidebar(df_act1)
    df_scope = structural_scope(df_act1, sel)

    st.caption(
        f"Cargado: {len(df_act1):,} filas · {df_act1['composition'].nunique()} composición(es) · "
        f"{df_act1['pair_id'].nunique()} arquetipos · seeds {rex.available_seeds(df_act1)}. "
        f"Filtrado a {len(df_scope):,} filas."
    )

    t1, t2, t3, t4, t5 = st.tabs(
        ["Detección", "Propagación", "Composición y dinámica", "Ablaciones", "Explorador de baskets"]
    )
    with t1:
        tab_detection(df_scope, df_placebo, sel)
    with t2:
        tab_propagation(df_scope, sel)
    with t3:
        tab_composition(df_scope, df_placebo, sel)
    with t4:
        tab_ablations(df_scope, sel)
    with t5:
        tab_explorer(df_scope, sel)


if __name__ == "__main__":
    main()
