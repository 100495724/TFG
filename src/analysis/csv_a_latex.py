#!/usr/bin/env python3
"""Genera fragmentos LaTeX (uno por tabla) a partir de los CSV de resultados.

Uso:
    python csv_a_latex.py --entrada outputs/ --salida memoria/anexos/tablas/

Cada CSV produce un fichero .tex con un entorno longtable listo para \\input.
Las tablas anchas se marcan para envolverlas en landscape.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

# Columnas a conservar en las tablas demasiado anchas para una pagina.
# Si una tabla no aparece aqui, se vuelca completa.
SUBCONJUNTOS = {
    "T01": ["stage", "composition", "n_runs_completed", "n_subject_rows",
            "rate_parse_error", "rate_was_normalized", "rate_descartadas",
            "p_binom_asimetria_parseo"],
    "T04": ["composition", "instruction_level", "sensitive_attr", "n_pairs",
            "sd_visible", "sd_ciego", "F", "ci95_lo_F", "ci95_hi_F",
            "p_fisher", "p_levene", "q_global", "sig_BH_global"],
    "T07": ["composition", "n_treatment", "slope_treatment", "se_treatment",
            "slope_placebo", "se_placebo", "z_diff", "p_diff", "q_value",
            "sig_BH"],
    "T08": ["composition", "sensitive_attr", "level_a", "level_b", "n_pairs",
            "mean_diff", "p_signflip", "q_value", "sig_BH", "spearman_rho",
            "spearman_p", "spearman_n"],
    "T12": ["instruction_level", "sensitive_attr", "vaccine", "n_pairs",
            "sd_visible", "sd_ciego", "F", "ci95_lo_F", "ci95_hi_F",
            "F_ratio_vacuna_vs_sin", "mean_gap"],
}

# Abreviaturas de valores repetidos: reducen mucho el ancho de las tablas.
ABREVIATURAS = {
    "level_0_neutral": "L0",
    "level_1_professional": "L1",
    "level_2_identity": "L2",
    "homo_llama": "h-llama",
    "homo_qwen": "h-qwen",
    "homo_mistral": "h-mistral",
    "homo_gpt": "h-gpt",
    "hetero_local": "het-local",
    "hetero_api_gpt": "het-gpt",
    "detection": "detección",
    "mitigation": "mitigación",
}

TITULOS = {
    "T01": "Calidad de los datos por composición",
    "T02": "Suelo de ruido de la condición placebo",
    "T03": "Diferencias medias en génesis con corrección BH",
    "T04": "Ratios de varianzas en génesis con corrección BH",
    "T05": "Correlación intraclase entre semillas",
    "T06": "Varianza por turno de deliberación",
    "T07": "Pendientes de transmisión con errores robustos",
    "T08": "Contraste entre niveles de instrucción",
    "T09": "Comparación entre composiciones",
    "T10": "Comités heterogéneos frente a sus modelos miembros",
    "T11": "Tasas de verbalización del atributo",
    "T12": "Campaña de mitigación",
    "T13": "Validación del null interno",
}

ANCHO_MAX_VERTICAL = 8  # columnas que caben en pagina vertical

# Cabeceras largas que desbordan la caja de texto.
CABECERAS = {
    "p_binom_asimetria_parseo": "p binom asim.",
    "rate_was_normalized": "tasa normaliz.",
    "rate_descartadas": "tasa descart.",
    "rate_parse_error": "tasa error",
    "n_runs_completed": "n ejec.",
    "n_subject_rows": "n filas suj.",
    "n_pairs_semillas_incompletas": "n semillas inc.",
    "instruction_level": "nivel",
    "sensitive_attr": "atributo",
    "member_compositions": "miembros",
    "hetero_composition": "comité hetero",
    "F_ratio_vacuna_vs_sin": "F vac./sin",
    "sd_visible_sin_vacuna": "sd vis. sin vac.",
    "correlacion_entre_atributos": "corr. atributos",
    "n_parejas_identicas": "n parejas idént.",
    "max_diferencia_abs": "máx. dif. abs.",
    "ciego_identico_entre_atributos": "ciego idéntico",
    "sig_BH_composicion": "sig BH comp.",
    "sd_entre_parejas": "sd entre par.",
    "sd_dentro_pareja": "sd dentro par.",
    "n_pairs_descartadas": "n descart.",
    "composition": "composición",
}


def escapar(texto: str) -> str:
    """Escapa los caracteres que LaTeX interpreta."""
    reemplazos = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    for viejo, nuevo in reemplazos.items():
        texto = texto.replace(viejo, nuevo)
    return texto


def formatear(valor, columna: str) -> str:
    """Convierte un valor a su representacion LaTeX con coma decimal."""
    if valor is None or (isinstance(valor, float) and math.isnan(valor)):
        return "---"
    if isinstance(valor, (bool,)) or str(valor) in ("True", "False"):
        return "Sí" if str(valor) == "True" else "No"
    if isinstance(valor, float):
        # p-valores y q-valores: cuatro decimales o notacion cientifica
        if columna.startswith(("p_", "q_")) or "p_value" in columna:
            if abs(valor) >= 1e-4:
                return f"{valor:.4f}".replace(".", ",")
            return f"{valor:.1e}"
        # magnitudes monetarias: separador de millar, sin decimales
        if abs(valor) >= 1000:
            entero = f"{valor:,.0f}".replace(",", ".")
            return entero
        # resto: dos decimales con coma
        return f"{valor:.2f}".replace(".", ",")
    texto = str(valor)
    for largo, corto in ABREVIATURAS.items():
        texto = texto.replace(largo, corto)
    # Listas separadas por comas: partirlas para que el texto pueda ajustarse
    if texto.count(",") >= 2:
        texto = texto.replace(",", ", ")
    return escapar(texto)


def cabecera_corta(columna: str) -> str:
    """Acorta nombres de columna largos para que quepan."""
    if columna in CABECERAS:
        return r"\textbf{" + escapar(CABECERAS[columna]) + "}"
    return r"\textbf{" + escapar(columna.replace("_", " ")) + "}"


def ancho_estimado(df: pd.DataFrame) -> float:
    """Ancho aproximado en cm, contando cabeceras y datos."""
    total = 0.0
    for col in df.columns:
        celdas = [formatear(v, col) for v in df[col]]
        cabecera = CABECERAS.get(col, col.replace("_", " "))
        n = max([len(c) for c in celdas] + [len(cabecera)])
        total += min(n, 20) * 0.17 + 0.35
    return total


def alineacion(df: pd.DataFrame) -> str:
    """Asigna l, r o p{} a cada columna segun el ancho de su contenido."""
    especificadores = []
    for col in df.columns:
        celdas = [formatear(v, col) for v in df[col]]
        ancho = max([len(c) for c in celdas] + [len(str(col))])
        if df[col].dtype.kind in "if" and not celdas[0].startswith(("L", "h")):
            especificadores.append("r")
        elif ancho > 18:
            especificadores.append("p{3.2cm}")
        else:
            especificadores.append("l")
    return "".join(especificadores)


def tabla_a_latex(df: pd.DataFrame, etiqueta: str, titulo: str) -> str:
    ancho = ancho_estimado(df)
    apaisada = ancho > 15.0  # caja de texto vertical aprox. 16 cm
    align = alineacion(df)

    lineas = []
    if apaisada:
        lineas.append(r"\begin{landscape}")
    lineas.append(r"\footnotesize" if not apaisada else r"\scriptsize")
    lineas.append(r"\begin{longtable}{" + align + "}")
    lineas.append(rf"\caption{{{titulo}.}}\label{{tab:{etiqueta.lower()}}}\\")
    lineas.append(r"\hline")
    lineas.append(" & ".join(cabecera_corta(c) for c in df.columns) + r" \\")
    lineas.append(r"\hline")
    lineas.append(r"\endfirsthead")
    lineas.append(r"\hline")
    lineas.append(" & ".join(cabecera_corta(c) for c in df.columns) + r" \\")
    lineas.append(r"\hline")
    lineas.append(r"\endhead")
    lineas.append(r"\hline")
    lineas.append(r"\endfoot")

    for _, fila in df.iterrows():
        celdas = [formatear(fila[c], c) for c in df.columns]
        lineas.append(" & ".join(celdas) + r" \\")

    lineas.append(r"\end{longtable}")
    if apaisada:
        lineas.append(r"\end{landscape}")
    return "\n".join(lineas) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--entrada", required=True, type=Path)
    p.add_argument("--salida", required=True, type=Path)
    args = p.parse_args()
    args.salida.mkdir(parents=True, exist_ok=True)

    generados = []
    for csv in sorted(args.entrada.glob("T*.csv")):
        codigo = csv.name.split("_")[0]
        df = pd.read_csv(csv)
        if codigo in SUBCONJUNTOS:
            cols = [c for c in SUBCONJUNTOS[codigo] if c in df.columns]
            df = df[cols]
        titulo = TITULOS.get(codigo, codigo)
        destino = args.salida / f"{codigo}.tex"
        destino.write_text(tabla_a_latex(df, codigo, titulo), encoding="utf-8")
        generados.append((codigo, len(df), len(df.columns)))
        print(f"  {destino.name}: {len(df)} filas, {len(df.columns)} columnas")

    print("\n% Pegar en el anexo:")
    for codigo, _, _ in generados:
        print(rf"\input{{anexos/tablas/{codigo}.tex}}")


if __name__ == "__main__":
    main()