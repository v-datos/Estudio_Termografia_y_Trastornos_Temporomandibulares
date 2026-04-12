"""
================================================================================
ANÁLISIS ESTADÍSTICO COMPLEMENTARIO — TERMOGRAFÍA INFRARROJA Y DOLOR OROFACIAL
(Script alterno — incorpora hallazgos del EDA exploratorio)
================================================================================
Diferencias respecto a analisis_termografia.py:
  - Variable térmica: temperatura basal-normalizada (T_regional − T_basal)
    como alternativa al ΔT crudo
  - Covariables: sexo y edad incluidos en modelos logísticos (Modelos C y E)
  - Análisis subgrupo masetero R3: la región con mayor concentración de dolor
  - Correlación con intensidad continua (0-10) a nivel de paciente
  - Tests de asimetría sistemática derecha (población)

Objetivos cubiertos (mismo marco que objetivos_del_estudio_con_hipotesis_estadisticas.pdf):
  a_alt) Asimetría derecha basal poblacional
  b_alt) Correlación temp normalizada ↔ intensidad dolor (Spearman + Bonferroni)
  b2_alt) Correlación intensidad continua nivel paciente
  c_alt) Utilidad diagnóstica: curvas ROC con temp normalizada
  d_alt) Modelos logísticos extendidos (A, C, D, E) con covariables
  e_alt) Subgrupo específico masetero R3
  f_alt) Análisis nivel paciente N=43 (intensidad continua)
  g)     Efectos del sexo y la edad sobre temperatura (confundidores)

Guía rápida de uso:
  1) Archivo de entrada requerido:
       datos_finales_termografia_procesados_todas_fotos.csv
  2) Ejecutar:
       python analisis_termografia_complementario.py
  3) Salidas principales (en resultado_complementario/):
       - resultados_analisis_termografia_alterno.pdf
       - analisis_termografia_alterno.png
       - asimetria_derecha_basal.csv
       - correlaciones_normalizadas.csv
       - correlaciones_intensidad_continua.csv
       - resultados_roc_normalizados.csv
       - resultados_logit_extendido.csv
       - comparacion_modelos_completa.csv
       - analisis_subgrupo_masetero_r3.csv
       - resultados_paciente_n45_alterno.csv
       - efectos_sexo_edad_temperatura.csv

Notas metodológicas:
  - Variable térmica principal: temperatura basal-normalizada (T_regional − T_basal).
    Corrección por diferencias individuales de termorregulación.
  - Spearman: apropiado para variables ordinales (dolor 0-10). P-valores por
    prueba de permutación con barajeo a nivel de bloque de paciente.
    Seed fijo: rng = random.Random(42).
  - Bonferroni: corrección sobre todas las pruebas de correlación conjuntamente.
  - Logística: errores estándar robustos (cov_type='cluster', groups=muestra).
    Pesos de clase balanceados. Modelos A, C, D, E con y sin covariables.
================================================================================
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
from sklearn.metrics import (
    roc_curve, auc, cohen_kappa_score, precision_recall_curve,
    average_precision_score, f1_score, precision_score, recall_score
)
try:
    import statsmodels.api as sm
except ImportError:
    print("\n[ERROR] El módulo 'statsmodels' no está instalado.")
    print("Por favor, instálelo con: pip install statsmodels")
    import sys
    sys.exit(1)
import textwrap
import random
from statsmodels.stats.multitest import multipletests

# Seed para reproducibilidad de permutaciones
random.seed(42)
np.random.seed(42)

warnings.filterwarnings('ignore')

from constantes import PAIN_COL_REGEX

# ── Paleta y estilo ──────────────────────────────────────────────────────────
BG       = '#0F1117'
AX_BG    = '#1A1D27'
TEXT     = '#E8EAF0'
GRID     = '#2C2F3E'
ACCENT   = '#F5C842'
RED      = '#FF5252'
BLUE     = '#4D9FEC'
GREEN    = '#4CAF50'
ORANGE   = '#FF9800'
PURPLE   = '#AB47BC'
CYAN     = '#26C6DA'

REGION_COLORS = {'r1': BLUE, 'r2': ORANGE, 'r3': GREEN, 'r4': PURPLE}
MUSCLE_COLORS = {
    'atm':                   '#26C6DA',
    'esternocleidomastoideo': '#EF5350',
    'masetero':              '#66BB6A',
    'temporal':              '#FFA726',
}


def p_sig_label(p_val: float, threshold: float = 0.05) -> str:
    """Etiqueta de significancia para público general."""
    if pd.isna(p_val):
        return "No evaluable"
    return f"Significativo (p < {threshold:.4f})" if p_val < threshold else "No significativo"


def interpretar_rho(rho: float) -> str:
    """Interpretación simple de magnitud de correlación."""
    if pd.isna(rho):
        return "No evaluable"
    mag = abs(rho)
    if mag < 0.10:
        fuerza = "muy débil"
    elif mag < 0.30:
        fuerza = "débil"
    elif mag < 0.50:
        fuerza = "moderada"
    else:
        fuerza = "fuerte"
    direccion = "positiva" if rho >= 0 else "negativa"
    return f"Correlación {direccion} {fuerza}"


def interpretar_auc(auc_val: float) -> str:
    """Interpretación clínica simple de AUC (ROC o PR)."""
    if pd.isna(auc_val):
        return "No evaluable"
    if auc_val < 0.60:
        return "Capacidad discriminativa baja"
    if auc_val < 0.70:
        return "Capacidad discriminativa aceptable"
    if auc_val < 0.85:
        return "Capacidad discriminativa buena"
    return "Capacidad discriminativa excelente"


def spearman_permutation_test(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    cluster_col: str = 'muestra',
    n_perms: int = 1000,
) -> tuple[float, float]:
    """
    Calcula la correlación de Spearman y un p-valor basado en prueba de
    permutación que respeta la estructura de clúster (paciente).

    La hipótesis nula es que no existe correlación monotónica entre `x_col`
    e `y_col`. En lugar de barajar observaciones individuales (lo que
    ignoraría la dependencia intra-sujeto), se barajan **bloques completos**
    de observaciones a nivel de paciente. Esto es válido porque `paired` se
    construye iterando pacientes y luego PAIN_COLS, por lo que los bloques
    de cada paciente son contiguos y del mismo tamaño.

    Algoritmo de la permutación (bucle activo al final de la función):
      1. Dividir el vector Y en `n_pacientes` bloques de tamaño `block_size`.
      2. Barajar el orden de los bloques aleatoriamente.
      3. Recalcular la correlación de Spearman entre X (fijo) e Y permutado.
      4. Contar cuántas permutaciones producen |ρ| ≥ |ρ_observado|.
      5. p = (count_extreme + 1) / (n_perms + 1)  [corrección +1 para evitar p=0]

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame con las observaciones pareadas. Debe estar ordenado de
        forma que las filas de cada paciente sean contiguas (tal como lo
        produce el bucle de construcción de `paired`).
    x_col : str
        Nombre de la columna predictora (e.g., 'delta_t', 'temp_max').
    y_col : str
        Nombre de la columna de respuesta (e.g., 'intensidad').
    cluster_col : str, optional
        Columna que identifica el clúster/paciente. Se usa sólo para contar
        el número de clústeres y derivar el tamaño de bloque. Por defecto
        'muestra'.
    n_perms : int, optional
        Número de permutaciones. Por defecto 1000. Aumentar a 5000+ para
        p-valores más precisos en muestras pequeñas.

    Returns
    -------
    rho_obs : float
        Correlación de Spearman observada. NaN si no es calculable.
    p_perm : float
        P-valor de la prueba de permutación (dos colas). NaN si no es
        calculable. Siempre ≥ 1/(n_perms+1) por la corrección +1.

    Notes
    -----
    Los bloques por clúster se construyen una vez antes del bucle de
    permutaciones agrupando filas según `cluster_col`. Esto es correcto
    incluso cuando los pacientes no aportan el mismo número de observaciones
    (bloques de tamaño variable son concatenados y alineados con `x_values`).
    """
    if len(data) < 2:
        return np.nan, np.nan

    # 1. Correlación observada
    rho_obs, _ = stats.spearmanr(data[x_col], data[y_col])
    if pd.isna(rho_obs):
        return np.nan, np.nan

    # 2. Estructura de clúster: agrupar valores Y por paciente preservando el
    # orden de aparición (no el orden de unique(), que puede variar).
    seen = {}
    for c in data[cluster_col]:
        if c not in seen:
            seen[c] = None
    ordered_clusters = list(seen.keys())

    y_blocks = [data.loc[data[cluster_col] == c, y_col].values for c in ordered_clusters]

    # 3. Barajeo de bloques por paciente — respeta dependencia intra-sujeto.
    count_extreme = 0
    x_values = data[x_col].values

    for _ in range(n_perms):
        shuffled = y_blocks.copy()
        random.shuffle(shuffled)
        y_perm = np.concatenate(shuffled)
        r_p, _ = stats.spearmanr(x_values, y_perm)
        if abs(r_p) >= abs(rho_obs):
            count_extreme += 1

    return rho_obs, (count_extreme + 1) / (n_perms + 1)


def _draw_text_page(pdf, title: str, sections: list, fig_size: tuple = (8.27, 11.69)) -> None:
    """
    Dibuja una página de texto estructurada en el reporte PDF (formato A4).

    Gestiona el salto de página automáticamente cuando el contenido supera
    el margen inferior de la página.

    Parameters
    ----------
    pdf : matplotlib.backends.backend_pdf.PdfPages
        Objeto PdfPages abierto al que se añaden las páginas.
    title : str
        Título principal de la página (tamaño 16, negrita).
    sections : list of tuple[str, list[str]]
        Lista de tuplas ``(título_sección, líneas_cuerpo)``.
        Cada elemento de ``líneas_cuerpo`` es un string que se envuelve
        automáticamente a 108 caracteres por línea.
    fig_size : tuple[float, float], optional
        Dimensiones de la página en pulgadas (ancho, alto). Por defecto A4:
        (8.27, 11.69). Cambiar solo si se genera en formato distinto.
    """
    fig = plt.figure(figsize=fig_size)
    fig.patch.set_facecolor('white')
    ax = fig.add_axes([0.06, 0.06, 0.88, 0.88])
    ax.axis('off')

    y = 0.98
    ax.text(0.0, y, title, fontsize=16, fontweight='bold', va='top', color='#111111')
    y -= 0.05

    for sec_title, body_lines in sections:
        ax.text(0.0, y, sec_title, fontsize=12, fontweight='bold', va='top', color='#222222')
        y -= 0.028
        for line in body_lines:
            wrapped = textwrap.wrap(str(line), width=108) or [""]
            for wline in wrapped:
                ax.text(0.01, y, wline, fontsize=10.5, va='top', color='#222222')
                y -= 0.020
                if y < 0.08:
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)
                    fig = plt.figure(figsize=fig_size)
                    fig.patch.set_facecolor('white')
                    ax = fig.add_axes([0.06, 0.06, 0.88, 0.88])
                    ax.axis('off')
                    y = 0.98
        y -= 0.016
        if y < 0.10:
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            fig = plt.figure(figsize=fig_size)
            fig.patch.set_facecolor('white')
            ax = fig.add_axes([0.06, 0.06, 0.88, 0.88])
            ax.axis('off')
            y = 0.98

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def generar_reporte_pdf_general(path_pdf: str, resumen: dict, fig_path: str | None = None) -> None:
    """
    Genera un reporte PDF legible para público general (versión alterno).

    Incluye:
    - Resumen de muestra y objetivo del análisis.
    - Resultados con temperatura basal-normalizada.
    - Análisis de covariables (sexo, edad).
    - Subgrupo masetero R3.
    - Glosario rápido para entender p, rho, AUC, OR.
    - Figura resumen si está disponible.

    Parameters
    ----------
    path_pdf : str
        Ruta de destino del PDF generado.
    resumen : dict
        Diccionario con los resultados del análisis.
    fig_path : str or None, optional
        Ruta a una imagen PNG que se incluye como última página del PDF.
    """
    with PdfPages(path_pdf) as pdf:
        _draw_text_page(
            pdf,
            "Reporte Alterno: Termografía y Dolor Orofacial (Temperatura Normalizada)",
            [
                ("Contexto del estudio", [
                    "Este reporte complementario incorpora hallazgos del EDA exploratorio. La variable térmica "
                    "principal es la temperatura basal-normalizada (T_regional - T_basal) que corrige por "
                    "diferencias individuales en termorregulación.",
                    f"Pacientes analizados: {resumen.get('n_total', 'N/A')} "
                    f"(Con Dolor={resumen.get('n_con_dolor', 'N/A')}, Sin Dolor={resumen.get('n_sin_dolor_p', 'N/A')}).",
                    f"Observaciones pareadas: {resumen.get('n_paired', 'N/A')}.",
                ]),
                ("Diferencias respecto al análisis original", [
                    "1) Variable térmica: T_norm = T_regional - T_basal (temperatura normalizada por individuo).",
                    "2) Covariables: sexo y edad incluidos en Modelos C y E.",
                    "3) Análisis subgrupo masetero R3: región con mayor concentración de dolor.",
                    "4) Correlación con intensidad continua (0-10) a nivel de paciente.",
                    "5) Tests de asimetría sistemática derecha (hipótesis a_alt).",
                ]),
            ]
        )

        _draw_text_page(
            pdf,
            "Resultados: Temperatura Normalizada y Asimetría",
            [
                ("Obj. a_alt — Asimetría derecha basal poblacional", [
                    resumen.get('asimetria_text', 'No evaluado.'),
                ]),
                ("Obj. b_alt — Correlación temp normalizada ↔ dolor", [
                    f"Resultado global: rho={resumen.get('rho_norm_g', float('nan')):+.3f}, "
                    f"p={resumen.get('p_norm_g', float('nan')):.4f}.",
                    f"Interpretación: {interpretar_rho(resumen.get('rho_norm_g', float('nan')))}. "
                    f"{p_sig_label(resumen.get('p_norm_g', float('nan')))}.",
                    "Comparado con ΔT crudo del análisis original.",
                ]),
                ("Obj. b2_alt — Correlación intensidad continua (N=43)", [
                    resumen.get('corr_intensidad_text', 'No evaluado.'),
                ]),
                ("Obj. c_alt — ROC con temperatura normalizada", [
                    resumen.get('roc_norm_text', 'No evaluado.'),
                ]),
            ]
        )

        _draw_text_page(
            pdf,
            "Resultados: Modelos Logísticos Extendidos y Subgrupos",
            [
                ("Obj. d_alt — Modelos logísticos (A, C, D, E)", [
                    resumen.get('modelo_comp_text', 'No evaluado.'),
                ]),
                ("Obj. e_alt — Subgrupo Masetero R3", [
                    resumen.get('masetero_r3_text', 'No evaluado.'),
                ]),
                ("Obj. f_alt — Nivel paciente N=43 (intensidad continua)", [
                    resumen.get('paciente_alt_text', 'No evaluado.'),
                ]),
                ("Obj. g — Efectos sexo y edad sobre temperatura", [
                    resumen.get('confounders_text', 'No evaluado.'),
                ]),
            ]
        )

        _draw_text_page(
            pdf,
            "Guía rápida para interpretar los números",
            [
                ("Qué significa p-valor", [
                    "p < 0.05: resultado estadísticamente significativo.",
                    "p >= 0.05: no hay evidencia suficiente para descartar que el resultado sea por azar.",
                ]),
                ("Qué significa rho (Spearman)", [
                    "Mide relación monotónica entre dos variables.",
                    "Cercano a 0: relación débil; cercano a +1 o -1: relación más fuerte.",
                ]),
                ("Qué significa AUC (ROC)", [
                    "0.5: similar al azar. 0.6–0.7: limitada. 0.7–0.8: aceptable. >0.8: buena o excelente.",
                ]),
                ("Qué significa OR (odds ratio)", [
                    "OR > 1: mayor probabilidad del evento (aquí, presencia de dolor).",
                    "OR < 1: menor probabilidad del evento.",
                ]),
                ("Nota metodológica", [
                    "Este reporte resume asociaciones estadísticas y rendimiento diagnóstico.",
                    "No sustituye juicio clínico ni evaluación médica integral.",
                ]),
            ]
        )

        if fig_path:
            fig = plt.figure(figsize=(11.69, 8.27))
            fig.patch.set_facecolor('white')
            ax = fig.add_axes([0.03, 0.03, 0.94, 0.94])
            ax.axis('off')
            try:
                img = plt.imread(fig_path)
                ax.imshow(img)
                ax.set_title("Figura resumen del análisis estadístico (Alterno)", fontsize=14, pad=10)
                pdf.savefig(fig, bbox_inches='tight')
            finally:
                plt.close(fig)


def style_ax(ax, title=''):
    """
    Aplica formato visual homogéneo a ejes de la figura final.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Eje a estilizar.
    title : str
        Título opcional del panel.
    """
    ax.set_facecolor(AX_BG)
    ax.tick_params(colors=TEXT, labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
    ax.yaxis.label.set_color(TEXT)
    ax.xaxis.label.set_color(TEXT)
    ax.grid(color=GRID, linewidth=0.4, alpha=0.7)
    if title:
        ax.set_title(title, color=TEXT, fontsize=9, fontweight='bold', pad=7)


def es_columna_dolor_anatomico(col: str) -> bool:
    """
    Valida si una columna corresponde a intensidad de dolor siguiendo el
    patrón anatómico robusto: {musculo}_{lado}_r{N}p{M}
    """
    if not isinstance(col, str):
        return False
    return bool(PAIN_COL_REGEX.match(col.strip().lower()))


def seleccionar_columnas_dolor(df_in: pd.DataFrame) -> list[str]:
    """
    Identifica y selecciona todas las columnas de dolor presentes en el
    DataFrame basándose en el patrón anatómico definido.
    """
    return [c for c in df_in.columns if es_columna_dolor_anatomico(c)]


def parse_pain_col(col: str):
    """
    Parsea una columna de dolor y extrae metadatos anatómicos.

    Ejemplo
    -------
    'masetero_derecho_r3p2'
      -> musculo='masetero', lado='derecha', region='r3', punto='p2'

    Returns
    -------
    tuple[str, str, str, str] | None
        (musculo, lado, region, punto) o None si la columna no cumple patrón.
    """
    m = re.search(r'(r\d+)p(\d+)', col)
    if not m:
        return None
    region = m.group(1)
    punto  = f"p{m.group(2)}"
    parts  = col.split('_')
    musculo = parts[0]

    # Aceptar variantes gramaticales: derecho/derecha e izquierdo/izquierda.
    lado_match = re.search(r'(derech[oa]|izquierd[oa])', col)
    if not lado_match:
        return None
    lado_raw = lado_match.group(1)
    lado = 'derecha' if lado_raw.startswith('derech') else 'izquierda'
    return musculo, lado, region, punto


def temp_col_name(region: str, lado: str) -> str:
    """
    Construye el nombre de la columna de temperatura máxima regional del CSV.

    Parameters
    ----------
    region : str   — Código de región (e.g., 'r1', 'r2', 'r3', 'r4').
    lado   : str   — 'derecha' o 'izquierda'.

    Returns
    -------
    str  — Nombre de columna tal como aparece en el CSV procesado.
           Ejemplo: 'r1: temperatura maxima derecha_primera'
    """
    return f"{region}: temperatura maxima {lado}_primera"


def temp_col_name_segunda(region: str, lado: str) -> str:
    """
    Construye el nombre de la columna de temperatura máxima regional (Foto 2).

    Parameters
    ----------
    region : str   — Código de región (e.g., 'r1').
    lado   : str   — 'derecha' o 'izquierda'.

    Returns
    -------
    str  — Nombre de columna (e.g., 'r1: temperatura maxima derecha_segunda').
    """
    return f"{region}: temperatura maxima {lado}_segunda"


def delta_col_name(region: str) -> str:
    """
    Construye el nombre de la columna de asimetría térmica (ΔT) regional.

    Parameters
    ----------
    region : str   — Código de región (e.g., 'r1').

    Returns
    -------
    str  — Nombre de columna. Ejemplo: 'delta_t_r1_primera'
    """
    return f"delta_t_{region.lower()}_primera"


def temp_punto_col_name(col_dolor: str) -> str:
    """
    Devuelve el nombre de la columna de temperatura puntual mapeada.

    Parameters
    ----------
    col_dolor : str   — Nombre de columna de dolor.

    Returns
    -------
    str  — Nombre de columna térmica puntual.
    """
    return f"temp_punto__{col_dolor}"


def ajustar_logit_cluster(
    data: pd.DataFrame,
    incluir_temp_punto: bool = False,
    col_temp_max: str = 'temp_max',
    covariates: list[str] = None
) -> dict:
    """
    Ajusta un modelo de regresión logística con errores estándar robustos
    por clúster de paciente (sandwich estimator).

    Fórmula del modelo (Modelo A):
        tiene_dolor ~ const + temp_max + region (dummies) + lado (dummies)

    Si `incluir_temp_punto=True` (Modelo B):
        tiene_dolor ~ const + temp_max + temp_punto + region + lado

    Si `covariates` es una lista (Modelo C: sexo+edad; Modelo D: temp_norm):
        tiene_dolor ~ const + temp_max + region + lado + [covariates]

    Modelo C (con sexo/edad): covariates=['sexo_num', 'edad']
    Modelo D (temp basal-normalizada): col_temp_max='temp_norm', covariates=None
    Modelo E: col_temp_max='temp_norm', covariates=['sexo_num', 'edad']

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame con las observaciones pareadas.
    incluir_temp_punto : bool, optional
        Si True, incluye temperatura puntual (Modelo B). Por defecto False.
    col_temp_max : str, optional
        Nombre de la columna de temperatura predictora. Por defecto 'temp_max'.
        Usar 'temp_norm' para Modelos D y E.
    covariates : list[str] or None, optional
        Lista de columnas adicionales a incluir en el modelo (e.g., ['sexo_num', 'edad']).
        Si None, no se añaden covariables (Modelos A y D).
        Modelo C: ['sexo_num', 'edad'] con col_temp_max='temp_max'.
        Modelo E: ['sexo_num', 'edad'] con col_temp_max='temp_norm'.

    Returns
    -------
    dict con claves:
        'ok'      : bool   — True si el modelo convergió sin excepciones.
        'rows'    : list[dict] — Términos del modelo con coef, SE, p, OR, IC95%.
        'metrics' : dict   — AUC-ROC, AUC-PR, F1, Brier, Pseudo-R², AIC, BIC.
        'model'   : objeto statsmodels (para LRT).
        'error'   : str | None — Mensaje de error si 'ok' es False.
    """
    out = {'ok': False, 'rows': [], 'metrics': {}, 'model': None, 'error': None}

    if data.empty:
        out['error'] = 'Sin datos para ajustar.'
        return out

    d = data.copy()
    base_cols = [col_temp_max, 'region', 'lado']
    if incluir_temp_punto:
        base_cols.insert(1, 'temp_punto')

    # Add covariates before dummies
    extra_cols = []
    if covariates is not None:
        for cov in covariates:
            if cov in d.columns:
                extra_cols.append(cov)

    x_in = d[base_cols]
    X = pd.get_dummies(x_in, columns=['region', 'lado'], drop_first=True, dtype=float)

    # Append covariates after dummies
    for cov in extra_cols:
        X[cov] = d[cov].values

    X = sm.add_constant(X, has_constant='add')
    y = d['tiene_dolor'].astype(float)
    clusters = d['muestra']

    # Drop rows with NaN in any predictor or response
    valid_mask = X.notna().all(axis=1) & y.notna()
    X = X[valid_mask]
    y = y[valid_mask]
    clusters = clusters[valid_mask]

    # Validación: se requieren ambas clases
    if y.nunique() < 2:
        out['error'] = (
            f"Variable respuesta con una sola clase presente ({y.unique()}). "
            f"Se requieren observaciones con dolor (1) y sin dolor (0)."
        )
        return out

    # Pesos de clase (equivalente a class_weight='balanced')
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    weight_pos = (n_neg + n_pos) / (2.0 * n_pos)
    weight_neg = (n_neg + n_pos) / (2.0 * n_neg)
    weights = y.apply(lambda val: weight_pos if val == 1 else weight_neg)

    try:
        model = sm.Logit(y, X).fit(
            disp=0,
            cov_type='cluster',
            cov_kwds={'groups': clusters},
            weights=weights
        )
        conf = model.conf_int()
        conf.columns = ['ci_low', 'ci_high']
        params = model.params
        pvals  = model.pvalues
        ses    = model.bse

        rows = []
        for term in params.index:
            rows.append({
                'termino'             : term,
                'coef_logit'          : params[term],
                'se_robusto_cluster'  : ses[term],
                'p_valor'             : pvals[term],
                'odds_ratio'          : np.exp(params[term]),
                'or_ci95_low'         : np.exp(conf.loc[term, 'ci_low']),
                'or_ci95_high'        : np.exp(conf.loc[term, 'ci_high']),
                'significativo_0_05'  : 'Sí' if pvals[term] < 0.05 else 'No',
            })

        y_prob   = model.predict(X)
        fpr_l, tpr_l, _ = roc_curve(y, y_prob)
        auc_logit  = auc(fpr_l, tpr_l)
        avg_prec   = average_precision_score(y, y_prob)
        y_pred     = (y_prob >= 0.5).astype(int)
        f1_val     = f1_score(y, y_pred)
        prec_val   = precision_score(y, y_pred, zero_division=0)
        rec_val    = recall_score(y, y_pred, zero_division=0)
        brier      = float(np.mean((y_prob - y) ** 2))

        out['ok']    = True
        out['rows']  = rows
        out['model'] = model
        out['metrics'] = {
            'n_obs'               : int(model.nobs),
            'pseudo_r2_mcfadden'  : float(model.prsquared),
            'auc_prob_logit'      : float(auc_logit),
            'pr_auc_logit'        : float(avg_prec),
            'f1_score'            : float(f1_val),
            'precision'           : float(prec_val),
            'recall'              : float(rec_val),
            'brier_score'         : brier,
            'llf'                 : float(model.llf),
            'llnull'              : float(model.llnull),
            'aic'                 : float(model.aic),
            'bic'                 : float(model.bic),
            'incluye_temp_punto'  : 'Sí' if incluir_temp_punto else 'No',
        }
    except Exception as e:
        out['error'] = str(e)

    return out


# ════════════════════════════════════════════════════════════════════════════
# 0. CONFIGURACIÓN Y CARGA DE DATOS
# ════════════════════════════════════════════════════════════════════════════
CSV_ENTRADA = 'datos/datos_finales_termografia_procesados_todas_fotos.csv'

OUTPUT_DIR = 'resultado_complementario'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

PNG_RESULTADOS = os.path.join(OUTPUT_DIR, 'analisis_termografia_alterno_complementario.png')
PDF_REPORTE    = os.path.join(OUTPUT_DIR, 'resultados_analisis_termografia_alterno_complementario.pdf')

df = pd.read_csv(CSV_ENTRADA)
df.columns = df.columns.str.strip().str.lower()

# Validación de columnas delta requeridas (primeras fotos)
_delta_cols_requeridas = [f'delta_t_r{r}_primera' for r in [1, 2, 3, 4]]
_delta_faltantes = [c for c in _delta_cols_requeridas if c not in df.columns]
if _delta_faltantes:
    print(f"\n[ERROR] Columnas delta_t de primeras fotos no encontradas: {_delta_faltantes}")
    print(f"Columnas disponibles con 'delta': {[c for c in df.columns if 'delta' in c]}")
    import sys; sys.exit(1)

# Validación de columna de identificador de paciente
ID_COL = 'numero de muestra'
if ID_COL not in df.columns:
    print(f"\n[ERROR] Columna identificadora '{ID_COL}' no encontrada en el CSV.")
    import sys; sys.exit(1)

PAIN_COLS  = seleccionar_columnas_dolor(df)
TEMP_COLS  = [c for c in df.columns if 'temperatura maxima' in c and 'imagen' not in c]
TEMP_PUNTO_COLS = [c for c in df.columns if str(c).startswith('temp_punto__')]

# NaN en dolor → 0 (ausencia de dolor)
df[PAIN_COLS] = df[PAIN_COLS].fillna(0)

# ── Grupo principal ────────────────────────────────────────────────────────
_dolor_por_paciente = df.set_index(ID_COL)[PAIN_COLS].max(axis=1) > 0
_pacientes_con_dolor = set(_dolor_por_paciente[_dolor_por_paciente].index)
df['grupo'] = np.where(
    df[ID_COL].isin(_pacientes_con_dolor),
    'Con Dolor', 'Sin Dolor'
)

n_con_dolor = (df['grupo'] == 'Con Dolor').sum()
n_sin_dolor = (df['grupo'] == 'Sin Dolor').sum()
n_ttm = df['diagnosticado con ttm'].astype(str).str.strip().str.lower().isin(['si','sí','yes','1']).sum()
n_control = len(df) - n_ttm

print("=" * 72)
print("  TERMOGRAFÍA IR — ANÁLISIS COMPLEMENTARIO (TEMPERATURA NORMALIZADA)")
print("=" * 72)
print(f"  Pacientes totales     : {len(df)}")
print(f"  Pacientes con dolor   : {n_con_dolor}")
print(f"  Pacientes sin dolor   : {n_sin_dolor}")
print(f"  [Ref. diagnóstico TTM]: {n_ttm} TTM | {n_control} Control")
print(f"  Columnas de dolor detectadas: {len(PAIN_COLS)}")

# ════════════════════════════════════════════════════════════════════════════
# 1. VARIABLES DERIVADAS (temperatura normalizada + categorías de dolor)
# ════════════════════════════════════════════════════════════════════════════
print("\n  Calculando variables derivadas (temperatura basal-normalizada)...")

# Temperatura basal-normalizada: T_regional - T_basal (per patient)
for region in ['r1','r2','r3','r4']:
    for lado in ['derecha','izquierda']:
        col_med = f"{region}: temperatura media {lado}_primera"
        col_norm = f"temp_norm_{region}_{lado}"
        if col_med in df.columns and 'temperatura basal_primera' in df.columns:
            df[col_norm] = df[col_med] - df['temperatura basal_primera']

# Normalized ΔT = normalized_derecha - normalized_izquierda
for region in ['r1','r2','r3','r4']:
    col_d = f"temp_norm_{region}_derecha"
    col_i = f"temp_norm_{region}_izquierda"
    col_nd = f"delta_t_norm_{region}"
    if col_d in df.columns and col_i in df.columns:
        df[col_nd] = df[col_d] - df[col_i]

# Sex as numeric: 0=Masculino, 1=Femenino
if 'sexo' in df.columns:
    df['sexo_num'] = (df['sexo'].str.strip().str.lower() == 'femenino').astype(float)
else:
    df['sexo_num'] = np.nan

# Pain intensity category (0=none, 1=mild 1-4, 2=moderate+ 5+) — patient level
df['intensidad_max_paciente'] = df[PAIN_COLS].max(axis=1)
df['intensidad_total_paciente'] = df[PAIN_COLS].sum(axis=1)
df['n_puntos_dolor'] = (df[PAIN_COLS] > 0).sum(axis=1)
df['categoria_dolor'] = pd.cut(
    df['intensidad_max_paciente'],
    bins=[-1, 0, 4, 10],
    labels=['Sin Dolor', 'Leve (1-4)', 'Moderado-Severo (5+)']
)

print(f"  Columnas temp_norm creadas: {len([c for c in df.columns if c.startswith('temp_norm_')])}")
print(f"  Columnas delta_t_norm creadas: {len([c for c in df.columns if c.startswith('delta_t_norm_')])}")

# ════════════════════════════════════════════════════════════════════════════
# 2. CONSTRUCCIÓN DEL DATASET PAREADO (con variables adicionales)
# ════════════════════════════════════════════════════════════════════════════
records = []
for _, row in df.iterrows():
    for col in PAIN_COLS:
        parsed = parse_pain_col(col)
        if parsed is None:
            continue
        musculo, lado, region, punto = parsed

        col_temp = temp_col_name(region, lado)
        if col_temp not in df.columns:
            continue

        col_temp_segunda = temp_col_name_segunda(region, lado)
        temp_segunda_val = row[col_temp_segunda] if col_temp_segunda in df.columns else np.nan

        col_temp_punto = temp_punto_col_name(col)
        temp_punto_val = row[col_temp_punto] if col_temp_punto in df.columns else np.nan

        col_delta = delta_col_name(region)
        if col_delta not in df.columns:
            continue

        temp_val  = row[col_temp]
        delta_val = row[col_delta]
        dolor_val = float(row[col])

        if pd.isna(temp_val) or pd.isna(delta_val):
            continue

        # Normalized temperature for this region/side
        col_norm = f"temp_norm_{region}_{lado}"
        temp_norm_val = row[col_norm] if col_norm in df.columns else np.nan

        # Normalized delta_t for this region
        col_delta_norm = f"delta_t_norm_{region}"
        delta_norm_val = row[col_delta_norm] if col_delta_norm in df.columns else np.nan

        # Pain intensity category for this patient
        intensidad_cat_val = str(row['categoria_dolor']) if 'categoria_dolor' in df.columns else 'unknown'

        records.append({
            'muestra'        : row['numero de muestra'],
            'grupo'          : row['grupo'],
            'musculo'        : musculo,
            'lado'           : lado,
            'region'         : region,
            'punto'          : punto,
            'temp_max'       : float(temp_val),
            'temp_max_segunda': float(temp_segunda_val),
            'delta_t'        : float(delta_val),
            'intensidad'     : dolor_val,
            'tiene_dolor'    : 1 if dolor_val > 0 else 0,
            'temp_punto'     : float(temp_punto_val),
            'temp_norm'      : float(temp_norm_val) if pd.notna(temp_norm_val) else np.nan,
            'delta_t_norm'   : float(delta_norm_val) if pd.notna(delta_norm_val) else np.nan,
            'edad'           : float(row['edad']) if 'edad' in df.columns and pd.notna(row['edad']) else np.nan,
            'sexo_num'       : float(row['sexo_num']) if 'sexo_num' in df.columns and pd.notna(row['sexo_num']) else np.nan,
            'intensidad_cat' : intensidad_cat_val,
        })

paired = pd.DataFrame(records)
print(f"\n  Observaciones pareadas : {len(paired)}")
print(f"  Puntos CON dolor       : {paired['tiene_dolor'].sum()}")
print(f"  Puntos SIN dolor       : {(paired['tiene_dolor']==0).sum()}\n")


# ── Helper: calc_roc ──────────────────────────────────────────────────────
def calc_roc(data: pd.DataFrame, label: str = '', score_col: str = 'delta_t') -> dict | None:
    """
    Calcula curva ROC, curva Precision-Recall y umbral óptimo T*.

    Parameters
    ----------
    data : pd.DataFrame
        Debe contener 'tiene_dolor' y `score_col`.
    label : str
        Etiqueta del subgrupo.
    score_col : str
        Columna numérica usada como score predictor.

    Returns
    -------
    dict | None
    """
    y = data['tiene_dolor'].values
    s = data[score_col].values
    mask = ~np.isnan(s)
    y = y[mask]
    s = s[mask]
    if len(y) == 0:
        return None
    if y.sum() == 0 or y.sum() == len(y):
        return None

    fpr, tpr, thresholds = roc_curve(y, s)
    thresholds = np.clip(thresholds, None, 1e6)
    roc_auc = auc(fpr, tpr)

    prec_pts, rec_pts, _ = precision_recall_curve(y, s)
    pr_auc = auc(rec_pts, prec_pts)

    J = tpr - fpr
    J_search = J.copy()
    if len(J_search) > 1:
        J_search[0] = -np.inf
    best = int(np.argmax(J_search))
    t_star = thresholds[best] if J[best] > 0 else np.nan

    return {
        'label'      : label,
        'auc'        : roc_auc,
        'pr_auc'     : pr_auc,
        'T_star'     : t_star,
        'sensitivity': tpr[best],
        'specificity': 1 - fpr[best],
        'youden_J'   : J[best],
        'fpr'        : fpr,
        'tpr'        : tpr,
        'prec_pts'   : prec_pts,
        'rec_pts'    : rec_pts,
    }


# ════════════════════════════════════════════════════════════════════════════
# OBJ. a_alt — Asimetría derecha sistemática (nueva hipótesis)
# ════════════════════════════════════════════════════════════════════════════
print("─" * 72)
print("  OBJ. a_alt — Asimetría derecha basal poblacional")
print("─" * 72)

asimetria_rows = []
for region in ['r1', 'r2', 'r3', 'r4']:
    col_delta = delta_col_name(region)
    if col_delta not in df.columns:
        continue
    vals = df[col_delta].dropna().values
    if len(vals) < 5:
        continue

    mean_val = float(np.mean(vals))
    sd_val   = float(np.std(vals, ddof=1))
    pct_pos  = float((vals > 0).mean() * 100)

    # One-sample Wilcoxon: is mean ΔT > 0?
    try:
        w_stat, p_wilcox = stats.wilcoxon(vals, alternative='greater')
    except Exception:
        w_stat, p_wilcox = np.nan, np.nan

    # Mann-Whitney: |ΔT| Con Dolor vs Sin Dolor
    # Use normalized ΔT if available
    col_delta_norm = f"delta_t_norm_{region}"
    if col_delta_norm in df.columns:
        vals_norm_cd = df.loc[df['grupo'] == 'Con Dolor', col_delta_norm].dropna().abs()
        vals_norm_sd = df.loc[df['grupo'] == 'Sin Dolor', col_delta_norm].dropna().abs()
    else:
        vals_norm_cd = pd.Series(dtype=float)
        vals_norm_sd = pd.Series(dtype=float)

    if len(vals_norm_cd) >= 3 and len(vals_norm_sd) >= 3:
        u_stat_mw, p_mw = stats.mannwhitneyu(vals_norm_cd, vals_norm_sd, alternative='two-sided')
    else:
        u_stat_mw, p_mw = np.nan, np.nan

    sig_w = '★' if pd.notna(p_wilcox) and p_wilcox < 0.05 else '✗'
    sig_mw = '★' if pd.notna(p_mw) and p_mw < 0.05 else '✗'
    print(f"  {region.upper()}  mean={mean_val:+.3f}  SD={sd_val:.3f}  %pos={pct_pos:.1f}%  "
          f"Wilcoxon p={p_wilcox:.4f} {sig_w}  MW(|ΔT|) p={p_mw:.4f} {sig_mw}")

    asimetria_rows.append({
        'region'          : region,
        'media_delta_t'   : mean_val,
        'sd_delta_t'      : sd_val,
        'pct_pacientes_pos': pct_pos,
        'wilcoxon_stat'   : w_stat,
        'wilcoxon_p'      : p_wilcox,
        'mw_cd_vs_sd_stat': u_stat_mw,
        'mw_cd_vs_sd_p'   : p_mw,
    })

asimetria_df = pd.DataFrame(asimetria_rows)
asimetria_df.to_csv(os.path.join(OUTPUT_DIR, 'asimetria_derecha_basal_complementario.csv'), index=False)
print(f"  ✓ Guardado: {os.path.join(OUTPUT_DIR, 'asimetria_derecha_basal_complementario.csv')}")

# Summary text for report
if not asimetria_df.empty:
    sig_asim = asimetria_df[asimetria_df['wilcoxon_p'] < 0.05]
    asimetria_text = (
        f"Se evaluó si la temperatura media del lado derecho supera sistemáticamente al izquierdo "
        f"(Wilcoxon one-sample, H1: ΔT > 0). "
        f"{'Regiones con asimetría derecha significativa: ' + ', '.join(sig_asim['region'].tolist()) if len(sig_asim) > 0 else 'Ninguna región mostró asimetría derecha sistemática significativa.'}. "
        f"Región R1: mean={asimetria_df[asimetria_df['region']=='r1']['media_delta_t'].values[0]:+.3f} si existe."
    )
else:
    asimetria_text = "No fue posible evaluar asimetría derecha basal."


# ════════════════════════════════════════════════════════════════════════════
# OBJ. b_alt — Correlación Temperatura NORMALIZADA ↔ Dolor
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 72)
print("  OBJ. b_alt — Correlación Temperatura Normalizada ↔ Intensidad Dolor")
print("  (Bonferroni sobre todas las pruebas de correlación)")
print("─" * 72)

norm_corr_results = []
p_values_norm = []
test_keys_norm = []

# Global: delta_t_norm vs intensidad
paired_norm = paired.dropna(subset=['delta_t_norm', 'intensidad'])
rho_norm_g, p_norm_g = spearman_permutation_test(paired_norm, 'delta_t_norm', 'intensidad')
p_values_norm.append(p_norm_g)
test_keys_norm.append(('Global', 'Todos', rho_norm_g))

# Por región: delta_t_norm vs intensidad
for reg, grp in paired.groupby('region'):
    sub = grp.dropna(subset=['delta_t_norm', 'intensidad'])
    if len(sub) < 5:
        continue
    r, p = spearman_permutation_test(sub, 'delta_t_norm', 'intensidad')
    p_values_norm.append(p)
    test_keys_norm.append(('Región', reg, r))

# Por músculo: delta_t_norm vs intensidad
for musc, grp in paired.groupby('musculo'):
    sub = grp.dropna(subset=['delta_t_norm', 'intensidad'])
    if len(sub) < 5:
        continue
    r, p = spearman_permutation_test(sub, 'delta_t_norm', 'intensidad')
    p_values_norm.append(p)
    test_keys_norm.append(('Músculo', musc, r))

# Por punto: temp_norm vs intensidad
for col in PAIN_COLS:
    parsed = parse_pain_col(col)
    if parsed is None:
        continue
    musculo, lado, region, punto = parsed
    col_norm = f"temp_norm_{region}_{lado}"
    if col_norm not in df.columns:
        continue
    temps_n = df[col_norm].dropna()
    dolors_n = df[col].loc[temps_n.index].astype(float)
    valid = dolors_n.notna() & temps_n.notna()
    if valid.sum() < 5 or dolors_n[valid].max() == 0:
        continue
    r, p = stats.spearmanr(temps_n[valid], dolors_n[valid])
    p_values_norm.append(p)
    test_keys_norm.append(('Punto', col.upper(), r, col_norm))

# Corrección de Bonferroni
if p_values_norm:
    reject_n, p_corr_n, _, _ = multipletests(p_values_norm, alpha=0.05, method='bonferroni')
else:
    reject_n, p_corr_n = [], []

# Organizar resultados
for i, (key, p_corr, is_sig) in enumerate(zip(test_keys_norm, p_corr_n, reject_n)):
    nivel = key[0]
    subgrupo = key[1]
    rho = key[2]
    p_orig = p_values_norm[i]
    sig_mark = '★' if is_sig else '✗'

    res = {
        'nivel': nivel,
        'subgrupo': subgrupo,
        'rho': rho,
        'p_orig': p_orig,
        'p_bonf': p_corr,
        'significativo': is_sig
    }
    norm_corr_results.append(res)

    if nivel == 'Global':
        print(f"\n  [GLOBAL NORM]  ρ = {rho:+.4f}   p_orig = {p_orig:.4f}   p_bonf = {p_corr:.4f}  {sig_mark}")
    elif nivel == 'Región':
        if subgrupo == paired['region'].unique()[0]:
            print("\n  Por región (p-valores permutados + Bonferroni):")
        print(f"    {subgrupo.upper():<10} ρ = {rho:+.4f}   p_orig = {p_orig:.4f}   p_bonf = {p_corr:.4f}  {sig_mark}")
    elif nivel == 'Músculo':
        first_musc = next((k[1] for k in test_keys_norm if k[0] == 'Músculo'), None)
        if subgrupo == first_musc:
            print("\n  Por músculo (p-valores permutados + Bonferroni):")
        print(f"    {subgrupo:<40} ρ = {rho:+.4f}   p_orig = {p_orig:.4f}   p_bonf = {p_corr:.4f}  {sig_mark}")

pd.DataFrame(norm_corr_results).to_csv(
    os.path.join(OUTPUT_DIR, 'correlaciones_normalizadas_complementario.csv'), index=False
)
print(f"\n  ✓ Guardado: {os.path.join(OUTPUT_DIR, 'correlaciones_normalizadas_complementario.csv')}")

# Get global rho for report
res_norm_global = next((r for r in norm_corr_results if r['nivel'] == 'Global'), {})
rho_norm_g_rep = res_norm_global.get('rho', np.nan)
p_norm_g_rep   = res_norm_global.get('p_bonf', np.nan)


# ════════════════════════════════════════════════════════════════════════════
# OBJ. b2_alt — Correlación con Intensidad Continua (nivel de paciente)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 72)
print("  OBJ. b2_alt — Correlación con Intensidad Continua (Nivel Paciente N=43)")
print("─" * 72)

# Aggregate at patient level
paciente_agg = paired.groupby('muestra').agg(
    delta_t_max=('delta_t', 'max'),
    delta_t_mean=('delta_t', 'mean'),
    delta_t_norm_max=('delta_t_norm', lambda x: x.abs().max()),
    delta_t_r3=('delta_t', lambda x: x[paired.loc[x.index, 'region'] == 'r3'].mean() if (paired.loc[x.index, 'region'] == 'r3').any() else np.nan),
    intensidad_max=('intensidad', 'max'),
    intensidad_total=('intensidad', 'sum'),
    n_puntos_dolor=('tiene_dolor', 'sum'),
    presencia_dolor=('tiene_dolor', 'max'),
).reset_index()

# Merge with patient-level covariates
pat_covs = df[['numero de muestra', 'sexo_num', 'edad', 'categoria_dolor',
               'intensidad_max_paciente', 'intensidad_total_paciente']].copy()
pat_covs = pat_covs.rename(columns={'numero de muestra': 'muestra'})
paciente_agg = paciente_agg.merge(pat_covs, on='muestra', how='left')

corr_intensidad_rows = []

tests_b2 = [
    ('delta_t_max', 'intensidad_max', 'ΔT_max vs intensidad_max'),
    ('delta_t_max', 'intensidad_total', 'ΔT_max vs intensidad_total'),
    ('delta_t_r3', 'n_puntos_dolor', 'ΔT_R3 vs n_puntos_dolor'),
    ('delta_t_norm_max', 'intensidad_max', 'ΔT_norm_max vs intensidad_max'),
    ('delta_t_norm_max', 'intensidad_total', 'ΔT_norm_max vs intensidad_total'),
]

for xcol, ycol, label in tests_b2:
    sub = paciente_agg[[xcol, ycol]].dropna()
    if len(sub) < 5:
        continue
    rho_b2, p_b2 = stats.spearmanr(sub[xcol], sub[ycol])
    sig = '★' if pd.notna(p_b2) and p_b2 < 0.05 else '✗'
    print(f"  {label:<40} ρ={rho_b2:+.4f}  p={p_b2:.4f}  {sig}")
    corr_intensidad_rows.append({
        'predictor'  : xcol,
        'outcome'    : ycol,
        'label'      : label,
        'n'          : len(sub),
        'rho'        : rho_b2,
        'p_valor'    : p_b2,
        'significativo': p_b2 < 0.05 if pd.notna(p_b2) else False,
    })

# Binary vs continuous comparison
print("\n  Comparación: intensidad binaria (0/1) vs continua (0-10):")
for xcol in ['delta_t_max', 'delta_t_norm_max']:
    sub_bin = paciente_agg[[xcol, 'presencia_dolor']].dropna()
    if len(sub_bin) >= 5:
        rho_bin, p_bin = stats.spearmanr(sub_bin[xcol], sub_bin['presencia_dolor'])
        print(f"    {xcol} vs presencia_dolor (binaria): ρ={rho_bin:+.4f}  p={p_bin:.4f}")
        corr_intensidad_rows.append({
            'predictor'  : xcol,
            'outcome'    : 'presencia_dolor',
            'label'      : f"{xcol} vs presencia_dolor (binaria)",
            'n'          : len(sub_bin),
            'rho'        : rho_bin,
            'p_valor'    : p_bin,
            'significativo': p_bin < 0.05 if pd.notna(p_bin) else False,
        })

pd.DataFrame(corr_intensidad_rows).to_csv(
    os.path.join(OUTPUT_DIR, 'correlaciones_intensidad_continua_complementario.csv'), index=False
)
print(f"\n  ✓ Guardado: {os.path.join(OUTPUT_DIR, 'correlaciones_intensidad_continua_complementario.csv')}")

sig_b2 = [r for r in corr_intensidad_rows if r.get('significativo')]
if sig_b2:
    sig_b2_desc = '; '.join([r['label'] + ' rho={:+.3f} p={:.4f}'.format(r['rho'], r['p_valor']) for r in sig_b2])
    corr_intensidad_text = (
        f"A nivel de paciente (N={len(paciente_agg)}): Correlaciones significativas: {sig_b2_desc}"
    )
else:
    corr_intensidad_text = (
        f"A nivel de paciente (N={len(paciente_agg)}): Sin correlaciones significativas entre "
        f"ΔT y intensidad continua."
    )


# ════════════════════════════════════════════════════════════════════════════
# OBJ. c_alt — Utilidad diagnóstica con temperatura NORMALIZADA (ROC)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 72)
print("  OBJ. c_alt — Curvas ROC con Temperatura Normalizada")
print("─" * 72)

roc_norm_results = []

# Global: delta_t_norm
roc_norm_global = calc_roc(paired.dropna(subset=['delta_t_norm']), 'Global_Norm', score_col='delta_t_norm')
roc_raw_global  = calc_roc(paired, 'Global_Raw', score_col='delta_t')

print(f"\n  {'Subgrupo':<42} {'AUC_raw':>7}  {'AUC_norm':>8}  {'Δ AUC':>6}")
print("  " + "─" * 70)

if roc_raw_global and roc_norm_global:
    d_auc = roc_norm_global['auc'] - roc_raw_global['auc']
    print(f"  {'Global':<42} {roc_raw_global['auc']:>7.3f}  {roc_norm_global['auc']:>8.3f}  {d_auc:>+6.3f}")
    roc_norm_results.append({
        'subgrupo': 'Global', 'auc_raw': roc_raw_global['auc'],
        'auc_norm': roc_norm_global['auc'], 'delta_auc': d_auc,
        'T_star_norm': roc_norm_global['T_star'],
        'sens_norm': roc_norm_global['sensitivity'], 'spec_norm': roc_norm_global['specificity'],
    })

# By region
for reg, grp in paired.groupby('region'):
    roc_r_raw  = calc_roc(grp, f'{reg}_raw', score_col='delta_t')
    roc_r_norm = calc_roc(grp.dropna(subset=['delta_t_norm']), f'{reg}_norm', score_col='delta_t_norm')
    auc_raw_v  = roc_r_raw['auc']  if roc_r_raw  else np.nan
    auc_norm_v = roc_r_norm['auc'] if roc_r_norm else np.nan
    d_auc = auc_norm_v - auc_raw_v if pd.notna(auc_raw_v) and pd.notna(auc_norm_v) else np.nan
    print(f"  {reg.upper():<42} {auc_raw_v:>7.3f}  {auc_norm_v:>8.3f}  {d_auc:>+6.3f}")
    roc_norm_results.append({
        'subgrupo': reg, 'auc_raw': auc_raw_v, 'auc_norm': auc_norm_v, 'delta_auc': d_auc,
        'T_star_norm': roc_r_norm['T_star'] if roc_r_norm else np.nan,
        'sens_norm': roc_r_norm['sensitivity'] if roc_r_norm else np.nan,
        'spec_norm': roc_r_norm['specificity'] if roc_r_norm else np.nan,
    })

# By muscle
for musc, grp in paired.groupby('musculo'):
    if len(grp) < 10:
        continue
    roc_m_raw  = calc_roc(grp, f'{musc}_raw', score_col='delta_t')
    roc_m_norm = calc_roc(grp.dropna(subset=['delta_t_norm']), f'{musc}_norm', score_col='delta_t_norm')
    auc_raw_v  = roc_m_raw['auc']  if roc_m_raw  else np.nan
    auc_norm_v = roc_m_norm['auc'] if roc_m_norm else np.nan
    d_auc = auc_norm_v - auc_raw_v if pd.notna(auc_raw_v) and pd.notna(auc_norm_v) else np.nan
    print(f"  {musc:<42} {auc_raw_v:>7.3f}  {auc_norm_v:>8.3f}  {d_auc:>+6.3f}")
    roc_norm_results.append({
        'subgrupo': musc, 'auc_raw': auc_raw_v, 'auc_norm': auc_norm_v, 'delta_auc': d_auc,
        'T_star_norm': roc_m_norm['T_star'] if roc_m_norm else np.nan,
        'sens_norm': roc_m_norm['sensitivity'] if roc_m_norm else np.nan,
        'spec_norm': roc_m_norm['specificity'] if roc_m_norm else np.nan,
    })

pd.DataFrame(roc_norm_results).to_csv(
    os.path.join(OUTPUT_DIR, 'resultados_roc_normalizados_complementario.csv'), index=False
)
print(f"\n  ✓ Guardado: {os.path.join(OUTPUT_DIR, 'resultados_roc_normalizados_complementario.csv')}")

roc_norm_text = (
    f"ROC con temp_norm (global): AUC_raw={roc_raw_global['auc']:.3f} vs AUC_norm={roc_norm_global['auc']:.3f} "
    f"(Δ={roc_norm_global['auc'] - roc_raw_global['auc']:+.3f}). "
    f"T*_norm={roc_norm_global['T_star']:.3f}, Sens={roc_norm_global['sensitivity']:.3f}, "
    f"Spec={roc_norm_global['specificity']:.3f}."
    if roc_norm_global and roc_raw_global else "Datos insuficientes para ROC normalizado."
)


# ════════════════════════════════════════════════════════════════════════════
# OBJ. d_alt — Modelos Logísticos Extendidos con Covariables
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 72)
print("  OBJ. d_alt — Modelos Logísticos Extendidos (A, C, D, E)")
print("─" * 72)

paired_for_models = paired.dropna(subset=['temp_max', 'region', 'lado', 'tiene_dolor', 'muestra'])
paired_norm_models = paired.dropna(subset=['temp_norm', 'region', 'lado', 'tiene_dolor', 'muestra'])

logit_extendido_rows = []
model_comparison_rows = []

models_config = [
    ('A', 'temp_max', None, 'tiene_dolor ~ temp_max + region + lado'),
    ('C', 'temp_max', ['sexo_num', 'edad'], 'tiene_dolor ~ temp_max + region + lado + sexo + edad'),
    ('D', 'temp_norm', None, 'tiene_dolor ~ temp_norm + region + lado'),
    ('E', 'temp_norm', ['sexo_num', 'edad'], 'tiene_dolor ~ temp_norm + region + lado + sexo + edad'),
]

fitted_models = {}
for model_name, col_temp, covariates, formula_desc in models_config:
    if col_temp == 'temp_norm':
        data_m = paired_norm_models
    else:
        data_m = paired_for_models

    print(f"\n  Modelo {model_name}: {formula_desc}")
    fit = ajustar_logit_cluster(data_m, col_temp_max=col_temp, covariates=covariates)
    fitted_models[model_name] = fit

    if fit['ok']:
        m = fit['metrics']
        print(f"    N={m['n_obs']}  Pseudo-R²={m['pseudo_r2_mcfadden']:.4f}  "
              f"AUC={m['auc_prob_logit']:.4f}  AIC={m['aic']:.2f}")
        # print key coefficient
        main_term = next((r for r in fit['rows'] if r['termino'] == col_temp), None)
        if main_term:
            print(f"    {col_temp}: OR={main_term['odds_ratio']:.3f}  "
                  f"(IC95% {main_term['or_ci95_low']:.3f}–{main_term['or_ci95_high']:.3f})  "
                  f"p={main_term['p_valor']:.4f}  {'★' if main_term['p_valor'] < 0.05 else '✗'}")
        for row in fit['rows']:
            logit_extendido_rows.append({
                'modelo'   : model_name,
                'formula'  : formula_desc,
                **row
            })
        model_comparison_rows.append({
            'modelo'       : model_name,
            'formula'      : formula_desc,
            'n_obs'        : m['n_obs'],
            'pseudo_r2'    : m['pseudo_r2_mcfadden'],
            'auc_roc'      : m['auc_prob_logit'],
            'pr_auc'       : m['pr_auc_logit'],
            'aic'          : m['aic'],
            'bic'          : m['bic'],
            'f1'           : m['f1_score'],
            'brier'        : m['brier_score'],
        })
    else:
        print(f"    No convergió: {fit['error']}")
        model_comparison_rows.append({
            'modelo': model_name, 'formula': formula_desc,
            'n_obs': np.nan, 'pseudo_r2': np.nan, 'auc_roc': np.nan,
            'pr_auc': np.nan, 'aic': np.nan, 'bic': np.nan,
            'f1': np.nan, 'brier': np.nan,
        })

# LRT: C vs A
print("\n  LRT: Modelo C vs A (efecto de sexo+edad sobre temp_max):")
if fitted_models.get('A', {}).get('ok') and fitted_models.get('C', {}).get('ok'):
    llr_ca = 2 * (fitted_models['C']['model'].llf - fitted_models['A']['model'].llf)
    df_ca  = int(fitted_models['C']['model'].df_model - fitted_models['A']['model'].df_model)
    p_ca   = stats.chi2.sf(llr_ca, df_ca) if df_ca > 0 else np.nan
    print(f"    χ²={llr_ca:.3f}  df={df_ca}  p={p_ca:.4f}  {'★' if pd.notna(p_ca) and p_ca < 0.05 else '✗'}")
else:
    p_ca = np.nan
    print("    No fue posible calcular LRT C vs A.")

# LRT: E vs D
print("  LRT: Modelo E vs D (efecto de sexo+edad sobre temp_norm):")
if fitted_models.get('D', {}).get('ok') and fitted_models.get('E', {}).get('ok'):
    llr_ed = 2 * (fitted_models['E']['model'].llf - fitted_models['D']['model'].llf)
    df_ed  = int(fitted_models['E']['model'].df_model - fitted_models['D']['model'].df_model)
    p_ed   = stats.chi2.sf(llr_ed, df_ed) if df_ed > 0 else np.nan
    print(f"    χ²={llr_ed:.3f}  df={df_ed}  p={p_ed:.4f}  {'★' if pd.notna(p_ed) and p_ed < 0.05 else '✗'}")
else:
    p_ed = np.nan
    print("    No fue posible calcular LRT E vs D.")

pd.DataFrame(logit_extendido_rows).to_csv(
    os.path.join(OUTPUT_DIR, 'resultados_logit_extendido_complementario.csv'), index=False
)
pd.DataFrame(model_comparison_rows).to_csv(
    os.path.join(OUTPUT_DIR, 'comparacion_modelos_completa_complementario.csv'), index=False
)
print(f"\n  ✓ Guardado: {os.path.join(OUTPUT_DIR, 'resultados_logit_extendido_complementario.csv')}")
print(f"  ✓ Guardado: {os.path.join(OUTPUT_DIR, 'comparacion_modelos_completa_complementario.csv')}")

# Build model comparison text for report
mc_df = pd.DataFrame(model_comparison_rows)
if not mc_df.empty and mc_df['auc_roc'].notna().any():
    best_model = mc_df.loc[mc_df['auc_roc'].idxmax(), 'modelo'] if mc_df['auc_roc'].notna().any() else 'N/A'
    modelo_comp_text = (
        "Comparación de modelos logísticos (A, C, D, E): "
        + "; ".join([
            f"Mod{r['modelo']}: AUC={r['auc_roc']:.3f} AIC={r['aic']:.1f}"
            for _, r in mc_df.iterrows() if pd.notna(r['auc_roc'])
        ])
        + f". Mejor AUC: Modelo {best_model}."
    )
else:
    modelo_comp_text = "No fue posible ajustar los modelos logísticos extendidos."


# ════════════════════════════════════════════════════════════════════════════
# OBJ. e_alt — Análisis Subgrupo Masetero R3
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 72)
print("  OBJ. e_alt — Análisis Subgrupo Masetero R3 (anatómicamente específico)")
print("─" * 72)

masetero_r3 = paired[(paired['musculo'] == 'masetero') & (paired['region'] == 'r3')].copy()
print(f"  N observaciones masetero R3: {len(masetero_r3)}")
print(f"  Pacientes con dolor masetero R3: {masetero_r3[masetero_r3['tiene_dolor']==1]['muestra'].nunique()}")

masetero_r3_rows = []

if len(masetero_r3) >= 5:
    # Spearman: temp_max_r3 vs masetero pain
    sub_m = masetero_r3[['temp_max', 'intensidad']].dropna()
    if len(sub_m) >= 5:
        rho_m_raw, p_m_raw = stats.spearmanr(sub_m['temp_max'], sub_m['intensidad'])
        sig_m = '★' if p_m_raw < 0.05 else '✗'
        print(f"  Spearman temp_max_R3 vs intensidad: ρ={rho_m_raw:+.4f}  p={p_m_raw:.4f}  {sig_m}")
        masetero_r3_rows.append({
            'analisis': 'spearman_temp_max_r3_vs_intensidad', 'n': len(sub_m),
            'valor_1': rho_m_raw, 'valor_2': p_m_raw, 'significativo': p_m_raw < 0.05,
        })

    # Spearman: delta_t_norm_r3 vs masetero pain
    sub_mn = masetero_r3[['delta_t_norm', 'intensidad']].dropna()
    if len(sub_mn) >= 5:
        rho_m_norm, p_m_norm = stats.spearmanr(sub_mn['delta_t_norm'], sub_mn['intensidad'])
        sig_mn = '★' if p_m_norm < 0.05 else '✗'
        print(f"  Spearman delta_t_norm_R3 vs intensidad: ρ={rho_m_norm:+.4f}  p={p_m_norm:.4f}  {sig_mn}")
        masetero_r3_rows.append({
            'analisis': 'spearman_delta_t_norm_r3_vs_intensidad', 'n': len(sub_mn),
            'valor_1': rho_m_norm, 'valor_2': p_m_norm, 'significativo': p_m_norm < 0.05,
        })

    # ROC: temp_max_r3 vs tiene_dolor_masetero
    roc_m_r3 = calc_roc(masetero_r3, 'Masetero_R3_TempMax', score_col='temp_max')
    if roc_m_r3:
        print(f"  ROC temp_max_R3: AUC={roc_m_r3['auc']:.3f}  T*={roc_m_r3['T_star']:.2f}  "
              f"Sens={roc_m_r3['sensitivity']:.3f}  Spec={roc_m_r3['specificity']:.3f}")
        masetero_r3_rows.append({
            'analisis': 'roc_temp_max_r3', 'n': len(masetero_r3),
            'valor_1': roc_m_r3['auc'], 'valor_2': roc_m_r3['T_star'],
            'valor_3': roc_m_r3['sensitivity'], 'valor_4': roc_m_r3['specificity'],
            'significativo': roc_m_r3['auc'] > 0.6,
        })

    # Per-point analysis: each masetero R3 point
    print("  Por punto masetero R3:")
    for punto, grp_p in masetero_r3.groupby('punto'):
        if len(grp_p) < 5:
            continue
        sub_p = grp_p[['temp_max', 'intensidad']].dropna()
        if len(sub_p) < 5 or sub_p['intensidad'].max() == 0:
            continue
        rho_p, p_p = stats.spearmanr(sub_p['temp_max'], sub_p['intensidad'])
        sig_p = '★' if p_p < 0.05 else '✗'
        print(f"    {punto}: n={len(sub_p)}  ρ={rho_p:+.4f}  p={p_p:.4f}  {sig_p}")
        masetero_r3_rows.append({
            'analisis': f'spearman_punto_{punto}', 'n': len(sub_p),
            'valor_1': rho_p, 'valor_2': p_p, 'significativo': p_p < 0.05,
        })

    # Logit restricted to masetero R3
    if masetero_r3['tiene_dolor'].nunique() >= 2:
        fit_m_r3 = ajustar_logit_cluster(masetero_r3, col_temp_max='temp_max')
        if fit_m_r3['ok']:
            m_m = fit_m_r3['metrics']
            print(f"  Logit masetero R3: AUC={m_m['auc_prob_logit']:.4f}  "
                  f"Pseudo-R²={m_m['pseudo_r2_mcfadden']:.4f}  AIC={m_m['aic']:.2f}")
            masetero_r3_rows.append({
                'analisis': 'logit_masetero_r3_auc', 'n': int(m_m['n_obs']),
                'valor_1': m_m['auc_prob_logit'], 'valor_2': m_m['pseudo_r2_mcfadden'],
                'valor_3': m_m['aic'], 'significativo': m_m['auc_prob_logit'] > 0.6,
            })
else:
    print("  Datos insuficientes para análisis subgrupo masetero R3.")
    roc_m_r3 = None

pd.DataFrame(masetero_r3_rows).to_csv(
    os.path.join(OUTPUT_DIR, 'analisis_subgrupo_masetero_r3_complementario.csv'), index=False
)
print(f"  ✓ Guardado: {os.path.join(OUTPUT_DIR, 'analisis_subgrupo_masetero_r3_complementario.csv')}")

sig_m3 = [r for r in masetero_r3_rows if r.get('significativo')]
masetero_r3_text = (
    f"Subgrupo masetero R3 (N={len(masetero_r3)} observaciones): "
    f"{'Resultados significativos: ' + '; '.join([r['analisis'] for r in sig_m3]) if sig_m3 else 'Sin resultados significativos en este subgrupo.'}"
)


# ════════════════════════════════════════════════════════════════════════════
# OBJ. f_alt — Análisis Nivel Paciente (N=43) con Intensidad Continua
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 72)
print("  OBJ. f_alt — Análisis Nivel Paciente N=43 (Intensidad Continua)")
print("─" * 72)

paciente_n45_rows = []

# Kruskal-Wallis: ΔT_max by pain category (none/mild/severe)
if 'categoria_dolor' in paciente_agg.columns and paciente_agg['categoria_dolor'].notna().any():
    grupos_cat = []
    for cat in ['Sin Dolor', 'Leve (1-4)', 'Moderado-Severo (5+)']:
        g = paciente_agg[paciente_agg['categoria_dolor'] == cat]['delta_t_max'].dropna().values
        grupos_cat.append(g)
        print(f"  {cat}: n={len(g)}  mediana_ΔT={np.median(g):.3f}" if len(g) > 0 else f"  {cat}: n=0")

    grupos_valid = [g for g in grupos_cat if len(g) >= 2]
    if len(grupos_valid) >= 2:
        kw_stat, kw_p = stats.kruskal(*grupos_valid)
        sig_kw = '★' if kw_p < 0.05 else '✗'
        print(f"  Kruskal-Wallis ΔT_max por categoría dolor: H={kw_stat:.3f}  p={kw_p:.4f}  {sig_kw}")
        paciente_n45_rows.append({
            'analisis': 'kruskal_wallis_delta_t_max_by_cat', 'n': len(paciente_agg),
            'valor_1': kw_stat, 'valor_2': kw_p, 'significativo': kw_p < 0.05,
        })

# ROC: ΔT_max_norm vs presencia_dolor
roc_dt_norm_n45 = calc_roc(
    paciente_agg.rename(columns={'presencia_dolor': 'tiene_dolor'}).dropna(subset=['delta_t_norm_max']),
    label='ΔT_norm_max (N=43)',
    score_col='delta_t_norm_max'
)
if roc_dt_norm_n45:
    print(f"  ROC ΔT_norm_max vs presencia_dolor (N=43): AUC={roc_dt_norm_n45['auc']:.4f}  "
          f"T*={roc_dt_norm_n45['T_star']:.3f}")
    paciente_n45_rows.append({
        'analisis': 'roc_delta_t_norm_max_n45', 'n': len(paciente_agg),
        'valor_1': roc_dt_norm_n45['auc'], 'valor_2': roc_dt_norm_n45['T_star'],
        'valor_3': roc_dt_norm_n45['sensitivity'], 'valor_4': roc_dt_norm_n45['specificity'],
        'significativo': roc_dt_norm_n45['auc'] > 0.6,
    })

# Spearman: ΔT_max vs intensidad_total at patient level
sub_f = paciente_agg[['delta_t_max', 'intensidad_total']].dropna()
if len(sub_f) >= 5:
    rho_f, p_f = stats.spearmanr(sub_f['delta_t_max'], sub_f['intensidad_total'])
    sig_f = '★' if p_f < 0.05 else '✗'
    print(f"  Spearman ΔT_max vs intensidad_total (N=43): ρ={rho_f:+.4f}  p={p_f:.4f}  {sig_f}")
    paciente_n45_rows.append({
        'analisis': 'spearman_delta_t_max_vs_intensidad_total', 'n': len(sub_f),
        'valor_1': rho_f, 'valor_2': p_f, 'significativo': p_f < 0.05,
    })

# Mann-Whitney: ΔT_max Con Dolor vs Sin Dolor
con_d = paciente_agg[paciente_agg['presencia_dolor'] == 1]['delta_t_max'].dropna()
sin_d = paciente_agg[paciente_agg['presencia_dolor'] == 0]['delta_t_max'].dropna()
if len(con_d) >= 2 and len(sin_d) >= 2:
    u_f, p_mw_f = stats.mannwhitneyu(con_d, sin_d, alternative='two-sided')
    sig_mw_f = '★' if p_mw_f < 0.05 else '✗'
    print(f"  Mann-Whitney ΔT_max (Con Dolor vs Sin Dolor): p={p_mw_f:.4f}  {sig_mw_f}")
    print(f"    Mediana Con Dolor={con_d.median():.3f}  Mediana Sin Dolor={sin_d.median():.3f}")
    paciente_n45_rows.append({
        'analisis': 'mann_whitney_delta_t_max_cd_vs_sd', 'n': len(paciente_agg),
        'valor_1': u_f, 'valor_2': p_mw_f,
        'valor_extra': f'Med_CD={con_d.median():.3f}_SD={sin_d.median():.3f}',
        'significativo': p_mw_f < 0.05,
    })

pd.DataFrame(paciente_n45_rows).to_csv(
    os.path.join(OUTPUT_DIR, 'resultados_paciente_n45_alterno_complementario.csv'), index=False
)
print(f"  ✓ Guardado: {os.path.join(OUTPUT_DIR, 'resultados_paciente_n45_alterno_complementario.csv')}")

paciente_alt_text = (
    f"N=43 pacientes. ROC ΔT_norm_max: AUC={roc_dt_norm_n45['auc']:.3f} "
    f"({interpretar_auc(roc_dt_norm_n45['auc'])})."
    if roc_dt_norm_n45 else "No fue posible evaluar ROC normalizado a nivel de paciente."
)


# ════════════════════════════════════════════════════════════════════════════
# OBJ. g — Efecto del Sexo y Edad sobre Temperatura (confundidores)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 72)
print("  OBJ. g — Efectos del Sexo y la Edad sobre Temperatura (Confundidores)")
print("─" * 72)

confounders_rows = []

# Mann-Whitney: regional temperatures by sex
if 'sexo_num' in df.columns and df['sexo_num'].notna().any():
    print("  Mann-Whitney: temperaturas regionales por sexo")
    for region in ['r1', 'r2', 'r3', 'r4']:
        for lado in ['derecha', 'izquierda']:
            col_temp = temp_col_name(region, lado)
            if col_temp not in df.columns:
                continue
            masc = df.loc[df['sexo_num'] == 0, col_temp].dropna()
            fem  = df.loc[df['sexo_num'] == 1, col_temp].dropna()
            if len(masc) < 3 or len(fem) < 3:
                continue
            u_s, p_s = stats.mannwhitneyu(masc, fem, alternative='two-sided')
            sig_s = '★' if p_s < 0.05 else '✗'
            if p_s < 0.05:
                print(f"    {region} {lado}: p={p_s:.4f}  Med_M={masc.median():.2f}  Med_F={fem.median():.2f}  {sig_s}")
            confounders_rows.append({
                'analisis': f'mw_sexo_{region}_{lado}', 'region': region, 'lado': lado,
                'n_masc': len(masc), 'n_fem': len(fem),
                'mediana_masc': masc.median(), 'mediana_fem': fem.median(),
                'u_stat': u_s, 'p_valor': p_s, 'significativo': p_s < 0.05,
            })

# Spearman: age vs regional temperatures
if 'edad' in df.columns and df['edad'].notna().any():
    print("  Spearman: edad vs temperaturas regionales (R1-R4, ambos lados)")
    for region in ['r1', 'r2', 'r3', 'r4']:
        for lado in ['derecha', 'izquierda']:
            col_temp = temp_col_name(region, lado)
            if col_temp not in df.columns:
                continue
            sub_age = df[['edad', col_temp]].dropna()
            if len(sub_age) < 5:
                continue
            rho_age, p_age = stats.spearmanr(sub_age['edad'], sub_age[col_temp])
            sig_age = '★' if p_age < 0.05 else '✗'
            if p_age < 0.05:
                print(f"    {region} {lado}: ρ={rho_age:+.4f}  p={p_age:.4f}  {sig_age}")
            confounders_rows.append({
                'analisis': f'spearman_edad_{region}_{lado}', 'region': region, 'lado': lado,
                'n': len(sub_age),
                'rho': rho_age, 'p_valor': p_age, 'significativo': p_age < 0.05,
            })

sig_conf = [r for r in confounders_rows if r.get('significativo')]
n_sig_sexo = sum(1 for r in sig_conf if 'mw_sexo' in r.get('analisis', ''))
n_sig_edad = sum(1 for r in sig_conf if 'spearman_edad' in r.get('analisis', ''))
print(f"\n  Resumen confundidores:")
print(f"    Regiones significativas por sexo: {n_sig_sexo}")
print(f"    Regiones significativas por edad: {n_sig_edad}")

pd.DataFrame(confounders_rows).to_csv(
    os.path.join(OUTPUT_DIR, 'efectos_sexo_edad_temperatura_complementario.csv'), index=False
)
print(f"  ✓ Guardado: {os.path.join(OUTPUT_DIR, 'efectos_sexo_edad_temperatura_complementario.csv')}")

confounders_text = (
    f"Confundidores: {n_sig_sexo} regiones con diferencia significativa por sexo (Mann-Whitney); "
    f"{n_sig_edad} regiones con correlación significativa edad-temperatura (Spearman). "
    f"Estos efectos justifican la inclusión de sexo y edad en Modelos C y E."
)


# ════════════════════════════════════════════════════════════════════════════
# FIGURA PANEL 3×3
# ════════════════════════════════════════════════════════════════════════════
print("\n  Generando figura panel 3×3...")

fig = plt.figure(figsize=(21, 17))
fig.patch.set_facecolor(BG)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.38)

# ── Panel a: Boxplot ΔT_norm by Con/Sin Dolor (4 regions) ───────────────
ax_a = fig.add_subplot(gs[0, 0])
regions = sorted(paired['region'].unique())
pos_list, data_con_list, data_sin_list = [], [], []
for i, reg in enumerate(regions):
    grp_reg = paired[paired['region'] == reg]
    d_con = grp_reg[grp_reg['grupo'] == 'Con Dolor']['delta_t_norm'].dropna().values
    d_sin = grp_reg[grp_reg['grupo'] == 'Sin Dolor']['delta_t_norm'].dropna().values
    data_con_list.append(d_con)
    data_sin_list.append(d_sin)
    pos_list.append(i)

bp_a_con = ax_a.boxplot(data_con_list,
    positions=[i - 0.2 for i in range(len(regions))], widths=0.35,
    patch_artist=True,
    boxprops=dict(facecolor=RED, alpha=0.7),
    medianprops=dict(color=TEXT, lw=1.5),
    whiskerprops=dict(color=TEXT), capprops=dict(color=TEXT),
    flierprops=dict(markerfacecolor=TEXT, markersize=2))
bp_a_sin = ax_a.boxplot(data_sin_list,
    positions=[i + 0.2 for i in range(len(regions))], widths=0.35,
    patch_artist=True,
    boxprops=dict(facecolor=BLUE, alpha=0.7),
    medianprops=dict(color=TEXT, lw=1.5),
    whiskerprops=dict(color=TEXT), capprops=dict(color=TEXT),
    flierprops=dict(markerfacecolor=TEXT, markersize=2))
ax_a.set_xticks(range(len(regions)))
ax_a.set_xticklabels([r.upper() for r in regions])
ax_a.set_ylabel('ΔT_norm (°C)')
ax_a.legend([bp_a_con['boxes'][0], bp_a_sin['boxes'][0]], ['Con Dolor', 'Sin Dolor'],
            fontsize=8, facecolor=AX_BG, labelcolor=TEXT, framealpha=0.8)
style_ax(ax_a, 'a) ΔT_norm por Región y Grupo')

# ── Panel b: Scatter delta_t_norm R3 vs intensidad ──────────────────────
ax_b = fig.add_subplot(gs[0, 1])
r3_sub = paired[paired['region'] == 'r3'].dropna(subset=['delta_t_norm', 'intensidad'])
if len(r3_sub) >= 5:
    cmap_group = {'Con Dolor': RED, 'Sin Dolor': BLUE}
    for grupo, grp_b in r3_sub.groupby('grupo'):
        ax_b.scatter(grp_b['delta_t_norm'], grp_b['intensidad'],
                     color=cmap_group.get(grupo, BLUE), alpha=0.35, s=12,
                     label=grupo, edgecolors='none')
    m_b, b_b, *_ = stats.linregress(r3_sub['delta_t_norm'], r3_sub['intensidad'])
    x_b = np.linspace(r3_sub['delta_t_norm'].min(), r3_sub['delta_t_norm'].max(), 200)
    rho_b, p_b = stats.spearmanr(r3_sub['delta_t_norm'], r3_sub['intensidad'])
    ax_b.plot(x_b, m_b * x_b + b_b, color=ACCENT, lw=1.5,
              label=f'ρ={rho_b:+.3f}  p={p_b:.3f}')
ax_b.set_xlabel('ΔT_norm R3 (°C)')
ax_b.set_ylabel('Intensidad Dolor (0–10)')
ax_b.legend(fontsize=7.5, facecolor=AX_BG, labelcolor=TEXT, framealpha=0.8)
style_ax(ax_b, 'b) ΔT_norm R3 vs Intensidad Dolor')

# ── Panel c: ROC comparison (raw vs normalized, global) ──────────────────
ax_c = fig.add_subplot(gs[0, 2])
if roc_raw_global:
    ax_c.plot(roc_raw_global['fpr'], roc_raw_global['tpr'], color=BLUE, lw=2,
              label=f'ΔT crudo AUC={roc_raw_global["auc"]:.3f}')
if roc_norm_global:
    ax_c.plot(roc_norm_global['fpr'], roc_norm_global['tpr'], color=ACCENT, lw=2,
              label=f'ΔT_norm AUC={roc_norm_global["auc"]:.3f}')
ax_c.plot([0, 1], [0, 1], color=GRID, ls=':', lw=1)
ax_c.set_xlabel('1 – Especificidad')
ax_c.set_ylabel('Sensibilidad')
ax_c.legend(fontsize=8, facecolor=AX_BG, labelcolor=TEXT, framealpha=0.8)
style_ax(ax_c, 'c) ROC: ΔT crudo vs ΔT_norm (Global)')

# ── Panel d: ROC Masetero R3 subgroup ────────────────────────────────────
ax_d = fig.add_subplot(gs[1, 0])
roc_m_r3_obj = calc_roc(masetero_r3, 'Masetero_R3', score_col='temp_max') if len(masetero_r3) >= 5 else None
if roc_m_r3_obj:
    ax_d.plot(roc_m_r3_obj['fpr'], roc_m_r3_obj['tpr'], color=GREEN, lw=2,
              label=f'Masetero R3 AUC={roc_m_r3_obj["auc"]:.3f}')
    ax_d.scatter([1 - roc_m_r3_obj['specificity']], [roc_m_r3_obj['sensitivity']],
                 color=RED, s=60, zorder=5, label=f'T*={roc_m_r3_obj["T_star"]:.1f}°C')
ax_d.plot([0, 1], [0, 1], color=GRID, ls=':', lw=1)
ax_d.set_xlabel('1 – Especificidad')
ax_d.set_ylabel('Sensibilidad')
ax_d.legend(fontsize=8, facecolor=AX_BG, labelcolor=TEXT, framealpha=0.8)
style_ax(ax_d, 'd) ROC Masetero R3 (subgrupo)')

# ── Panel e: Scatter age vs T_R2 ─────────────────────────────────────────
ax_e = fig.add_subplot(gs[1, 1])
col_r2_d = temp_col_name('r2', 'derecha')
if 'edad' in df.columns and col_r2_d in df.columns:
    sub_e = df[['edad', col_r2_d]].dropna()
    if len(sub_e) >= 5:
        ax_e.scatter(sub_e['edad'], sub_e[col_r2_d], color=CYAN, alpha=0.6, s=25, edgecolors='none')
        m_e, b_e, *_ = stats.linregress(sub_e['edad'], sub_e[col_r2_d])
        x_e = np.linspace(sub_e['edad'].min(), sub_e['edad'].max(), 200)
        rho_e, p_e = stats.spearmanr(sub_e['edad'], sub_e[col_r2_d])
        ax_e.plot(x_e, m_e * x_e + b_e, color=ACCENT, lw=1.5,
                  label=f'ρ={rho_e:+.3f}  p={p_e:.3f}')
        ax_e.legend(fontsize=8, facecolor=AX_BG, labelcolor=TEXT, framealpha=0.8)
ax_e.set_xlabel('Edad (años)')
ax_e.set_ylabel('Temp. R2 Derecha (°C)')
style_ax(ax_e, 'e) Edad vs Temperatura R2 (confundidor)')

# ── Panel f: Boxplot T_R3 by sex ──────────────────────────────────────────
ax_f = fig.add_subplot(gs[1, 2])
col_r3_d = temp_col_name('r3', 'derecha')
if 'sexo_num' in df.columns and col_r3_d in df.columns:
    masc_t = df.loc[df['sexo_num'] == 0, col_r3_d].dropna().values
    fem_t  = df.loc[df['sexo_num'] == 1, col_r3_d].dropna().values
    bp_f = ax_f.boxplot([masc_t, fem_t], patch_artist=True, widths=0.5,
                        medianprops=dict(color=ACCENT, lw=2))
    colors_f = [BLUE, RED]
    for patch, color in zip(bp_f['boxes'], colors_f):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax_f.set_xticks([1, 2])
    ax_f.set_xticklabels(['Masculino', 'Femenino'])
    if len(masc_t) >= 2 and len(fem_t) >= 2:
        _, p_sex_r3 = stats.mannwhitneyu(masc_t, fem_t, alternative='two-sided')
        y_top = max(np.percentile(masc_t, 95), np.percentile(fem_t, 95)) * 1.02
        ax_f.text(1.5, y_top, f'p={p_sex_r3:.4f}', ha='center', va='bottom',
                  color=ACCENT, fontsize=9, fontweight='bold')
ax_f.set_ylabel('Temp. R3 Derecha (°C)')
style_ax(ax_f, 'f) Temperatura R3 por Sexo (confundidor)')

# ── Panel g: Boxplot T_norm by intensidad category ───────────────────────
ax_g = fig.add_subplot(gs[2, 0])
categories = ['Sin Dolor', 'Leve (1-4)', 'Moderado-Severo (5+)']
data_g = []
labels_g = []
for cat in categories:
    vals_g = paired[paired['intensidad_cat'] == cat]['delta_t_norm'].dropna().values
    if len(vals_g) > 0:
        data_g.append(vals_g)
        labels_g.append(cat[:10])
if data_g:
    bp_g = ax_g.boxplot(data_g, patch_artist=True, widths=0.5,
                        medianprops=dict(color=ACCENT, lw=2))
    cat_colors = [BLUE, ORANGE, RED]
    for patch, color in zip(bp_g['boxes'], cat_colors[:len(data_g)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax_g.set_xticks(range(1, len(labels_g) + 1))
    ax_g.set_xticklabels(labels_g, fontsize=7)
ax_g.set_ylabel('ΔT_norm (°C)')
style_ax(ax_g, 'g) ΔT_norm por Categoría de Dolor')

# ── Panel h: Heatmap ρ(T_norm_region × pain_muscle) 4×4 ──────────────────
ax_h = fig.add_subplot(gs[2, 1])
regions_h  = ['r1', 'r2', 'r3', 'r4']
muscles_h  = ['atm', 'esternocleidomastoideo', 'masetero', 'temporal']
rho_matrix = np.full((len(muscles_h), len(regions_h)), np.nan)

for i_m, musc in enumerate(muscles_h):
    for i_r, reg in enumerate(regions_h):
        sub_h = paired[(paired['musculo'] == musc) & (paired['region'] == reg)].dropna(
            subset=['temp_norm', 'intensidad'])
        if len(sub_h) < 5 or sub_h['intensidad'].max() == 0:
            continue
        rho_h, _ = stats.spearmanr(sub_h['temp_norm'], sub_h['intensidad'])
        rho_matrix[i_m, i_r] = rho_h

im_h = ax_h.imshow(rho_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax_h.set_xticks(range(len(regions_h)))
ax_h.set_xticklabels([r.upper() for r in regions_h], fontsize=8)
ax_h.set_yticks(range(len(muscles_h)))
ax_h.set_yticklabels([m[:8].upper() for m in muscles_h], fontsize=7)
for i_m in range(len(muscles_h)):
    for i_r in range(len(regions_h)):
        if not np.isnan(rho_matrix[i_m, i_r]):
            ax_h.text(i_r, i_m, f'{rho_matrix[i_m, i_r]:+.2f}',
                      ha='center', va='center', fontsize=7.5,
                      color='white' if abs(rho_matrix[i_m, i_r]) > 0.5 else TEXT)
plt.colorbar(im_h, ax=ax_h, shrink=0.8)
style_ax(ax_h, 'h) Heatmap ρ(T_norm_región × músculo)')

# ── Panel i: Model comparison AIC bar chart ───────────────────────────────
ax_i = fig.add_subplot(gs[2, 2])
mc_df_plot = pd.DataFrame(model_comparison_rows)
valid_mc = mc_df_plot.dropna(subset=['aic'])
if not valid_mc.empty:
    bars_i = ax_i.bar(
        valid_mc['modelo'],
        valid_mc['aic'],
        color=[BLUE, GREEN, ORANGE, PURPLE][:len(valid_mc)],
        alpha=0.8, edgecolor='none'
    )
    # Annotate AUC values
    for bar, (_, row) in zip(bars_i, valid_mc.iterrows()):
        ax_i.text(bar.get_x() + bar.get_width() / 2,
                  bar.get_height() + 5,
                  f"AUC={row['auc_roc']:.3f}",
                  ha='center', va='bottom', color=TEXT, fontsize=7.5)
    ax_i.set_ylabel('AIC (menor es mejor)')
ax_i.set_xlabel('Modelo')
style_ax(ax_i, 'i) Comparación AIC Modelos A, C, D, E')

fig.suptitle(
    'Termografía Infrarroja — Análisis Complementario (Temperatura Normalizada)\n'
    'Variable principal: T_norm = T_regional − T_basal  ·  Covariables: sexo, edad',
    color=TEXT, fontsize=13, fontweight='bold', y=0.985
)

plt.savefig(PNG_RESULTADOS, dpi=160, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"\n  ✓ Figura guardada: {PNG_RESULTADOS}")


# ════════════════════════════════════════════════════════════════════════════
# REPORTE PDF
# ════════════════════════════════════════════════════════════════════════════
reporte_resumen = {
    'n_total'             : len(df),
    'n_con_dolor'         : int(n_con_dolor),
    'n_sin_dolor_p'       : int(n_sin_dolor),
    'n_paired'            : int(len(paired)),
    'asimetria_text'      : asimetria_text,
    'rho_norm_g'          : float(rho_norm_g_rep) if pd.notna(rho_norm_g_rep) else np.nan,
    'p_norm_g'            : float(p_norm_g_rep) if pd.notna(p_norm_g_rep) else np.nan,
    'corr_intensidad_text': corr_intensidad_text,
    'roc_norm_text'       : roc_norm_text,
    'modelo_comp_text'    : modelo_comp_text,
    'masetero_r3_text'    : masetero_r3_text,
    'paciente_alt_text'   : paciente_alt_text,
    'confounders_text'    : confounders_text,
}

generar_reporte_pdf_general(PDF_REPORTE, reporte_resumen, fig_path=PNG_RESULTADOS)
print(f"  ✓ Reporte PDF: {PDF_REPORTE}")


# ════════════════════════════════════════════════════════════════════════════
# CONCLUSIÓN
# ════════════════════════════════════════════════════════════════════════════
print()
print("=" * 72)
print("  CONCLUSIÓN ESTADÍSTICA — ANÁLISIS COMPLEMENTARIO")
print("=" * 72)

res_norm_g_conc = next((r for r in norm_corr_results if r['nivel'] == 'Global'), {})
sig_norm = [r for r in norm_corr_results if r.get('significativo')]

if res_norm_g_conc:
    print(f"\n  Obj. b_alt — Correlación global (T_norm): ρ={res_norm_g_conc.get('rho', np.nan):+.3f}")
    print(f"               p_bonf = {res_norm_g_conc.get('p_bonf', np.nan):.4f} → "
          f"{'Significativa ★' if res_norm_g_conc.get('significativo') else 'No significativa'}")

if sig_norm:
    print("\n  Subgrupos significativos (temp_norm, tras Bonferroni):")
    for r in sig_norm:
        if r['nivel'] == 'Global':
            continue
        print(f"    {r['nivel']} {r['subgrupo']}: ρ={r['rho']:+.3f}  p_bonf={r['p_bonf']:.4f}")
else:
    print("\n  Ningún subgrupo mantuvo significancia tras Bonferroni con temperatura normalizada.")

print(f"\n  Confundidores detectados:")
print(f"    Sexo: {n_sig_sexo} regiones significativas")
print(f"    Edad: {n_sig_edad} regiones significativas")

if model_comparison_rows:
    mc_auc = [(r['modelo'], r['auc_roc']) for r in model_comparison_rows if pd.notna(r.get('auc_roc'))]
    if mc_auc:
        best = max(mc_auc, key=lambda x: x[1])
        print(f"\n  Mejor modelo por AUC: Modelo {best[0]} (AUC={best[1]:.4f})")

print()
print("=" * 72)
print("  ARCHIVOS GENERADOS EN:", OUTPUT_DIR)
print("=" * 72)
outputs = [
    'asimetria_derecha_basal_complementario.csv',
    'correlaciones_normalizadas_complementario.csv',
    'correlaciones_intensidad_continua_complementario.csv',
    'resultados_roc_normalizados_complementario.csv',
    'resultados_logit_extendido_complementario.csv',
    'comparacion_modelos_completa_complementario.csv',
    'analisis_subgrupo_masetero_r3_complementario.csv',
    'resultados_paciente_n45_alterno_complementario.csv',
    'efectos_sexo_edad_temperatura_complementario.csv',
    'analisis_termografia_alterno_complementario.png',
    'resultados_analisis_termografia_alterno_complementario.pdf',
]
for f in outputs:
    path_f = os.path.join(OUTPUT_DIR, f)
    exists = os.path.exists(path_f)
    print(f"  {'✓' if exists else '✗'} {f}")
