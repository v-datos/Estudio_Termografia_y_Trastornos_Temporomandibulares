"""
================================================================================
ANÁLISIS ESTADÍSTICO — TERMOGRAFÍA INFRARROJA Y DOLOR OROFACIAL
================================================================================
Propósito:
  Evaluar si la asimetría térmica pre-palpación (ΔT = |T_derecha − T_izquierda|,
  tomada de las primeras imágenes termográficas) se correlaciona con los puntos
  de dolor reportados durante el examen clínico de palpación muscular.

  La variable de agrupación principal es la presencia de dolor reportado por el
  propio paciente, no el diagnóstico clínico de TTM. Esto es metodológicamente
  más directo con los objetivos del estudio: correlacionar temperatura con dolor,
  no con una etiqueta diagnóstica externa.

Objetivos específicos implementados:
  a) Identificar las zonas de mayor asimetría térmica y los puntos anatómicos
     con mayor intensidad de dolor.
  b) Correlacionar ΔT asimetría con intensidad de dolor (Spearman + permutación
     + Bonferroni para múltiples comparaciones).
  c) Evaluar la utilidad diagnóstica: curvas ROC (AUC, T*, Youden J) y
     curvas Precision-Recall para manejar el desequilibrio de clases.
  d) Ajustar un modelo logístico con SE robustos por clúster y pesos de clase
     balanceados para cuantificar la asociación ajustada temperatura-dolor.
  e) Comparar modelos: Modelo A (temperatura regional) vs. Modelo B
     (regional + temperatura puntual p1-p7), usando LRT y AIC.
  f) Análisis a nivel de paciente (N=43): Mann-Whitney U (ΔT Max en pacientes
     Con Dolor vs Sin Dolor) y ROC para predicción de presencia de dolor.

Decisiones metodológicas clave:
  - Variable térmica principal: ΔT asimetría regional (delta_t_r{N}_primera),
    que usa el lado contralateral como control interno por paciente. Más robusto
    que la temperatura absoluta, que varía entre individuos y sesiones.
  - Fuente temporal: temperaturas de las PRIMERAS imágenes (pre-palpación),
    evitando el sesgo térmico por contacto manual. El dolor se registra de las
    segundas imágenes (durante el examen clínico).
  - Agrupación: 'Con Dolor' (≥1 punto con intensidad > 0) vs 'Sin Dolor',
    en lugar de TTM/Control. El diagnóstico TTM está en los datos pero no es
    la variable de agrupación principal del análisis.
  - Corrección Bonferroni aplicada sobre TODAS las pruebas de correlación
    (global + 4 regiones + 4 músculos + puntos anatómicos individuales).

Guía rápida de uso:
  1) Ejecutar la cadena de preprocesamiento:
       python proceso_de_datos.py --modo merge
       python unificar_dolor_termografia.py   ← genera columnas temp_punto__*
  2) Archivo de entrada requerido:
       datos_finales_termografia_procesados_todas_fotos.csv
  3) Ejecutar:
       python analisis_termografia.py
  4) Salidas principales:
       - resultados_analisis_termografia.pdf  (Reporte ejecutivo PDF)
       - analisis_termografia_ttm_primera.png           (Panel visual 3×3)
       - correlaciones_globales.csv       (Spearman: global / región / músculo)
       - correlaciones_por_punto.csv      (Puntos significativos tras Bonferroni)
       - resultados_roc.csv               (AUC-ROC, AUC-PR, T*, Sens, Spec)
       - resultados_logit_cluster.csv     (Coeficientes y OR del Modelo A)
       - metricas_logit_cluster.csv       (AUC, F1, Brier, Pseudo-R² del Modelo A)
       - comparacion_modelos_a_vs_b_temp_punto.csv (LRT Modelo A vs B)
       - resultados_delta_termico.csv     (Correlación y ROC con ΔT por región)
       - resultados_temp_punto.csv        (Correlación y ROC con temp puntual)
       - concordancia_kappa.csv           (Kappa de Cohen y % concordancia)
       - temperatura_por_region_grupo.csv (Medias por región: Con/Sin Dolor)
       - top_puntos_dolorosos.csv         (Top-10 puntos con mayor dolor)
       - resultados_paciente_n45.csv      (Dataset agregado por paciente)

Notas metodológicas:
  - Spearman: apropiado para variables ordinales (dolor 0-10). P-valores por
    prueba de permutación con barajeo a nivel de bloque de paciente para
    respetar la dependencia intra-sujeto. Seed fijo: rng = random.Random(42).
  - Bonferroni: corrección sobre todas las pruebas de correlación conjuntamente.
    Los p-valores de ROC y métricas de modelo NO se ajustan (son métricas de
    rendimiento, no pruebas de hipótesis múltiples independientes).
  - Logística: errores estándar robustos (cov_type='cluster', groups=muestra).
    Pesos de clase balanceados: w_pos = N/(2·n_pos), w_neg = N/(2·n_neg).
    Modelo simplificado: tiene_dolor ~ temp_max + region + lado (sin grupo_ttm,
    ya que la variable de agrupación es el dolor mismo, no un diagnóstico).
  - Youden J: J = Sens + Spec − 1. Se excluye el punto trivial J[0] = −∞.
  - Nivel de paciente: Mann-Whitney U bilateral (no paramétrico, apropiado
    para N pequeño). ROC con ΔT_max como predictor de presencia de dolor.
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
import statsmodels.api as sm
import textwrap
import random
from statsmodels.stats.multitest import multipletests

from constantes import PAIN_COL_REGEX

# Seed para reproducibilidad de permutaciones
random.seed(42)
np.random.seed(42)

warnings.filterwarnings('ignore')

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
    # Los bloques se construyen una vez fuera del bucle (correcta separación por clúster,
    # sin asumir tamaño de bloque uniforme).
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
    Genera un reporte PDF legible para público general.

    Incluye:
    - Resumen de muestra y objetivo del análisis.
    - Resultados principales con interpretación no técnica.
    - Glosario rápido para entender p, rho, AUC, OR.
    - Figura resumen si está disponible.

    Parameters
    ----------
    path_pdf : str
        Ruta de destino del PDF generado.
    resumen : dict
        Diccionario con los resultados del análisis. Claves requeridas:

        Muestra
        -------
        n_total       : int  — Número total de pacientes.
        n_con_dolor   : int  — Pacientes que reportaron dolor (≥1 punto).
        n_sin_dolor_p : int  — Pacientes sin ningún punto de dolor.
        n_paired      : int  — Observaciones pareadas (muestra × punto).
        n_dolor       : int  — Puntos con presencia de dolor.
        n_sin_dolor   : int  — Puntos sin dolor.
        top5_text     : str  — Descripción de los 5 puntos más dolorosos.

        Correlación (Obj. b)
        --------------------
        rho_g : float  — ρ de Spearman global (ΔT vs intensidad).
        p_g   : float  — P-valor de la correlación global (permutación + Bonferroni).

        Concordancia espacial (Obj. b2)
        --------------------------------
        kappa_text : str  — Kappa de Cohen formateado (e.g., "-0.008").
        conc_text  : str  — Porcentaje de concordancia directa (e.g., "58.1%").

        Análisis de paciente (N=43)
        ---------------------------
        paciente_roc_text : str — AUC, T*, Sens, Spec para predicción de dolor.

        Utilidad diagnóstica ROC (Obj. c)
        ----------------------------------
        auc_text   : str  — AUC global formateado.
        auc_interp : str  — Interpretación clínica del AUC.
        tstar_text : str  — Umbral óptimo T* (con unidad °C).
        sens_text  : str  — Sensibilidad en T*.
        spec_text  : str  — Especificidad en T*.

        Modelos (Obj. c2 y extensión)
        ------------------------------
        logit_text           : str — Resumen del modelo logístico A (OR, IC95%, p).
        model_comp_text      : str — Comparación Modelo A vs B (LRT, ΔAUC).
        temp_punto_text      : str — Spearman temperatura puntual vs dolor.
        temp_punto_roc_text  : str — ROC con temperatura puntual.
        delta_text           : str — Spearman ΔT global vs dolor.
        delta_roc_text       : str — ROC con ΔT global.

    fig_path : str or None, optional
        Ruta a una imagen PNG que se incluye como última página del PDF.
        Si es None, no se añade figura.
    """
    with PdfPages(path_pdf) as pdf:
        _draw_text_page(
            pdf,
            "Reporte de Resultados: Termografía y Dolor Orofacial",
            [
                ("Contexto del estudio", [
                    "Este reporte resume, en lenguaje simple, los hallazgos estadísticos del análisis de "
                    "termografía infrarroja y su relación con el dolor reportado durante la palpación muscular.",
                    f"Pacientes analizados: {resumen['n_total']} "
                    f"(Con Dolor={resumen['n_con_dolor']}, Sin Dolor={resumen['n_sin_dolor_p']}).",
                    f"Observaciones pareadas (muestra × punto anatómico): {resumen['n_paired']}.",
                    f"Puntos con dolor: {resumen['n_dolor']} | Puntos sin dolor: {resumen['n_sin_dolor']}.",
                ]),
                ("Distribución de la muestra", [
                    "Se compararon temperaturas máximas por región entre pacientes Con Dolor y Sin Dolor.",
                    "Se identificaron los puntos anatómicos con mayor intensidad media de dolor en pacientes con reporte de dolor.",
                    f"Top-5 puntos dolorosos: {resumen['top5_text']}",
                ]),
            ]
        )

        _draw_text_page(
            pdf,
            "Resultados principales e interpretación",
            [
                ("Obj. b — Correlación asimetría térmica (ΔT) ↔ dolor", [
                    f"Resultado global: rho={resumen['rho_g']:+.3f}, p={resumen['p_g']:.4f}.",
                    f"Interpretación: {interpretar_rho(resumen['rho_g'])}. {p_sig_label(resumen['p_g'])}.",
                    "Una correlación positiva indicaría que, a mayor asimetría térmica, mayor intensidad de dolor.",
                    "La corrección de Bonferroni fue aplicada sobre todas las pruebas de correlación.",
                ]),
                ("Obj. b2 — Concordancia espacial", [
                    f"Kappa de Cohen: {resumen['kappa_text']}.",
                    f"Concordancia directa zona más caliente vs zona más dolorosa: {resumen['conc_text']}.",
                    "Evalúa si las zonas de mayor asimetría térmica coinciden con las zonas de mayor dolor.",
                ]),
                ("Análisis de paciente (N=43) — Predicción de dolor", [
                    "Se evalúa si la asimetría térmica máxima (ΔT Max) predice si el paciente reportará dolor.",
                    resumen.get('paciente_roc_text', 'No evaluado.')
                ]),
                ("Obj. c — Utilidad diagnóstica (ROC)", [
                    f"AUC global: {resumen['auc_text']} ({resumen['auc_interp']}).",
                    f"Umbral térmico sugerido (T*): {resumen['tstar_text']}.",
                    f"Sensibilidad: {resumen['sens_text']} | Especificidad: {resumen['spec_text']}.",
                ]),
                ("Obj. c2 — Modelo multivariable (regresión logística)", [
                    resumen['logit_text'],
                    "Modelo: tiene_dolor ~ temp_max + region + lado. SE robustos por clúster. Pesos balanceados.",
                ]),
                ("Extensión con temperatura puntual (Modelo A vs B)", [
                    resumen['model_comp_text'],
                ]),
                ("Temperatura puntual mapeada (p1..p7)", [
                    resumen['temp_punto_text'],
                    resumen['temp_punto_roc_text'],
                ]),
                ("Análisis de aumento térmico (Delta T)", [
                    resumen['delta_text'],
                    resumen['delta_roc_text'],
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
                ax.set_title("Figura resumen del análisis estadístico", fontsize=14, pad=10)
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

# ════════════════════════════════════════════════════════════════════════════
# 0. CONFIGURACIÓN Y CARGA DE DATOS
# ════════════════════════════════════════════════════════════════════════════
# Archivo unificado: temperaturas primeras fotos + dolor segundas fotos.
# Generado por: proceso_de_datos.py --modo merge + unificar_dolor_termografia.py
CSV_ENTRADA = 'datos/datos_finales_termografia_procesados_todas_fotos.csv'

# Salidas principales
OUTPUT_DIR = 'resultados'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

PNG_RESULTADOS = os.path.join(OUTPUT_DIR, 'analisis_termografia_ttm_primera.png')
PDF_REPORTE    = os.path.join(OUTPUT_DIR, 'resultados_analisis_termografia.pdf')

# Variable térmica principal usada en correlación y ROC.
# Es la asimetría térmica regional: |T_media_derecha − T_media_izquierda|
# de las primeras imágenes (pre-palpación), sufijo '_primera'.
# Las columnas reales en el CSV tienen el formato: delta_t_r{N}_primera
# (e.g., delta_t_r1_primera, delta_t_r2_primera, ...).
# Esta constante documenta el patrón; la selección real ocurre en delta_col_name().
VAR_PRINCIPAL_PATRON = 'delta_t_r{N}_primera'

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
    print(f"Columnas disponibles: {list(df.columns)}")
    import sys; sys.exit(1)

PAIN_COLS  = seleccionar_columnas_dolor(df)
TEMP_COLS  = [c for c in df.columns if 'temperatura maxima' in c and 'imagen' not in c]
TEMP_PUNTO_COLS = [c for c in df.columns if str(c).startswith('temp_punto__')]

# NaN en dolor → 0 (ausencia de dolor)
df[PAIN_COLS] = df[PAIN_COLS].fillna(0)

# ── Definición del grupo principal: Con Dolor vs Sin Dolor ───────────────────
# La variable de agrupación es si el paciente reportó al menos un punto con
# intensidad de dolor > 0 durante el examen clínico. Esto es más directo con
# los objetivos del estudio (correlacionar temperatura con dolor) que usar el
# diagnóstico clínico de TTM como variable de agrupación.
#
# Nota: el diagnóstico TTM ('diagnosticado con ttm') sigue disponible en df
# para análisis secundarios o descriptivos, pero NO se usa como grupo principal.
_dolor_por_paciente = df.set_index(ID_COL)[PAIN_COLS].max(axis=1) > 0
_pacientes_con_dolor = set(_dolor_por_paciente[_dolor_por_paciente].index)
df['grupo'] = np.where(
    df[ID_COL].isin(_pacientes_con_dolor),
    'Con Dolor', 'Sin Dolor'
)

n_con_dolor = (df['grupo'] == 'Con Dolor').sum()
n_sin_dolor = (df['grupo'] == 'Sin Dolor').sum()
# Mantener conteo TTM como información descriptiva secundaria
n_ttm     = df['diagnosticado con ttm'].astype(str).str.strip().str.lower().isin(['si','sí','yes','1']).sum()
n_control = len(df) - n_ttm

print("=" * 72)
print("  TERMOGRAFÍA IR — ANÁLISIS DE CORRELACIÓN TEMPERATURA-DOLOR")
print("=" * 72)
print(f"  Pacientes totales     : {len(df)}")
print(f"  Pacientes con dolor   : {n_con_dolor}  (reportaron ≥1 punto durante palpación)")
print(f"  Pacientes sin dolor   : {n_sin_dolor}")
print(f"  [Ref. diagnóstico TTM]: {n_ttm} TTM | {n_control} Control (variable secundaria)")
print(f"  Columnas de dolor detectadas (regex anatómico): {len(PAIN_COLS)}")
print(f"  Columnas temp_punto detectadas: {len(TEMP_PUNTO_COLS)}")

# ── Parser de columnas de dolor ──────────────────────────────────────────────
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
    # Si no se detecta lado, no forzar un valor por defecto (evita mapeos erróneos).
    lado_match = re.search(r'(derech[oa]|izquierd[oa])', col)
    if not lado_match:
        return None
    lado_raw = lado_match.group(1)
    lado = 'derecha' if lado_raw.startswith('derech') else 'izquierda'
    return musculo, lado, region, punto

def temp_col_name(region: str, lado: str) -> str:
    """
    Construye el nombre de la columna de temperatura máxima regional del CSV.

    Se usa el sufijo `_primera` para referenciar las imágenes termográficas
    tomadas ANTES de la palpación, evitando el sesgo térmico producido por
    el contacto manual durante la exploración clínica.

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

    ΔT = Temperatura_lado_derecho − Temperatura_lado_izquierdo para la región
    dada. Un valor positivo indica mayor temperatura en el lado derecho.
    El sufijo `_primera` indica que corresponde a la imagen pre-palpación.

    Parameters
    ----------
    region : str   — Código de región (e.g., 'r1'). Se convierte a minúsculas.

    Returns
    -------
    str  — Nombre de columna. Ejemplo: 'delta_t_r1_primera'
    """
    return f"delta_t_{region.lower()}_primera"


def temp_punto_col_name(col_dolor: str) -> str:
    """
    Devuelve el nombre de la columna de temperatura puntual mapeada.

    El mapeo anatómico punto-de-dolor → píxel-térmico se define en
    `unificar_dolor_termografia.py` y genera columnas con el prefijo
    `temp_punto__` seguido del nombre de la columna de dolor original.

    Parameters
    ----------
    col_dolor : str   — Nombre de columna de dolor (e.g., 'masetero_derecho_r3p2').

    Returns
    -------
    str  — Nombre de columna térmica puntual (e.g., 'temp_punto__masetero_derecho_r3p2').
    """
    return f"temp_punto__{col_dolor}"


def ajustar_logit_cluster(data: pd.DataFrame, incluir_temp_punto: bool = False, col_temp_max: str = 'temp_max') -> dict:
    """
    Ajusta un modelo de regresión logística con errores estándar robustos
    por clúster de paciente (sandwich estimator).

    Este enfoque es necesario porque cada paciente aporta múltiples
    observaciones (una por cada punto anatómico evaluado), lo que viola el
    supuesto de independencia de la logística estándar. Al especificar
    `cov_type='cluster'` con `groups=muestra`, los errores estándar se
    corrigen para la correlación intra-sujeto sin eliminar datos.

    Fórmula del modelo (Modelo A):
        tiene_dolor ~ const + temp_max + region (dummies) + lado (dummies)

    La variable de grupo (Con Dolor / Sin Dolor) NO se incluye como predictor
    porque es la variable de respuesta recodificada a nivel de observación:
    `tiene_dolor` ya captura si ese punto específico fue doloroso. Incluir
    la agrupación del paciente crearía redundancia con la respuesta.

    Si `incluir_temp_punto=True` (Modelo B), se añade `temp_punto` como
    predictor adicional para evaluar si la temperatura exactamente en el
    píxel del punto de palpación mejora la predicción respecto a la
    temperatura máxima regional.

    Manejo del desequilibrio de clases:
        La variable respuesta (`tiene_dolor`) está desbalanceada (~4.7% positivos).
        Se aplican pesos de clase inversos a la frecuencia:

            w_pos = (n_neg + n_pos) / (2 · n_pos)   [peso para clase=1]
            w_neg = (n_neg + n_pos) / (2 · n_neg)   [peso para clase=0]

        Esto es equivalente al parámetro `class_weight='balanced'` de
        scikit-learn y hace que cada clase contribuya igual al log-likelihood,
        penalizando más los errores sobre los puntos dolorosos (minoritarios).

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame con las observaciones pareadas. Debe contener las columnas:
        'temp_max', 'region', 'lado', 'tiene_dolor', 'muestra'.
        Si `incluir_temp_punto=True`, también debe contener 'temp_punto'.
    incluir_temp_punto : bool, optional
        Si True, incluye temperatura puntual mapeada (p1..p7) como predictor
        adicional (Modelo B). Por defecto False (Modelo A).

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

    x_in = d[base_cols]
    X = pd.get_dummies(x_in, columns=['region', 'lado'], drop_first=True, dtype=float)
    X = sm.add_constant(X, has_constant='add')
    y = d['tiene_dolor'].astype(float)
    clusters = d['muestra']

    # Validación: se requieren ambas clases para ajustar el modelo
    if y.nunique() < 2:
        out['error'] = (
            f"Variable respuesta con una sola clase presente ({y.unique()}). "
            f"Se requieren observaciones con dolor (1) y sin dolor (0). "
            f"Posible causa: submuestra sin puntos dolorosos."
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

# ── Construcción del dataset pareado (muestra × punto doloroso) ──────────────
# Este DataFrame es la unidad de análisis de TODOS los objetivos posteriores.
# Cada fila = un punto anatómico de palpación de un paciente específico.
# Dimensionalidad esperada: n_pacientes × n_PAIN_COLS filas.
#
# Variables incluidas por registro:
#   muestra      — Identificador de paciente (clave de clúster en logit/permutación)
#   grupo        — 'Con Dolor' o 'Sin Dolor' (agrupación principal del estudio)
#   musculo      — Músculo evaluado (atm, masetero, temporal, esternocleidomastoideo)
#   lado         — 'derecha' o 'izquierda'
#   region       — Zona termográfica asociada (r1..r4)
#   punto        — Sub-punto de palpación (p1..p7)
#   temp_max     — Temperatura máxima regional (°C), PRIMERAS imágenes (pre-palpación)
#   delta_t      — Asimetría térmica ΔT = T_media_der − T_media_izq (°C), primeras fotos
#                  Puede ser negativa (mayor temperatura en lado izquierdo). El nombre
#                  "asimetría" refleja la magnitud relativa, no un valor absoluto.
#   intensidad   — Intensidad de dolor reportada (0–10); 0 si NaN (ausencia de dolor)
#   tiene_dolor  — Binaria: 1 si intensidad > 0, 0 en caso contrario
#   temp_punto   — Temperatura exacta en el píxel del punto anatómico.
#                  NaN actualmente: las columnas temp_punto__ provienen de las
#                  segundas fotos, pero las coordenadas de píxel no han sido
#                  re-extraídas de las primeras imágenes. Para activar este
#                  análisis, generar columnas temp_punto_primera__ en
#                  unificar_dolor_termografia.py.
#
# Filas excluidas: puntos sin temperatura regional válida o sin delta_t disponible.
records = []
for _, row in df.iterrows():
    for col in PAIN_COLS:
        parsed = parse_pain_col(col)
        if parsed is None:
            continue
        musculo, lado, region, punto = parsed
        
        # Temperatura regional (Foto 1)
        col_temp = temp_col_name(region, lado)
        if col_temp not in df.columns:
            continue

        # Temperatura regional (Foto 2) - Para Obj. e
        col_temp_segunda = temp_col_name_segunda(region, lado)
        temp_segunda_val = row[col_temp_segunda] if col_temp_segunda in df.columns else np.nan

        # Temperatura puntual (píxel específico) - Para Obj. e
        col_temp_punto = temp_punto_col_name(col)
        temp_punto_val = row[col_temp_punto] if col_temp_punto in df.columns else np.nan
        
        # Asimetría térmica regional (Foto 1)
        col_delta = delta_col_name(region)
        if col_delta not in df.columns:
            continue
            
        temp_val  = row[col_temp]
        delta_val = row[col_delta]
        dolor_val = float(row[col])
        
        if pd.isna(temp_val) or pd.isna(delta_val):
            continue
            
        records.append({
            'muestra'   : row['numero de muestra'],
            'grupo'     : row['grupo'],
            'musculo'   : musculo,
            'lado'      : lado,
            'region'    : region,
            'punto'     : punto,
            'temp_max'  : float(temp_val),
            'temp_max_segunda': float(temp_segunda_val),
            'delta_t'   : float(delta_val),
            'intensidad': dolor_val,
            'tiene_dolor': 1 if dolor_val > 0 else 0,
            'temp_punto': float(temp_punto_val),
        })

paired = pd.DataFrame(records)
print(f"\n  Observaciones pareadas : {len(paired)}")
print(f"  Puntos CON dolor       : {paired['tiene_dolor'].sum()}")
print(f"  Puntos SIN dolor       : {(paired['tiene_dolor']==0).sum()}\n")

# ════════════════════════════════════════════════════════════════════════════
# OBJETIVO ESPECÍFICO a — Zonas de mayor temperatura y puntos dolorosos
# ════════════════════════════════════════════════════════════════════════════
print("─" * 72)
print("  OBJ. a — Temperaturas máximas promedio por región y grupo")
print("─" * 72)

temp_region_grupo = (
    paired.groupby(['grupo', 'region'])['temp_max']
    .mean().unstack('grupo').round(2)
)
# Diferencia Con Dolor − Sin Dolor (interpretación directa del efecto)
if 'Con Dolor' in temp_region_grupo.columns and 'Sin Dolor' in temp_region_grupo.columns:
    temp_region_grupo['Δ (Con Dolor − Sin Dolor)'] = (
        temp_region_grupo['Con Dolor'] - temp_region_grupo['Sin Dolor']
    ).round(3)
print(temp_region_grupo.to_string())

print("\n  Top-5 puntos anatómicos con mayor intensidad media de dolor:")
# Todos los pacientes que reportaron dolor (sin filtrar por diagnóstico)
top_pain = (
    paired[paired['tiene_dolor'] == 1]
    .groupby(['musculo','lado','region','punto'])['intensidad']
    .mean().sort_values(ascending=False).head(5)
)
print(top_pain.to_string())

# ════════════════════════════════════════════════════════════════════════════
# OBJETIVO ESPECÍFICO b — Correlación Asimetría Térmica ↔ dolor (Spearman)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 72)
print("  OBJ. b — Correlación Spearman: Asimetría Térmica (Delta T) ↔ intensidad dolor")
print("  (Ajuste de Bonferroni aplicado para comparaciones múltiples)")
print("─" * 72)

corr_results = []

# 1. Recolección de p-valores para corrección múltiple
p_values_to_correct = []
test_keys = [] # Para mapear resultados después

# Global
rho_g, p_g = spearman_permutation_test(paired, 'delta_t', 'intensidad')
p_values_to_correct.append(p_g)
test_keys.append(('Global', 'Todos', rho_g))

# Por región
for reg, grp in paired.groupby('region'):
    r, p = spearman_permutation_test(grp, 'delta_t', 'intensidad')
    p_values_to_correct.append(p)
    test_keys.append(('Región', reg, r))

# Por músculo
for musc, grp in paired.groupby('musculo'):
    if len(grp) < 5:
        continue
    r, p = spearman_permutation_test(grp, 'delta_t', 'intensidad')
    p_values_to_correct.append(p)
    test_keys.append(('Músculo', musc, r))

# Por columna específica (Puntos anatómicos)
col_test_indices = []
for col in PAIN_COLS:
    parsed = parse_pain_col(col)
    if parsed is None:
        continue
    musculo, lado, region, punto = parsed
    col_temp = temp_col_name(region, lado)
    if col_temp not in df.columns:
        continue
    temps  = df[col_temp].dropna()
    dolors = df[col].loc[temps.index].astype(float)
    if dolors.max() == 0 or len(temps) < 5:
        continue
    r, p = stats.spearmanr(temps, dolors)
    p_values_to_correct.append(p)
    col_test_indices.append(len(test_keys))
    test_keys.append(('Punto', col.upper(), r, col_temp))

# 2. Aplicar Corrección de Bonferroni
if p_values_to_correct:
    reject, p_corrected, _, alphacBonf = multipletests(
        p_values_to_correct, alpha=0.05, method='bonferroni'
    )
else:
    reject, p_corrected = [], []

# 3. Organizar y mostrar resultados
for i, (key, p_corr, is_sig) in enumerate(zip(test_keys, p_corrected, reject)):
    nivel = key[0]
    subgrupo = key[1]
    rho = key[2]
    
    p_orig = p_values_to_correct[i]
    sig_mark = '★' if is_sig else '✗'
    
    res = {
        'nivel': nivel,
        'subgrupo': subgrupo,
        'rho': rho,
        'p_orig': p_orig,
        'p_bonf': p_corr,
        'significativo': is_sig
    }
    corr_results.append(res)
    
    if nivel == 'Global':
        print(f"\n  [GLOBAL]  ρ = {rho:+.4f}   p_orig = {p_orig:.4f}   p_bonf = {p_corr:.4f}  {sig_mark}")
    elif nivel == 'Región':
        if subgrupo == paired['region'].unique()[0]:
            print("\n  Por región (p-valores permutados + Bonferroni):")
        print(f"    {subgrupo.upper():<10} ρ = {rho:+.4f}   p_orig = {p_orig:.4f}   p_bonf = {p_corr:.4f}  {sig_mark}")
    elif nivel == 'Músculo':
        # Encontrar el primer músculo que entró en test_keys
        first_musc = next(k[1] for k in test_keys if k[0] == 'Músculo')
        if subgrupo == first_musc:
            print("\n  Por músculo (p-valores permutados + Bonferroni):")
        print(f"    {subgrupo:<40} ρ = {rho:+.4f}   p_orig = {p_orig:.4f}   p_bonf = {p_corr:.4f}  {sig_mark}")

# Mostrar puntos anatómicos significativos (si queda alguno)
print("\n  Por punto anatómico específico (p_bonf < 0.05):")
col_corrs = []
for i in col_test_indices:
    key = test_keys[i]
    p_corr = p_corrected[i]
    is_sig = reject[i]
    if is_sig:
        col_corrs.append({
            'Punto': key[1],
            'ROI Térmico': key[3],
            'ρ': round(key[2], 3),
            'p_orig': round(p_values_to_correct[i], 4),
            'p_bonf': round(p_corr, 4),
            'Sig': '★'
        })

if col_corrs:
    df_col_corrs = pd.DataFrame(col_corrs)
    print(df_col_corrs.to_string(index=False))
else:
    print("    No se encontraron correlaciones significativas tras corrección de Bonferroni.")

# ════════════════════════════════════════════════════════════════════════════
# OBJETIVO ESPECÍFICO b2 — Concordancia espacial (Kappa de Cohen)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 72)
print("  OBJ. b2 — Concordancia espacial: zona más caliente ↔ región más dolorosa")
print("─" * 72)

kappa_records = []
for (muestra, lado), grp in paired.groupby(['muestra', 'lado']):
    if grp.empty:
        continue
    hot_region = grp.loc[grp['temp_max'].idxmax(), 'region']
    pain_grp = grp[grp['tiene_dolor'] == 1]
    if pain_grp.empty:
        continue
    painful_region = pain_grp.loc[pain_grp['intensidad'].idxmax(), 'region']
    concordante = 1 if hot_region == painful_region else 0
    kappa_records.append({
        'hot_region': hot_region,
        'pain_region': painful_region,
        'concordancia': concordante,
    })

kdf = pd.DataFrame(kappa_records)
if len(kdf) >= 2:
    kappa_val = cohen_kappa_score(kdf['hot_region'], kdf['pain_region'])
    concordancia_pct = kdf['concordancia'].mean() * 100
    print(f"  κ de Cohen = {kappa_val:+.4f}")
    print(f"  Concordancia directa = {concordancia_pct:.1f}%")
else:
    kappa_val = np.nan
    concordancia_pct = np.nan
    print("  Datos insuficientes para calcular Kappa.")

# ════════════════════════════════════════════════════════════════════════════
# OBJETIVO ESPECÍFICO c — Termografía como herramienta diagnóstica (ROC)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 72)
print("  OBJ. c — Utilidad diagnóstica: Curva ROC + Umbral T* (Índice de Youden)")
print("─" * 72)

def calc_roc(data: pd.DataFrame, label: str = '', score_col: str = 'delta_t') -> dict | None:
    """
    Calcula curva ROC, curva Precision-Recall y umbral óptimo T*.

    El umbral óptimo T* se determina maximizando el Índice de Youden:
        J = Sensibilidad + Especificidad − 1
    Se excluye el punto inicial de la curva ROC (J[0] = −∞) para evitar
    seleccionar el umbral trivial donde todo se clasifica como positivo.

    Parameters
    ----------
    data : pd.DataFrame
        Debe contener la columna binaria 'tiene_dolor' (0/1) y la columna
        `score_col` con valores numéricos continuos. Las filas con NaN en
        `score_col` se eliminan automáticamente.
    label : str, optional
        Etiqueta del subgrupo para identificación en reportes y gráficas.
    score_col : str, optional
        Columna numérica usada como score predictor. Por defecto 'delta_t'
        (asimetría térmica regional). Puede ser 'temp_max', 'temp_punto', etc.

    Returns
    -------
    dict | None
        None si no hay datos, o si la variable respuesta es constante
        (todas las observaciones pertenecen a la misma clase).
        Si hay datos válidos, retorna un diccionario con:
        'label', 'auc', 'pr_auc', 'T_star', 'sensitivity', 'specificity',
        'youden_J', 'fpr', 'tpr' (arrays para graficar ROC),
        'prec_pts', 'rec_pts' (arrays para graficar Precision-Recall).
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
    
    # ROC
    fpr, tpr, thresholds = roc_curve(y, s)
    thresholds = np.clip(thresholds, None, 1e6)
    roc_auc = auc(fpr, tpr)
    
    # Precision-Recall
    prec_pts, rec_pts, _ = precision_recall_curve(y, s)
    pr_auc = auc(rec_pts, prec_pts) # AUC de la curva PR
    
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

roc_global  = calc_roc(paired, 'Global')
roc_by_reg  = {r: calc_roc(g, r) for r, g in paired.groupby('region')}
roc_by_musc = {m: calc_roc(g, m) for m, g in paired.groupby('musculo') if len(g) >= 10}
roc_by_musc = {k: v for k, v in roc_by_musc.items() if v}

all_rocs = ([roc_global] if roc_global else []) + \
           [v for v in roc_by_reg.values() if v] + \
           list(roc_by_musc.values())

print(f"\n  {'Subgrupo':<42} {'AUC':>5}  {'PR-AUC':>6}  {'T* (°C)':>8}  {'Sens':>5}  {'Spec':>5}")
print("  " + "─" * 78)
for r in all_rocs:
    interp = '►' if r['auc'] >= 0.65 else ''
    print(f"  {r['label']:<42} {r['auc']:>5.3f}  {r['pr_auc']:>6.3f}  {r['T_star']:>8.2f}  "
          f"{r['sensitivity']:>5.3f}  {r['specificity']:>5.3f} {interp}")

# ════════════════════════════════════════════════════════════════════════════
# OBJETIVO ESPECÍFICO c2 — Regresión logística robusta por clúster (muestra)
# ════════════════════════════════════════════════════════════════════════════
# Modelo A (todas las observaciones):
#   tiene_dolor ~ const + temp_max + region (dummies) + lado (dummies)
#
# Se usa temp_max (temperatura absoluta) en lugar de delta_t porque a nivel
# de punto individual la temperatura puntual absoluta es el predictor natural,
# mientras que delta_t es más apropiado para comparaciones inter-individuales.
# El grupo (Con Dolor / Sin Dolor) NO se incluye como predictor porque es
# la variable de respuesta recodificada — su inclusión sería redundante.
#
# SE robustos por clúster: corrigen la dependencia intra-sujeto (múltiples
# puntos por paciente) sin requerir un modelo mixto completo.
# Pesos de clase balanceados: compensan el desequilibrio (4.7% positivos).
print("\n" + "─" * 72)
print("  OBJ. c2 — Regresión Logística (SE robustos por clúster de paciente)")
print("─" * 72)

logit_rows = []
logit_metrics = {}
logit_ok = False
model_comparison = {}
model_comp_ok = False

fit_a_full = ajustar_logit_cluster(paired, incluir_temp_punto=False)
if fit_a_full['ok']:
    logit_ok = True
    logit_rows = fit_a_full['rows']
    logit_metrics = fit_a_full['metrics']
    logit_model = fit_a_full['model']
    params = logit_model.params
    pvals = logit_model.pvalues
    conf = logit_model.conf_int()
    conf.columns = ['ci_low', 'ci_high']

    print(f"  N observaciones                : {logit_metrics['n_obs']}")
    print(f"  Pseudo-R² (McFadden)           : {logit_metrics['pseudo_r2_mcfadden']:.4f}")
    print(f"  AUC-ROC (probabilidad)         : {logit_metrics['auc_prob_logit']:.4f}")
    print(f"  AUC-PR (Prec-Rec)              : {logit_metrics['pr_auc_logit']:.4f}")
    print(f"  F1-Score (umbral 0.5)          : {logit_metrics['f1_score']:.4f}")
    print(f"  Precisión                      : {logit_metrics['precision']:.4f}")
    print(f"  Recall                         : {logit_metrics['recall']:.4f}")
    print(f"  Brier score                    : {logit_metrics['brier_score']:.4f}")
    if 'temp_max' in params.index:
        print("  Efecto principal temp_max:")
        print(f"    OR={np.exp(params['temp_max']):.3f} "
              f"(IC95% {np.exp(conf.loc['temp_max','ci_low']):.3f}–{np.exp(conf.loc['temp_max','ci_high']):.3f}), "
              f"p={pvals['temp_max']:.4f}")
else:
    print(f"  No fue posible ajustar la regresión logística robusta: {fit_a_full['error']}")
    logit_ok = False

# Comparación A vs B sobre la misma submuestra con temp_punto disponible
# Para el Objetivo e, comparamos regional vs puntual de la misma toma (segunda).
print("\n  Comparación de modelos (A vs B) en submuestra con temp_punto (Foto 2):")
sub_temp = paired.dropna(subset=['temp_punto', 'temp_max_segunda']).copy()
print(f"    N registros con temp_punto: {len(sub_temp)}")
fit_a_sub = ajustar_logit_cluster(sub_temp, incluir_temp_punto=False, col_temp_max='temp_max_segunda')
fit_b_sub = ajustar_logit_cluster(sub_temp, incluir_temp_punto=True, col_temp_max='temp_max_segunda')
if fit_a_sub['ok'] and fit_b_sub['ok']:
    llr = 2 * (fit_b_sub['model'].llf - fit_a_sub['model'].llf)
    df_diff = int(fit_b_sub['model'].df_model - fit_a_sub['model'].df_model)
    lr_p = stats.chi2.sf(llr, df_diff) if df_diff > 0 else np.nan
    model_comparison = {
        'n_submuestra': len(sub_temp),
        'auc_A': fit_a_sub['metrics']['auc_prob_logit'],
        'auc_B': fit_b_sub['metrics']['auc_prob_logit'],
        'delta_auc': fit_b_sub['metrics']['auc_prob_logit'] - fit_a_sub['metrics']['auc_prob_logit'],
        'pseudo_r2_A': fit_a_sub['metrics']['pseudo_r2_mcfadden'],
        'pseudo_r2_B': fit_b_sub['metrics']['pseudo_r2_mcfadden'],
        'delta_pseudo_r2': fit_b_sub['metrics']['pseudo_r2_mcfadden'] - fit_a_sub['metrics']['pseudo_r2_mcfadden'],
        'brier_A': fit_a_sub['metrics']['brier_score'],
        'brier_B': fit_b_sub['metrics']['brier_score'],
        'delta_brier_B_menos_A': fit_b_sub['metrics']['brier_score'] - fit_a_sub['metrics']['brier_score'],
        'aic_A': fit_a_sub['metrics']['aic'],
        'aic_B': fit_b_sub['metrics']['aic'],
        'delta_aic_B_menos_A': fit_b_sub['metrics']['aic'] - fit_a_sub['metrics']['aic'],
        'lr_stat': llr,
        'lr_df': df_diff,
        'lr_p_value': lr_p,
    }
    model_comp_ok = True
    print(f"    AUC A={model_comparison['auc_A']:.4f} vs B={model_comparison['auc_B']:.4f} "
          f"(Δ={model_comparison['delta_auc']:+.4f})")
    print(f"    Pseudo-R² A={model_comparison['pseudo_r2_A']:.4f} vs B={model_comparison['pseudo_r2_B']:.4f} "
          f"(Δ={model_comparison['delta_pseudo_r2']:+.4f})")
    print(f"    AIC A={model_comparison['aic_A']:.2f} vs B={model_comparison['aic_B']:.2f} "
          f"(Δ={model_comparison['delta_aic_B_menos_A']:+.2f})")
    print(f"    LRT (B mejora A): χ²={model_comparison['lr_stat']:.3f}, "
          f"df={model_comparison['lr_df']}, p={model_comparison['lr_p_value']:.4f}")
else:
    msg = fit_a_sub['error'] if not fit_a_sub['ok'] else fit_b_sub['error']
    print(f"    No fue posible comparar modelos A vs B: {msg}")

# ════════════════════════════════════════════════════════════════════════════
# OBJETIVO ESPECÍFICO b3/c3 — Aumento térmico (Delta T) y dolor
# ════════════════════════════════════════════════════════════════════════════
# Este bloque alinea explícitamente el análisis con hipótesis redactadas en
# términos de "aumento de temperatura" (delta_t_*), no solo temp_max absoluta.
print("\n" + "─" * 72)
print("  OBJ. b3/c3 — Aumento térmico (delta_t_*) como indicador de dolor")
print("─" * 72)

delta_results = []

# Correlación global: delta_t vs intensidad de dolor
delta_global_df = paired[['delta_t', 'intensidad', 'tiene_dolor']].dropna()
if len(delta_global_df) >= 3 and delta_global_df['delta_t'].nunique() > 1:
    rho_delta_g, p_delta_g = stats.spearmanr(delta_global_df['delta_t'], delta_global_df['intensidad'])
else:
    rho_delta_g, p_delta_g = np.nan, np.nan

delta_results.append({
    'nivel': 'Global',
    'subgrupo': 'delta_t_regional_primera',
    'metrica': 'spearman_delta_vs_intensidad',
    'valor_1': rho_delta_g,
    'valor_2': p_delta_g,
    'valor_3': np.nan,
    'valor_4': np.nan,
})
print(f"  Spearman ΔT regional vs dolor: ρ={rho_delta_g:+.4f}  p={p_delta_g:.4f}")

# Correlación por región: delta_t vs intensidad
print("  Spearman por región (ΔT_región vs dolor):")
for reg, grp in paired.groupby('region'):
    sub = grp[['delta_t', 'intensidad']].dropna()
    if len(sub) >= 3 and sub['delta_t'].nunique() > 1:
        r, p = stats.spearmanr(sub['delta_t'], sub['intensidad'])
    else:
        r, p = np.nan, np.nan
    delta_results.append({
        'nivel': 'Región',
        'subgrupo': reg,
        'metrica': 'spearman_delta_region_vs_intensidad',
        'valor_1': r,
        'valor_2': p,
        'valor_3': np.nan,
        'valor_4': np.nan,
    })
    sig = '★' if pd.notna(p) and p < 0.05 else '✗'
    print(f"    {reg.upper()}  ρ={r:+.4f}  p={p:.4f}  {sig}")

# ROC global usando delta_t como score
roc_delta_global = calc_roc(paired, 'Delta_Regional', score_col='delta_t')
if roc_delta_global:
    print(f"  ROC ΔT regional: AUC={roc_delta_global['auc']:.3f}, "
          f"T*={roc_delta_global['T_star']:.3f}, "
          f"Sens={roc_delta_global['sensitivity']:.3f}, "
          f"Spec={roc_delta_global['specificity']:.3f}")
    delta_results.append({
        'nivel': 'Global',
        'subgrupo': 'delta_t_regional_primera',
        'metrica': 'roc_delta_global',
        'valor_1': roc_delta_global['auc'],
        'valor_2': roc_delta_global['T_star'],
        'valor_3': roc_delta_global['sensitivity'],
        'valor_4': roc_delta_global['specificity'],
    })
else:
    print("  ROC ΔT regional: datos insuficientes.")

# ROC por región usando delta_t
for reg, grp in paired.groupby('region'):
    roc_reg = calc_roc(grp, f'Delta_{reg.upper()}', score_col='delta_t')
    if not roc_reg:
        continue
    delta_results.append({
        'nivel': 'Región',
        'subgrupo': reg,
        'metrica': 'roc_delta_region',
        'valor_1': roc_reg['auc'],
        'valor_2': roc_reg['T_star'],
        'valor_3': roc_reg['sensitivity'],
        'valor_4': roc_reg['specificity'],
    })

# ════════════════════════════════════════════════════════════════════════════
# OBJETIVO ESPECÍFICO b4/c4 — Temperatura puntual mapeada (p1..p7) y dolor
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 72)
print("  OBJ. b4/c4 — Temperatura puntual mapeada (p1..p7) como indicador de dolor")
print("─" * 72)

temp_punto_rows = []
paired_tp = paired.dropna(subset=['temp_punto']).copy()
if len(paired_tp) >= 3 and paired_tp['temp_punto'].nunique() > 1:
    rho_tp, p_tp = stats.spearmanr(paired_tp['temp_punto'], paired_tp['intensidad'])
else:
    rho_tp, p_tp = np.nan, np.nan
print(f"  Spearman temp_punto vs dolor: ρ={rho_tp:+.4f}  p={p_tp:.4f}")
temp_punto_rows.append({
    'nivel': 'Global',
    'metrica': 'spearman_temp_punto_vs_intensidad',
    'subgrupo': 'todos',
    'n': len(paired_tp),
    'valor_1': rho_tp,
    'valor_2': p_tp,
    'valor_3': np.nan,
    'valor_4': np.nan,
})

roc_temp_punto = calc_roc(paired_tp, 'Temp_Punto', score_col='temp_punto') if len(paired_tp) else None
if roc_temp_punto:
    print(f"  ROC temp_punto: AUC={roc_temp_punto['auc']:.3f}, "
          f"T*={roc_temp_punto['T_star']:.3f}, "
          f"Sens={roc_temp_punto['sensitivity']:.3f}, "
          f"Spec={roc_temp_punto['specificity']:.3f}")
    temp_punto_rows.append({
        'nivel': 'Global',
        'metrica': 'roc_temp_punto',
        'subgrupo': 'todos',
        'n': len(paired_tp),
        'valor_1': roc_temp_punto['auc'],
        'valor_2': roc_temp_punto['T_star'],
        'valor_3': roc_temp_punto['sensitivity'],
        'valor_4': roc_temp_punto['specificity'],
    })
else:
    print("  ROC temp_punto: datos insuficientes.")

for musc, grp in paired_tp.groupby('musculo'):
    if len(grp) < 10 or grp['temp_punto'].nunique() <= 1:
        continue
    r_m, p_m = stats.spearmanr(grp['temp_punto'], grp['intensidad'])
    temp_punto_rows.append({
        'nivel': 'Músculo',
        'metrica': 'spearman_temp_punto_vs_intensidad',
        'subgrupo': musc,
        'n': len(grp),
        'valor_1': r_m,
        'valor_2': p_m,
        'valor_3': np.nan,
        'valor_4': np.nan,
    })

# ════════════════════════════════════════════════════════════════════════════
# ANÁLISIS A NIVEL DE PACIENTE (N=43) - Con Dolor vs Sin Dolor
# ════════════════════════════════════════════════════════════════════════════
# Este bloque colapsa el dataset pareado al nivel de paciente (N=43),
# permitiendo comparaciones directas sin el problema de dependencia
# intra-sujeto que afecta a los análisis de punto anatómico.
#
# Variable de agrupación: presencia de dolor (coherente con el análisis principal).
#   'Con Dolor' — paciente reportó ≥1 punto con intensidad > 0
#   'Sin Dolor' — ningún punto de dolor durante toda la exploración
#
# Variables agregadas por paciente:
#   delta_t_max     — ΔT máximo en cualquier región (indicador clínico más
#                     sensible: captura la región más afectada del paciente)
#   delta_t_mean    — ΔT medio (todas las regiones)
#   intensidad_max  — Peor dolor reportado en cualquier punto (0-10)
#   presencia_dolor — 1 si el paciente reportó dolor en algún punto
#
# Pruebas estadísticas:
#   Mann-Whitney U  — Comparación de delta_t_max entre Con Dolor y Sin Dolor.
#                     No paramétrica: apropiada para N pequeño y sin supuesto
#                     de normalidad.
#   ROC (ΔT Max)   — Capacidad del ΔT máximo para predecir si el paciente
#                     reportará algún punto de dolor durante la exploración.
print("\n" + "═" * 72)
print("  ANÁLISIS A NIVEL DE PACIENTE (N=43) - Con Dolor vs Sin Dolor")
print("═" * 72)

# Agregación de datos por paciente
pacientes_n45_agg = paired.groupby(['muestra', 'grupo']).agg({
    'delta_t': ['max', 'mean'],
    'intensidad': 'max',
    'tiene_dolor': 'max'
})

# Agregación por región para cada paciente (ΔT y temp_max por región)
reg_agg = paired.groupby(['muestra', 'grupo', 'region']).agg({
    'delta_t': 'mean',
    'temp_max': 'mean'
}).unstack()
reg_agg.columns = [f"{col}_{reg}" for col, reg in reg_agg.columns]

pacientes_n45 = pd.concat([pacientes_n45_agg, reg_agg], axis=1).reset_index()
# NOTA: La asignación de columnas a continuación asume que el MultiIndex de
# pacientes_n45_agg se aplana en el orden exacto:
#   (delta_t, max) → delta_t_max
#   (delta_t, mean) → delta_t_mean
#   (intensidad, max) → intensidad_max
#   (tiene_dolor, max) → presencia_dolor
# Esto depende del orden en que se declaró el dict en .agg(). Si se modifica
# dicho dict, también hay que actualizar este listado de nombres.
pacientes_n45.columns = (
    ['muestra', 'grupo', 'delta_t_max', 'delta_t_mean', 'intensidad_max', 'presencia_dolor']
    + list(reg_agg.columns)
)

# ROC: ΔT máximo como predictor de presencia de dolor (Con/Sin Dolor)
roc_dolor_paciente = calc_roc(
    pacientes_n45.rename(columns={'presencia_dolor': 'tiene_dolor'}),
    label='Presencia Dolor (N=43)',
    score_col='delta_t_max'
)

# Separar grupos para pruebas
con_dolor_n45 = pacientes_n45[pacientes_n45['grupo'] == 'Con Dolor']
sin_dolor_n45 = pacientes_n45[pacientes_n45['grupo'] == 'Sin Dolor']

print(f"  Pacientes Con Dolor : {len(con_dolor_n45)}")
print(f"  Pacientes Sin Dolor : {len(sin_dolor_n45)}")

if roc_dolor_paciente:
    print(f"\n  [Capacidad Predictiva de Dolor (N=43)] Predictor: ΔT Max")
    print(f"    AUC-ROC         : {roc_dolor_paciente['auc']:.4f}")
    print(f"    AUC-PR          : {roc_dolor_paciente['pr_auc']:.4f}")
    print(f"    Umbral Óptimo T*: {roc_dolor_paciente['T_star']:.2f} °C")
    print(f"    Sensibilidad    : {roc_dolor_paciente['sensitivity']:.4f}")
    print(f"    Especificidad   : {roc_dolor_paciente['specificity']:.4f}")

# Mann-Whitney U: ΔT Max — Con Dolor vs Sin Dolor
u_stat, p_mw = stats.mannwhitneyu(
    con_dolor_n45['delta_t_max'], sin_dolor_n45['delta_t_max'],
    alternative='two-sided'
)
median_con = con_dolor_n45['delta_t_max'].median()
median_sin = sin_dolor_n45['delta_t_max'].median()

print(f"\n  [Mann-Whitney U] ΔT Máximo — Con Dolor vs Sin Dolor:")
print(f"    Mediana Con Dolor : {median_con:.2f} °C")
print(f"    Mediana Sin Dolor : {median_sin:.2f} °C")
print(f"    p-valor           : {p_mw:.4f} {'★' if p_mw < 0.05 else '✗'}")

# Guardar resultados N=43
pacientes_n45.to_csv(os.path.join(OUTPUT_DIR, 'resultados_paciente_n45.csv'), index=False)

# ════════════════════════════════════════════════════════════════════════════
# VISUALIZACIONES
# ════════════════════════════════════════════════════════════════════════════
# Figura resumen multipanel para reporte clínico-estadístico.
fig = plt.figure(figsize=(21, 17))
fig.patch.set_facecolor(BG)
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.38)

# ── A: Temperaturas máximas por región (Con Dolor vs Sin Dolor) ─────────────
ax_a = fig.add_subplot(gs[0, 0])
regions    = sorted(paired['region'].unique())
x          = np.arange(len(regions))
w          = 0.35
con_means  = [paired[(paired['region']==r)&(paired['grupo']=='Con Dolor')]['temp_max'].mean() for r in regions]
sin_means  = [paired[(paired['region']==r)&(paired['grupo']=='Sin Dolor')]['temp_max'].mean() for r in regions]
ax_a.bar(x - w/2, con_means, w, color=RED,  alpha=0.8, label='Con Dolor', edgecolor='none')
ax_a.bar(x + w/2, sin_means, w, color=BLUE, alpha=0.8, label='Sin Dolor', edgecolor='none')
ax_a.set_xticks(x)
ax_a.set_xticklabels([r.upper() for r in regions])
ax_a.set_ylabel('Temperatura Máx. Prom. (°C)')
style_ax(ax_a, 'a) Temp. Máxima por Región y Grupo')
ax_a.legend(fontsize=8, facecolor=AX_BG, labelcolor=TEXT, framealpha=0.8)

# ── B: Top puntos dolorosos (pacientes con dolor) ────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])
top10 = (
    paired[(paired['grupo']=='Con Dolor') & (paired['tiene_dolor']==1)]
    .groupby(['musculo','punto'])['intensidad'].mean()
    .sort_values(ascending=True).tail(10)
)
labels_b = [f"{m.split('_')[0].upper()} {p}" for m, p in top10.index]
colors_b  = [MUSCLE_COLORS.get(m.split('_')[0], '#888') for m, _ in top10.index]
ax_b.barh(range(len(top10)), top10.values, color=colors_b, height=0.6, edgecolor='none')
ax_b.set_yticks(range(len(top10)))
ax_b.set_yticklabels(labels_b, fontsize=7.5)
ax_b.set_xlabel('Intensidad Media de Dolor (1–10)')
style_ax(ax_b, 'b) Top-10 Puntos Dolorosos (Con Dolor)')
patches = [mpatches.Patch(color=c, label=m.upper()) for m, c in MUSCLE_COLORS.items()]
ax_b.legend(handles=patches, fontsize=7, facecolor=AX_BG, labelcolor=TEXT, framealpha=0.8)

# ── C: Scatter Delta T vs intensidad ─────────────────────────────────────
ax_c = fig.add_subplot(gs[0, 2])
res_g = next(r for r in corr_results if r['nivel'] == 'Global')
cmap_group = {'Con Dolor': RED, 'Sin Dolor': BLUE}
for grupo, grp in paired.groupby('grupo'):
    ax_c.scatter(grp['delta_t'], grp['intensidad'],
                 color=cmap_group[grupo], alpha=0.35, s=12, label=grupo, edgecolors='none')
m_fit, b_fit, *_ = stats.linregress(paired['delta_t'], paired['intensidad'])
x_line = np.linspace(paired['delta_t'].min(), paired['delta_t'].max(), 200)
ax_c.plot(x_line, m_fit * x_line + b_fit, color=ACCENT, lw=1.5,
          label=f'ρ={res_g["rho"]:+.3f}  p_adj={res_g["p_bonf"]:.3f}')
ax_c.set_xlabel('Asimetría Térmica (Delta T) (°C)')
ax_c.set_ylabel('Intensidad del Dolor (0–10)')
style_ax(ax_c, 'c) Delta T vs Intensidad de Dolor')
ax_c.legend(fontsize=7.5, facecolor=AX_BG, labelcolor=TEXT, framealpha=0.8)

# ── D: ρ Spearman por músculo ─────────────────────────────────────────────────
ax_d = fig.add_subplot(gs[1, 0])
musc_rhos = [(r['subgrupo'], r['rho'], r['p_bonf'], r['significativo'])
             for r in corr_results if r['nivel'] == 'Músculo']
musc_rhos.sort(key=lambda x: x[1])
if musc_rhos:
    names, rhos, p_bonfs, sigs = zip(*musc_rhos)
    bar_c = [GREEN if s else '#546E7A' for s in sigs]
    ax_d.barh(names, rhos, color=bar_c, height=0.5, edgecolor='none')
    ax_d.axvline(0, color=TEXT, lw=0.8, alpha=0.4)
    for i, (r, is_sig) in enumerate(zip(rhos, sigs)):
        if is_sig:
            ax_d.text(r + 0.003, i, '★', va='center', color=ACCENT, fontsize=10)
    ax_d.set_xlabel('ρ de Spearman')
    ax_d.tick_params(axis='y', labelsize=7.5)
    style_ax(ax_d, 'd) Correlación Spearman por Músculo (Adj.)')

# ── E: Curva ROC para Detección de Dolor (N=43) ──────────────────────────────
ax_e = fig.add_subplot(gs[1, 1])
if roc_dolor_paciente:
    ax_e.plot(roc_dolor_paciente['fpr'], roc_dolor_paciente['tpr'], color=ACCENT, lw=2.5,
              label=f"Predictor: ΔT Max\nAUC={roc_dolor_paciente['auc']:.3f}")
    ax_e.scatter([1 - roc_dolor_paciente['specificity']], [roc_dolor_paciente['sensitivity']],
                 color=RED, s=60, zorder=5, label=f"T*={roc_dolor_paciente['T_star']:.2f}°C")
    ax_e.plot(roc_dolor_paciente['rec_pts'], roc_dolor_paciente['prec_pts'],
              color=CYAN, lw=1.5, ls='--',
              label=f"PR-AUC={roc_dolor_paciente['pr_auc']:.2f}")

ax_e.plot([0, 1], [0, 1], color=GRID, ls=':', lw=1)
ax_e.set_xlabel('1 – Especificidad / Recall')
ax_e.set_ylabel('Sensibilidad / Precisión')
style_ax(ax_e, 'e) Capacidad Predictiva para Dolor (N=43)')
ax_e.legend(fontsize=7.5, facecolor=AX_BG, labelcolor=TEXT, framealpha=0.8)

# ── F: Curvas ROC por músculo ─────────────────────────────────────────────────
ax_f = fig.add_subplot(gs[1, 2])
for musc, res in roc_by_musc.items():
    col = MUSCLE_COLORS.get(musc, '#888')
    ax_f.plot(res['fpr'], res['tpr'], color=col, lw=1.8,
              label=f"{musc[:6].upper()} AUC={res['auc']:.2f} T*={res['T_star']:.1f}°C")
    ax_f.scatter([1 - res['specificity']], [res['sensitivity']],
                 color=col, s=50, zorder=5)
ax_f.plot([0,1],[0,1], color=GRID, ls='--', lw=1)
ax_f.set_xlabel('1 – Especificidad')
ax_f.set_ylabel('Sensibilidad')
style_ax(ax_f, 'f) Curvas ROC por Músculo')
ax_f.legend(fontsize=7.5, facecolor=AX_BG, labelcolor=TEXT, framealpha=0.8)

# ── G: Histograma temperatura con umbral T* ───────────────────────────────────
ax_g = fig.add_subplot(gs[2, 0])
bins = np.linspace(paired['temp_max'].min(), paired['temp_max'].max(), 32)
ax_g.hist(paired[paired['tiene_dolor']==0]['temp_max'], bins=bins,
          alpha=0.6, color=BLUE, label='Sin dolor', edgecolor='none')
ax_g.hist(paired[paired['tiene_dolor']==1]['temp_max'], bins=bins,
          alpha=0.75, color=RED, label='Con dolor', edgecolor='none')
if roc_global:
    ax_g.axvline(roc_global['T_star'], color=ACCENT, lw=2, ls='--',
                 label=f"T* = {roc_global['T_star']:.2f}°C")
ax_g.set_xlabel('Temperatura Máxima (°C)')
ax_g.set_ylabel('Frecuencia')
style_ax(ax_g, 'g) Distribución Temperatura por Presencia de Dolor')
ax_g.legend(fontsize=8, facecolor=AX_BG, labelcolor=TEXT, framealpha=0.8)

# ── H: Boxplot temperatura por músculo y dolor ───────────────────────────────
ax_h = fig.add_subplot(gs[2, 1])
musculos = sorted(paired['musculo'].unique())
pos_no, pos_si = [], []
data_no, data_si = [], []
tick_pos, tick_labels = [], []
offset = 0
for m in musculos:
    g  = paired[paired['musculo'] == m]
    d0 = g[g['tiene_dolor']==0]['temp_max'].dropna().values
    d1 = g[g['tiene_dolor']==1]['temp_max'].dropna().values
    pos_no.append(offset); pos_si.append(offset + 1)
    data_no.append(d0);    data_si.append(d1)
    tick_pos.append(offset + 0.5)
    tick_labels.append(m[:6].upper())
    offset += 3.2

bp0 = ax_h.boxplot(data_no, positions=pos_no, widths=0.7, patch_artist=True,
    boxprops=dict(facecolor=BLUE, alpha=0.7),
    medianprops=dict(color=TEXT, lw=1.5),
    whiskerprops=dict(color=TEXT), capprops=dict(color=TEXT),
    flierprops=dict(markerfacecolor=TEXT, markersize=2))
bp1 = ax_h.boxplot(data_si, positions=pos_si, widths=0.7, patch_artist=True,
    boxprops=dict(facecolor=RED, alpha=0.7),
    medianprops=dict(color=TEXT, lw=1.5),
    whiskerprops=dict(color=TEXT), capprops=dict(color=TEXT),
    flierprops=dict(markerfacecolor=TEXT, markersize=2))
ax_h.set_xticks(tick_pos)
ax_h.set_xticklabels(tick_labels, fontsize=8)
ax_h.set_ylabel('Temperatura Máxima (°C)')
style_ax(ax_h, 'h) Temperatura por Músculo y Dolor')
ax_h.legend([bp0['boxes'][0], bp1['boxes'][0]], ['Sin dolor', 'Con dolor'],
            fontsize=8, facecolor=AX_BG, labelcolor=TEXT, framealpha=0.8)

# ── I: Comparación Sistémica (N=43) Delta T Max ──────────────────────────────
gs_right_bottom = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[2, 2], hspace=0.3)
ax_i = fig.add_subplot(gs_right_bottom[0, 0])
data_to_plot = [sin_dolor_n45['delta_t_max'], con_dolor_n45['delta_t_max']]
bp = ax_i.boxplot(data_to_plot, patch_artist=True, widths=0.5,
                  medianprops=dict(color=ACCENT, lw=2))
colors_i = [BLUE, RED]
for patch, color in zip(bp['boxes'], colors_i):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

ax_i.set_xticks([1, 2])
ax_i.set_xticklabels(['Sin Dolor', 'Con Dolor'])
ax_i.set_ylabel('Asimetría Térmica Máxima (°C)')
style_ax(ax_i, 'i) ΔT Máximo: Con Dolor vs Sin Dolor (N=43)')

# Anotación de p-valor
y_max = pacientes_n45['delta_t_max'].max()
ax_i.plot([1, 1, 2, 2], [y_max*0.92, y_max*0.95, y_max*0.95, y_max*0.92], color=TEXT, lw=1)
ax_i.text(1.5, y_max*0.96, f"p = {p_mw:.4f}", ha='center', va='bottom', color=ACCENT, fontsize=10, fontweight='bold')

# ── J: Tabla resumen ─────────────────────────────────────────────────────────
ax_j = fig.add_subplot(gs_right_bottom[1, 0])
ax_j.set_facecolor(AX_BG)
ax_j.axis('off')
rows_table = []
for r in corr_results:
    if r['nivel'] == 'Global':
        rows_table.append(['GLOBAL', f'ρ={r["rho"]:+.3f}', f'p_adj={r["p_bonf"]:.4f}', '★' if r['significativo'] else '—'])
    elif r['nivel'] in ('Región', 'Músculo'):
        rows_table.append([
            r['subgrupo'].upper()[:14],
            f"ρ={r['rho']:+.3f}",
            f"p_adj={r['p_bonf']:.4f}",
            '★' if r['significativo'] else '—'
        ])
if roc_global:
    rows_table.append(['ROC Global',
                       f"AUC={roc_global['auc']:.3f}",
                       f"T*={roc_global['T_star']:.1f}°C",
                       f"J={roc_global['youden_J']:.3f}"])
if not np.isnan(kappa_val):
    rows_table.append(['Kappa (H2)',
                       f"κ={kappa_val:.3f}",
                       f"{concordancia_pct:.1f}% conc.",
                       '—'])
if logit_ok and logit_rows:
    logit_or_temp = next((r for r in logit_rows if r['termino'] == 'temp_max'), None)
    if logit_or_temp:
        rows_table.append(['Logit temp_max',
                           f"OR={logit_or_temp['odds_ratio']:.3f}",
                           f"p={logit_or_temp['p_valor']:.4f}",
                           '★' if logit_or_temp['p_valor'] < 0.05 else '—'])
    rows_table.append(['Logit (Imbalanced)',
                       f"AUC-PR={logit_metrics['pr_auc_logit']:.3f}",
                       f"F1={logit_metrics['f1_score']:.3f}",
                       '—'])
if pd.notna(rho_delta_g):
    rows_table.append(['ΔTglobal vs dolor',
                       f"ρ={rho_delta_g:+.3f}",
                       f"p={p_delta_g:.4f}",
                       '★' if p_delta_g < 0.05 else '—'])
# NOTA: Los p-valores de ROC y Logit no se ajustan por Bonferroni por defecto aquí
# ya que son métricas de rendimiento de modelo, no pruebas de hipótesis múltiples de correlación.
if roc_delta_global:
    rows_table.append(['ROC ΔTglobal',
                       f"AUC={roc_delta_global['auc']:.3f}",
                       f"T*={roc_delta_global['T_star']:.3f}",
                       f"J={roc_delta_global['youden_J']:.3f}"])
if pd.notna(rho_tp):
    rows_table.append(['TempPunto vs dolor',
                       f"ρ={rho_tp:+.3f}",
                       f"p={p_tp:.4f}",
                       '★' if p_tp < 0.05 else '—'])
if roc_temp_punto:
    rows_table.append(['ROC TempPunto',
                       f"AUC={roc_temp_punto['auc']:.3f}",
                       f"T*={roc_temp_punto['T_star']:.3f}",
                       f"J={roc_temp_punto['youden_J']:.3f}"])

if roc_dolor_paciente:
    rows_table.append(['DOLOR (N=43)',
                       f"AUC={roc_dolor_paciente['auc']:.3f}",
                       f"T*={roc_dolor_paciente['T_star']:.1f}°C",
                       '★' if roc_dolor_paciente['auc'] > 0.6 else '—'])
tbl = ax_j.table(
    cellText=rows_table,
    colLabels=['Subgrupo', 'Estadístico', 'p / Valor', 'Sig.'],
    cellLoc='center', loc='center', bbox=[0, 0, 1, 1]
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(7.5)
for (row, col), cell in tbl.get_celld().items():
    cell.set_facecolor('#252836' if row % 2 == 0 else AX_BG)
    cell.set_text_props(color=TEXT)
    cell.set_edgecolor(GRID)
    if row == 0:
        cell.set_facecolor('#303450')
        cell.set_text_props(color=ACCENT, fontweight='bold')
ax_j.set_title('Resumen Estadístico', color=TEXT, fontsize=9, fontweight='bold', pad=7)

fig.suptitle(
    'Termografía Infrarroja — Correlación Temperatura ↔ Dolor Orofacial\n'
    'Grupo: Con Dolor vs Sin Dolor  ·  ΔT asimetría pre-palpación (primeras imágenes)',
    color=TEXT, fontsize=13, fontweight='bold', y=0.985
)

plt.savefig(PNG_RESULTADOS, dpi=160, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"\n  ✓ Figura guardada: {PNG_RESULTADOS}")

# ── Exportar CSVs ─────────────────────────────────────────────────────────────
# Se exportan tablas intermedias y finales para trazabilidad y replicación.
pd.DataFrame(col_corrs).to_csv(os.path.join(OUTPUT_DIR, 'correlaciones_por_punto.csv'), index=False)
print(f"  ✓ Correlaciones por punto (Bonferroni): {os.path.join(OUTPUT_DIR, 'correlaciones_por_punto.csv')}")

roc_rows = [{
    'subgrupo'    : r['label'],
    'AUC-ROC'     : round(r['auc'], 4),
    'AUC-PR'      : round(r['pr_auc'], 4),
    'T_star_C'    : round(r['T_star'], 2),
    'sensibilidad': round(r['sensitivity'], 4),
    'especificidad': round(r['specificity'], 4),
    'youden_J'    : round(r['youden_J'], 4),
} for r in all_rocs]
pd.DataFrame(roc_rows).to_csv(os.path.join(OUTPUT_DIR, 'resultados_roc.csv'), index=False)
print(f"  ✓ Resultados ROC: {os.path.join(OUTPUT_DIR, 'resultados_roc.csv')}")

if logit_ok and logit_rows:
    pd.DataFrame(logit_rows).to_csv(os.path.join(OUTPUT_DIR, 'resultados_logit_cluster.csv'), index=False)
    pd.DataFrame([logit_metrics]).to_csv(os.path.join(OUTPUT_DIR, 'metricas_logit_cluster.csv'), index=False)
    print(f"  ✓ Coeficientes Logit robusto: {os.path.join(OUTPUT_DIR, 'resultados_logit_cluster.csv')}")
    print(f"  ✓ Métricas Logit robusto: {os.path.join(OUTPUT_DIR, 'metricas_logit_cluster.csv')}")
if fit_b_sub['ok']:
    pd.DataFrame(fit_b_sub['rows']).to_csv(os.path.join(OUTPUT_DIR, 'resultados_logit_cluster_modelo_b_temp_punto.csv'), index=False)
    pd.DataFrame([fit_b_sub['metrics']]).to_csv(os.path.join(OUTPUT_DIR, 'metricas_logit_cluster_modelo_b_temp_punto.csv'), index=False)
    print(f"  ✓ Coeficientes Logit modelo B (incluye temp_punto): {os.path.join(OUTPUT_DIR, 'resultados_logit_cluster_modelo_b_temp_punto.csv')}")
    print(f"  ✓ Métricas Logit modelo B: {os.path.join(OUTPUT_DIR, 'metricas_logit_cluster_modelo_b_temp_punto.csv')}")
if model_comp_ok:
    pd.DataFrame([model_comparison]).to_csv(os.path.join(OUTPUT_DIR, 'comparacion_modelos_a_vs_b_temp_punto.csv'), index=False)
    print(f"  ✓ Comparación modelos A vs B: {os.path.join(OUTPUT_DIR, 'comparacion_modelos_a_vs_b_temp_punto.csv')}")
if delta_results:
    pd.DataFrame(delta_results).to_csv(os.path.join(OUTPUT_DIR, 'resultados_delta_termico.csv'), index=False)
    print(f"  ✓ Resultados Delta T: {os.path.join(OUTPUT_DIR, 'resultados_delta_termico.csv')}")
if temp_punto_rows:
    pd.DataFrame(temp_punto_rows).to_csv(os.path.join(OUTPUT_DIR, 'resultados_temp_punto.csv'), index=False)
    print(f"  ✓ Resultados temperatura puntual: {os.path.join(OUTPUT_DIR, 'resultados_temp_punto.csv')}")

# Exportar tabla de temperaturas por región y grupo
temp_export = temp_region_grupo.reset_index()
temp_export.columns = [str(c) for c in temp_export.columns]
temp_export.to_csv(os.path.join(OUTPUT_DIR, 'temperatura_por_region_grupo.csv'), index=False)
print(f"  ✓ Temperatura por región y grupo: {os.path.join(OUTPUT_DIR, 'temperatura_por_region_grupo.csv')}")

# Exportar Top-10 puntos dolorosos (todos los pacientes que reportaron dolor)
top10_export = (
    paired[(paired['grupo'] == 'Con Dolor') & (paired['tiene_dolor'] == 1)]
    .groupby(['musculo','lado','region','punto'])['intensidad']
    .mean().sort_values(ascending=False).head(10)
    .reset_index()
)
top10_export.columns = ['musculo','lado','region','punto','intensidad_media']
top10_export.to_csv(os.path.join(OUTPUT_DIR, 'top_puntos_dolorosos.csv'), index=False)
print(f"  ✓ Top puntos dolorosos: {os.path.join(OUTPUT_DIR, 'top_puntos_dolorosos.csv')}")

# Exportar concordancia espacial (Kappa)
kappa_export = {
    'kappa_cohen': kappa_val if not np.isnan(kappa_val) else None,
    'concordancia_pct': concordancia_pct if not np.isnan(concordancia_pct) else None,
    'n_pares': len(kdf),
}
pd.DataFrame([kappa_export]).to_csv(os.path.join(OUTPUT_DIR, 'concordancia_kappa.csv'), index=False)
print(f"  ✓ Concordancia Kappa: {os.path.join(OUTPUT_DIR, 'concordancia_kappa.csv')}")

# Exportar correlaciones globales / región / músculo
pd.DataFrame(corr_results).to_csv(os.path.join(OUTPUT_DIR, 'correlaciones_globales.csv'), index=False)
print(f"  ✓ Correlaciones globales: {os.path.join(OUTPUT_DIR, 'correlaciones_globales.csv')}")

# ── Reporte PDF en lenguaje general ──────────────────────────────────────────
top5_text = (
    '; '.join([f"{'/'.join(map(str, idx))}: {val:.2f}" for idx, val in top_pain.items()])
    if not top_pain.empty else "No se identificaron puntos dolorosos con datos suficientes."
)

if roc_global:
    auc_text = f"{roc_global['auc']:.3f}"
    auc_interp = interpretar_auc(roc_global['auc'])
    tstar_text = f"{roc_global['T_star']:.2f} °C"
    sens_text = f"{roc_global['sensitivity']:.3f}"
    spec_text = f"{roc_global['specificity']:.3f}"
else:
    auc_text = "No evaluable"
    auc_interp = "Datos insuficientes"
    tstar_text = "No evaluable"
    sens_text = "No evaluable"
    spec_text = "No evaluable"

if logit_ok and logit_rows:
    logit_or_temp = next((r for r in logit_rows if r['termino'] == 'temp_max'), None)
    if logit_or_temp:
        logit_text = (
            f"Modelo ajustado: OR temp_max={logit_or_temp['odds_ratio']:.3f} "
            f"(IC95% {logit_or_temp['or_ci95_low']:.3f}–{logit_or_temp['or_ci95_high']:.3f}), "
            f"p={logit_or_temp['p_valor']:.4f}; "
            f"Pseudo-R²={logit_metrics['pseudo_r2_mcfadden']:.3f}, "
            f"AUC prob={logit_metrics['auc_prob_logit']:.3f}."
        )
    else:
        logit_text = (
            f"Modelo ajustado estimado. Pseudo-R²={logit_metrics['pseudo_r2_mcfadden']:.3f}, "
            f"AUC prob={logit_metrics['auc_prob_logit']:.3f}."
        )
else:
    logit_text = "No fue posible ajustar el modelo logístico robusto con los datos disponibles."

if model_comp_ok:
    mejora = "mejora" if model_comparison['delta_auc'] > 0 else "no mejora"
    model_comp_text = (
        f"En la submuestra de la segunda toma (N={model_comparison['n_submuestra']}), "
        f"el Modelo B (regional + puntual) {mejora} respecto al Modelo A (solo regional): "
        f"AUC A={model_comparison['auc_A']:.3f} vs B={model_comparison['auc_B']:.3f} "
        f"(Δ={model_comparison['delta_auc']:+.3f}), "
        f"Pseudo-R² A={model_comparison['pseudo_r2_A']:.3f} vs B={model_comparison['pseudo_r2_B']:.3f}, "
        f"LRT p={model_comparison['lr_p_value']:.4f}. "
        "Nota: esta comparación usa datos de la segunda toma ya que los puntos p1-p7 solo se mapearon en dicha fase."
    )
else:
    model_comp_text = "No fue posible comparar el Modelo A (sin temp_punto) vs Modelo B (con temp_punto)."

temp_punto_text = (
    f"Con temperatura puntual mapeada (N={len(paired_tp)}), "
    f"la relación con intensidad de dolor fue rho={rho_tp:+.3f}, p={p_tp:.4f}. "
    f"{interpretar_rho(rho_tp)}. {p_sig_label(p_tp)}."
    if pd.notna(rho_tp) else
    "No hubo datos suficientes para estimar correlación entre temperatura puntual e intensidad de dolor."
)
if roc_temp_punto:
    temp_punto_roc_text = (
        f"ROC con temperatura puntual: AUC={roc_temp_punto['auc']:.3f} "
        f"({interpretar_auc(roc_temp_punto['auc'])}), "
        f"T*={roc_temp_punto['T_star']:.3f}, "
        f"Sens={roc_temp_punto['sensitivity']:.3f}, Spec={roc_temp_punto['specificity']:.3f}."
    )
else:
    temp_punto_roc_text = "No hubo datos suficientes para curva ROC con temperatura puntual."

delta_text = (
    f"Delta T global vs dolor: rho={rho_delta_g:+.3f}, p={p_delta_g:.4f}. "
    f"{interpretar_rho(rho_delta_g)}. {p_sig_label(p_delta_g)}."
    if pd.notna(rho_delta_g) else
    "No se pudo evaluar adecuadamente la relación Delta T global vs dolor."
)
if roc_delta_global:
    delta_roc_text = (
        f"ROC con Delta T global: AUC={roc_delta_global['auc']:.3f} "
        f"({interpretar_auc(roc_delta_global['auc'])}), "
        f"T*={roc_delta_global['T_star']:.3f}, "
        f"Sens={roc_delta_global['sensitivity']:.3f}, Spec={roc_delta_global['specificity']:.3f}."
    )
else:
    delta_roc_text = "No hubo datos suficientes para ROC basada en Delta T global."

reporte_resumen = {
    'n_total'          : len(df),
    'n_con_dolor'      : int(n_con_dolor),
    'n_sin_dolor_p'    : int(n_sin_dolor),
    'n_paired'         : int(len(paired)),
    'n_dolor'          : int(paired['tiene_dolor'].sum()),
    'n_sin_dolor'      : int((paired['tiene_dolor'] == 0).sum()),
    'top5_text'        : top5_text,
    'rho_g'            : float(rho_g) if pd.notna(rho_g) else np.nan,
    'p_g'              : float(p_g) if pd.notna(p_g) else np.nan,
    'kappa_text'       : f"{kappa_val:.3f}" if pd.notna(kappa_val) else "No evaluable",
    'conc_text'        : f"{concordancia_pct:.1f}%" if pd.notna(concordancia_pct) else "No evaluable",
    'auc_text'         : auc_text,
    'auc_interp'       : auc_interp,
    'tstar_text'       : tstar_text,
    'sens_text'        : sens_text,
    'spec_text'        : spec_text,
    'logit_text'       : logit_text,
    'model_comp_text'  : model_comp_text,
    'temp_punto_text'  : temp_punto_text,
    'temp_punto_roc_text': temp_punto_roc_text,
    'delta_text'       : delta_text,
    'delta_roc_text'   : delta_roc_text,
    'paciente_roc_text': (
        f"Predicción Dolor (N=43): AUC={roc_dolor_paciente['auc']:.3f} "
        f"({interpretar_auc(roc_dolor_paciente['auc'])}), "
        f"PR-AUC={roc_dolor_paciente['pr_auc']:.3f}, "
        f"T*={roc_dolor_paciente['T_star']:.2f} °C, "
        f"Sens={roc_dolor_paciente['sensitivity']:.3f}, "
        f"Spec={roc_dolor_paciente['specificity']:.3f}."
        if roc_dolor_paciente
        else "Datos insuficientes para predecir presencia de dolor a nivel de paciente."
    ),
}

generar_reporte_pdf_general(PDF_REPORTE, reporte_resumen, fig_path=PNG_RESULTADOS)
print(f"  ✓ Reporte PDF ejecutivo: {PDF_REPORTE}")

# ── Conclusión ────────────────────────────────────────────────────────────────
print()
print("=" * 72)
print("  CONCLUSIÓN ESTADÍSTICA (CON CORRECCIÓN DE BONFERRONI)")
print("=" * 72)
res_g = next(r for r in corr_results if r['nivel'] == 'Global')
sig_items = [r for r in corr_results if r['significativo']]

print(f"\n  Obj. b — Correlación global: ρ={res_g['rho']:+.3f}")
print(f"           p_orig = {res_g['p_orig']:.4f}")
print(f"           p_bonf = {res_g['p_bonf']:.4f} → {'Significativa ★' if res_g['significativo'] else 'No significativa'}")

if sig_items:
    print("\n  Subgrupos significativos (tras Bonferroni):")
    for r in sig_items:
        if r['nivel'] == 'Global': continue
        print(f"           {r['nivel']} {r['subgrupo']}: ρ={r['rho']:+.3f} p_bonf={r['p_bonf']:.4f}")
else:
    print("\n  Ningún subgrupo mantuvo significancia estadística tras la corrección de Bonferroni.")

if roc_global:
    util = "ACEPTABLE ►" if roc_global['auc'] >= 0.65 else "LIMITADA"
    print(f"\n  Obj. c — Termografía como herramienta diagnóstica: {util}")
    print(f"           AUC global = {roc_global['auc']:.3f}")
    print(f"           Umbral T*  = {roc_global['T_star']:.2f}°C "
          f"(Sens={roc_global['sensitivity']:.3f}, Spec={roc_global['specificity']:.3f})")
if not np.isnan(kappa_val):
    print(f"\n  Obj. b2 — Concordancia espacial (H2): κ={kappa_val:.3f}")
    print(f"            Coincidencia directa zona caliente↔dolor: {concordancia_pct:.1f}%")
if logit_ok and logit_rows:
    logit_or_temp = next((r for r in logit_rows if r['termino'] == 'temp_max'), None)
    print("\n  Obj. c2 — Regresión logística robusta por clúster:")
    print(f"            Modelo: tiene_dolor ~ temp_max + region + lado")
    print(f"            Pseudo-R²={logit_metrics['pseudo_r2_mcfadden']:.3f}, "
          f"AUC_prob={logit_metrics['auc_prob_logit']:.3f}, "
          f"Brier={logit_metrics['brier_score']:.3f}")
    if logit_or_temp:
        print(f"            temp_max OR={logit_or_temp['odds_ratio']:.3f} "
              f"(IC95% {logit_or_temp['or_ci95_low']:.3f}–{logit_or_temp['or_ci95_high']:.3f}), "
              f"p={logit_or_temp['p_valor']:.4f}")
if model_comp_ok:
    print("\n  Extensión A vs B — Integrando temperatura puntual (p1..p7 mapeada):")
    print(f"            N submuestra={model_comparison['n_submuestra']}")
    print(f"            AUC A={model_comparison['auc_A']:.3f} vs B={model_comparison['auc_B']:.3f} "
          f"(Δ={model_comparison['delta_auc']:+.3f})")
    print(f"            LRT p={model_comparison['lr_p_value']:.4f}")
if pd.notna(rho_tp):
    print(f"\n  Obj. b4 — Temp. puntual mapeada vs dolor: ρ={rho_tp:+.3f}, p={p_tp:.4f}")
if roc_temp_punto:
    print(f"  Obj. c4 — ROC temp_punto: T*={roc_temp_punto['T_star']:.3f} "
          f"(AUC={roc_temp_punto['auc']:.3f}, Sens={roc_temp_punto['sensitivity']:.3f}, "
          f"Spec={roc_temp_punto['specificity']:.3f})")
if pd.notna(rho_delta_g):
    print(f"\n  Obj. b3 — Aumento térmico (ΔT_global) vs dolor: ρ={rho_delta_g:+.3f}, p={p_delta_g:.4f}")
if roc_delta_global:
    print(f"  Obj. c3 — Umbral con ΔT_global: T*={roc_delta_global['T_star']:.3f} "
          f"(AUC={roc_delta_global['auc']:.3f}, Sens={roc_delta_global['sensitivity']:.3f}, "
          f"Spec={roc_delta_global['specificity']:.3f})")
print()
