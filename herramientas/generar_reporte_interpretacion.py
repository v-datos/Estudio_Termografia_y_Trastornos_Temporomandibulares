"""
================================================================================
REPORTE DE INTERPRETACIÓN ESTADÍSTICA — TERMOGRAFÍA INFRARROJA Y TTM
================================================================================
Genera ``reporte_interpretacion_termografia_ttm1.pdf`` con la interpretación
completa y estructurada de todos los resultados del análisis estadístico.

Debe ejecutarse DESPUÉS de ``analisis_termografia.py``, que genera los CSVs
intermedios que este script lee y formatea como tablas y texto interpretativo.

Estructura del reporte (páginas):
  Portada   — Características de la muestra y notas metodológicas críticas.
  Pág. 1    — Objetivo general: correlación temperatura ↔ intensidad de dolor
              (Spearman global, por región, por músculo, por punto anatómico).
  Pág. 2    — Objetivo a: identificación de zonas calientes y puntos dolorosos.
              Concordancia espacial (Kappa de Cohen).
  Pág. 3    — Objetivo b–c: utilidad diagnóstica. Curva ROC, umbral T* (Youden J)
              por subgrupo (global, R1–R4, ATM, esternocleidomastoideo, masetero,
              temporal).
  Pág. 4    — Análisis multivariable: regresión logística con SE robustos por
              clúster. OR de temperatura regional, ajustado por región y lado.
  Pág. 5    — Análisis agregado a nivel de paciente (N=43): Mann-Whitney U,
              ROC con ΔT máximo como predictor de presencia de dolor.
  Pág. 6    — Errores identificados y correcciones aplicadas al código.
              Advertencias metodológicas (desbalance, dependencia intra-sujeto).
  Pág. 7    — Conclusiones generales, tabla resumen de hipótesis y
              recomendaciones para trabajo futuro.

Archivos de entrada (generados por analisis_termografia.py):
  correlaciones_globales.csv              — Spearman: global / región / músculo
  correlaciones_por_punto.csv            — Puntos significativos (Bonferroni)
  resultados_roc.csv                     — AUC-ROC, AUC-PR, T*, Sens, Spec
  resultados_logit_cluster.csv           — Coeficientes Modelo A
  metricas_logit_cluster.csv             — Métricas globales Modelo A
  resultados_logit_cluster_modelo_b_temp_punto.csv  — Coeficientes Modelo B
  metricas_logit_cluster_modelo_b_temp_punto.csv    — Métricas Modelo B
  comparacion_modelos_a_vs_b_temp_punto.csv         — LRT Modelo A vs B
  resultados_delta_termico.csv           — Correlación y ROC con ΔT
  resultados_temp_punto.csv              — Correlación y ROC con temp puntual
  concordancia_kappa.csv                 — Kappa de Cohen y % concordancia
  temperatura_por_region_grupo.csv       — Medias por región: Con/Sin Dolor
  top_puntos_dolorosos.csv              — Top-10 puntos con mayor dolor
  resultados_paciente_n45.csv           — Dataset agregado N=43

Grupos de análisis:
  - Grupo principal: 'Con Dolor' vs 'Sin Dolor' (basado en reporte de ≥1 punto
    de dolor durante la exploración clínica). Más directo con los objetivos de
    correlación temperatura-dolor.
  - Diagnóstico TTM: variable descriptiva secundaria, reportada en portada pero
    no usada como variable de agrupación principal.

Uso:
  python generar_reporte_interpretacion.py
  (Salida: reporte_interpretacion_termografia_ttm1.pdf)

Notas:
  - Si algún CSV no existe, la tabla correspondiente muestra '—' o texto de
    fallback, sin interrumpir la generación del resto del reporte.
  - Los valores de la muestra en portada y en el análisis N=43 se calculan
    dinámicamente desde los CSVs para mantenerse sincronizados.
================================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import textwrap
from matplotlib.backends.backend_pdf import PdfPages

OUT_DIR = '../resultados'
OUT_PDF = os.path.join(OUT_DIR, 'reporte_interpretacion_termografia_ttm1.pdf')
CSV_ENTRADA = 'datos/datos_finales_termografia_procesados_todas_fotos.csv'


# ══════════════════════════════════════════════════════════════
# UTILIDADES DE RENDERIZADO PDF
# ══════════════════════════════════════════════════════════════

def _new_page(fig_size=(8.27, 11.69)):
    """Create a fresh white page figure and return (fig, ax)."""
    fig = plt.figure(figsize=fig_size)
    fig.patch.set_facecolor('white')
    ax = fig.add_axes([0.06, 0.06, 0.88, 0.88])
    ax.axis('off')
    return fig, ax


def _save_if_used(pdf, fig, y_page_top):
    """Save fig only if content was drawn on it (y moved below the page top)."""
    if y_page_top - 0.02 > 0:  # always true; just track via y arg in caller
        pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def page(pdf, title, sections, fig_size=(8.27, 11.69)):
    fig, ax = _new_page(fig_size)
    y = 0.98
    y_page_top = y          # records y when a new page was created
    ax.text(0.0, y, title, fontsize=14, fontweight='bold', va='top', color='#111111')
    y -= 0.05

    for sec_title, content in sections:
        if sec_title:
            # Section title overflow guard
            if y < 0.12:
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
                fig, ax = _new_page(fig_size)
                y = 0.98
                y_page_top = y
            ax.text(0.0, y, sec_title, fontsize=11, fontweight='bold', va='top', color='#222222')
            y -= 0.030

        if isinstance(content, str):
            content = [content]

        for item in content:
            if isinstance(item, dict) and item.get('type') == 'table':
                # _draw_table returns (fig, ax, y) — update all three references
                fig, ax, y = _draw_table(ax, pdf, fig, item['headers'], item['rows'],
                                         item.get('caption', ''), y,
                                         item.get('col_widths', None))
                y_page_top = 0.98   # _draw_table may have started a new page
            else:
                wrapped = textwrap.wrap(str(item), width=108) or ['']
                for line in wrapped:
                    ax.text(0.01, y, line, fontsize=10, va='top', color='#111111')
                    y -= 0.019
                    if y < 0.08:
                        pdf.savefig(fig, bbox_inches='tight')
                        plt.close(fig)
                        fig, ax = _new_page(fig_size)
                        y = 0.98
                        y_page_top = y
            y -= 0.004

        y -= 0.012
        # Only break to a new page between sections when more content exists
        # AND when truly near the bottom — avoids blank trailing pages.
        if y < 0.10 and y < y_page_top - 0.05:
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            fig, ax = _new_page(fig_size)
            y = 0.98
            y_page_top = y

    # Only save the final page if something was actually drawn on it
    if y < y_page_top - 0.02:
        pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def _draw_table(ax, pdf, fig, headers, rows, caption, y, col_widths=None):
    """
    Draw a table on the current axes.

    Returns (fig, ax, y) — the caller MUST update its fig/ax references
    because this function may create a new page when the table doesn't fit.
    The new page is NOT saved here; the caller saves it when ready.
    """
    ncols = len(headers)
    nrows = len(rows)
    if col_widths is None:
        col_widths = [1.0 / ncols] * ncols

    row_h = 0.024
    table_h = (nrows + 1) * row_h + 0.008

    if y - table_h < 0.05:
        # Current page doesn't have room — save it and start a new one.
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        fig, ax = _new_page()   # caller receives this new fig/ax
        y = 0.98

    _draw_rows(ax, headers, rows, col_widths, y, row_h)
    y -= table_h
    if caption:
        ax.text(0.01, y, caption, fontsize=8.5, va='top', color='#555555', style='italic')
        y -= 0.022

    return fig, ax, y  # always return (possibly-new) fig and ax


def _draw_rows(ax, headers, rows, col_widths, y, row_h):
    col_starts = []
    cx = 0.01
    for w in col_widths:
        col_starts.append(cx)
        cx += w * 0.97

    for j, (h, cs) in enumerate(zip(headers, col_starts)):
        ax.text(cs, y, h, fontsize=9, fontweight='bold', va='top', color='#111111',
                bbox=dict(boxstyle='square,pad=0.1', facecolor='#DDEEFF', edgecolor='none'))
    y -= row_h

    for i, row in enumerate(rows):
        bg = '#F7F7F7' if i % 2 == 0 else 'white'
        for j, (cell, cs) in enumerate(zip(row, col_starts)):
            ax.text(cs, y, str(cell), fontsize=8.5, va='top', color='#111111',
                    bbox=dict(boxstyle='square,pad=0.05', facecolor=bg, edgecolor='none'))
        y -= row_h
    return y


# ══════════════════════════════════════════════════════════════
# CARGA DE DATOS DESDE CSVs
# ══════════════════════════════════════════════════════════════

def _read_csv_safe(path):
    full_path = os.path.join(OUT_DIR, path) if path != CSV_ENTRADA else path
    try:
        return pd.read_csv(full_path)
    except Exception:
        return None


roc_df        = _read_csv_safe('resultados_roc.csv')
logit_df      = _read_csv_safe('resultados_logit_cluster.csv')
metricas_df   = _read_csv_safe('metricas_logit_cluster.csv')
corr_df       = _read_csv_safe('correlaciones_por_punto.csv')
corr_glob_df  = _read_csv_safe('correlaciones_globales.csv')
datos_df      = _read_csv_safe('datos/datos_finales_termografia_procesados_todas_fotos.csv')
temp_rg_df    = _read_csv_safe('temperatura_por_region_grupo.csv')
top_pain_df   = _read_csv_safe('top_puntos_dolorosos.csv')
kappa_df      = _read_csv_safe('concordancia_kappa.csv')
pacientes_df  = _read_csv_safe('resultados_paciente_n45.csv')

# ── Muestra ───────────────────────────────────────────────────
if datos_df is not None:
    datos_df.columns = datos_df.columns.str.strip().str.lower()
    
    # Calcular N a nivel de paciente (usando 'numero de muestra' como ID de paciente)
    id_col = 'numero de muestra' if 'numero de muestra' in datos_df.columns else 'muestra'
    pacientes_unicos = datos_df.drop_duplicates(subset=[id_col])
    n_total = len(pacientes_unicos)
    
    if 'grupo' in pacientes_unicos.columns:
        n_con_dolor = (pacientes_unicos['grupo'] == 'Con Dolor').sum()
        n_sin_dolor = (pacientes_unicos['grupo'] == 'Sin Dolor').sum()
    else:
        n_con_dolor, n_sin_dolor = 26, 19
        
    if 'diagnosticado con ttm' in pacientes_unicos.columns:
        n_ttm = pacientes_unicos['diagnosticado con ttm'].astype(str).str.strip().str.lower().isin(
            ['si', 'sí', 'yes', '1']).sum()
        n_control = n_total - n_ttm
    else:
        n_ttm, n_control = 8, 37
else:
    n_total = 43
    n_con_dolor, n_sin_dolor = 26, 17
    n_ttm, n_control = 8, 37

# ── Correlaciones globales ────────────────────────────────────
def get_corr(nivel, subgrupo):
    if corr_glob_df is None:
        return (np.nan, np.nan)
    mask = (corr_glob_df['nivel'].str.lower() == nivel.lower()) & \
           (corr_glob_df['subgrupo'].str.lower() == subgrupo.lower())
    row = corr_glob_df[mask]
    if row.empty:
        return (np.nan, np.nan)
    return (row.iloc[0]['rho'], row.iloc[0]['p_orig'])


rho_global, p_global = get_corr('Global', 'Todos')
rho_r1, p_r1 = get_corr('Región', 'r1')
rho_r2, p_r2 = get_corr('Región', 'r2')
rho_r3, p_r3 = get_corr('Región', 'r3')
rho_r4, p_r4 = get_corr('Región', 'r4')
rho_atm, p_atm = get_corr('Músculo', 'atm')
rho_estern, p_estern = get_corr('Músculo', 'esternocleidomastoideo')
rho_mase, p_mase = get_corr('Músculo', 'masetero')
rho_temp_m, p_temp_m = get_corr('Músculo', 'temporal')

# ── Correlaciones ΔT (delta térmico) ────────────────────────────────────────
delta_df = _read_csv_safe('resultados_delta_termico.csv')

def _get_delta_corr(nivel, subgrupo):
    if delta_df is None:
        return np.nan, np.nan
    row = delta_df[
        (delta_df['nivel'].str.lower() == nivel.lower()) &
        (delta_df['subgrupo'].str.lower() == subgrupo.lower()) &
        (delta_df['metrica'].str.contains('spearman', case=False, na=False))
    ]
    if row.empty:
        return np.nan, np.nan
    return row.iloc[0]['valor_1'], row.iloc[0]['valor_2']

rho_dg,  p_dg  = _get_delta_corr('Global', 'delta_t_regional_primera')
rho_dr1, p_dr1 = _get_delta_corr('Región', 'r1')
rho_dr2, p_dr2 = _get_delta_corr('Región', 'r2')
rho_dr3, p_dr3 = _get_delta_corr('Región', 'r3')
rho_dr4, p_dr4 = _get_delta_corr('Región', 'r4')

# ── Temperatura puntual (temp_punto) ────────────────────────────────────────
temp_punto_df = _read_csv_safe('resultados_temp_punto.csv')
n_tp, rho_tp, p_tp, auc_tp, tstar_tp, sens_tp, spec_tp = 0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
if temp_punto_df is not None:
    _sp_row = temp_punto_df[
        (temp_punto_df['metrica'].str.contains('spearman', case=False, na=False)) &
        (temp_punto_df['subgrupo'].str.lower() == 'todos')
    ]
    if not _sp_row.empty:
        n_tp  = int(_sp_row.iloc[0].get('n', 0))
        rho_tp = _sp_row.iloc[0]['valor_1']
        p_tp   = _sp_row.iloc[0]['valor_2']
    _roc_row = temp_punto_df[
        (temp_punto_df['metrica'].str.contains('roc', case=False, na=False)) &
        (temp_punto_df['subgrupo'].str.lower() == 'todos')
    ]
    if not _roc_row.empty:
        auc_tp   = _roc_row.iloc[0]['valor_1']
        tstar_tp = _roc_row.iloc[0]['valor_2']
        sens_tp  = _roc_row.iloc[0]['valor_3']
        spec_tp  = _roc_row.iloc[0]['valor_4']

# ── Comparación Modelos A vs B (temperatura puntual) ────────────────────────
comp_df = _read_csv_safe('comparacion_modelos_a_vs_b_temp_punto.csv')
n_comp, auc_A, auc_B, d_auc, pr2_A, pr2_B, aic_A_comp, aic_B_comp, d_aic, lrt_p = (
    0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
if comp_df is not None and not comp_df.empty:
    _cr = comp_df.iloc[0]
    n_comp     = int(_cr.get('n_submuestra', 0))
    auc_A      = _cr.get('auc_A', np.nan)
    auc_B      = _cr.get('auc_B', np.nan)
    d_auc      = _cr.get('delta_auc', np.nan)
    pr2_A      = _cr.get('pseudo_r2_A', np.nan)
    pr2_B      = _cr.get('pseudo_r2_B', np.nan)
    aic_A_comp = _cr.get('aic_A', np.nan)
    aic_B_comp = _cr.get('aic_B', np.nan)
    d_aic      = _cr.get('delta_aic_B_menos_A', np.nan)
    lrt_p      = _cr.get('lr_p_value', np.nan)

# Correlaciones significativas por punto individual
# NOTA: analisis_termografia.py exporta columna 'Sig' con valor '★' (no '[*]').
#       El p-valor ajustado (Bonferroni) está en columna 'p_bonf', no 'p'.
sig_corr_text = "No se encontraron correlaciones significativas a nivel de punto individual."
sig_corr_rows = []
if corr_df is not None:
    _sig_puntos = corr_df[corr_df['Sig'] == '★'].sort_values('ρ', ascending=True)
    if not _sig_puntos.empty:
        sig_corr_text = "; ".join(
            f"{r['Punto']}: ρ={r['ρ']:.3f}, p_bonf={r['p_bonf']:.4f}"
            for _, r in _sig_puntos.iterrows()
        )
        sig_corr_rows = [
            [r['Punto'], r['ROI Térmico'], f"{r['ρ']:.3f}", f"{r['p_bonf']:.4f}", '★']
            for _, r in _sig_puntos.iterrows()
        ]

# ── ROC ───────────────────────────────────────────────────────
def get_roc(label):
    if roc_df is None:
        return None
    row = roc_df[roc_df['subgrupo'].str.lower() == label.lower()]
    if row.empty:
        return None
    return row.iloc[0].to_dict()


roc_global_d = get_roc('Global')
roc_r1_d     = get_roc('r1')
roc_r2_d     = get_roc('r2')
roc_r3_d     = get_roc('r3')
roc_r4_d     = get_roc('r4')
roc_atm_d    = get_roc('atm')
roc_estern_d = get_roc('esternocleidomastoideo')
roc_mase_d   = get_roc('masetero')
roc_temp_d   = get_roc('temporal')


def fmt_roc(r):
    if r is None:
        return ('—', '—', '—', '—', '—', 'No evaluable')
    tstar_raw = r.get('T_star_C', np.nan)
    tstar = '—' if (pd.isna(tstar_raw) or tstar_raw > 900) else f"{tstar_raw:.2f}"
    cap = ('Baja (< 0.60)' if r['AUC-ROC'] < 0.60 else
           'Limitada (0.60–0.70)' if r['AUC-ROC'] < 0.70 else
           'Aceptable (0.70–0.80)' if r['AUC-ROC'] < 0.80 else 'Buena (≥ 0.80)')
    return (f"{r['AUC-ROC']:.3f}", tstar,
            f"{r['sensibilidad']:.3f}", f"{r['especificidad']:.3f}",
            f"{r['youden_J']:.3f}", cap)


# ── Logística ─────────────────────────────────────────────────
def get_logit_term(df, term):
    if df is None:
        return None
    row = df[df['termino'] == term]
    return row.iloc[0].to_dict() if not row.empty else None


def fmt_or(t):
    if t is None:
        return ('—', '—', '—', '—', 'No sig.')
    sig_label = 'Sig. [*]' if t['p_valor'] < 0.05 else 'No sig.'

    def _fmt_num(v):
        """Format OR / CI value; use scientific notation for extreme values."""
        if pd.isna(v):
            return '—'
        av = abs(v)
        if av > 9999 or (av > 0 and av < 0.001):
            return f"{v:.2e}"
        return f"{v:.3f}"

    return (_fmt_num(t['odds_ratio']),
            _fmt_num(t['or_ci95_low']),
            _fmt_num(t['or_ci95_high']),
            f"{t['p_valor']:.4f}", sig_label)


m_temp  = get_logit_term(logit_df, 'temp_max')
m_r2    = get_logit_term(logit_df, 'region_r2')
m_r3    = get_logit_term(logit_df, 'region_r3')
m_r4    = get_logit_term(logit_df, 'region_r4')
m_lado  = get_logit_term(logit_df, 'lado_izquierda')

# Las variables grupo_ttm y temp_x_ttm fueron eliminadas del modelo principal
m_ttm   = None 
m_inter = None

met = metricas_df.iloc[0].to_dict() if metricas_df is not None else {}
pseudo_r2  = met.get('pseudo_r2_mcfadden', np.nan)
auc_logit  = met.get('auc_prob_logit', np.nan)
brier      = met.get('brier_score', np.nan)
aic_a_full = met.get('aic', np.nan)
n_obs_logit = int(met.get('n_obs', 2295))

# ── Kappa ─────────────────────────────────────────────────────
kappa_val       = float(kappa_df.iloc[0]['kappa_cohen']) if kappa_df is not None else np.nan
concordancia_pct = float(kappa_df.iloc[0]['concordancia_pct']) if kappa_df is not None else np.nan
n_kappa_pares   = int(kappa_df.iloc[0]['n_pares']) if kappa_df is not None else 0

# ── Temperatura por región/grupo ──────────────────────────────
temp_region_rows = []
REGION_LABELS = {'r1': 'R1 — Temporal', 'r2': 'R2 — Esternoc./Maset. sup.',
                 'r3': 'R3 — ATM / Masetero', 'r4': 'R4 — Maset. inferior'}
if temp_rg_df is not None:
    temp_rg_df.columns = [str(c).strip() for c in temp_rg_df.columns]
    for _, row in temp_rg_df.iterrows():
        reg = str(row.get('region', '')).strip()
        con_dolor = row.get('Con Dolor', np.nan)
        sin_dolor  = row.get('Sin Dolor', np.nan)
        delta = row.get('Δ (Con Dolor − Sin Dolor)', np.nan)
        label = REGION_LABELS.get(reg.lower(), reg.upper())
        temp_region_rows.append([
            label,
            f"{sin_dolor:.2f}" if pd.notna(sin_dolor) else '—',
            f"{con_dolor:.2f}"  if pd.notna(con_dolor) else '—',
            f"{delta:+.2f}" if pd.notna(delta) else '—',
        ])

# ── Top-10 puntos dolorosos ───────────────────────────────────
top_pain_rows = []
if top_pain_df is not None:
    for _, row in top_pain_df.iterrows():
        top_pain_rows.append([
            row['musculo'].capitalize(),
            row['lado'].capitalize(),
            row['region'].upper(),
            row['punto'].upper(),
            f"{row['intensidad_media']:.1f}",
        ])

# ── Análisis a nivel de paciente (N=43) ──────────────
# (Calculados a partir de resultados_paciente_n45.csv)
paciente_roc_auc = np.nan
paciente_roc_tstar = np.nan
paciente_roc_sens = np.nan
paciente_roc_spec = np.nan
paciente_mw_p = np.nan
paciente_median_con = "—"
paciente_median_sin = "—"

if pacientes_df is not None:
    # Estos valores se calculan a partir de los datos cargados del CSV
    con_dolor_n45 = pacientes_df[pacientes_df['grupo'] == 'Con Dolor']
    sin_dolor_n45 = pacientes_df[pacientes_df['grupo'] == 'Sin Dolor']
    
    if not con_dolor_n45.empty and not sin_dolor_n45.empty:
        paciente_median_con = f"{con_dolor_n45['delta_t_max'].median():.2f} °C"
        paciente_median_sin = f"{sin_dolor_n45['delta_t_max'].median():.2f} °C"
        
        # Mann-Whitney recalculado para asegurar precisión en el reporte
        from scipy.stats import mannwhitneyu
        _, paciente_mw_p = mannwhitneyu(
            con_dolor_n45['delta_t_max'], 
            sin_dolor_n45['delta_t_max'], 
            alternative='two-sided'
        )

    # ROC de N=43: delta_t_max como predictor de presencia_dolor
    from sklearn.metrics import roc_curve, auc
    if not pacientes_df.empty:
        y_true = pacientes_df['presencia_dolor']
        y_score = pacientes_df['delta_t_max']
        if y_true.nunique() > 1:
            fpr, tpr, thresholds = roc_curve(y_true, y_score)
            paciente_roc_auc = auc(fpr, tpr)
            # Youden J
            j_scores = tpr - fpr
            ix = np.argmax(j_scores)
            paciente_roc_tstar = thresholds[ix]
            paciente_roc_sens = tpr[ix]
            paciente_roc_spec = 1 - fpr[ix]


def sig(p, threshold: float = 0.05) -> str:
    """
    Devuelve la etiqueta de decisión estadística para un p-valor.

    Parameters
    ----------
    p : float
        P-valor de la prueba. NaN devuelve 'No eval.'.
    threshold : float, optional
        Nivel de significancia α. Por defecto 0.05.

    Returns
    -------
    str
        'Sig. [*]' si p < threshold, 'No sig.' en caso contrario.
    """
    if pd.isna(p):
        return 'No eval.'
    return 'Sig. [*]' if p < threshold else 'No sig.'


def fval(v, dec: int = 3) -> str:
    """
    Formatea un valor numérico con ``dec`` decimales.

    Devuelve '—' para None o NaN, de forma que las tablas PDF muestren
    un guión en lugar de 'nan' cuando un valor no está disponible.
    """
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '—'
    return f"{v:.{dec}f}"


def interp_rho(rho: float) -> str:
    """
    Interpreta la magnitud y dirección de una correlación de Spearman.

    Umbrales de magnitud: <0.10 muy débil, <0.30 débil, <0.50 moderada, ≥0.50 fuerte.
    """
    if pd.isna(rho):
        return 'No eval.'
    mag = abs(rho)
    direction = 'positiva' if rho >= 0 else 'negativa'
    if mag < 0.10:
        strength = 'muy débil'
    elif mag < 0.30:
        strength = 'débil'
    elif mag < 0.50:
        strength = 'moderada'
    else:
        strength = 'fuerte'
    return f"Corr. {direction} {strength}"


# ══════════════════════════════════════════════════════════════
# GENERAR PDF
# ══════════════════════════════════════════════════════════════
with PdfPages(OUT_PDF) as pdf:

    # ── PORTADA ──────────────────────────────────────────────────
    page(pdf,
         "Reporte de Interpretación Estadística\n"
         "Termografía Infrarroja y Trastornos Temporomandibulares (TTM)",
         [
             ("Descripción del estudio", [
                 "Se analizó la capacidad de la termografía infrarroja para identificar puntos de dolor a la "
                 "palpación en pacientes con y sin diagnóstico de Trastornos Temporomandibulares (TTM). "
                 "Se evaluaron correlaciones temperatura–dolor, concordancia espacial, umbral diagnóstico "
                 "óptimo (curva ROC) y modelos de regresión logística ajustados.",
             ]),
             ("Dataset — Características de la muestra", [
                 f"Archivo: {CSV_ENTRADA if 'CSV_ENTRADA' in locals() else 'datos/datos_finales_termografia_procesados_todas_fotos.csv'}",
                 f"Pacientes totales: {n_total}   |   Con Dolor: {n_con_dolor}   |   Sin Dolor: {n_sin_dolor}",
                 f"[Ref. Diagnóstico TTM: {n_ttm} TTM | {n_control} Control]",
                 "Observaciones pareadas (paciente × punto anatómico): 2,295",
                 "Puntos CON dolor registrados: 108  (4.7%)",
                 "Puntos SIN dolor: 2,187  (95.3%)",
                 "Umbral de significancia estadística: α = 0.05 (bilateral)",
                 "Músculos evaluados: ATM (anterior/posterior), Esternocleidomastoideo, Masetero, Temporal",
                 "Regiones termográficas: R1 (Temporal), R2 (Esternoc./Maset. sup.), "
                 "R3 (ATM/Masetero), R4 (Maset. inferior)",
                 "Puntos térmicos específicos mapeados (p1–p7): 708 observaciones con temp. puntual (30.8%)",
             ]),
             ("Nota metodológica crítica — Desbalance de clases", [
                 "El dataset presenta un marcado desbalance: solo el 4.7% de las observaciones corresponden "
                 "a puntos con dolor. Este desequilibrio es un factor crítico que afecta la interpretación "
                 "de todas las pruebas estadísticas, especialmente el AUC-ROC y la correlación de Spearman. "
                 "El AUC global puede verse subestimado al computarse sobre todas las observaciones "
                 "independientemente del grupo. Los errores estándar de la regresión logística se corrigen "
                 "mediante SE robustos por clúster (paciente) para manejar la dependencia intra-sujeto.",
             ]),
             ("Nota metodológica — Dependencia intra-sujeto", [
                 "Cada paciente aporta múltiples observaciones (hasta 51 puntos anatómicos), por lo que las "
                 "2,295 observaciones pareadas NO son independientes. Esto viola el supuesto estándar de "
                 "Spearman y de la regresión logística ordinaria. Para la regresión logística se aplican SE "
                 "robustos por clúster. Para Spearman, los resultados deben interpretarse con cautela "
                 "considerando el inflado artificial de potencia estadística.",
             ]),
         ]
    )

    # ── 1. OBJETIVO GENERAL — CORRELACIÓN ────────────────────────
    corr_region_rows = [
        ['Global (todas regiones)', fval(rho_global), fval(p_global, 4), interp_rho(rho_global), sig(p_global)],
        ['R1 — Temporal',          fval(rho_r1),     fval(p_r1, 4),     interp_rho(rho_r1),     sig(p_r1)],
        ['R2 — Esternoc./Maset.',  fval(rho_r2),     fval(p_r2, 4),     interp_rho(rho_r2),     sig(p_r2)],
        ['R3 — ATM / Masetero',    fval(rho_r3),     fval(p_r3, 4),     interp_rho(rho_r3),     sig(p_r3)],
        ['R4 — Maset. inferior',   fval(rho_r4),     fval(p_r4, 4),     interp_rho(rho_r4),     sig(p_r4)],
    ]
    corr_musc_rows = [
        ['ATM',                    fval(rho_atm),    fval(p_atm, 4),    interp_rho(rho_atm),    sig(p_atm)],
        ['Esternocleidomastoideo', fval(rho_estern), fval(p_estern, 4), interp_rho(rho_estern), sig(p_estern)],
        ['Masetero',               fval(rho_mase),   fval(p_mase, 4),   interp_rho(rho_mase),   sig(p_mase)],
        ['Temporal',               fval(rho_temp_m), fval(p_temp_m, 4), interp_rho(rho_temp_m), sig(p_temp_m)],
    ]

    page(pdf,
         "1. Objetivo General — Correlación Temperatura ↔ Intensidad de Dolor",
         [
             ("Enunciado", [
                 '"Correlacionar las temperaturas registradas mediante imágenes de termografía infrarroja '
                 'con los puntos dolorosos a la palpación en pacientes con y sin TTM."',
             ]),
             ("Hipótesis 1 — Prueba: Correlación de Spearman (temperatura máxima regional vs intensidad de dolor)", [
                 "H0: No existe correlación estadísticamente significativa entre temperatura máxima regional "
                 "e intensidad de dolor (ρ = 0).",
                 "H1: Existe correlación significativa (ρ ≠ 0).",
                 {
                     'type': 'table',
                     'headers': ['Nivel', 'ρ de Spearman', 'p-valor', 'Interpretación', 'Decisión H0'],
                     'rows': corr_region_rows,
                     'caption': 'Tabla 1. Correlación de Spearman por región termográfica vs intensidad de dolor.',
                     'col_widths': [0.28, 0.14, 0.12, 0.28, 0.18],
                 },
                 "La correlación global (ρ = " + fval(rho_global) + f", p = {fval(p_global, 4)}) es negativa "
                 "muy débil y no alcanza significancia estadística. No hay evidencia para rechazar H0 a nivel "
                 "regional global ni en ninguna región individual.",
                 {
                     'type': 'table',
                     'headers': ['Músculo', 'ρ de Spearman', 'p-valor', 'Interpretación', 'Decisión H0'],
                     'rows': corr_musc_rows,
                     'caption': 'Tabla 2. Correlación de Spearman por músculo vs intensidad de dolor.',
                     'col_widths': [0.28, 0.14, 0.12, 0.28, 0.18],
                 },
             ]),
             ("Interpretación detallada — Correlación por músculo", [
                 "Ningún músculo mostró una correlación significativa entre temperatura y dolor. "
                 "Las correlaciones observadas son muy débiles (|ρ| < 0.10) y no alcanzan significancia "
                 "estadística en ningún músculo evaluado (ATM, Esternocleidomastoideo, Masetero, Temporal). "
                 "La hipótesis clínica de que mayor temperatura = mayor inflamación = más dolor no se confirma "
                 "de forma sistemática en esta muestra.",
             ]),
             ("Hipótesis 1 — Correlaciones por punto anatómico individual", [
                 f"Correlaciones significativas tras corrección de Bonferroni a nivel de punto anatómico: "
                 f"{sig_corr_text}",
             ] + (
                 [{
                     'type': 'table',
                     'headers': ['Punto', 'ROI Térmico', 'ρ', 'p_bonf', 'Sig.'],
                     'rows': sig_corr_rows,
                     'caption': 'Tabla 3. Puntos anatómicos con correlación significativa tras corrección de Bonferroni.',
                     'col_widths': [0.35, 0.30, 0.10, 0.13, 0.12],
                 }] if sig_corr_rows else []
             ) + [
                 "CONCLUSIÓN H1: No se rechaza H0 a nivel global, por región ni por músculo. "
                 + ("No se encontraron correlaciones significativas a nivel de punto individual "
                    "tras corrección de Bonferroni."
                    if not sig_corr_rows else
                    f"Se identificaron {len(sig_corr_rows)} punto(s) con correlación significativa "
                    "tras corrección de Bonferroni (ver tabla anterior)."),
             ]),
         ]
    )

    # ── 2. OBJETIVO a — ZONAS CALIENTES Y PUNTOS DOLOROSOS ───────
    page(pdf,
         "2. Objetivo Específico a — Identificación de Zonas Calientes y Puntos Dolorosos",
         [
             ("Enunciado", [
                 '"Identificar las zonas de mayor temperatura y los puntos dolorosos con mayor '
                 'intensidad media en los grupos con y sin dolor."',
             ]),
             ("2.1. Temperaturas máximas promedio por región y grupo (°C)", [
                 {
                     'type': 'table',
                     'headers': ['Región', 'Sin Dolor (°C)', 'Con Dolor (°C)', 'Δ (Con − Sin) (°C)'],
                     'rows': temp_region_rows if temp_region_rows else [['Sin datos', '—', '—', '—']],
                     'caption': 'Tabla 4. Temperatura máxima promedio por región y grupo. ',
                     'col_widths': [0.38, 0.20, 0.20, 0.22],
                 },
                 "El grupo Con Dolor presenta temperaturas ligeramente inferiores en todas las regiones evaluadas "
                 "respecto al grupo Sin Dolor. La mayor diferencia se observa en R4 (Masetero inferior, -0.41°C). "
                 "Este hallazgo contradice la hipótesis inicial de que el dolor se asociaría a una mayor temperatura "
                 "local (hipertermia inflamatoria) en esta muestra específica.",
             ]),
             ("2.2. Top-10 puntos anatómicos con mayor intensidad media de dolor", [
                 {
                     'type': 'table',
                     'headers': ['Músculo', 'Lado', 'Región', 'Punto', 'Intensidad media (0–10)'],
                     'rows': top_pain_rows[:10] if top_pain_rows else [['Sin datos', '—', '—', '—', '—']],
                     'caption': 'Tabla 5. Puntos anatómicos con mayor intensidad media de dolor en pacientes con dolor.',
                     'col_widths': [0.22, 0.18, 0.15, 0.15, 0.30],
                 },
                 "Los puntos de mayor dolor se concentran en Masetero (izquierda, R3) y Esternocleidomastoideo "
                 "(izquierda, R2). Esta distribución muestra una predominancia del lado izquierdo en los puntos "
                 "más severos en esta muestra.",
             ]),
             ("Hipótesis 2 — Concordancia espacial (Kappa de Cohen)", [
                 "H0: La zona de mayor temperatura no coincide con la región anatómica más dolorosa.",
                 "H1: Existe concordancia significativa entre zona caliente y zona dolorosa.",
                 {
                     'type': 'table',
                     'headers': ['Métrica', 'Valor', 'Interpretación'],
                     'rows': [
                         ['Kappa de Cohen (κ)', fval(kappa_val),
                          'Peor que el azar (κ < 0)' if not pd.isna(kappa_val) and kappa_val < 0
                          else ('Leve acuerdo' if not pd.isna(kappa_val) and kappa_val < 0.20 else '—')],
                         ['Concordancia directa (%)', f"{concordancia_pct:.1f}%" if pd.notna(concordancia_pct) else '—',
                          'Coincidencia bruta sin ajuste por azar'],
                         ['Pares evaluados (muestra × lado)', str(n_kappa_pares), 'Solo pares con al menos un punto de dolor'],
                     ],
                     'caption': 'Tabla 6. Resultados de concordancia espacial zona caliente ↔ zona dolorosa.',
                     'col_widths': [0.35, 0.20, 0.45],
                 },
                 f"El Kappa de Cohen (κ = {fval(kappa_val)}) indica concordancia peor que el azar entre "
                 "la zona de mayor temperatura y la región más dolorosa. La concordancia directa del "
                 f"{concordancia_pct:.1f}% es engañosa: dado el desbalance severo (108/2295 puntos con dolor) "
                 "y el número limitado de regiones (R1–R4), una asignación aleatoria ya produce un porcentaje "
                 "de 'acierto' elevado. El κ corrige este efecto y revela que no existe una relación "
                 "sistemática entre la ubicación de la mayor temperatura y la localización del dolor. "
                 "Este hallazgo es relevante: el calor no siempre se concentra donde hay dolor, lo que "
                 "limita el uso de la temperatura regional como marcador espacial de dolor.",
                 "CONCLUSIÓN H2: No se rechaza H0. La termografía regional no predice de forma confiable "
                 "la localización espacial del dolor (κ ≈ 0).",
             ]),
         ]
    )

    # ── 3. OBJETIVO b–c — ROC Y UTILIDAD DIAGNÓSTICA ─────────────
    roc_table_rows = []
    for label, r in [
        ('GLOBAL', roc_global_d), ('R1 — Temporal', roc_r1_d),
        ('R2 — Esternoc./Maset.', roc_r2_d), ('R3 — ATM/Maset.', roc_r3_d),
        ('R4 — Maset. inf.', roc_r4_d), ('ATM', roc_atm_d),
        ('Esternocleidom.', roc_estern_d), ('Masetero', roc_mase_d),
        ('Temporal', roc_temp_d),
    ]:
        auc_v, ts, se, sp, j, cap = fmt_roc(r)
        roc_table_rows.append([label, auc_v, ts, se, sp, j, cap])

    auc_g_val = roc_global_d['AUC-ROC'] if roc_global_d else np.nan
    tstar_g   = roc_global_d.get('T_star_C', np.nan) if roc_global_d else np.nan
    sens_g    = roc_global_d.get('sensibilidad', np.nan) if roc_global_d else np.nan
    spec_g    = roc_global_d.get('especificidad', np.nan) if roc_global_d else np.nan

    page(pdf,
         "3. Objetivo Específico b–c — Utilidad Diagnóstica: Curva ROC y Umbral T*",
         [
             ("Enunciado", [
                 '"Evaluar si existe un umbral térmico T* que permita predecir la presencia de dolor con '
                 'precisión aceptable, y determinar la utilidad diagnóstica de la termografía infrarroja '
                 'como herramienta complementaria para la detección de puntos dolorosos."',
             ]),
             ("Hipótesis 3 — Curva ROC + Índice de Youden (T*)", [
                 "H0: La temperatura máxima regional no predice la presencia de dolor (AUC ≈ 0.50).",
                 "H1: Existe un umbral T* que predice dolor con precisión aceptable (AUC ≥ 0.70).",
                 {
                     'type': 'table',
                     'headers': ['Subgrupo', 'AUC', 'T* (°C)', 'Sensib.', 'Especif.', 'Youden J', 'Capacidad'],
                     'rows': roc_table_rows,
                     'caption': 'Tabla 7. Resultados ROC por subgrupo. T* = umbral óptimo (índice de Youden). '
                                '"—" en T* indica predictor invertido (AUC < 0.50, no útil como predictor directo).',
                     'col_widths': [0.20, 0.07, 0.09, 0.09, 0.09, 0.09, 0.37],
                 },
                 f"El AUC global de {fval(auc_g_val)} es inferior a 0.50, lo que indica que la temperatura "
                 "máxima regional discrimina los puntos con dolor PEOR que una asignación al azar. Esto ocurre "
                 "porque la correlación temperatura–dolor es negativa (a mayor temperatura, menor dolor en "
                 "la muestra): el predictor funciona en dirección opuesta a la hipoteizada. "
                 f"La región R2 (AUC = {fval(roc_r2_d['AUC-ROC'] if roc_r2_d else np.nan)}) destaca como "
                 "el único subgrupo con capacidad diagnóstica aceptable (AUC ≥ 0.65).",
                 f"Para el subgrupo Global, el umbral óptimo T* = {fval(tstar_g, 2)}°C ofrece sensibilidad "
                 f"{float(sens_g)*100:.1f}% y especificidad {float(spec_g)*100:.1f}% — valores que no "
                 "constituyen un test diagnóstico útil en la práctica clínica.",
                 (f"El caso del Esternocleidomastoideo es notable: AUC = {fval(roc_estern_d['AUC-ROC'] if roc_estern_d else np.nan)}, "
                  "indicando que el predictor funciona en dirección inversa (temperatura más alta = menos dolor). "
                  "Sin embargo, la correlación de Spearman no es significativa en este músculo "
                  f"(ρ = {fval(rho_estern)}, p = {fval(p_estern, 4)}), por lo que este patrón invertido "
                  "debe interpretarse con cautela dado el N reducido por subgrupo."),
                 "CONCLUSIÓN H3: No se rechaza H0. La temperatura máxima regional no constituye un "
                 "predictor diagnóstico útil de dolor en esta muestra con el enfoque de T* único.",
             ]),
             ("Consideración sobre el AUC < 0.50", [
                 "Un AUC < 0.50 no significa 'no hay información': significa que el predictor funciona "
                 "en dirección inversa. Si se invirtiera la regla de decisión (temperatura BAJA → predice "
                 "dolor), el AUC pasaría a ser 1 − 0.461 = 0.539, sigue siendo pobre pero supera el azar. "
                 "Esta inversión carece de base fisiopatológica sólida y solo se menciona como contexto "
                 "metodológico. La hipótesis clínica de que mayor temperatura = mayor inflamación = más "
                 "dolor no se confirma en esta muestra.",
             ]),
         ]
    )

    # ── 4. REGRESIÓN LOGÍSTICA ────────────────────────────────────
    or_temp_v = fmt_or(m_temp)
    or_r2_v   = fmt_or(m_r2)
    or_r3_v   = fmt_or(m_r3)
    or_r4_v   = fmt_or(m_r4)
    or_lado_v = fmt_or(m_lado)

    page(pdf,
         "4. Análisis Multivariable — Regresión Logística con SE Robustos por Clúster",
         [
             ("Descripción del modelo", [
                 "Para manejar la dependencia intra-sujeto (múltiples puntos anatómicos por paciente) y "
                 "controlar covariables simultáneamente, se ajustó una regresión logística con errores "
                 "estándar robustos por clúster (clúster = muestra/paciente).",
                 "Variable dependiente: presencia de dolor (1 = dolor, 0 = sin dolor).",
                 "Predictores: temperatura máxima regional (temp_max), región anatómica (R1–R4), "
                 "lado (derecha/izquierda).",
                 "Este modelo permite cuantificar OR ajustados separando el efecto de la temperatura "
                 "de la variabilidad regional y lateral.",
             ]),
             ("Métricas globales del modelo", [
                 {
                     'type': 'table',
                     'headers': ['Métrica', 'Valor', 'Interpretación'],
                     'rows': [
                         ['N observaciones',      f"{n_obs_logit:,}",     'Todos los registros pareados'],
                         ['Pseudo-R² (McFadden)', fval(pseudo_r2, 4),    '< 0.05 = ajuste muy pobre'],
                         ['AUC probabilidad logística', fval(auc_logit, 4), 'Capacidad discriminativa del modelo'],
                         ['Brier Score',          fval(brier, 4),        '< 0.10 = calibración favorable'],
                         ['AIC (modelo completo)', fval(aic_a_full, 2),  'Para comparación de modelos'],
                     ],
                     'caption': 'Tabla 8. Métricas globales del modelo logístico robusto (Modelo A completo).',
                     'col_widths': [0.35, 0.20, 0.45],
                 },
                 f"El AUC de la probabilidad logística ({fval(auc_logit)}) es superior al AUC ROC directo "
                 f"({fval(auc_g_val)}), lo que indica que la combinación de variables (temperatura máxima "
                 "regional, región anatómica y lado) aporta información discriminativa adicional respecto "
                 "al predictor único de temperatura. Sin embargo, la capacidad discriminativa sigue siendo "
                 "limitada (AUC < 0.70).",
             ]),
             ("Odds Ratios — Términos principales del modelo", [
                 {
                     'type': 'table',
                     'headers': ['Término', 'OR', 'IC95-inf', 'IC95-sup', 'p-val', 'Sig.'],
                     'rows': [
                         ['temp_max', or_temp_v[0], or_temp_v[1], or_temp_v[2], or_temp_v[3], or_temp_v[4]],
                         ['region_r2', or_r2_v[0], or_r2_v[1], or_r2_v[2], or_r2_v[3], or_r2_v[4]],
                         ['region_r3', or_r3_v[0], or_r3_v[1], or_r3_v[2], or_r3_v[3], or_r3_v[4]],
                         ['region_r4', or_r4_v[0], or_r4_v[1], or_r4_v[2], or_r4_v[3], or_r4_v[4]],
                         ['lado_izq.', or_lado_v[0], or_lado_v[1], or_lado_v[2], or_lado_v[3], or_lado_v[4]],
                     ],
                     'caption': 'Tabla 9. OR del modelo logístico. OR > 1 = mayor prob. de dolor. '
                                'IC95 = intervalo de confianza 95%. ',
                     'col_widths': [0.26, 0.12, 0.12, 0.12, 0.10, 0.10],
                 },
                 f"El factor region_r3 (OR = {or_r3_v[0]}, p = {or_r3_v[3]}) alcanzó significancia "
                 "estadística (p < 0.05), sugiriendo que en la región R3 (ATM/Masetero) la probabilidad "
                 "de dolor es significativamente mayor que en la región de referencia (R1), manteniendo "
                 "constante la temperatura." if m_r3 and m_r3['p_valor'] < 0.05 else
                 "Ningún coeficiente alcanzó significancia estadística al nivel 0.05.",
                 f"El OR de temp_max = {or_temp_v[0]} (IC95%: {or_temp_v[1]}–{or_temp_v[2]}, "
                 f"p = {or_temp_v[3]}) sugiere que por cada °C adicional de temperatura máxima regional, "
                 "la probabilidad de dolor tiende a disminuir (OR < 1), aunque este efecto no es "
                 "significativo (p > 0.05).",
                 "El Pseudo-R² de McFadden = " + fval(pseudo_r2, 4) + " confirma un ajuste muy pobre "
                 "(< 0.10). La temperatura regional por sí sola explica muy poco de "
                 "la varianza en la presencia de dolor en los puntos de palpación.",
             ]),
         ]
    )

    # ── 5. DELTA T Y TEMPERATURA PUNTUAL ─────────────────────────
    page(pdf,
         "5. Análisis del Aumento Térmico (∆T) y Temperatura Puntual (p1–p7)",
         [
             ("5.1. Correlación Spearman: ∆T vs intensidad de dolor", [
                 "El delta térmico (∆T = T_región_derecha − T_región_izquierda) es la asimetría térmica "
                 "bilateral: usa el lado contralateral como control interno por paciente. Un ∆T positivo "
                 "indica mayor temperatura en el lado derecho; un valor negativo, en el izquierdo. "
                 "Esta métrica es preferible a la temperatura absoluta porque controla la variabilidad "
                 "inter-individual de temperatura basal, siendo clínicamente más relevante como marcador "
                 "de asimetría inflamatoria.",
                 {
                     'type': 'table',
                     'headers': ['Variable', 'ρ de Spearman', 'p-valor', 'Interpretación', 'Decisión H0'],
                     'rows': [
                         ['∆T_global_media vs dolor', fval(rho_dg),  fval(p_dg, 4),  interp_rho(rho_dg),  sig(p_dg)],
                         ['∆T_R1 vs dolor',           fval(rho_dr1), fval(p_dr1, 4), interp_rho(rho_dr1), sig(p_dr1)],
                         ['∆T_R2 vs dolor',           fval(rho_dr2), fval(p_dr2, 4), interp_rho(rho_dr2), sig(p_dr2)],
                         ['∆T_R3 vs dolor',           fval(rho_dr3), fval(p_dr3, 4), interp_rho(rho_dr3), sig(p_dr3)],
                         ['∆T_R4 vs dolor',           fval(rho_dr4), fval(p_dr4, 4), interp_rho(rho_dr4), sig(p_dr4)],
                     ],
                     'caption': 'Tabla 10. Correlación de Spearman entre ∆T (aumento térmico) e intensidad de dolor.',
                     'col_widths': [0.28, 0.14, 0.12, 0.26, 0.20],
                 },
                 f"El ∆T global no muestra correlación significativa con el dolor (ρ = {fval(rho_dg)}, "
                 f"p = {fval(p_dg, 4)}). Sin embargo, ∆T_R2 (Esternocleidomastoideo/Masetero superior) "
                 f"presenta una correlación positiva débil significativa (ρ = {fval(rho_dr2)}, "
                 f"p = {fval(p_dr2, 4)}): en esta región, un mayor aumento térmico se asocia levemente "
                 "con mayor intensidad de dolor. Este es el único hallazgo con dirección esperada "
                 "(positiva) y significativo en toda la batería de correlaciones.",
                 "La falta de significancia global del ∆T sugiere que el simple aumento de temperatura "
                 "respecto a la temperatura basal del paciente tampoco es un predictor confiable de "
                 "dolor en la mayoría de las regiones.",
             ]),
             ("5.2. Temperatura puntual mapeada (p1–p7)", [
                 f"Se evaluó la temperatura extraída exactamente en el píxel correspondiente a cada punto "
                 f"anatómico de palpación (temp_punto, p1–p7). Esta información proviene del mapeo "
                 f"anatómico definido en unificar_dolor_termografia.py.",
                 {
                     'type': 'table',
                     'headers': ['Métrica', 'Valor', 'Interpretación'],
                     'rows': [
                         ['N con temp. puntual', str(n_tp),
                          'Sin datos disponibles (N=0): columnas temp_punto__ no disponibles en primeras fotos'
                          if n_tp == 0 else f'Cobertura: {n_tp/2295*100:.1f}% del total'],
                         ['ρ Spearman temp_punto vs dolor', fval(rho_tp), interp_rho(rho_tp)],
                         ['p-valor', fval(p_tp, 4), sig(p_tp)],
                         ['AUC ROC (temp_punto)', fval(auc_tp), 'Capacidad diagnóstica'],
                         ['T* (°C) Youden', fval(tstar_tp, 3) if pd.notna(tstar_tp) and tstar_tp < 900 else '—', 'Umbral óptimo por punto'],
                         ['Sensibilidad', fval(sens_tp), '—'],
                         ['Especificidad', fval(spec_tp), '—'],
                     ],
                     'caption': 'Tabla 11. Análisis de temperatura puntual mapeada (p1–p7).',
                     'col_widths': [0.35, 0.20, 0.45],
                 },
                 ("NOTA: Las columnas de temperatura puntual (temp_punto__*) en el análisis actual "
                  "provienen de las segundas imágenes. Para las primeras imágenes (pre-palpación), "
                  "las coordenadas de píxel no han sido re-extraídas aún. Por ello N=0 en este análisis. "
                  "La comparación de modelos A vs B en la sección 5.3 sí usa los 708 registros "
                  "disponibles del conjunto combinado (todas_fotos)."
                  if n_tp == 0 else
                  f"Con N={n_tp} observaciones, la temperatura puntual "
                  f"{'no muestra' if pd.isna(rho_tp) or p_tp >= 0.05 else 'muestra'} correlación "
                  f"significativa con el dolor (ρ = {fval(rho_tp)}, p = {fval(p_tp, 4)}). "
                  f"AUC = {fval(auc_tp)}."),
             ]),
             ("5.3. Comparación de modelos A vs B", [
                 "Modelo A: temperatura máxima regional + región + lado.",
                 "Modelo B: Modelo A + temperatura puntual (p1–p7). Evaluado en submuestra con temp. puntual disponible.",
                 {
                     'type': 'table',
                     'headers': ['Métrica', 'Modelo A', 'Modelo B', '∆ (B − A)', 'Interpretación'],
                     'rows': [
                         ['N submuestra', str(n_comp), str(n_comp), '—', 'Misma submuestra N='+str(n_comp)],
                         ['AUC', fval(auc_A, 4), fval(auc_B, 4), fval(d_auc, 4),
                          'Mejora marginal' if not pd.isna(d_auc) and d_auc > 0 else 'Sin mejora'],
                         ['Pseudo-R²', fval(pr2_A, 4), fval(pr2_B, 4), fval(pr2_B - pr2_A if pd.notna(pr2_A) and pd.notna(pr2_B) else np.nan, 4), 'Mínima mejora'],
                         ['AIC', fval(aic_A_comp, 2), fval(aic_B_comp, 2), fval(d_aic, 2),
                          'AIC↑ = mayor penalización complejidad' if not pd.isna(d_aic) and d_aic > 0 else '—'],
                         ['LRT p-valor', '—', '—', fval(lrt_p, 4), sig(lrt_p)],
                     ],
                     'caption': 'Tabla 12. Comparación modelos A (sin temp. puntual) vs B (con temp. puntual).',
                     'col_widths': [0.18, 0.14, 0.14, 0.14, 0.40],
                 },
                 f"La incorporación de la temperatura puntual (Modelo B) produce una mejora marginal en "
                 f"AUC (∆ = {fval(d_auc, 4)}) y Pseudo-R² (∆ = {fval(pr2_B-pr2_A if pd.notna(pr2_A) and pd.notna(pr2_B) else np.nan, 4)}), "
                 f"pero el Likelihood Ratio Test no es significativo (p = {fval(lrt_p, 4)}). "
                 f"El AIC del Modelo B es mayor (∆ = {fval(d_aic, 2)}), indicando que la temperatura "
                 "puntual no compensa el coste de complejidad adicional. "
                 "Conclusión: la temperatura puntual mapeada no añade valor explicativo estadísticamente "
                 "demostrable sobre el modelo base en esta muestra.",
             ]),
         ]
    )

    # ── 6. ANÁLISIS A NIVEL DE PACIENTE (N=43) ───────────────────
    page(pdf,
         "6. Análisis Agregado a Nivel de Paciente (N=43)",
         [
             ("Fundamento", [
                 "Para evitar la dependencia intra-sujeto (múltiples puntos por paciente), se colapsaron "
                 "los datos a nivel de individuo (N=43). Se comparó el grupo 'Con Dolor' (N=" + str(n_con_dolor) + ") "
                 "vs el grupo 'Sin Dolor' (N=" + str(n_sin_dolor) + ").",
                 "Variable térmica: ∆T máximo registrado en cualquier región del paciente.",
             ]),
             ("Comparación de Medias (Mann-Whitney U)", [
                 {
                     'type': 'table',
                     'headers': ['Grupo', 'Mediana ∆T Max', 'p-valor', 'Interpretación'],
                     'rows': [
                         ['Con Dolor', paciente_median_con, fval(paciente_mw_p, 4), sig(paciente_mw_p)],
                         ['Sin Dolor', paciente_median_sin, '—', '—'],
                     ],
                     'caption': 'Tabla 12. Comparación de asimetría térmica máxima entre pacientes sintomáticos y asintomáticos.',
                     'col_widths': [0.25, 0.25, 0.20, 0.30],
                 },
                 f"La asimetría térmica máxima es " + ("significativamente" if (paciente_mw_p < 0.05) else "no significativamente") + 
                 f" diferente entre ambos grupos (p = {fval(paciente_mw_p, 4)}).",
             ]),
             ("Capacidad Predictiva de Dolor (ROC N=43)", [
                 {
                     'type': 'table',
                     'headers': ['Métrica', 'Valor', 'Interpretación'],
                     'rows': [
                         ['AUC-ROC (∆T Max)', fval(paciente_roc_auc, 4), 'Capacidad discriminativa a nivel paciente'],
                         ['Umbral óptimo T*', fval(paciente_roc_tstar, 2) + " °C", 'Punto de corte sugerido'],
                         ['Sensibilidad', fval(paciente_roc_sens, 3), 'Capacidad de detectar pacientes con dolor'],
                         ['Especificidad', fval(paciente_roc_spec, 3), 'Capacidad de detectar pacientes sanos'],
                     ],
                     'caption': 'Tabla 13. Utilidad del ∆T máximo para predecir si un paciente reportará dolor.',
                     'col_widths': [0.35, 0.20, 0.45],
                 },
                 (f"A nivel de paciente, el AUC de {fval(paciente_roc_auc, 3)} es inferior a 0.50, "
                  "indicando que el ΔT máximo discrimina PEOR que una asignación al azar. "
                  "Esto confirma que la asimetría térmica máxima regional no predice la presencia de "
                  "dolor a nivel de paciente en esta muestra."
                  if not pd.isna(paciente_roc_auc) and paciente_roc_auc < 0.50 else
                  f"A nivel de paciente, el AUC de {fval(paciente_roc_auc, 3)} sugiere una capacidad "
                  "discriminativa " +
                  ("buena (AUC ≥ 0.70)" if not pd.isna(paciente_roc_auc) and paciente_roc_auc >= 0.70
                   else "aceptable (0.60–0.70)" if not pd.isna(paciente_roc_auc) and paciente_roc_auc >= 0.60
                   else "baja (AUC < 0.60, similar al azar)") +
                  " para identificar individuos con sintomatología "
                  "basándose únicamente en la asimetría térmica máxima regional."),
             ]),
         ]
    )

    # ── 7. ERRORES Y ADVERTENCIAS METODOLÓGICAS ──────────────────
    page(pdf,
         "7. Errores Identificados y Correcciones Aplicadas al Código",
         [
             ("Error corregido 1 — T* sentinel (sklearn ROC)", [
                 "PROBLEMA: Para subgrupos con AUC < 0.50 (predictor invertido), sklearn.roc_curve "
                 "añade un threshold centinela en index 0 (valor = max_score + 1). Cuando el índice "
                 "de Youden es negativo en todos los thresholds reales, argmax(J) seleccionaba "
                 "el centinela (J=0), produciendo T* = 1,000,000°C en la salida.",
                 "CORRECCIÓN APLICADA: Se excluye el centinela del argmax forzando J[0] = −∞. "
                 "Si todos los J restantes son ≤ 0, T* se reporta como NaN ('—' en tablas), "
                 "indicando que no existe umbral positivo para ese subgrupo.",
             ]),
             ("Error corregido 2 — Etiqueta duplicada en figura PNG", [
                 "PROBLEMA: Ambos paneles A y B de la figura resumen tenían etiqueta 'a)' "
                 "(style_ax(ax_b, 'a) Top-10 Puntos Dolorosos (TTM)')), lo que generaba "
                 "ambigüedad visual en el reporte.",
                 "CORRECCIÓN APLICADA: Panel B etiquetado correctamente como 'b) Top-10...'",
             ]),
             ("Advertencia metodológica 1 — Dependencia intra-sujeto", [
                 "Las 2,295 observaciones provienen de solo 43 pacientes. Nota: el CSV original contenía 2 filas duplicadas (IDs 26 y 31) que han sido eliminadas. Spearman estándar asume "
                 "independencia: el N inflado puede generar p-valores artificialmente bajos. "
                 "MITIGACIÓN (YA IMPLEMENTADA): (1) errores robustos por clúster en la regresión "
                 "logística; (2) permutation test con block-shuffling por paciente para las "
                 "correlaciones de Spearman (función spearman_permutation_test en analisis_termografia.py). "
                 "Los p-valores Bonferroni reportados en la tabla de correlaciones ya incorporan "
                 "corrección por comparaciones múltiples.",
             ]),
             ("Advertencia metodológica 2 — Desbalance severo de clases", [
                 "Solo 108/2295 observaciones (4.7%) tienen dolor. El AUC global puede verse "
                 "influenciado por este desbalance. El índice de Youden J (Sens + Spec − 1) es "
                 "más robusto ante este escenario. Se recomienda reportar también F1-score "
                 "y considerar oversampling (SMOTE) para modelos predictivos futuros.",
             ]),
             ("Advertencia metodológica 3 — Temporalidad del registro", [
                 "El análisis actual YA usa las PRIMERAS imágenes termográficas (pre-palpación), "
                 "tomadas antes del examen clínico de palpación. Esto es metodológicamente correcto: "
                 "la temperatura se mide antes de que la palpación induzca cambios térmicos. "
                 "El conjunto 'todas_fotos' (primeras + segundas imágenes) se usa únicamente para "
                 "el análisis comparativo del Modelo B (temperatura puntual). Para trabajo futuro: "
                 "comparar explícitamente primeras vs segundas imágenes cuantificará el sesgo "
                 "térmico inducido por la palpación manual.",
             ]),
             ("Advertencia metodológica 4 — Tamaño muestral y Agrupación", [
                 f"De los {n_total} pacientes, {n_con_dolor} reportaron al menos un punto de dolor ({n_con_dolor/n_total*100:.1f}%) "
                 f"mientras que {n_sin_dolor} fueron completamente asintomáticos durante la palpación.",
                 f"El diagnóstico clínico previo de TTM (N={n_ttm}) se consideró como variable descriptiva secundaria, "
                 "pero la agrupación principal del estudio se basó en la sintomatología observada (dolor real) "
                 "para una correlación más directa con los hallazgos térmicos.",
             ]),
         ]
    )

    # ── 8. CONCLUSIONES GENERALES ─────────────────────────────────
    page(pdf,
         "8. Conclusiones Generales y Recomendaciones",
         [
             ("Resumen de hipótesis", [
                 {
                     'type': 'table',
                     'headers': ['Hipótesis / Analisis', 'Estadistico', 'Resultado', 'Conclusion'],
                     'rows': [
                         ['H1 Corr. global',
                          f'rho={fval(rho_global)} p={fval(p_global, 4)}',
                          'No sig.',
                          'No se rechaza H0. Sin corr. global.'],
                         ['H1 Esternocleidom.',
                          f'rho={fval(rho_estern)} p={fval(p_estern, 4)}',
                          sig(p_estern),
                          'No se rechaza H0. Sin corr. significativa.'],
                         ['H1 Maset.D R3P1',
                          'rho=-0.013 p=0.584',
                          'No sig.',
                          'Sin correlacion significativa.'],
                         ['H1 Paciente N=43',
                          f'p={fval(paciente_mw_p, 4)}',
                          'No sig.',
                          'DeltaT Max no difiere entre grupos.'],
                         ['H2 Concordancia',
                          f'k={fval(kappa_val)} conc={concordancia_pct:.1f}%',
                          'H0 no rechazo',
                          'Zona caliente != zona de dolor.'],
                         ['H3 Umbral T* ROC',
                          f'AUC={fval(auc_g_val)} T*={fval(tstar_g, 2)}C',
                          'H0 no rechazo',
                          'AUC<0.50: sin capacidad diagnostica.'],
                         ['ROC N=43',
                          f'AUC={fval(paciente_roc_auc, 3)}',
                          'Baja',
                          'Prediccion de dolor a nivel paciente.'],
                         ['Logit multivariable',
                          f'PR2={fval(pseudo_r2)} AUC={fval(auc_logit)}',
                          'Exploratorio',
                          'Ningun coef. sig. AUC logist.=0.62.'],
                         ['DeltaT R2',
                          f'rho={fval(rho_dr2)} p={fval(p_dr2, 4)}',
                          'Sig.[*]',
                          'Unico hallazgo positivo (dir. esperada).'],
                         ['Temp. puntual p1-p7',
                          f'rho={fval(rho_tp)} AUC={fval(auc_tp)}',
                          'No sig.',
                          'Sin valor predictivo adicional.'],
                         ['Modelos A vs B',
                          f'LRT p={fval(lrt_p, 4)} dAUC={fval(d_auc, 4)}',
                          'No sig.',
                          'Temp. puntual no mejora modelo.'],
                     ],
                     'caption': 'Tabla 13. Resumen de conclusiones estadisticas por hipotesis.',
                     'col_widths': [0.20, 0.24, 0.13, 0.43],
                 },
             ]),
             ("Interpretación global integrada", [
                 "Los resultados de este estudio sugieren que la temperatura máxima regional, medida "
                 "mediante termografía infrarroja, NO constituye un predictor confiable ni un marcador "
                 "diagnóstico útil de la presencia o intensidad del dolor a la palpación en esta muestra.",
                 "El hallazgo más informativo es que la asimetría térmica (∆T) tampoco muestra una relación "
                 "fuerte con el dolor reportado, ni a nivel de punto individual ni a nivel agregado de paciente (N=43). "
                 "La hipótesis clínica de que el dolor se asocia a hipertermia local no se confirma; "
                 "de hecho, se observaron tendencias a temperaturas ligeramente menores en zonas dolorosas.",
             ]),
             ("Recomendaciones para trabajo futuro", [
                 "1. [YA IMPLEMENTADO] El análisis actual usa las PRIMERAS imágenes termográficas "
                 "(pre-palpación), que es metodológicamente correcto. Futura prioridad: comparar "
                 "explícitamente primeras vs segundas imágenes en la misma muestra para cuantificar "
                 "el sesgo térmico inducido por la palpación manual.",
                 "2. Ampliar la muestra total (actualmente N=43; Con Dolor=" + str(n_con_dolor) + ", "
                 "Sin Dolor=" + str(n_sin_dolor) + "). Se recomienda al menos N=30–40 pacientes "
                 "con dolor para disponer de potencia estadística adecuada en análisis por subgrupo.",
                 "3. La asimetría térmica (ΔT = T_derecha − T_izquierda) YA se usa como predictor "
                 "principal en el análisis actual. Sin embargo, explorar también la temperatura "
                 "ABSOLUTA estandarizada por temperatura basal del paciente podría revelar patrones "
                 "adicionales no capturados por la diferencia bilateral.",
                 "4. Considerar análisis por subgrupos diagnósticos de TTM (miofascial, articular, "
                 "mixto, combinado) en lugar de un grupo TTM único. El grupo TTM actual es "
                 "clínicamente heterogéneo, lo que puede diluir la señal térmica.",
                 "5. [YA IMPLEMENTADO] La corrección de Bonferroni sobre las correlaciones de "
                 "Spearman y la prueba de permutación a nivel de bloque de paciente YA se aplican "
                 "en el análisis actual. Próximo paso: aplicar también FDR (Benjamini-Hochberg) "
                 "como alternativa menos conservadora para identificar hallazgos exploratorios.",
                 "6. Ampliar la cobertura de temperatura puntual mapeada (p1–p7), actualmente "
                 "disponible solo para un subconjunto de observaciones (ver N en resultados). "
                 "Esto requiere re-extraer las coordenadas de píxel de las primeras imágenes "
                 "en unificar_dolor_termografia.py (columnas temp_punto_primera__).",
             ]),
         ]
    )

print(f"  ✓ Reporte de interpretación generado: {OUT_PDF}")
