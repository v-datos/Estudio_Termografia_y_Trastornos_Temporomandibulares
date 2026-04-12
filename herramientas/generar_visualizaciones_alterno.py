"""
generar_visualizaciones_alterno.py
Produces 6 publication-quality PNG figures for analisis_termografia_complementario.py results.
All saved in resultado_complementario/
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

warnings.filterwarnings('ignore')

# ─── Output directory ────────────────────────────────────────────────────────
OUTDIR = 'resultado_complementario'
RAWDIR = 'resultados'
os.makedirs(OUTDIR, exist_ok=True)

# ─── Design tokens ───────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.facecolor': '#F8F9FA', 'figure.facecolor': 'white',
    'grid.color': '#E2E8F0', 'grid.linewidth': 0.5, 'axes.grid': True,
})

NAVY   = '#0D2B4E'
TEAL   = '#1A7A8A'
SLATE  = '#4A5568'
GOLD   = '#C8962A'
RED    = '#9B2335'
GREEN  = '#1A6B3C'
LIGHT  = '#E8F4F8'
BORDER = '#CBD5E0'
C_CON  = '#9B2335'
C_SIN  = '#1A7A8A'

# ─── Load data ───────────────────────────────────────────────────────────────
df_asim    = pd.read_csv(f'{OUTDIR}/asimetria_derecha_basal_complementario.csv')
df_corr_n  = pd.read_csv(f'{OUTDIR}/correlaciones_normalizadas_complementario.csv')
df_modelos = pd.read_csv(f'{OUTDIR}/comparacion_modelos_completa_complementario.csv')
df_logit_e = pd.read_csv(f'{OUTDIR}/resultados_logit_extendido_complementario.csv')
df_roc_n   = pd.read_csv(f'{OUTDIR}/resultados_roc_normalizados_complementario.csv')
df_conf    = pd.read_csv(f'{OUTDIR}/efectos_sexo_edad_temperatura_complementario.csv')
df_pac_alt = pd.read_csv(f'{OUTDIR}/resultados_paciente_n45_alterno_complementario.csv')
df_roc_raw = pd.read_csv(f'{RAWDIR}/resultados_roc.csv')
df_pac     = pd.read_csv(f'{RAWDIR}/resultados_paciente_n45.csv')

df_raw = pd.read_csv('datos/datos_finales_termografia_procesados_todas_fotos.csv', encoding='utf-8-sig')
df_raw.columns = df_raw.columns.str.strip().str.lower()

# ─── Compute pain info from raw data ─────────────────────────────────────────
from herramientas.constantes import PAIN_COL_REGEX
PAIN_COLS = [c for c in df_raw.columns if PAIN_COL_REGEX.match(c.strip().lower())]
df_raw[PAIN_COLS] = df_raw[PAIN_COLS].fillna(0)
df_raw['total_pain'] = df_raw[PAIN_COLS].sum(axis=1)
df_raw['con_dolor'] = (df_raw[PAIN_COLS].max(axis=1) > 0).map({True: 'Con Dolor', False: 'Sin Dolor'})

# Normalized temps
for r in ['r1', 'r2', 'r3', 'r4']:
    for s in ['derecha', 'izquierda']:
        col = f'{r}: temperatura media {s}_primera'
        if col in df_raw.columns:
            df_raw[f'norm_{r}_{s}'] = df_raw[col] - df_raw['temperatura basal_primera']
    if f'norm_{r}_derecha' in df_raw.columns and f'norm_{r}_izquierda' in df_raw.columns:
        df_raw[f'delta_t_norm_{r}'] = df_raw[f'norm_{r}_derecha'] - df_raw[f'norm_{r}_izquierda']


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE A1 — Asimetría Derecha Sistemática
# ═══════════════════════════════════════════════════════════════════════════════
def figure_A1():
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(10, 6))
    fig.patch.set_facecolor('white')
    fig.suptitle(
        'Asimetría Térmica Derecha: Rasgo Poblacional vs. Marcador de Dolor',
        fontsize=13, fontweight='bold', y=0.97
    )

    region_labels = ['R1\n(Temporal)', 'R2\n(Esternoc./Maset.)', 'R3\n(ATM/Maset.)', 'R4\n(Maset. inf.)']
    x = np.arange(len(region_labels))
    width = 0.55

    # ── Left panel: mean ΔT with SD error bars ─────────────────────────────
    ax_l.set_facecolor('#F8F9FA')
    bars = ax_l.bar(
        x, df_asim['media_delta_t'], width,
        yerr=df_asim['sd_delta_t'], capsize=4,
        color=TEAL, alpha=0.8, edgecolor=NAVY, linewidth=0.7,
        error_kw=dict(ecolor=SLATE, elinewidth=1.2)
    )
    ax_l.axhline(0, color=GOLD, linestyle='--', linewidth=1.5, zorder=3, label='Simetría bilateral')

    # Annotate % > 0 and p-value
    for i, row in df_asim.iterrows():
        pct = row['pct_pacientes_pos']
        p   = row['wilcoxon_p']
        yv  = row['media_delta_t']
        sd  = row['sd_delta_t']
        ax_l.text(i, yv + sd + 0.04, f"{pct:.0f}%\n>0",
                  ha='center', va='bottom', fontsize=8.5, color=NAVY, fontweight='bold')
        ax_l.text(i, -0.05, f"p={p:.2e}",
                  ha='center', va='top', fontsize=7.5, color=SLATE, style='italic')

    ax_l.set_xticks(x)
    ax_l.set_xticklabels(region_labels, fontsize=9)
    ax_l.set_ylabel('ΔT medio (°C)  [T_derecha − T_izquierda]', fontsize=10)
    ax_l.set_title('ΔT medio por región (primeras imágenes)', fontsize=10, fontweight='bold')
    ax_l.set_ylim(-0.18, 2.0)
    ax_l.spines['top'].set_visible(False)
    ax_l.spines['right'].set_visible(False)

    # Annotation box
    ax_l.annotate(
        'Todas las regiones: ΔT > 0\n(lado derecho más caliente)',
        xy=(0.5, 0.88), xycoords='axes fraction', ha='center',
        fontsize=8.5, color=NAVY,
        bbox=dict(boxstyle='round,pad=0.4', fc=LIGHT, ec=BORDER, lw=1)
    )
    ax_l.legend(fontsize=8, loc='upper right')

    # ── Right panel: |ΔT| Con Dolor vs Sin Dolor ───────────────────────────
    ax_r.set_facecolor('#F8F9FA')
    region_cols = ['delta_t_r1_primera', 'delta_t_r2_primera',
                   'delta_t_r3_primera', 'delta_t_r4_primera']

    # Compute absolute delta from raw data by grupo
    medians_cd, medians_sd = [], []
    for col in region_cols:
        if col in df_raw.columns:
            abs_vals = df_raw[col].abs()
            medians_cd.append(abs_vals[df_raw['con_dolor'] == 'Con Dolor'].median())
            medians_sd.append(abs_vals[df_raw['con_dolor'] == 'Sin Dolor'].median())
        else:
            medians_cd.append(np.nan)
            medians_sd.append(np.nan)

    w2 = 0.35
    bars_cd = ax_r.bar(x - w2 / 2, medians_cd, w2,
                       color=C_CON, alpha=0.8, label='Con Dolor', edgecolor=NAVY, linewidth=0.7)
    bars_sd = ax_r.bar(x + w2 / 2, medians_sd, w2,
                       color=C_SIN, alpha=0.8, label='Sin Dolor', edgecolor=NAVY, linewidth=0.7)

    # Annotate MW p-values
    for i, row in df_asim.iterrows():
        p = row['mw_cd_vs_sd_p']
        ax_r.text(i, max(medians_cd[i] or 0, medians_sd[i] or 0) + 0.04,
                  f"p={p:.3f}", ha='center', va='bottom', fontsize=7.5, color=SLATE, style='italic')

    ax_r.set_xticks(x)
    ax_r.set_xticklabels(region_labels, fontsize=9)
    ax_r.set_ylabel('|ΔT| mediana (°C)', fontsize=10)
    ax_r.set_title('|ΔT| por grupo — Con Dolor vs Sin Dolor', fontsize=10, fontweight='bold')
    ax_r.legend(fontsize=9)
    ax_r.spines['top'].set_visible(False)
    ax_r.spines['right'].set_visible(False)

    ax_r.annotate(
        'Diferencias NS en todos los grupos',
        xy=(0.5, 0.88), xycoords='axes fraction', ha='center',
        fontsize=8.5, color=SLATE,
        bbox=dict(boxstyle='round,pad=0.4', fc=LIGHT, ec=BORDER, lw=1)
    )

    fig.text(
        0.5, 0.01,
        'Fig. A1. La asimetría derecha es sistemática en toda la muestra (Wilcoxon p<0.0001) '
        'pero no distingue pacientes con y sin dolor.',
        ha='center', va='bottom', fontsize=8.5, color=SLATE, style='italic',
        wrap=True
    )

    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    out = f'{OUTDIR}/viz_A1_asimetria_derecha_complementario.png'
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE A2 — Temperatura Normalizada vs Intensidad de Dolor
# ═══════════════════════════════════════════════════════════════════════════════
def figure_A2():
    region_names = {
        'r1': 'R1 (Temporal)',
        'r2': 'R2 (Esternoc./Maset.)',
        'r3': 'R3 (ATM/Maset.)',
        'r4': 'R4 (Maset. inf.)'
    }

    fig, axes = plt.subplots(1, 4, figsize=(12, 5), sharey=True)
    fig.patch.set_facecolor('white')
    fig.suptitle(
        'Temperatura Normalizada (T−T_basal) vs. Intensidad de Dolor por Región',
        fontsize=12, fontweight='bold', y=0.97
    )

    color_map = {'Con Dolor': C_CON, 'Sin Dolor': C_SIN}

    for ax, (r, rname) in zip(axes, region_names.items()):
        ax.set_facecolor('#F8F9FA')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        col_x = f'delta_t_norm_{r}'
        if col_x not in df_raw.columns:
            ax.set_title(rname, fontsize=9, fontweight='bold')
            ax.set_xlabel('ΔT norm (°C)', fontsize=9)
            continue

        valid = df_raw[[col_x, 'total_pain', 'con_dolor']].dropna()
        if len(valid) < 5:
            ax.set_title(rname, fontsize=9, fontweight='bold')
            continue

        x_vals = valid[col_x].values
        y_vals = valid['total_pain'].values
        groups = valid['con_dolor'].values

        for g in ['Con Dolor', 'Sin Dolor']:
            mask = groups == g
            ax.scatter(x_vals[mask], y_vals[mask],
                       color=color_map[g], alpha=0.65, s=28,
                       edgecolors='white', linewidths=0.4, label=g, zorder=3)

        # Linear regression line
        if len(x_vals) > 2:
            m, b = np.polyfit(x_vals, y_vals, 1)
            xfit = np.linspace(x_vals.min(), x_vals.max(), 100)
            ax.plot(xfit, m * xfit + b, '--', color=SLATE, linewidth=1.0, alpha=0.6, zorder=2)

        # Spearman from df_corr_n
        row = df_corr_n[(df_corr_n['nivel'] == 'Región') & (df_corr_n['subgrupo'] == r)]
        if len(row) > 0:
            rho  = row.iloc[0]['rho']
            p_bf = row.iloc[0]['p_bonf']
            p_or = row.iloc[0]['p_orig']
            ax.annotate(
                f"ρ={rho:+.3f}\np={p_or:.3f}\np_bonf={p_bf:.3f}",
                xy=(0.97, 0.97), xycoords='axes fraction',
                ha='right', va='top', fontsize=7.5,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=BORDER, lw=0.8, alpha=0.9)
            )

        ax.set_title(rname, fontsize=9, fontweight='bold')
        ax.set_xlabel('ΔT norm (°C)', fontsize=9)
        if ax == axes[0]:
            ax.set_ylabel('Intensidad total', fontsize=9)

    # Shared legend
    handles = [
        mpatches.Patch(color=C_CON, label='Con Dolor'),
        mpatches.Patch(color=C_SIN, label='Sin Dolor'),
        mlines.Line2D([], [], color=SLATE, linestyle='--', linewidth=1, label='Regresión lineal'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, 0.08), frameon=True)

    fig.text(
        0.5, 0.01,
        'Fig. A2. La normalización por temperatura basal no revela correlaciones significativas '
        'con el dolor en ninguna región (p_bonf=1.000 en todas).',
        ha='center', va='bottom', fontsize=8.5, color=SLATE, style='italic'
    )

    plt.tight_layout(rect=[0, 0.12, 1, 0.94])
    out = f'{OUTDIR}/viz_A2_temp_normalizada_dolor_complementario.png'
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE A3 — Comparación de Modelos Logísticos
# ═══════════════════════════════════════════════════════════════════════════════
def figure_A3():
    model_labels = ['Modelo A\n(base)', 'Modelo C\n(+sexo+edad)',
                    'Modelo D\n(norm)', 'Modelo E\n(norm+sexo+edad)']
    model_order = ['A', 'C', 'D', 'E']

    # Reorder df_modelos
    df_m = df_modelos.set_index('modelo').loc[model_order].reset_index()

    aic_vals = df_m['aic'].values
    auc_vals = df_m['auc_roc'].values
    aic_base = aic_vals[0]  # Model A reference

    # Colors: GREEN=best (lowest AIC), NAVY=reference A, TEAL=others
    best_idx = np.argmin(aic_vals)
    bar_colors = []
    for i, m in enumerate(model_order):
        if i == best_idx:
            bar_colors.append(GREEN)
        elif m == 'A':
            bar_colors.append(NAVY)
        else:
            bar_colors.append(TEAL)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11, 6))
    fig.patch.set_facecolor('white')
    fig.suptitle(
        'Modelos A, C, D, E — Impacto de Normalización y Covariables Demográficas',
        fontsize=12, fontweight='bold', y=0.97
    )

    y_pos = np.arange(len(model_labels))

    # ── Left panel: AIC ─────────────────────────────────────────────────────
    ax_l.set_facecolor('#F8F9FA')
    hbars = ax_l.barh(y_pos, aic_vals, color=bar_colors, alpha=0.85,
                      edgecolor='white', linewidth=0.8, height=0.55)
    ax_l.axvline(aic_base, color=GOLD, linestyle='--', linewidth=1.5, label='AIC Modelo A (base)')

    for i, (aic_v, color) in enumerate(zip(aic_vals, bar_colors)):
        delta = aic_v - aic_base
        sign = '+' if delta >= 0 else ''
        ax_l.text(aic_v + 0.3, i, f'ΔAIC={sign}{delta:.1f}',
                  va='center', ha='left', fontsize=8.5, color=SLATE)

    # Star annotation for best model
    ax_l.text(aic_vals[best_idx] - 1.5, best_idx,
              '★ LRT p=0.018', va='center', ha='right', fontsize=8.5,
              color=GREEN, fontweight='bold')

    ax_l.set_yticks(y_pos)
    ax_l.set_yticklabels(model_labels, fontsize=9)
    ax_l.set_xlabel('AIC (menor = mejor ajuste)', fontsize=10)
    ax_l.set_title('Comparación AIC — Modelos Logísticos', fontsize=10, fontweight='bold')
    ax_l.spines['top'].set_visible(False)
    ax_l.spines['right'].set_visible(False)
    ax_l.legend(fontsize=8)
    ax_l.set_xlim(840, max(aic_vals) + 12)

    # ── Right panel: AUC ────────────────────────────────────────────────────
    ax_r.set_facecolor('#F8F9FA')
    ax_r.barh(y_pos, auc_vals, color=bar_colors, alpha=0.85,
              edgecolor='white', linewidth=0.8, height=0.55)
    ax_r.axvline(0.50, color='gray', linestyle='--', linewidth=1.0, alpha=0.6, label='Azar (0.50)')
    ax_r.axvline(0.70, color=GOLD, linestyle='--', linewidth=1.2, alpha=0.8, label='Umbral clínico (0.70)')

    for i, auc_v in enumerate(auc_vals):
        ax_r.text(auc_v + 0.002, i, f'{auc_v:.3f}',
                  va='center', ha='left', fontsize=9, color=SLATE)

    ax_r.set_yticks(y_pos)
    ax_r.set_yticklabels(model_labels, fontsize=9)
    ax_r.set_xlabel('AUC-ROC', fontsize=10)
    ax_r.set_title('Comparación AUC — Modelos Logísticos', fontsize=10, fontweight='bold')
    ax_r.set_xlim(0.55, 0.72)
    ax_r.spines['top'].set_visible(False)
    ax_r.spines['right'].set_visible(False)
    ax_r.legend(fontsize=8, loc='lower right')

    # Annotation box
    fig.text(
        0.5, 0.82,
        'Modelo C (raw + sexo + edad) es el mejor:\nmenor AIC, mayor AUC, LRT p=0.018',
        ha='center', va='top', fontsize=8.5, color=GREEN,
        bbox=dict(boxstyle='round,pad=0.4', fc='#E8F5E9', ec=GREEN, lw=1.2),
        transform=fig.transFigure
    )

    fig.text(
        0.5, 0.01,
        'Fig. A3. La inclusión de sexo y edad (Modelo C) mejora significativamente el ajuste '
        'respecto al Modelo A (LRT χ²=8.04, p=0.018).',
        ha='center', va='bottom', fontsize=8.5, color=SLATE, style='italic'
    )

    plt.tight_layout(rect=[0, 0.06, 1, 0.90])
    out = f'{OUTDIR}/viz_A3_comparacion_modelos_complementario.png'
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE A4 — Efectos de Sexo y Edad sobre Temperatura
# ═══════════════════════════════════════════════════════════════════════════════
def figure_A4():
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('white')
    fig.suptitle(
        'Confundidores Demográficos: Sexo y Edad afectan la Temperatura Facial',
        fontsize=12, fontweight='bold', y=0.97
    )

    # ── Left panel: Temperature by sex (Cleveland dot plot) ─────────────────
    ax_l.set_facecolor('#F8F9FA')
    df_sex = df_conf[df_conf['analisis'].str.startswith('mw_sexo')].copy()

    row_labels = []
    for _, row in df_sex.iterrows():
        r = row['region'].upper()
        l = 'Der.' if 'derecha' in row['lado'] else 'Izq.'
        row_labels.append(f'{r} {l}')

    y_pos = np.arange(len(row_labels))

    for i, (_, row) in enumerate(df_sex.iterrows()):
        masc = row['mediana_masc']
        fem  = row['mediana_fem']
        sig  = row['significativo']
        p    = row['p_valor']
        seg_color = GREEN if sig else '#BBBBBB'

        # connecting segment
        ax_l.plot([masc, fem], [i, i], color=seg_color, linewidth=2.0, zorder=1)
        # dots
        ax_l.scatter([masc], [i], marker='^', color=NAVY, s=60, zorder=3,
                     label='Masculino' if i == 0 else '')
        ax_l.scatter([fem], [i], marker='o', color=RED, s=60, zorder=3,
                     label='Femenino' if i == 0 else '')
        if sig:
            ax_l.text(max(masc, fem) + 0.05, i, f'★ p={p:.3f}',
                      va='center', ha='left', fontsize=8, color=GREEN, fontweight='bold')

    ax_l.set_yticks(y_pos)
    ax_l.set_yticklabels(row_labels, fontsize=9)
    ax_l.set_xlabel('Temperatura mediana (°C)', fontsize=10)
    ax_l.set_title('Temperatura Regional por Sexo', fontsize=10, fontweight='bold')
    ax_l.spines['top'].set_visible(False)
    ax_l.spines['right'].set_visible(False)

    masc_dot = mlines.Line2D([], [], marker='^', color=NAVY, markersize=7,
                              linestyle='none', label='Masculino')
    fem_dot  = mlines.Line2D([], [], marker='o', color=RED, markersize=7,
                              linestyle='none', label='Femenino')
    sig_line = mlines.Line2D([], [], color=GREEN, linewidth=2, label='Sig. (p<0.05)')
    ns_line  = mlines.Line2D([], [], color='#BBBBBB', linewidth=2, label='NS')
    ax_l.legend(handles=[masc_dot, fem_dot, sig_line, ns_line], fontsize=8, loc='lower right')

    # ── Right panel: Spearman edad vs temperature ────────────────────────────
    ax_r.set_facecolor('#F8F9FA')
    df_age = df_conf[df_conf['analisis'].str.startswith('spearman_edad')].copy()

    age_labels = []
    for _, row in df_age.iterrows():
        r = row['region'].upper()
        l = 'Der.' if 'derecha' in row['lado'] else 'Izq.'
        age_labels.append(f'{r} {l}')

    y_pos2 = np.arange(len(age_labels))

    for i, (_, row) in enumerate(df_age.iterrows()):
        rho = row['rho']
        p   = row['p_valor']
        sig = row['significativo']
        if sig:
            dot_c = GREEN
        elif p < 0.10:
            dot_c = GOLD
        else:
            dot_c = SLATE

        ax_r.plot([0, rho], [i, i], color=dot_c, linewidth=2.0, zorder=1)
        ax_r.scatter([rho], [i], color=dot_c, s=70, zorder=3)
        if sig:
            xoff = rho - 0.012 if rho < 0 else rho + 0.012
            ha   = 'right' if rho < 0 else 'left'
            ax_r.text(xoff, i, f'p={p:.3f}', va='center', ha=ha,
                      fontsize=8, color=GREEN, fontweight='bold')
        elif p < 0.10:
            xoff = rho - 0.012 if rho < 0 else rho + 0.012
            ha   = 'right' if rho < 0 else 'left'
            ax_r.text(xoff, i, f'p={p:.3f}', va='center', ha=ha,
                      fontsize=7.5, color=GOLD)

    ax_r.axvline(0, color=SLATE, linewidth=1.0, linestyle='-', alpha=0.5)
    ax_r.set_yticks(y_pos2)
    ax_r.set_yticklabels(age_labels, fontsize=9)
    ax_r.set_xlabel('Spearman ρ (edad vs temperatura)', fontsize=10)
    ax_r.set_title('Correlación Edad–Temperatura Regional', fontsize=10, fontweight='bold')
    ax_r.spines['top'].set_visible(False)
    ax_r.spines['right'].set_visible(False)

    g_dot  = mlines.Line2D([], [], marker='o', color=GREEN, markersize=7,
                            linestyle='none', label='p<0.05')
    go_dot = mlines.Line2D([], [], marker='o', color=GOLD, markersize=7,
                            linestyle='none', label='p<0.10')
    ns_dot = mlines.Line2D([], [], marker='o', color=SLATE, markersize=7,
                            linestyle='none', label='NS')
    ax_r.legend(handles=[g_dot, go_dot, ns_dot], fontsize=8, loc='lower right')

    fig.text(
        0.5, 0.01,
        'Fig. A4. Hombres tienen temperatura significativamente mayor en R4 (p<0.05). '
        'Edad correlaciona negativamente con R2 y R4 izquierdas.',
        ha='center', va='bottom', fontsize=8.5, color=SLATE, style='italic'
    )

    plt.tight_layout(rect=[0, 0.07, 1, 0.94])
    out = f'{OUTDIR}/viz_A4_confundidores_sexo_edad_complementario.png'
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE A5 — ROC: Temperatura Cruda vs Normalizada
# ═══════════════════════════════════════════════════════════════════════════════
def figure_A5():
    # Merge raw and normalized AUC
    df_roc_merged = df_roc_n.copy()
    df_roc_merged = df_roc_merged.sort_values('auc_raw', ascending=False)

    n_groups = len(df_roc_merged)
    subgrupos = df_roc_merged['subgrupo'].values
    auc_raw  = df_roc_merged['auc_raw'].values
    auc_norm = df_roc_merged['auc_norm'].values
    delta    = df_roc_merged['delta_auc'].values

    fig = plt.figure(figsize=(10, 7))
    fig.patch.set_facecolor('white')

    # Main axes
    ax = fig.add_axes([0.12, 0.12, 0.60, 0.75])
    ax.set_facecolor('#F8F9FA')

    def auc_color(v):
        if v < 0.50:   return RED
        if v < 0.60:   return GOLD
        if v < 0.70:   return TEAL
        return GREEN

    y_pos = np.arange(n_groups)
    bar_h = 0.35

    for i in range(n_groups):
        c_raw  = auc_color(auc_raw[i])
        c_norm = auc_color(auc_norm[i])

        ax.barh(i + bar_h / 2, auc_raw[i], bar_h,
                color=c_raw, alpha=0.85, edgecolor='white', linewidth=0.5)
        ax.barh(i - bar_h / 2, auc_norm[i], bar_h,
                color=c_norm, alpha=0.65, hatch='///', edgecolor=c_norm, linewidth=0.4)

        # Delta annotation
        xmax = max(auc_raw[i], auc_norm[i]) + 0.005
        sign = '+' if delta[i] >= 0 else ''
        ax.text(xmax + 0.002, i, f'Δ={sign}{delta[i]:.3f}',
                va='center', ha='left', fontsize=8, color=SLATE)

    ax.axvline(0.50, color='gray', linestyle='--', linewidth=1.0, alpha=0.7, label='Azar (0.50)')
    ax.axvline(0.70, color=GOLD, linestyle='--', linewidth=1.3, alpha=0.8, label='Umbral clínico (0.70)')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(subgrupos, fontsize=9)
    ax.set_xlabel('AUC-ROC', fontsize=10)
    ax.set_title(
        'AUC-ROC: ΔT Crudo vs. ΔT Normalizado (T−T_basal)\npor Subgrupo Anatómico',
        fontsize=10, fontweight='bold'
    )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(0.25, 0.80)

    raw_patch  = mpatches.Patch(color=TEAL, alpha=0.85, label='AUC crudo')
    norm_patch = mpatches.Patch(facecolor=TEAL, alpha=0.6, hatch='///', label='AUC normalizado')
    ax.legend(handles=[raw_patch, norm_patch], fontsize=8, loc='lower right')

    # Inset scatter (top-right)
    ax_ins = fig.add_axes([0.77, 0.55, 0.20, 0.28])
    ax_ins.set_facecolor('#F8F9FA')
    ax_ins.scatter(auc_raw, auc_norm, color=TEAL, s=35, alpha=0.8, zorder=3)

    all_v = np.concatenate([auc_raw, auc_norm])
    vmin, vmax = all_v.min() - 0.02, all_v.max() + 0.02
    ax_ins.plot([vmin, vmax], [vmin, vmax], '--', color=SLATE, linewidth=1.0,
                alpha=0.6, label='y=x')

    # Label outliers
    # R2: largest loss (most negative delta)
    i_worst = np.argmin(delta)
    # R1: largest gain
    i_best = np.argmax(delta)
    for idx, lab in [(i_worst, subgrupos[i_worst]), (i_best, subgrupos[i_best])]:
        ax_ins.annotate(lab, (auc_raw[idx], auc_norm[idx]),
                        fontsize=7, textcoords='offset points', xytext=(3, 3))

    ax_ins.set_xlabel('AUC crudo', fontsize=7.5)
    ax_ins.set_ylabel('AUC norm', fontsize=7.5)
    ax_ins.set_title('Raw vs Norm', fontsize=8, fontweight='bold')
    ax_ins.tick_params(labelsize=7)
    ax_ins.spines['top'].set_visible(False)
    ax_ins.spines['right'].set_visible(False)

    fig.suptitle('', y=0.98)
    fig.text(
        0.45, 0.01,
        'Fig. A5. La normalización por temperatura basal no mejora consistentemente la capacidad diagnóstica. '
        'R2 empeora significativamente con normalización (ΔAUC=−0.309).',
        ha='center', va='bottom', fontsize=8.5, color=SLATE, style='italic'
    )

    out = f'{OUTDIR}/viz_A5_roc_raw_vs_norm_complementario.png'
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE A6 — Forest Plot Comparativo: Modelos A vs C
# ═══════════════════════════════════════════════════════════════════════════════
def figure_A6():
    term_map = {
        'temp_max':        'Temperatura máxima',
        'region_r2':       'Región R2 vs R1',
        'region_r3':       'Región R3 vs R1',
        'region_r4':       'Región R4 vs R1',
        'lado_izquierda':  'Lado izquierdo',
        'sexo_num':        'Sexo femenino [solo C]',
        'edad':            'Edad (por año) [solo C]',
    }

    shared_terms = ['temp_max', 'region_r2', 'region_r3', 'region_r4', 'lado_izquierda']
    extra_terms  = ['sexo_num', 'edad']
    all_terms    = shared_terms + extra_terms

    df_A = df_logit_e[df_logit_e['modelo'] == 'A'].set_index('termino')
    df_C = df_logit_e[df_logit_e['modelo'] == 'C'].set_index('termino')

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#F8F9FA')

    n_shared = len(shared_terms)
    n_extra  = len(extra_terms)
    n_total  = n_shared + n_extra

    # y positions: shared terms top, then separator, then extra
    y_shared = np.arange(n_shared, 0, -1)                         # 5,4,3,2,1
    y_extra  = np.arange(-0.5, -0.5 - n_extra, -1)                # -0.5, -1.5
    y_all    = np.concatenate([y_shared, y_extra])

    offset = 0.15  # vertical offset between A and C dots

    for i, term in enumerate(shared_terms):
        y = y_shared[i]
        for df_m, color, yo, mname in [(df_A, NAVY, offset, 'A'), (df_C, TEAL, -offset, 'C')]:
            if term not in df_m.index:
                continue
            row = df_m.loc[term]
            OR  = row['odds_ratio']
            lo  = row['or_ci95_low']
            hi  = row['or_ci95_high']
            sig = str(row['significativo_0_05']).lower() in ('sí', 'si', 'yes', 'true', '1')

            dot_c  = GREEN if sig else SLATE
            marker = 'o' if sig else 'o'
            fc     = dot_c if sig else 'white'

            ax.errorbar(OR, y + yo, xerr=[[OR - lo], [hi - OR]],
                        fmt='o', color=dot_c, mfc=fc, mec=dot_c,
                        markersize=7, capsize=3, linewidth=1.2,
                        label=f'Modelo {mname}' if i == 0 else '')

    for i, term in enumerate(extra_terms):
        y = y_extra[i]
        if term not in df_C.index:
            continue
        row = df_C.loc[term]
        OR  = row['odds_ratio']
        lo  = row['or_ci95_low']
        hi  = row['or_ci95_high']
        sig = str(row['significativo_0_05']).lower() in ('sí', 'si', 'yes', 'true', '1')
        dot_c = GREEN if sig else SLATE
        fc    = dot_c if sig else 'white'

        ax.errorbar(OR, y, xerr=[[OR - lo], [hi - OR]],
                    fmt='D', color=TEAL, mfc=fc, mec=TEAL,
                    markersize=7, capsize=3, linewidth=1.2)

    # Reference line OR=1
    ax.axvline(1.0, color=GOLD, linestyle='--', linewidth=1.5, zorder=1, label='OR=1.0')

    # Separator line between shared and extra
    ax.axhline(0.5, color=BORDER, linestyle=':', linewidth=1.2)
    ax.text(ax.get_xlim()[0] if ax.get_xlim()[0] > 0 else 0.1,
            0.35, '── Términos solo en Modelo C ──',
            fontsize=8, color=SLATE, style='italic', va='center')

    # y-tick labels
    ytick_pos    = list(y_shared) + list(y_extra)
    ytick_labels = [term_map.get(t, t) for t in all_terms]
    ax.set_yticks(ytick_pos)
    ax.set_yticklabels(ytick_labels, fontsize=9)
    ax.set_xscale('log')
    ax.set_xlabel('Odds Ratio (escala log)', fontsize=10)
    ax.set_title('Forest Plot Comparativo — Modelo A (base) vs Modelo C (+sexo+edad)',
                 fontsize=11, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend
    mod_A = mlines.Line2D([], [], marker='o', color=NAVY, markersize=7,
                           linestyle='none', label='Modelo A')
    mod_C = mlines.Line2D([], [], marker='o', color=TEAL, markersize=7,
                           linestyle='none', label='Modelo C')
    sig_dot = mlines.Line2D([], [], marker='o', color=GREEN, markersize=7,
                             linestyle='none', label='Sig. (p<0.05)')
    ns_dot  = mlines.Line2D([], [], marker='o', mfc='white', mec=SLATE, markersize=7,
                             linestyle='none', label='NS')
    ax.legend(handles=[mod_A, mod_C, sig_dot, ns_dot], fontsize=8.5,
              loc='upper right', framealpha=0.9)

    # Annotation box
    ax.annotate(
        'Modelo C: AUC=0.642, AIC=850.9\n'
        'Modelo A: AUC=0.612, AIC=854.6\n'
        'LRT: χ²=8.04, p=0.018 ★',
        xy=(0.98, 0.05), xycoords='axes fraction',
        ha='right', va='bottom', fontsize=8.5, color=GREEN,
        bbox=dict(boxstyle='round,pad=0.4', fc='#E8F5E9', ec=GREEN, lw=1.2)
    )

    fig.text(
        0.5, 0.01,
        'Fig. A6. Comparación de Odds Ratios entre Modelo A y Modelo C. '
        'La inclusión de sexo y edad mejora el ajuste sin alterar sustancialmente los OR de temperatura y región.',
        ha='center', va='bottom', fontsize=8.5, color=SLATE, style='italic'
    )

    plt.tight_layout(rect=[0, 0.07, 1, 0.97])
    out = f'{OUTDIR}/viz_A6_forest_modelos_A_C_complementario.png'
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return out


# ─── Main ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import os

    figures = [
        ('A1', figure_A1),
        ('A2', figure_A2),
        ('A3', figure_A3),
        ('A4', figure_A4),
        ('A5', figure_A5),
        ('A6', figure_A6),
    ]

    for code, fn in figures:
        print(f'Generating Figure {code}...', end=' ', flush=True)
        try:
            out = fn()
            size_kb = os.path.getsize(out) / 1024
            print(f'✓  {out}  ({size_kb:.1f} KB)')
        except Exception as e:
            import traceback
            print(f'✗  ERROR:')
            traceback.print_exc()

    print('\nDone. Files in resultado_complementario/:')
    for f in sorted(os.listdir(OUTDIR)):
        if f.endswith('.png'):
            sz = os.path.getsize(f'{OUTDIR}/{f}') / 1024
            print(f'  {f:50s}  {sz:7.1f} KB')
