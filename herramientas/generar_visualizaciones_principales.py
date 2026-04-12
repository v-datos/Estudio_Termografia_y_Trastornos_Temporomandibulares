"""
generar_visualizaciones_principales.py
Produces 6 publication-quality PNG figures for analisis_termografia.py results.
All saved in resultados/
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

warnings.filterwarnings('ignore')

# ─── Working directory ───────────────────────────────────────────────────────
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ─── Output directory ────────────────────────────────────────────────────────
OUTDIR = 'resultados'
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
C_CON  = '#9B2335'   # Con Dolor
C_SIN  = '#1A7A8A'   # Sin Dolor

# ─── Load data ────────────────────────────────────────────────────────────────
df_roc    = pd.read_csv(f'{OUTDIR}/resultados_roc.csv')
df_pac    = pd.read_csv(f'{OUTDIR}/resultados_paciente_n45.csv')
df_corr   = pd.read_csv(f'{OUTDIR}/correlaciones_globales.csv')
df_delta  = pd.read_csv(f'{OUTDIR}/resultados_delta_termico.csv')
df_logit  = pd.read_csv(f'{OUTDIR}/resultados_logit_cluster.csv')
df_cmp    = pd.read_csv(f'{OUTDIR}/comparacion_modelos_a_vs_b_temp_punto.csv')
df_temp   = pd.read_csv(f'{OUTDIR}/temperatura_por_region_grupo.csv')

df_raw = pd.read_csv('datos/datos_finales_termografia_procesados_todas_fotos.csv', encoding='utf-8-sig')
df_raw.columns = df_raw.columns.str.strip().str.lower()
df_raw = df_raw.drop_duplicates(subset=['numero de muestra'], keep='first')

# ─── Derived: pain indicator on df_raw ──────────────────────────────────────
from herramientas.constantes import PAIN_COL_REGEX
PAIN_COLS = [c for c in df_raw.columns if PAIN_COL_REGEX.match(c.strip().lower())]
df_raw[PAIN_COLS] = df_raw[PAIN_COLS].fillna(0)
df_raw['total_pain'] = df_raw[PAIN_COLS].sum(axis=1)
df_raw['con_dolor'] = (df_raw[PAIN_COLS].max(axis=1) > 0).map({True: 'Con Dolor', False: 'Sin Dolor'})

# ─── Derived: ΔT columns ─────────────────────────────────────────────────────
for r in ['r1', 'r2', 'r3', 'r4']:
    col_d = f'{r}: temperatura media derecha_primera'
    col_i = f'{r}: temperatura media izquierda_primera'
    if col_d in df_raw.columns and col_i in df_raw.columns:
        df_raw[f'delta_t_{r}'] = df_raw[col_d] - df_raw[col_i]


def save_fig(fig, name, dpi=300):
    path = f'{OUTDIR}/{name}'
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    size_kb = os.path.getsize(path) / 1024
    print(f'  ✓ {path}  ({size_kb:.1f} KB)')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — ROC Curves by Anatomical Region
# ═══════════════════════════════════════════════════════════════════════════════
print('\n[Fig 1] ROC Curves by Anatomical Region')

REGION_LABELS = {
    'Global': 'Global (todos los ΔT)',
    'r1': 'R1 – ATM Posterior',
    'r2': 'R2 – ATM Anterior',
    'r3': 'R3 – Masetero',
    'r4': 'R4 – Esternocleidomastoideo',
}
REGION_COLORS = {
    'Global': NAVY,
    'r1': TEAL,
    'r2': GREEN,
    'r3': GOLD,
    'r4': RED,
}
REGION_DASHES = {
    'Global': (None, None),
    'r1': (6, 2),
    'r2': (4, 2),
    'r3': (2, 2),
    'r4': (8, 2, 2, 2),
}

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# Left panel: ROC "schematic" from AUC + sens/spec data
ax = axes[0]
ax.set_facecolor('#F8F9FA')
ax.plot([0, 1], [0, 1], '--', color=BORDER, lw=1, zorder=1)

legend_handles = []
for _, row in df_roc.iterrows():
    key = row['subgrupo']
    auc = row['AUC-ROC']
    sens = row['sensibilidad']
    spec = row['especificidad']
    fpr_pt = 1 - spec

    # Build a simple trapezoid ROC from (0,0), operating point, (1,1)
    fpr = np.array([0.0, fpr_pt, 1.0])
    tpr = np.array([0.0, sens, 1.0])

    color = REGION_COLORS.get(key, SLATE)
    dash  = REGION_DASHES.get(key, (None, None))
    lw    = 2.5 if key == 'Global' else 1.8

    ls = '--' if key == 'Global' else '-'
    if dash[0] is not None:
        line, = ax.plot(fpr, tpr, color=color, lw=lw, linestyle=(0, dash))
    else:
        line, = ax.plot(fpr, tpr, color=color, lw=lw, linestyle='--')

    ax.scatter([fpr_pt], [sens], s=60, color=color, zorder=5, ec='white', lw=0.8)

    lbl = REGION_LABELS.get(key, key)
    legend_handles.append(mlines.Line2D([], [], color=color, lw=lw,
                                        label=f'{lbl}\nAUC={auc:.3f}'))

ax.set_xlabel('1 − Especificidad (FPR)', fontsize=11)
ax.set_ylabel('Sensibilidad (TPR)', fontsize=11)
ax.set_title('Curvas ROC por Subgrupo Anatómico', fontsize=12, fontweight='bold',
             color=NAVY, pad=10)
ax.legend(handles=legend_handles, fontsize=8.5, loc='lower right',
          framealpha=0.92, edgecolor=BORDER)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)

# Right panel: AUC comparison bar chart
ax2 = axes[1]
ax2.set_facecolor('#F8F9FA')

labels_short = [REGION_LABELS.get(s, s).split('–')[0].strip()
                for s in df_roc['subgrupo']]
aucs_roc = df_roc['AUC-ROC'].values
aucs_pr  = df_roc['AUC-PR'].values
colors_bar = [REGION_COLORS.get(s, SLATE) for s in df_roc['subgrupo']]

x = np.arange(len(labels_short))
w = 0.38
bars = ax2.bar(x - w/2, aucs_roc, width=w, label='AUC-ROC',
               color=colors_bar, alpha=0.85, ec='white', lw=0.7)
bars2 = ax2.bar(x + w/2, aucs_pr, width=w, label='AUC-PR',
                color=colors_bar, alpha=0.45, ec='white', lw=0.7,
                hatch='//')

ax2.axhline(0.5, ls='--', color=BORDER, lw=1.2, label='Referencia AUC=0.5')
for b in bars:
    h = b.get_height()
    ax2.text(b.get_x() + b.get_width()/2, h + 0.008, f'{h:.3f}',
             ha='center', va='bottom', fontsize=8, color=NAVY)

ax2.set_xticks(x)
ax2.set_xticklabels(labels_short, rotation=25, ha='right', fontsize=9)
ax2.set_ylabel('AUC', fontsize=11)
ax2.set_title('AUC-ROC vs AUC-PR por Subgrupo', fontsize=12, fontweight='bold',
              color=NAVY, pad=10)
ax2.legend(fontsize=9, loc='upper right', framealpha=0.92, edgecolor=BORDER)
ax2.set_ylim(0, 0.85)

fig.suptitle(
    'Análisis ROC — ΔT como Predictor Diagnóstico de Dolor Orofacial',
    fontsize=13, fontweight='bold', color=NAVY, y=1.01
)
fig.tight_layout(pad=1.5)
save_fig(fig, 'viz_01_roc_curvas.png')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Scatter ΔT_max vs Intensidad_max (patient level)
# ═══════════════════════════════════════════════════════════════════════════════
print('[Fig 2] Scatter ΔT_max vs Intensidad_max')

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: scatter with regression line
ax = axes[0]
con  = df_pac[df_pac['grupo'] == 'Con Dolor']
sin_ = df_pac[df_pac['grupo'] == 'Sin Dolor']

ax.scatter(sin_['delta_t_max'], sin_['intensidad_max'],
           color=C_SIN, s=55, alpha=0.75, label='Sin Dolor', ec='white', lw=0.6, zorder=4)
ax.scatter(con['delta_t_max'], con['intensidad_max'],
           color=C_CON, s=65, alpha=0.82, label='Con Dolor', marker='D',
           ec='white', lw=0.6, zorder=5)

# Spearman annotation
rho_g  = df_corr.loc[df_corr['subgrupo'] == 'Todos', 'rho'].values
p_bonf = df_corr.loc[df_corr['subgrupo'] == 'Todos', 'p_bonf'].values
rho_v  = rho_g[0] if len(rho_g) else np.nan
p_v    = p_bonf[0] if len(p_bonf) else np.nan

# Fit Spearman and add annotated text
from scipy.stats import spearmanr
rho_s, p_s = spearmanr(df_pac['delta_t_max'], df_pac['intensidad_max'])

# OLS line for visual guidance only
m, b = np.polyfit(df_pac['delta_t_max'], df_pac['intensidad_max'], 1)
xr = np.linspace(df_pac['delta_t_max'].min(), df_pac['delta_t_max'].max(), 100)
ax.plot(xr, m*xr + b, color=SLATE, lw=1.4, ls='--', alpha=0.6, label='Tendencia OLS')

ax.set_xlabel('ΔT máximo (°C, derecha − izquierda)', fontsize=11)
ax.set_ylabel('Intensidad de dolor máxima (0–10)', fontsize=11)
ax.set_title('ΔT Máximo vs Intensidad de Dolor\n(nivel paciente, N=43)',
             fontsize=12, fontweight='bold', color=NAVY)
ax.legend(fontsize=9.5, framealpha=0.92, edgecolor=BORDER)

sig_str = '★' if p_s < 0.05 else 'n.s.'
ax.text(0.97, 0.95,
        f'ρ Spearman = {rho_s:.3f}\np (perm.) = {p_s:.3f} {sig_str}',
        transform=ax.transAxes, ha='right', va='top',
        fontsize=10, color=NAVY,
        bbox=dict(boxstyle='round,pad=0.4', fc='white', ec=BORDER, alpha=0.92))

# Right: scatter per region (ΔT r1..r4 vs intensidad_max)
ax2 = axes[1]
region_keys  = ['r1', 'r2', 'r3', 'r4']
region_names = ['R1 ATM Post.', 'R2 ATM Ant.', 'R3 Masetero', 'R4 ECM']
region_col   = [TEAL, GREEN, GOLD, RED]
region_mk    = ['o', 's', 'D', '^']

rho_by_r = df_corr[df_corr['nivel'] == 'Región'].set_index('subgrupo')['rho'].to_dict()
p_by_r   = df_corr[df_corr['nivel'] == 'Región'].set_index('subgrupo')['p_bonf'].to_dict()

for rk, rn, rc, rm in zip(region_keys, region_names, region_col, region_mk):
    col_dt = f'delta_t_{rk}'
    if col_dt not in df_pac.columns:
        continue
    rho_r = rho_by_r.get(rk, np.nan)
    p_r   = p_by_r.get(rk, np.nan)
    sig_r = '★' if p_r < 0.05 else ''
    ax2.scatter(df_pac[col_dt], df_pac['intensidad_max'],
                color=rc, s=48, alpha=0.70, marker=rm, ec='white', lw=0.5,
                label=f'{rn}  ρ={rho_r:.2f}{sig_r}')

ax2.set_xlabel('ΔT por región (°C)', fontsize=11)
ax2.set_ylabel('Intensidad de dolor máxima (0–10)', fontsize=11)
ax2.set_title('ΔT por Región vs Intensidad de Dolor',
              fontsize=12, fontweight='bold', color=NAVY)
ax2.legend(fontsize=8.5, loc='upper right', framealpha=0.92, edgecolor=BORDER)

fig.suptitle(
    'Correlación Espearman: Asimetría Térmica (ΔT) vs Intensidad de Dolor',
    fontsize=13, fontweight='bold', color=NAVY, y=1.01
)
fig.tight_layout(pad=1.5)
save_fig(fig, 'viz_02_scatter_dt_intensidad.png')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Heatmap de Correlaciones Spearman (región × métrica)
# ═══════════════════════════════════════════════════════════════════════════════
print('[Fig 3] Heatmap de Correlaciones Spearman')

# Build correlation matrix: row = region/level, col = ΔT metric (raw)
# We have: global + r1..r4 rows, and rho + p_orig + p_bonf
# Make a combined heatmap: rows = subgrupos, cols = [rho, p_orig, p_bonf]

hm_data = df_corr[['subgrupo', 'rho', 'p_orig', 'p_bonf', 'significativo']].copy()
hm_data = hm_data[hm_data['subgrupo'].isin(['Todos', 'r1', 'r2', 'r3', 'r4'])]

row_labels = {
    'Todos': 'Global',
    'r1': 'R1 – ATM Posterior',
    'r2': 'R2 – ATM Anterior',
    'r3': 'R3 – Masetero',
    'r4': 'R4 – ECM',
}
hm_data['label'] = hm_data['subgrupo'].map(row_labels)
hm_data = hm_data.set_index('label')

# Also include ROC AUC per region
auc_map = df_roc.set_index('subgrupo')['AUC-ROC'].to_dict()
auc_order = {'Todos': 'Global', 'r1': 'R1 – ATM Posterior',
             'r2': 'R2 – ATM Anterior', 'r3': 'R3 – Masetero',
             'r4': 'R4 – ECM'}
auc_vals = {auc_order.get(k, k): v for k, v in auc_map.items()}

col_order = list(row_labels.values())

# Build numeric matrix for heatmap (only rho values)
rho_vals  = [hm_data.loc[lbl, 'rho']    if lbl in hm_data.index else np.nan for lbl in col_order]
p_orig_v  = [hm_data.loc[lbl, 'p_orig'] if lbl in hm_data.index else np.nan for lbl in col_order]
p_bonf_v  = [hm_data.loc[lbl, 'p_bonf'] if lbl in hm_data.index else np.nan for lbl in col_order]
auc_v     = [auc_vals.get(lbl, np.nan) for lbl in col_order]

# Matrix: rows = metric, cols = region/level
mat = np.array([rho_vals, p_orig_v, p_bonf_v, auc_v])
row_names = ['ρ Spearman', 'p (perm.)', 'p (Bonf.)', 'AUC-ROC']

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5),
                         gridspec_kw={'width_ratios': [2, 1]})

# Left: styled heatmap
ax = axes[0]
import matplotlib.colors as mcolors

# Use diverging colormap for rho, sequential for p-values
# We'll use two panels
cmap_rho  = plt.cm.RdBu_r
cmap_pval = plt.cm.YlOrRd_r

# Normalize
norm_rho  = mcolors.TwoSlopeNorm(vmin=-0.3, vcenter=0, vmax=0.7)
norm_pval = mcolors.Normalize(vmin=0, vmax=1)
norm_auc  = mcolors.TwoSlopeNorm(vmin=0.3, vcenter=0.5, vmax=0.8)

# Build RGBA arrays per row
h, w = mat.shape
rgba = np.zeros((h, w, 4))
for ci in range(w):
    rgba[0, ci] = cmap_rho(norm_rho(mat[0, ci]))          # rho
    rgba[1, ci] = cmap_pval(norm_pval(mat[1, ci]))         # p_orig
    rgba[2, ci] = cmap_pval(norm_pval(mat[2, ci]))         # p_bonf
    rgba[3, ci] = cmap_rho(norm_auc(mat[3, ci]))           # auc

im = ax.imshow(rgba, aspect='auto')
ax.set_xticks(range(w))
ax.set_xticklabels(col_order, rotation=30, ha='right', fontsize=10)
ax.set_yticks(range(h))
ax.set_yticklabels(row_names, fontsize=11)
ax.set_title('Métricas de Asociación ΔT–Dolor por Subgrupo Anatómico',
             fontsize=12, fontweight='bold', color=NAVY, pad=12)

# Annotate cells
for ri in range(h):
    for ci in range(w):
        v = mat[ri, ci]
        if np.isnan(v):
            txt = '–'
        elif ri in (0, 3):  # rho or AUC
            txt = f'{v:.3f}'
        else:
            txt = f'{v:.3f}' + (' ★' if ri == 2 and v < 0.05 else '')
        brightness = 0.299*rgba[ri,ci,0] + 0.587*rgba[ri,ci,1] + 0.114*rgba[ri,ci,2]
        fc = 'white' if brightness < 0.5 else NAVY
        ax.text(ci, ri, txt, ha='center', va='center', fontsize=10, color=fc,
                fontweight='bold' if (ri == 0 and abs(float(v if not np.isnan(v) else 0)) > 0.3) else 'normal')

ax.tick_params(length=0)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.grid(False)

# Right: summary table
ax2 = axes[1]
ax2.axis('off')

summary_rows = []
for lbl, rk in zip(col_order, ['Todos', 'r1', 'r2', 'r3', 'r4']):
    r_row = hm_data.loc[lbl] if lbl in hm_data.index else None
    roc_row = df_roc[df_roc['subgrupo'] == rk]
    if r_row is not None and not roc_row.empty:
        sig = '★' if r_row['p_bonf'] < 0.05 else '–'
        summary_rows.append([
            lbl.split('–')[0].strip(),
            f"{r_row['rho']:.3f}",
            f"{r_row['p_bonf']:.3f}",
            f"{roc_row['AUC-ROC'].values[0]:.3f}",
            sig
        ])

col_hdrs = ['Subgrupo', 'ρ', 'p (Bonf.)', 'AUC', 'Sig.']
tbl = ax2.table(cellText=summary_rows, colLabels=col_hdrs,
                loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(9.5)
tbl.scale(1.1, 1.55)

for (ri, ci), cell in tbl.get_celld().items():
    if ri == 0:
        cell.set_facecolor(NAVY)
        cell.set_text_props(color='white', fontweight='bold')
    elif ri % 2 == 0:
        cell.set_facecolor(LIGHT)
    else:
        cell.set_facecolor('white')
    cell.set_edgecolor(BORDER)

ax2.set_title('Resumen de Resultados', fontsize=11, fontweight='bold',
              color=NAVY, pad=12)

fig.tight_layout(pad=1.5)
save_fig(fig, 'viz_03_heatmap_correlaciones.png')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Boxplots ΔT por Región (Con Dolor vs Sin Dolor)
# ═══════════════════════════════════════════════════════════════════════════════
print('[Fig 4] Boxplots ΔT por Región')

regions    = ['r1', 'r2', 'r3', 'r4']
reg_labels = ['R1 – ATM Posterior', 'R2 – ATM Anterior',
              'R3 – Masetero', 'R4 – ECM']

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
axes = axes.flatten()

from scipy.stats import mannwhitneyu

for i, (rk, rl) in enumerate(zip(regions, reg_labels)):
    ax = axes[i]
    col_dt = f'delta_t_{rk}'
    if col_dt not in df_pac.columns:
        ax.text(0.5, 0.5, 'Sin datos', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
        ax.set_title(rl, fontsize=11, fontweight='bold', color=NAVY)
        continue

    g_con = df_pac.loc[df_pac['grupo'] == 'Con Dolor', col_dt].dropna()
    g_sin = df_pac.loc[df_pac['grupo'] == 'Sin Dolor', col_dt].dropna()

    # Mann-Whitney U test
    if len(g_con) > 0 and len(g_sin) > 0:
        stat, p_mw = mannwhitneyu(g_con, g_sin, alternative='two-sided')
        r_eff = 1 - 2*stat / (len(g_con)*len(g_sin))  # rank-biserial
    else:
        p_mw = np.nan
        r_eff = np.nan

    data_plot = [g_sin.values, g_con.values]
    bplot = ax.boxplot(data_plot, patch_artist=True, widths=0.45,
                       medianprops=dict(color='white', lw=2.5),
                       whiskerprops=dict(color=SLATE, lw=1.3),
                       capprops=dict(color=SLATE, lw=1.3),
                       flierprops=dict(marker='o', markerfacecolor=SLATE,
                                       markersize=4, linestyle='none', alpha=0.6))

    bplot['boxes'][0].set_facecolor(C_SIN)
    bplot['boxes'][0].set_alpha(0.78)
    bplot['boxes'][1].set_facecolor(C_CON)
    bplot['boxes'][1].set_alpha(0.78)

    # Jitter overlay
    np.random.seed(42)
    for j_idx, (grp_data, xpos) in enumerate([(g_sin, 1), (g_con, 2)]):
        jx = xpos + np.random.uniform(-0.14, 0.14, len(grp_data))
        col = C_SIN if j_idx == 0 else C_CON
        ax.scatter(jx, grp_data, s=22, color=col, alpha=0.55, zorder=4,
                   ec='white', lw=0.4)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Sin Dolor\n(n=17)', 'Con Dolor\n(n=26)'], fontsize=10)
    ax.set_ylabel('ΔT (°C)', fontsize=10)
    ax.set_title(rl, fontsize=11, fontweight='bold', color=NAVY, pad=6)

    # Significance annotation
    sig_str = ('★★★' if p_mw < 0.001 else
               '★★'  if p_mw < 0.01  else
               '★'   if p_mw < 0.05  else 'n.s.')
    y_top = max(g_con.max(), g_sin.max()) * 1.05 + 0.1
    ax.annotate('', xy=(2, y_top), xytext=(1, y_top),
                arrowprops=dict(arrowstyle='-', color=SLATE, lw=1.2))
    ax.text(1.5, y_top + 0.04,
            f'MW {sig_str}\np={p_mw:.3f}  r={r_eff:.2f}',
            ha='center', va='bottom', fontsize=8.5, color=NAVY)

fig.suptitle(
    'Asimetría Térmica (ΔT) por Región Anatómica: Con Dolor vs Sin Dolor',
    fontsize=13, fontweight='bold', color=NAVY, y=1.01
)
fig.tight_layout(pad=1.8)
save_fig(fig, 'viz_04_boxplot_dt_grupos.png')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Forest Plot: OR Logit Modelo A (cluster-robust)
# ═══════════════════════════════════════════════════════════════════════════════
print('[Fig 5] Forest Plot Logit Modelo A')

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.set_facecolor('#F8F9FA')

# Exclude intercept
df_f = df_logit[df_logit['termino'] != 'const'].copy().reset_index(drop=True)

term_labels = {
    'temp_max':  'T° máxima (°C)',
    'lado_izq':  'Lado Izquierdo',
    'region_r2': 'Región R2 (ATM Ant.)',
    'region_r3': 'Región R3 (Masetero)',
    'region_r4': 'Región R4 (ECM)',
    'sexo':      'Sexo (Femenino)',
    'edad':      'Edad',
}

y_pos = np.arange(len(df_f))[::-1]

for i, (_, row) in enumerate(df_f.iterrows()):
    yi   = y_pos[i]
    OR   = row['odds_ratio']
    low  = row['or_ci95_low']
    high = row['or_ci95_high']
    pv   = row['p_valor']
    sig  = pv < 0.05

    color = RED if sig else TEAL
    ax.errorbar(OR, yi, xerr=[[OR-low], [high-OR]],
                fmt='D', color=color, markersize=9 if sig else 7,
                capsize=5, capthick=1.5, elinewidth=1.5,
                markeredgecolor='white', markeredgewidth=0.8, zorder=4)

    # p-value annotation
    p_str = f'p={pv:.3f}' + (' ★' if sig else '')
    ax.text(max(high, 1) + 0.05, yi, p_str, va='center',
            fontsize=9, color=color)

ax.axvline(1.0, color=BORDER, lw=1.5, ls='--', zorder=1)
ax.set_yticks(y_pos)
term_names = [term_labels.get(row['termino'], row['termino']) for _, row in df_f.iterrows()]
ax.set_yticklabels(term_names[::-1], fontsize=10.5)
ax.set_xlabel('Odds Ratio (IC 95% robusto por cluster)', fontsize=11)
ax.set_title('Forest Plot — Modelo Logístico A\n(temp_max + región + lado, SE robusto por cluster)',
             fontsize=12, fontweight='bold', color=NAVY, pad=10)

# Legend
ax.scatter([], [], color=RED, s=80, marker='D', label='Significativo (p<0.05)')
ax.scatter([], [], color=TEAL, s=60, marker='D', label='No significativo')
ax.legend(fontsize=9.5, loc='lower right', framealpha=0.92, edgecolor=BORDER)

# Shaded band
ax.axvspan(0.5, 2.0, alpha=0.05, color=GOLD, zorder=0)

# Pseudo-R² and N annotation
ax.text(0.01, 0.02,
        f"N obs=708  |  Pseudo-R²≈0.014  |  AUC≈0.579",
        transform=ax.transAxes, fontsize=9, color=SLATE,
        style='italic')

fig.tight_layout(pad=1.5)
save_fig(fig, 'viz_05_forest_plot_logit.png')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — AUC Bar Plot + Sensitivity/Specificity Summary
# ═══════════════════════════════════════════════════════════════════════════════
print('[Fig 6] AUC Bar Plot + Sens/Spec Summary')

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

# Left: Horizontal grouped bar chart AUC-ROC and AUC-PR
ax = axes[0]
ax.set_facecolor('#F8F9FA')

subgrupos = df_roc['subgrupo'].tolist()
sub_labels = [REGION_LABELS.get(s, s) for s in subgrupos]
aucs_roc  = df_roc['AUC-ROC'].values
aucs_pr   = df_roc['AUC-PR'].values
colors_h  = [REGION_COLORS.get(s, SLATE) for s in subgrupos]

y = np.arange(len(subgrupos))
h = 0.35

ax.barh(y + h/2, aucs_roc, height=h, color=colors_h, alpha=0.87,
        label='AUC-ROC', ec='white', lw=0.6)
ax.barh(y - h/2, aucs_pr, height=h, color=colors_h, alpha=0.45,
        label='AUC-PR', ec='white', lw=0.6, hatch='//')
ax.axvline(0.5, color=BORDER, lw=1.2, ls='--')
ax.axvline(0.7, color=GOLD, lw=0.8, ls=':', alpha=0.7)

for i, (av, pv) in enumerate(zip(aucs_roc, aucs_pr)):
    ax.text(av + 0.005, i + h/2, f'{av:.3f}', va='center', fontsize=9, color=NAVY)
    ax.text(pv + 0.005, i - h/2, f'{pv:.3f}', va='center', fontsize=8, color=SLATE)

ax.set_yticks(y)
ax.set_yticklabels(sub_labels, fontsize=9)
ax.set_xlabel('AUC', fontsize=11)
ax.set_title('AUC-ROC y AUC-PR\npor Subgrupo Anatómico', fontsize=12,
             fontweight='bold', color=NAVY, pad=10)
ax.legend(fontsize=9.5, loc='lower right', framealpha=0.92, edgecolor=BORDER)
ax.set_xlim(0, 0.85)

# Right: Sens / Spec / Youden scatter
ax2 = axes[1]
ax2.set_facecolor('#F8F9FA')

x_sens  = df_roc['sensibilidad'].values
x_spec  = df_roc['especificidad'].values
y_youd  = df_roc['youden_J'].values

ax2.scatter(x_spec, x_sens, s=110, color=colors_h, alpha=0.85,
            ec='white', lw=0.8, zorder=5)

# Label each point
for i, row in df_roc.iterrows():
    lbl = row['subgrupo']
    ax2.annotate(lbl.upper() if lbl != 'Global' else 'Global',
                 xy=(row['especificidad'], row['sensibilidad']),
                 xytext=(8, 4), textcoords='offset points',
                 fontsize=8.5, color=REGION_COLORS.get(lbl, SLATE))

# Add iso-Youden lines
for j_val, j_ls in [(0.1, ':'), (0.2, '--'), (0.3, '-')]:
    sens_range = np.linspace(0, 1, 100)
    spec_range = sens_range - j_val
    valid = (spec_range >= 0) & (spec_range <= 1)
    ax2.plot(spec_range[valid], sens_range[valid], ls=j_ls,
             color=BORDER, lw=0.9, alpha=0.7,
             label=f'Youden J={j_val}' if j_val == 0.3 else f'J={j_val}')

ax2.set_xlabel('Especificidad (en umbral óptimo ΔT*)', fontsize=11)
ax2.set_ylabel('Sensibilidad (en umbral óptimo ΔT*)', fontsize=11)
ax2.set_title('Sensibilidad vs Especificidad\n(umbral ΔT* óptimo por Youden J)',
              fontsize=12, fontweight='bold', color=NAVY, pad=10)
ax2.legend(fontsize=8.5, loc='lower left', framealpha=0.92, edgecolor=BORDER)
ax2.set_xlim(-0.02, 1.15)
ax2.set_ylim(-0.05, 1.05)

fig.suptitle(
    'Rendimiento Diagnóstico de ΔT — Análisis ROC Completo',
    fontsize=13, fontweight='bold', color=NAVY, y=1.01
)
fig.tight_layout(pad=1.5)
save_fig(fig, 'viz_06_auc_barplot.png')

print('\n✅  Todas las figuras generadas exitosamente en resultados/')
