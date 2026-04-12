"""
generar_reporte_analisis_complementario_imagenes_termograficas.py
Genera un reporte científico de publicación en PDF sobre termografía y TMD.
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer,
    HRFlowable, PageBreak, KeepTogether, Image
)
from reportlab.lib.colors import HexColor

# ─────────────────────────────────────────────
# COLOR PALETTE
# ─────────────────────────────────────────────
NAVY      = HexColor('#0D2B4E')
TEAL      = HexColor('#1A7A8A')
SLATE     = HexColor('#4A5568')
LIGHT_BG  = HexColor('#F7FAFC')
WHITE     = HexColor('#FFFFFF')
GOLD      = HexColor('#D4A017')
RED_SOFT  = HexColor('#C53030')
GREEN_OK  = HexColor('#276749')
BORDER    = HexColor('#CBD5E0')
BLACK     = HexColor('#000000')

# ─────────────────────────────────────────────
# PARAGRAPH STYLES
# ─────────────────────────────────────────────
def make_styles():
    styles = {}

    styles['title'] = ParagraphStyle(
        'title',
        fontName='Helvetica-Bold',
        fontSize=22,
        textColor=WHITE,
        alignment=TA_CENTER,
        spaceAfter=6,
        leading=28,
    )
    styles['subtitle'] = ParagraphStyle(
        'subtitle',
        fontName='Helvetica',
        fontSize=13,
        textColor=WHITE,
        alignment=TA_CENTER,
        spaceAfter=4,
        leading=17,
    )
    styles['cover_meta'] = ParagraphStyle(
        'cover_meta',
        fontName='Helvetica',
        fontSize=10,
        textColor=WHITE,
        alignment=TA_CENTER,
        spaceAfter=3,
        leading=14,
    )
    styles['section'] = ParagraphStyle(
        'section',
        fontName='Helvetica-Bold',
        fontSize=13,
        textColor=TEAL,
        spaceBefore=12,
        spaceAfter=2,
        leading=17,
    )
    styles['subsection'] = ParagraphStyle(
        'subsection',
        fontName='Helvetica-Bold',
        fontSize=11,
        textColor=NAVY,
        spaceBefore=8,
        spaceAfter=3,
        leading=15,
    )
    styles['body'] = ParagraphStyle(
        'body',
        fontName='Times-Roman',
        fontSize=10,
        textColor=SLATE,
        alignment=TA_JUSTIFY,
        spaceAfter=4,
        leading=14,
    )
    styles['body_bold'] = ParagraphStyle(
        'body_bold',
        fontName='Times-Bold',
        fontSize=10,
        textColor=SLATE,
        alignment=TA_JUSTIFY,
        spaceAfter=4,
        leading=14,
    )
    styles['caption'] = ParagraphStyle(
        'caption',
        fontName='Times-Italic',
        fontSize=9,
        textColor=SLATE,
        alignment=TA_CENTER,
        spaceAfter=3,
        leading=12,
    )
    styles['footnote'] = ParagraphStyle(
        'footnote',
        fontName='Times-Italic',
        fontSize=8,
        textColor=SLATE,
        alignment=TA_LEFT,
        spaceAfter=2,
        leading=11,
    )
    styles['callout'] = ParagraphStyle(
        'callout',
        fontName='Times-Roman',
        fontSize=9,
        textColor=SLATE,
        alignment=TA_JUSTIFY,
        spaceAfter=3,
        leading=13,
        leftIndent=4,
        rightIndent=4,
    )
    styles['callout_bold'] = ParagraphStyle(
        'callout_bold',
        fontName='Times-Bold',
        fontSize=9,
        textColor=NAVY,
        alignment=TA_JUSTIFY,
        spaceAfter=3,
        leading=13,
        leftIndent=4,
        rightIndent=4,
    )
    styles['bullet'] = ParagraphStyle(
        'bullet',
        fontName='Times-Roman',
        fontSize=10,
        textColor=SLATE,
        alignment=TA_LEFT,
        spaceAfter=3,
        leading=14,
        leftIndent=12,
        bulletIndent=0,
    )
    styles['numbered'] = ParagraphStyle(
        'numbered',
        fontName='Times-Roman',
        fontSize=10,
        textColor=SLATE,
        alignment=TA_JUSTIFY,
        spaceAfter=4,
        leading=14,
        leftIndent=16,
        firstLineIndent=-16,
    )
    styles['exec_header'] = ParagraphStyle(
        'exec_header',
        fontName='Helvetica-Bold',
        fontSize=11,
        textColor=WHITE,
        alignment=TA_CENTER,
        spaceAfter=2,
        leading=15,
    )
    styles['col_header'] = ParagraphStyle(
        'col_header',
        fontName='Helvetica-Bold',
        fontSize=10,
        textColor=NAVY,
        alignment=TA_LEFT,
        spaceAfter=4,
        leading=14,
    )
    styles['metric_label'] = ParagraphStyle(
        'metric_label',
        fontName='Helvetica-Bold',
        fontSize=11,
        textColor=WHITE,
        alignment=TA_CENTER,
        leading=15,
    )
    styles['metric_sub'] = ParagraphStyle(
        'metric_sub',
        fontName='Helvetica',
        fontSize=9,
        textColor=WHITE,
        alignment=TA_CENTER,
        leading=12,
    )
    return styles


# ─────────────────────────────────────────────
# TABLE STYLE HELPERS
# ─────────────────────────────────────────────
def base_table_style(has_header=True, alt_rows=True, num_rows=0):
    cmds = [
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, -1), 'Times-Roman'),
        ('TEXTCOLOR', (0, 0), (-1, -1), SLATE),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1), [WHITE, WHITE]),
    ]
    if has_header:
        cmds += [
            ('BACKGROUND', (0, 0), (-1, 0), TEAL),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('TEXTCOLOR', (0, 0), (-1, 0), WHITE),
        ]
    if alt_rows and num_rows > 0:
        start = 1 if has_header else 0
        for i in range(start, num_rows):
            if (i - start) % 2 == 1:
                cmds.append(('BACKGROUND', (0, i), (-1, i), LIGHT_BG))
    return TableStyle(cmds)


def section_header(text, styles):
    """Return a list of [Paragraph, HRFlowable] for a section header."""
    return [
        Paragraph(text, styles['section']),
        HRFlowable(width='100%', thickness=1, color=TEAL, spaceAfter=4),
    ]


def subsection_header(text, styles):
    return [Paragraph(text, styles['subsection'])]


def callout_box(paragraphs_or_text, styles, bg=LIGHT_BG, border_color=TEAL):
    """Wrap content in a single-cell table callout box."""
    if isinstance(paragraphs_or_text, str):
        content = Paragraph(paragraphs_or_text, styles['callout'])
    else:
        content = paragraphs_or_text
    t = Table([[content]], colWidths=['100%'])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), bg),
        ('BOX', (0, 0), (-1, -1), 1.5, border_color),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    return t


# ─────────────────────────────────────────────
# PAGE 1 — COVER
# ─────────────────────────────────────────────
def build_cover(styles, col_width):
    story = []

    # Top navy banner
    banner_data = [
        [Paragraph('Termografía Infrarroja y Evaluación del Dolor Orofacial en\nTrastornos Temporomandibulares', styles['title'])],
        [Paragraph('Análisis Estadístico Complementario: Temperatura Normalizada, Covariables Demográficas y Subgrupo Masetero', styles['subtitle'])],
        [Paragraph('Abril 2026', styles['cover_meta'])],
        [Paragraph('Análisis computacional — Datos clínicos N=43 pacientes', styles['cover_meta'])],
    ]
    banner = Table(banner_data, colWidths=[col_width])
    banner.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), NAVY),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('RIGHTPADDING', (0, 0), (-1, -1), 12),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (0, 0), 18),
        ('BOTTOMPADDING', (0, -1), (0, -1), 18),
    ]))
    story.append(banner)
    story.append(Spacer(1, 8*mm))

    # Three metric boxes
    w3 = col_width / 3 - 3*mm
    box1_content = [
        Paragraph('N = 43 pacientes', styles['metric_label']),
        Paragraph('26 Con Dolor | 17 Sin Dolor', styles['metric_sub']),
    ]
    box2_content = [
        Paragraph('2,295 observaciones pareadas', styles['metric_label']),
        Paragraph('108 puntos dolorosos (4.7%)', styles['metric_sub']),
    ]
    box3_content = [
        Paragraph('4 modelos logísticos comparados', styles['metric_label']),
        Paragraph('Mejor AUC: 0.642 (Modelo C)', styles['metric_sub']),
    ]

    def make_metric_cell(content_list):
        inner = Table([[p] for p in content_list], colWidths=[w3 - 8*mm])
        return inner

    metrics_data = [[make_metric_cell(box1_content),
                     make_metric_cell(box2_content),
                     make_metric_cell(box3_content)]]
    metrics = Table(metrics_data, colWidths=[w3, w3, w3])
    metrics.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, 0), TEAL),
        ('BACKGROUND', (1, 0), (1, 0), NAVY),
        ('BACKGROUND', (2, 0), (2, 0), GOLD),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('BOX', (0, 0), (0, 0), 1, WHITE),
        ('BOX', (1, 0), (1, 0), 1, WHITE),
        ('BOX', (2, 0), (2, 0), 1, WHITE),
    ]))
    story.append(metrics)
    story.append(Spacer(1, 8*mm))

    # Description paragraph
    desc = (
        "El presente reporte compila los resultados del análisis estadístico integral de la "
        "relación entre la asimetría térmica facial bilateral (ΔT) medida mediante termografía "
        "infrarroja y la presencia e intensidad del dolor durante el examen de palpación muscular "
        "orofacial en 43 pacientes. Se evaluaron cuatro regiones de interés (ROI) termográficas "
        "y múltiples métodos estadísticos, incluyendo correlaciones de Spearman con corrección "
        "de Bonferroni, análisis de concordancia espacial (kappa de Cohen), curvas ROC, "
        "regresión logística con errores robustos por clúster y análisis de covariables "
        "demográficas (sexo, edad). El hallazgo central es que la asimetría térmica pre-palpación "
        "no predice significativamente el dolor en palpación en ningún nivel de análisis, aunque "
        "sexo y edad emergen como confundidores estadísticamente relevantes."
    )
    story.append(Paragraph(desc, styles['body']))

    story.append(Spacer(1, 6*mm))

    # Key content summary box
    summary_items = [
        Paragraph('<b>Regiones evaluadas:</b> R1 (Temporal), R2 (Esternoc/Maset sup), R3 (ATM/Masetero), R4 (Maset inferior)', styles['callout']),
        Paragraph('<b>Músculos:</b> ATM (anterior/posterior), Esternocleidomastoideo, Masetero, Temporal', styles['callout']),
        Paragraph('<b>Período:</b> Imágenes termográficas pre-palpación (primera imagen)', styles['callout']),
        Paragraph('<b>Resultado principal:</b> Ninguna correlación ΔT–dolor supera la corrección de Bonferroni (56 pruebas)', styles['callout']),
    ]
    inner_tbl = Table([[item] for item in summary_items], colWidths=[col_width - 20*mm])
    story.append(callout_box(inner_tbl, styles, bg=LIGHT_BG, border_color=TEAL))

    story.append(PageBreak())
    return story


# ─────────────────────────────────────────────
# PAGE 2 — EXECUTIVE SUMMARY
# ─────────────────────────────────────────────
def build_executive_summary(styles, col_width):
    story = []

    # Full-width TEAL strip header
    hdr_data = [[Paragraph('RESUMEN EJECUTIVO', styles['exec_header'])]]
    hdr = Table(hdr_data, colWidths=[col_width])
    hdr.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), TEAL),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(hdr)
    story.append(Spacer(1, 6*mm))

    # Three-column content
    col_w = col_width / 3 - 2*mm

    # Col 1 — Pregunta
    c1 = [
        Paragraph('Pregunta de investigación', styles['col_header']),
        Paragraph(
            '¿La asimetría térmica facial pre-palpación predice la presencia e intensidad '
            'del dolor durante el examen clínico orofacial?',
            styles['body']
        ),
    ]

    # Col 2 — Hallazgos
    hallazgos = [
        '★ La correlación ΔT–dolor es NO significativa (ρ=−0.013, p<sub>bonf</sub>=1.000) en todos los análisis',
        '★ La normalización por temperatura basal NO mejora la predicción del dolor',
        '★ El sexo y la edad son confundidores significativos (LRT p=0.018)',
        '★ Asimetría derecha sistemática en los 4 ROI (Wilcoxon p&lt;0.0001 en todos)',
        '★ La región más caliente (R1/R2) NO coincide con la más dolorosa (R3) en el 88.5% de casos',
    ]
    c2 = [Paragraph('Hallazgos principales', styles['col_header'])]
    for h in hallazgos:
        c2.append(Paragraph(f'• {h}', styles['bullet']))

    # Col 3 — Implicaciones
    impl = [
        'La termografía regional NO es un predictor confiable del dolor en palpación muscular orofacial con los métodos actuales.',
        'Los modelos deben ajustar por sexo y edad en estudios futuros.',
        'Se requiere N mayor y análisis específico de masetero R3.',
    ]
    c3 = [Paragraph('Implicaciones clínicas', styles['col_header'])]
    for i in impl:
        c3.append(Paragraph(f'• {i}', styles['bullet']))

    def col_table(items, w):
        rows = [[item] for item in items]
        t = Table(rows, colWidths=[w])
        t.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING', (0, 0), (-1, -1), 2),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('RIGHTPADDING', (0, 0), (-1, -1), 0),
        ]))
        return t

    three_col = Table(
        [[col_table(c1, col_w), col_table(c2, col_w), col_table(c3, col_w)]],
        colWidths=[col_w, col_w, col_w],
        hAlign='LEFT'
    )
    three_col.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ('LINEAFTER', (0, 0), (1, 0), 0.5, BORDER),
    ]))
    story.append(three_col)
    story.append(Spacer(1, 6*mm))

    # Tabla 0 — Resumen ejecutivo de hipótesis
    story += section_header('Tabla 0 — Resumen ejecutivo de hipótesis', styles)

    h0_data = [
        ['Hipótesis', 'Prueba', 'Resultado', 'Significativo'],
        ['H1: ΔT ↔ dolor (global)', 'Spearman + permutación + Bonferroni', 'ρ=−0.013, p_bonf=1.000', 'No'],
        ['H2: Concordancia espacial', 'Kappa de Cohen', 'κ=−0.008, 58.1% concordancia', 'No'],
        ['H3: Utilidad diagnóstica (AUC)', 'ROC global', 'AUC=0.482', 'No (< 0.5)'],
        ['H4: ΔT_max paciente', 'Mann-Whitney U (N=43)', 'p=0.900', 'No'],
        ['H_alt: Sexo+edad en logit', 'LRT (Modelo C vs A)', 'χ²=8.04, p=0.018', 'Sí ★'],
    ]
    col_ws = [col_width*0.28, col_width*0.33, col_width*0.24, col_width*0.15]
    h0_tbl = Table(h0_data, colWidths=col_ws)
    h0_tbl.setStyle(base_table_style(has_header=True, alt_rows=True, num_rows=len(h0_data)))
    # Color "Sí" in green and "No" in red
    for i, row in enumerate(h0_data[1:], 1):
        sig = row[3]
        if 'Sí' in sig:
            h0_tbl.setStyle(TableStyle([('TEXTCOLOR', (3, i), (3, i), GREEN_OK),
                                        ('FONTNAME', (3, i), (3, i), 'Helvetica-Bold')]))
        else:
            h0_tbl.setStyle(TableStyle([('TEXTCOLOR', (3, i), (3, i), RED_SOFT)]))
    story.append(h0_tbl)
    story.append(PageBreak())
    return story


# ─────────────────────────────────────────────
# PAGES 3–4 — INTRODUCTION + METHODS
# ─────────────────────────────────────────────
def build_intro_methods(styles, col_width):
    story = []

    # INTRODUCCIÓN
    story += section_header('1. INTRODUCCIÓN', styles)

    intro_text = [
        ("Los trastornos temporomandibulares (TTM) representan el grupo de condiciones musculoesqueléticas "
         "más prevalente en la región orofacial, afectando entre el 5% y el 12% de la población general, "
         "con un impacto significativo en la calidad de vida, la función masticatoria y el bienestar "
         "psicosocial de los pacientes. El diagnóstico clínico de los TTM se basa fundamentalmente en "
         "la palpación muscular y articular, que provee información sobre la presencia e intensidad del "
         "dolor durante la compresión de los tejidos musculares."),
        ("La termografía infrarroja ha sido propuesta como biomarcador no invasivo del dolor y la "
         "inflamación muscular, dado que el incremento en la actividad metabólica y el flujo sanguíneo "
         "asociados a la inflamación producen elevaciones locales de temperatura detectables mediante "
         "cámaras termográficas de alta resolución. En el contexto de los TTM, la hipótesis es que la "
         "asimetría térmica bilateral (ΔT = T_derecha − T_izquierda) en las regiones musculares "
         "masticatorias podría reflejar el estado de dolor actual del paciente."),
        ("Sin embargo, los estudios previos muestran resultados inconsistentes: algunos reportan "
         "diferencias térmicas significativas entre grupos con y sin dolor, mientras otros no encuentran "
         "correlaciones clínicamente relevantes. Una limitación frecuente es la ausencia de corrección "
         "por multiplicidad de pruebas, la falta de análisis a nivel de punto anatómico individual, y "
         "el no considerar confundidores demográficos como sexo y edad. El presente estudio proporciona "
         "un análisis sistemático de resultados nulos, con corrección rigurosa de Bonferroni sobre 56 "
         "pruebas de correlación simultáneas."),
        ("<b>Objetivo del estudio:</b> Evaluar si la asimetría térmica bilateral (ΔT) pre-palpación, "
         "medida mediante termografía infrarroja en cuatro regiones de interés orofaciales, constituye "
         "un predictor estadísticamente válido de la presencia e intensidad del dolor durante el examen "
         "de palpación muscular orofacial en pacientes con y sin trastornos temporomandibulares."),
    ]
    for text in intro_text:
        story.append(Paragraph(text, styles['body']))

    story.append(Spacer(1, 4*mm))

    # MÉTODOS
    story += section_header('2. MÉTODOS', styles)

    # 2.1
    story += subsection_header('2.1 Diseño del estudio y participantes', styles)
    story.append(Paragraph(
        "Estudio transversal, observacional. Se evaluaron N=43 pacientes asignados a dos grupos "
        "según la presencia de puntos dolorosos a la palpación: <b>Con Dolor</b> (≥1 punto doloroso, "
        "n=26) y <b>Sin Dolor</b> (0 puntos dolorosos, n=17). La distribución por sexo fue 32.6% "
        "masculino y 67.4% femenino, con rango de edad 18–78 años. El diagnóstico de TTM según DC/TMD "
        "se registró como variable secundaria: 8/43 pacientes (18.6%), predominantemente en el grupo "
        "Sin Dolor (6/17 vs 2/26), lo que confirma que el diagnóstico de TTM no es equivalente a la "
        "presencia de dolor agudo durante la palpación. Por razones metodológicas, el diagnóstico TTM "
        "se excluyó como variable de agrupación primaria.",
        styles['body']
    ))

    # 2.2
    story += subsection_header('2.2 Adquisición de datos termográficos', styles)
    story.append(Paragraph(
        "Las imágenes termográficas fueron adquiridas en condiciones controladas de temperatura ambiente "
        "(20.3°C constante en todos los pacientes). Se utilizó la <b>primera imagen</b> (pre-palpación) "
        "como estándar de oro termográfico, evitando la influencia del calor por fricción mecánica "
        "asociado a la palpación. Se definieron cuatro regiones de interés (ROI):",
        styles['body']
    ))
    roi_items = [
        '<b>R1</b> — Temporal (músculo temporal)',
        '<b>R2</b> — Esternoc/Maset superior (esternocleidomastoideo y masetero superior)',
        '<b>R3</b> — ATM/Masetero (articulación temporomandibular y masetero central)',
        '<b>R4</b> — Masetero inferior (porción inferior del masetero)',
    ]
    for item in roi_items:
        story.append(Paragraph(f'• {item}', styles['bullet']))

    # 2.3 Variables table
    story += subsection_header('2.3 Variables', styles)
    var_data = [
        ['Variable', 'Descripción'],
        ['ΔT_regional', 'T_derecha − T_izquierda por ROI (pre-palpación)'],
        ['ΔT_norm', '(T_media − T_basal)_der − (T_media − T_basal)_izq'],
        ['temp_max', 'Temperatura máxima regional (°C)'],
        ['intensidad', 'Intensidad de dolor 0–10 en punto anatómico'],
        ['tiene_dolor', 'Binaria: 1 si intensidad > 0'],
        ['sexo_num', '0=Masculino, 1=Femenino'],
        ['edad', 'Edad en años'],
    ]
    var_w = [col_width * 0.22, col_width * 0.78]
    var_tbl = Table(var_data, colWidths=var_w)
    var_tbl.setStyle(base_table_style(has_header=True, alt_rows=True, num_rows=len(var_data)))
    story.append(var_tbl)
    story.append(Spacer(1, 4*mm))

    # 2.4 Statistical methods
    story += subsection_header('2.4 Análisis estadístico', styles)
    stat_data = [
        ['Objetivo', 'Método', 'Software'],
        ['Correlación ΔT ↔ dolor', 'Spearman + prueba de permutación por bloques + Bonferroni', 'Python / scipy'],
        ['Concordancia espacial', 'Kappa de Cohen', 'sklearn'],
        ['Utilidad diagnóstica', 'ROC, AUC-PR, umbral Youden J', 'sklearn'],
        ['Regresión', 'Logística con SE robustos por clúster (sandwich estimator)', 'statsmodels'],
        ['Comparación modelos', 'LRT (log-likelihood ratio test), AIC, AUC', 'statsmodels'],
        ['Asimetría poblacional', 'Wilcoxon one-sample; Mann-Whitney U', 'scipy'],
        ['Confundidores', 'Spearman (edad vs temp), Mann-Whitney (sexo vs temp)', 'scipy'],
    ]
    stat_ws = [col_width * 0.28, col_width * 0.49, col_width * 0.23]
    stat_tbl = Table(stat_data, colWidths=stat_ws)
    stat_tbl.setStyle(base_table_style(has_header=True, alt_rows=True, num_rows=len(stat_data)))
    story.append(stat_tbl)

    stat_notes = [
        '• α = 0.05 (bilateral)',
        '• Corrección de Bonferroni aplicada sobre todas las pruebas de correlación conjuntamente (56 pruebas)',
        '• Clúster = paciente (ID) para todos los análisis clusterizados (n=43 clústeres)',
        '• Desequilibrio de clases: 4.7% puntos dolorosos → pesos balanceados en regresión logística',
    ]
    for note in stat_notes:
        story.append(Paragraph(note, styles['footnote']))

    story.append(PageBreak())
    return story


# ─────────────────────────────────────────────
# PAGES 5–7 — RESULTS
# ─────────────────────────────────────────────
def build_results(styles, col_width):
    story = []

    story += section_header('3. RESULTADOS', styles)

    # 3.1 Características de la muestra
    story += subsection_header('3.1 Características de la muestra', styles)

    sample_data = [
        ['Variable', 'Total (N=43)', 'Con Dolor (n=26)', 'Sin Dolor (n=17)'],
        ['Sexo femenino, n (%)', '29 (67.4%)', '17 (65.4%)', '12 (70.6%)'],
        ['Edad, rango', '18–78 años', '—', '—'],
        ['Diagnóstico TTM, n (%)', '8 (18.6%)', '2 (7.7%)', '6 (35.3%)'],
        ['Puntos anatómicos evaluados', '2,295 totales', '108 dolorosos', '2,187 sin dolor'],
        ['T° ambiente (°C)', '20.3 ± 0.0', '—', '—'],
    ]
    s_ws = [col_width * 0.34, col_width * 0.22, col_width * 0.22, col_width * 0.22]
    s_tbl = Table(sample_data, colWidths=s_ws)
    s_tbl.setStyle(base_table_style(has_header=True, alt_rows=True, num_rows=len(sample_data)))
    story.append(s_tbl)

    story.append(Paragraph(
        "Los dos grupos (Con Dolor / Sin Dolor) fueron similares en distribución por sexo (Fisher p&gt;0.05). "
        "El diagnóstico de TTM fue más frecuente en el grupo Sin Dolor (31.6% vs 7.7%), confirmando que el "
        "diagnóstico de TTM no equivale a la presencia de dolor agudo durante la palpación muscular.",
        styles['body']
    ))

    # 3.2 H1 Correlación
    story += subsection_header('3.2 Hipótesis H1 — Correlación ΔT ↔ Intensidad de Dolor', styles)
    story.append(Paragraph(
        "<i>Ningún análisis de correlación Spearman alcanzó significancia estadística tras corrección de "
        "Bonferroni (56 pruebas).</i>",
        styles['body']
    ))

    # Two-column layout: table left, interpretation right
    spear_data = [
        ['Nivel', 'Subgrupo', 'ρ', 'p_orig', 'p_bonf', 'Sig.'],
        ['Global', 'Todos', '−0.013', '0.584', '1.000', 'No'],
        ['Región', 'R1 (Temporal)', '−0.037', '0.599', '1.000', 'No'],
        ['Región', 'R2 (Esternoc/Maset)', '+0.094', '0.144', '1.000', 'No'],
        ['Región', 'R3 (ATM/Masetero)', '−0.047', '0.268', '1.000', 'No'],
        ['Región', 'R4 (Maset. inf.)', '+0.013', '0.788', '1.000', 'No'],
        ['Músculo', 'ATM', '−0.049', '0.331', '1.000', 'No'],
        ['Músculo', 'Esternocleidomastoideo', '+0.047', '0.615', '1.000', 'No'],
        ['Músculo', 'Masetero', '+0.003', '0.918', '1.000', 'No'],
        ['Músculo', 'Temporal', '−0.037', '0.599', '1.000', 'No'],
        ['Punto más cercano', 'Masetero D R3P1', '−0.387', '0.009', '0.489', 'No*'],
    ]
    left_w = col_width * 0.57
    right_w = col_width * 0.40
    sp_ws = [left_w*0.18, left_w*0.30, left_w*0.12, left_w*0.13, left_w*0.13, left_w*0.12]
    sp_tbl = Table(spear_data, colWidths=sp_ws)
    sp_tbl.setStyle(base_table_style(has_header=True, alt_rows=True, num_rows=len(spear_data)))

    interp_text = (
        "<b>INTERPRETACIÓN:</b> Los valores de |ρ| ≤ 0.094 indican correlaciones muy débiles. "
        "El mejor candidato (Masetero D R3P1, ρ=−0.387) pierde significancia tras Bonferroni. "
        "Se concluye que <b>no se rechaza H0</b> en ningún nivel de análisis.\n\n"
        "* Pre-Bonferroni p=0.009 se convierte en p_bonf=0.489; la corrección es necesaria "
        "dada la multiplicidad de pruebas."
    )
    interp_para = Paragraph(interp_text, styles['callout'])
    interp_box = callout_box(interp_para, styles, bg=LIGHT_BG)

    two_col = Table([[sp_tbl, interp_box]], colWidths=[left_w, right_w])
    two_col.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
        ('TOPPADDING', (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ('LEFTPADDING', (1, 0), (1, 0), 8),
    ]))
    story.append(two_col)
    story.append(Spacer(1, 4*mm))

    # 3.3 ΔT Normalizada
    story += subsection_header('3.3 Análisis Alternativo — Temperatura Normalizada (T − T_basal)', styles)
    story.append(Paragraph(
        "Como extensión, se calculó ΔT_norm = (T_regional − T_basal)_der − (T_regional − T_basal)_izq, "
        "donde T_basal es la temperatura corporal basal (°C). Esta normalización controla las diferencias "
        "individuales en termorregulación.",
        styles['body']
    ))
    norm_data = [
        ['Subgrupo', 'ρ_raw', 'ρ_norm', 'Δρ', 'Sig.'],
        ['Global', '−0.013', '−0.024', '−0.011', 'No en ambos'],
        ['R1', '−0.037', '+0.066', '+0.103', 'No en ambos'],
        ['R2', '+0.094', '−0.083', '−0.177', 'No en ambos'],
        ['R3', '−0.047', '−0.026', '+0.021', 'No en ambos'],
        ['R4', '+0.013', '+0.014', '+0.001', 'No en ambos'],
    ]
    n_ws = [col_width * 0.20, col_width * 0.17, col_width * 0.17, col_width * 0.17, col_width * 0.29]
    n_tbl = Table(norm_data, colWidths=n_ws)
    n_tbl.setStyle(base_table_style(has_header=True, alt_rows=True, num_rows=len(norm_data)))
    story.append(n_tbl)
    story.append(Paragraph(
        "La normalización no mejora la correlación con el dolor. Las correlaciones siguen siendo no "
        "significativas en todos los niveles (p_bonf=1.000).",
        styles['body']
    ))

    # 3.4 H2 Concordancia
    story += subsection_header('3.4 Hipótesis H2 — Concordancia Espacial', styles)
    story.append(Paragraph(
        "Se evaluó si la región termográficamente más caliente (por temperatura media bilateral) coincide "
        "con la región más dolorosa durante la palpación.",
        styles['body']
    ))
    kappa_items = [
        Paragraph('• κ de Cohen = −0.008 (concordancia casi aleatoria)', styles['callout']),
        Paragraph('• Concordancia directa: 58.1% (25/43 pares válidos)', styles['callout']),
        Paragraph('• Región más caliente: <b>R1 (Temporal)</b> en la mayoría de pacientes', styles['callout']),
        Paragraph('• Región más dolorosa: <b>R3 (ATM/Masetero)</b> en 81% de pacientes con dolor', styles['callout']),
    ]
    inner = Table([[item] for item in kappa_items], colWidths=[col_width - 16*mm])
    story.append(callout_box(inner, styles, bg=LIGHT_BG))
    story.append(Paragraph(
        "Se rechaza H2 afirmativa: existe <b>disociación espacial sistemática</b> entre el hotspot "
        "térmico y el sitio de dolor. La termografía regional no localiza el dolor muscular.",
        styles['body']
    ))

    # 3.5 H3 ROC
    story += subsection_header('3.5 Hipótesis H3 — Utilidad Diagnóstica (ROC)', styles)
    roc_data = [
        ['Subgrupo', 'AUC-ROC', 'AUC-PR', 'T* (°C)', 'Sens', 'Spec', 'Youden J'],
        ['Global', '0.482', '0.044', '0.8', '0.213', '0.802', '0.015'],
        ['R1 (Temporal)', '0.443', '0.034', '0.2', '0.909', '0.158', '0.067'],
        ['R2 (Esternoc.)', '0.663', '0.044', '0.5', '0.800', '0.543', '0.343'],
        ['R3 (ATM/Maset.)', '0.447', '0.083', '1.6', '0.064', '0.957', '0.020'],
        ['R4 (Maset. inf.)', '0.514', '0.049', '0.7', '0.316', '0.806', '0.122'],
        ['ATM', '0.442', '0.076', '1.6', '0.059', '0.956', '0.015'],
        ['Esternocleid.', '0.577', '0.034', '0.5', '0.727', '0.542', '0.269'],
        ['Masetero', '0.505', '0.052', '0.7', '0.308', '0.792', '0.099'],
        ['Temporal', '0.443', '0.034', '0.2', '0.909', '0.158', '0.067'],
    ]
    r_ws = [col_width*0.22, col_width*0.11, col_width*0.11, col_width*0.10,
            col_width*0.10, col_width*0.10, col_width*0.12]
    # Adjust to fill width
    total_w = sum(r_ws)
    scale = col_width / total_w
    r_ws = [w * scale for w in r_ws]
    r_tbl = Table(roc_data, colWidths=r_ws)
    r_tbl.setStyle(base_table_style(has_header=True, alt_rows=True, num_rows=len(roc_data)))
    # Highlight R2 as best AUC
    r_tbl.setStyle(TableStyle([('TEXTCOLOR', (1, 3), (1, 3), GREEN_OK),
                                ('FONTNAME', (1, 3), (1, 3), 'Helvetica-Bold')]))
    story.append(r_tbl)
    story.append(Paragraph(
        "Ningún subgrupo alcanza AUC ≥ 0.70. El AUC global (0.482) es inferior a 0.50, indicando "
        "rendimiento peor que el azar. <b>No se rechaza H0 en H3.</b> R2 muestra el AUC más alto "
        "(0.663) pero con Youden J bajo (0.343) — no constituye un test diagnóstico clínicamente útil.",
        styles['body']
    ))

    story.append(Spacer(1, 3*mm))
    story.append(Paragraph('<b>ROC: temperatura normalizada vs cruda</b>', styles['subsection']))
    roc_norm_data = [
        ['Subgrupo', 'AUC_crudo', 'AUC_norm', 'Δ'],
        ['Global', '0.482', '0.467', '−0.015'],
        ['R1', '0.443', '0.598', '+0.155 ★'],
        ['R2', '0.663', '0.354', '−0.309'],
        ['R3', '0.447', '0.467', '+0.020'],
        ['R4', '0.514', '0.518', '+0.004'],
    ]
    rn_ws = [col_width*0.28, col_width*0.20, col_width*0.20, col_width*0.20]
    rn_tbl = Table(roc_norm_data, colWidths=rn_ws)
    rn_tbl.setStyle(base_table_style(has_header=True, alt_rows=True, num_rows=len(roc_norm_data)))
    rn_tbl.setStyle(TableStyle([('TEXTCOLOR', (3, 2), (3, 2), GREEN_OK),
                                 ('FONTNAME', (3, 2), (3, 2), 'Helvetica-Bold')]))
    story.append(rn_tbl)
    story.append(Paragraph(
        "La normalización mejora marginalmente R1 y R3 pero empeora R2. No hay un patrón consistente "
        "que justifique la normalización como estrategia global.",
        styles['body']
    ))

    # 3.6 H4
    story += subsection_header('3.6 Hipótesis H4 — Análisis a Nivel de Paciente (N=43)', styles)
    story.append(Paragraph(
        "A nivel de paciente, se comparó el ΔT_max (máxima asimetría entre regiones) entre grupos "
        "Con Dolor y Sin Dolor.",
        styles['body']
    ))
    box_a = callout_box(
        Paragraph('<b>Mann-Whitney U (bilateral):</b> p = 0.900 → <b>No significativo</b>', styles['callout']),
        styles, bg=LIGHT_BG
    )
    box_b = callout_box(
        Paragraph('<b>ROC ΔT_max → dolor:</b> AUC = 0.507 → Sin capacidad discriminativa', styles['callout']),
        styles, bg=LIGHT_BG
    )
    h4_row = Table([[box_a, box_b]], colWidths=[col_width/2 - 2*mm, col_width/2 - 2*mm])
    h4_row.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
        ('TOPPADDING', (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (0, 0), 4),
        ('LEFTPADDING', (1, 0), (1, 0), 4),
    ]))
    story.append(h4_row)
    story.append(Paragraph(
        "La asimetría térmica máxima no predice la presencia de dolor a nivel de paciente. "
        "Se confirma la hipótesis nula H4.",
        styles['body']
    ))

    story.append(PageBreak())

    # 3.7 Modelos logísticos
    story += subsection_header('3.7 Modelos Logísticos Extendidos', styles)
    story.append(Paragraph(
        "Se ajustaron cuatro modelos de regresión logística con errores estándar robustos por clúster "
        "(paciente). Los pesos de clase balanceados compensan el desequilibrio severo (4.7% positivos).",
        styles['body']
    ))

    mod_data = [
        ['Modelo', 'Fórmula', 'AUC', 'AIC', 'Pseudo-R²', 'LRT vs. ref', 'p_LRT'],
        ['A (referencia)', 'ΔT + región + lado', '0.615', '863.8', '0.022', '—', '—'],
        ['C ★', '+ sexo + edad', '0.642', '859.7', '0.031', 'χ²=8.04, df=2', '0.018'],
        ['D', 'ΔT_norm + región + lado', '0.614', '866.1', '0.019', '—', '—'],
        ['E', 'D + sexo + edad', '0.632', '863.5', '0.027', 'χ²=6.67, df=2', '0.036'],
    ]
    m_ws = [col_width*0.12, col_width*0.24, col_width*0.09, col_width*0.09,
            col_width*0.12, col_width*0.19, col_width*0.12]
    m_tbl = Table(mod_data, colWidths=m_ws)
    m_tbl.setStyle(base_table_style(has_header=True, alt_rows=True, num_rows=len(mod_data)))
    # Highlight Model C
    m_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 2), (-1, 2), HexColor('#E6F4F1')),
        ('FONTNAME', (0, 2), (-1, 2), 'Times-Bold'),
        ('TEXTCOLOR', (6, 2), (6, 2), GREEN_OK),
        ('TEXTCOLOR', (6, 4), (6, 4), GREEN_OK),
    ]))
    story.append(m_tbl)
    story.append(Paragraph(
        "★ <b>Hallazgo clave:</b> El Modelo C (sexo + edad añadidos) mejora significativamente el "
        "ajuste sobre el Modelo A (LRT p=0.018). El AUC mejora de 0.615 → 0.642. Aunque el AUC global "
        "sigue siendo limitado, esto demuestra que <b>sexo y edad son confundidores significativos</b> "
        "que deben incluirse en cualquier modelo futuro.",
        styles['body']
    ))

    story.append(Spacer(1, 3*mm))
    story.append(Paragraph('<b>Coeficientes del Modelo C (tabla detallada)</b>', styles['subsection']))
    coef_data = [
        ['Término', 'OR', 'IC 95%', 'p', 'Sig.'],
        ['Intercepto', '18,053', '(0.005, 6.4×10¹⁰)', '0.203', 'No'],
        ['temp_max', '0.675', '(0.435, 1.047)', '0.080', '†'],
        ['Región R2 vs R1', '0.589', '(0.268, 1.292)', '0.186', 'No'],
        ['Región R3 vs R1', '1.716', '(0.989, 2.979)', '0.055', '†'],
        ['Región R4 vs R1', '0.668', '(0.194, 2.293)', '0.521', 'No'],
        ['Lado izquierdo', '0.956', '(0.658, 1.390)', '0.815', 'No'],
        ['Sexo femenino', '1.935', '(0.929, 4.029)', '0.078', '†'],
        ['Edad (por año)', '0.999', '(0.974, 1.024)', '0.927', 'No'],
    ]
    c_ws = [col_width*0.28, col_width*0.12, col_width*0.30, col_width*0.12, col_width*0.10]
    c_tbl = Table(coef_data, colWidths=c_ws)
    c_tbl.setStyle(base_table_style(has_header=True, alt_rows=True, num_rows=len(coef_data)))
    story.append(c_tbl)
    story.append(Paragraph(
        "† p &lt; 0.10; Sí = p &lt; 0.05. SE robustos por clúster (sandwich estimator, n=43 clústeres). "
        "Pesos balanceados: w_pos=10.64, w_neg=0.526.",
        styles['footnote']
    ))

    # 3.8 Asimetría derecha
    story += subsection_header('3.8 Nuevo Hallazgo — Asimetría Derecha Sistemática', styles)
    story.append(Paragraph(
        "Un hallazgo no anticipado es la <b>asimetría derecha sistemática</b> en las cuatro regiones "
        "termográficas: el lado derecho es consistentemente más caliente que el izquierdo.",
        styles['body']
    ))
    asim_data = [
        ['Región', 'ΔT medio (°C)', 'DE', '% Der > Izq', 'Wilcoxon p', 'MW ConD vs SinD'],
        ['R1 (Temporal)', '+0.751', '0.851', '97.8%', '< 0.0001', 'p=0.182 (NS)'],
        ['R2 (Esternoc/Maset)', '+0.520', '0.490', '93.3%', '< 0.0001', 'p=0.209 (NS)'],
        ['R3 (ATM/Masetero)', '+0.469', '0.400', '91.1%', '< 0.0001', 'p=1.000 (NS)'],
        ['R4 (Maset. inf.)', '+0.464', '0.461', '97.8%', '< 0.0001', 'p=0.572 (NS)'],
    ]
    a_ws = [col_width*0.22, col_width*0.15, col_width*0.10,
            col_width*0.15, col_width*0.17, col_width*0.21]
    a_tbl = Table(asim_data, colWidths=a_ws)
    a_tbl.setStyle(base_table_style(has_header=True, alt_rows=True, num_rows=len(asim_data)))
    # Color Wilcoxon p values in green
    for i in range(1, 5):
        a_tbl.setStyle(TableStyle([('TEXTCOLOR', (4, i), (4, i), GREEN_OK),
                                    ('FONTNAME', (4, i), (4, i), 'Times-Bold')]))
    story.append(a_tbl)
    story.append(Paragraph(
        "Esta asimetría es un rasgo <b>poblacional basal</b> — no es específica del dolor. Una posible "
        "explicación es la lateralización funcional (dominancia manual derecha) o diferencias anatómicas "
        "sutiles. <b>Importancia metodológica:</b> los valores de ΔT no son simétricos alrededor de cero; "
        "el valor esperado bajo H0 es +0.55°C (promedio entre regiones), no 0.",
        styles['body']
    ))

    # 3.9 Efectos de Sexo y Edad
    story += subsection_header('3.9 Efectos de Sexo y Edad (Confundidores)', styles)

    # Two mini-tables side by side
    sex_data = [
        ['Región/Lado', 'Med. Masc.', 'Med. Fem.', 'p (MW)', 'Sig.'],
        ['R4 derecha', '33.7°C', '32.5°C', '0.015', 'Sí ★'],
        ['R4 izquierda', '33.4°C', '32.4°C', '0.002', 'Sí ★'],
        ['R1-R3 (todos)', '—', '—', '>0.20', 'No'],
    ]
    age_data = [
        ['Región/Lado', 'ρ', 'p', 'Sig.'],
        ['R2 izquierda', '−0.299', '0.046', 'Sí ★'],
        ['R4 izquierda', '−0.300', '0.045', 'Sí ★'],
        ['R2 derecha', '−0.221', '0.145', 'No'],
        ['R4 derecha', '−0.278', '0.064', '†'],
    ]

    half_w = col_width / 2 - 4*mm
    sx_ws = [half_w*0.32, half_w*0.18, half_w*0.18, half_w*0.18, half_w*0.14]
    sx_tbl = Table(sex_data, colWidths=sx_ws)
    sx_tbl.setStyle(base_table_style(has_header=True, alt_rows=True, num_rows=len(sex_data)))
    for i in [1, 2]:
        sx_tbl.setStyle(TableStyle([('TEXTCOLOR', (4, i), (4, i), GREEN_OK),
                                     ('FONTNAME', (4, i), (4, i), 'Times-Bold')]))

    age_ws = [half_w*0.40, half_w*0.18, half_w*0.18, half_w*0.14]
    age_tbl = Table(age_data, colWidths=age_ws)
    age_tbl.setStyle(base_table_style(has_header=True, alt_rows=True, num_rows=len(age_data)))
    for i in [1, 2]:
        age_tbl.setStyle(TableStyle([('TEXTCOLOR', (3, i), (3, i), GREEN_OK),
                                      ('FONTNAME', (3, i), (3, i), 'Times-Bold')]))

    def labeled_box(title, tbl):
        title_para = Paragraph(f'<b>Tabla — {title}</b>', styles['caption'])
        inner = Table([[title_para], [tbl]], colWidths=[half_w])
        inner.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING', (0, 0), (-1, -1), 2),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('RIGHTPADDING', (0, 0), (-1, -1), 0),
        ]))
        return inner

    sex_box = labeled_box('Sexo vs Temperatura', sx_tbl)
    age_box = labeled_box('Edad vs Temperatura (Spearman)', age_tbl)

    two_confound = Table([[sex_box, age_box]], colWidths=[half_w, half_w])
    two_confound.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
        ('TOPPADDING', (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (0, 0), 6),
        ('LEFTPADDING', (1, 0), (1, 0), 6),
    ]))
    story.append(two_confound)
    story.append(Paragraph(
        "Hombres tienen temperatura significativamente mayor en R4 (Masetero inferior). A mayor edad, "
        "menor temperatura en R2 y R4 izquierdas. Estas asociaciones, aunque moderadas, justifican "
        "su inclusión como covariables en modelos ajustados.",
        styles['body']
    ))

    story.append(PageBreak())
    return story


# ─────────────────────────────────────────────
# PAGE 8 — DISCUSSION
# ─────────────────────────────────────────────
def build_discussion(styles, col_width):
    story = []

    story += section_header('4. DISCUSIÓN', styles)

    story += subsection_header('4.1 Ausencia de correlación ΔT–dolor: hallazgo robusto', styles)
    story.append(Paragraph(
        "El resultado nulo es consistente a través de todos los niveles de análisis: global, regional, "
        "muscular y de punto anatómico individual, tanto con temperaturas crudas como normalizadas, y "
        "tanto a nivel de observación (2,295 pares) como de paciente (N=43). Esta convergencia de "
        "resultados nulos sugiere que la asimetría térmica bilateral pre-palpación no codifica el estado "
        "de dolor actual de forma detectable con el presente tamaño muestral y metodología de medición. "
        "La magnitud de las correlaciones (|ρ| ≤ 0.094 en todos los análisis excepto el punto Masetero "
        "D R3P1) es consistente con ausencia de efecto clínicamente relevante.",
        styles['body']
    ))

    story += subsection_header('4.2 Implicaciones de la disociación espacial', styles)
    story.append(Paragraph(
        "La región termográficamente más caliente es sistemáticamente R1/R2 (temporal, "
        "esternocleidomastoideo/masetero superior), mientras que el dolor se concentra en R3 "
        "(ATM/masetero). Esta disociación espacial es un hallazgo negativo clave: el análisis "
        "termográfico regional de la cara no localiza el dolor muscular masticatorio. La concordancia "
        "kappa de −0.008 confirma que la asociación espacial es esencialmente aleatoria. Desde una "
        "perspectiva clínica, esto implica que la identificación de la región más caliente en una "
        "imagen termográfica facial no orienta hacia la región más dolorosa.",
        styles['body']
    ))

    story += subsection_header('4.3 Asimetría derecha: variable de confusión potencial', styles)
    story.append(Paragraph(
        "La dominancia derecha sistemática (+0.46 a +0.75°C) significa que la línea de base poblacional "
        "de ΔT no es cero. Los análisis futuros deberían utilizar |ΔT| centrado en la media poblacional "
        "(+0.55°C aprox.) o valores normalizados respecto a la media del grupo, en lugar de tratar "
        "ΔT=0 como la hipótesis nula. No hacerlo introduce un sesgo sistemático en la dirección "
        "derecha que puede confundirse con efectos de dolor.",
        styles['body']
    ))

    story += subsection_header('4.4 Sexo y edad como confundidores', styles)
    story.append(Paragraph(
        "Los resultados del LRT (p=0.018 para Modelo C vs A) establecen estadísticamente que sexo y "
        "edad mejoran el modelo. Aunque los tamaños de efecto son modestos, estas covariables deben "
        "incluirse en cualquier modelo termográfico multivariado. El efecto del sexo en R4 (masetero "
        "inferior) es anatómicamente plausible: los hombres generalmente tienen músculos maseteros "
        "más grandes con mayor temperatura basal metabólica. La asociación negativa de la edad con la "
        "temperatura en R2 y R4 puede reflejar la reducción fisiológica en la densidad microvascular "
        "y la actividad metabólica muscular con el envejecimiento.",
        styles['body']
    ))

    story += subsection_header('4.5 Limitaciones', styles)
    limitaciones = [
        "N=43 es modesto; la potencia estadística es limitada para detectar correlaciones pequeñas "
        "(potencia del 80% requiere N≈85 para ρ=0.30).",
        "Desequilibrio severo de clases (4.7% puntos dolorosos) limita las métricas de predicción binaria.",
        "Sesión única (sin datos longitudinales); el dolor medido durante la palpación y la temperatura "
        "medida antes pueden tener desalineación temporal.",
        "temp_punto (temperatura a nivel de píxel en el sitio de palpación) no estaba disponible para "
        "las imágenes pre-palpación.",
    ]
    for i, lim in enumerate(limitaciones, 1):
        story.append(Paragraph(f'{i}. {lim}', styles['numbered']))

    story += subsection_header('4.6 Comparación con literatura', styles)
    story.append(Paragraph(
        "Estudios termográficos previos en TTM reportan resultados inconsistentes, con algunos reportando "
        "AUC de 0.65–0.80 para regiones específicas. El presente resultado nulo puede reflejar: "
        "(a) diferentes fenotipos de dolor evaluados, (b) problemas de temporalidad en la medición, "
        "o (c) resolución espacial insuficiente con el análisis a nivel de ROI. La fuerza del presente "
        "estudio es la aplicación rigurosa de corrección de Bonferroni sobre 56 pruebas simultáneas y "
        "el análisis a múltiples niveles de agregación.",
        styles['body']
    ))

    story.append(PageBreak())
    return story


# ─────────────────────────────────────────────
# PAGE 9 — CONCLUSIONS + RECOMMENDATIONS
# ─────────────────────────────────────────────
def build_conclusions(styles, col_width):
    story = []

    story += section_header('5. CONCLUSIONES', styles)

    conclusiones = [
        "La asimetría térmica bilateral (ΔT) pre-palpación no se correlaciona significativamente con "
        "la intensidad ni la presencia de dolor durante el examen de palpación muscular orofacial "
        "(ρ ≈ 0, p_bonf = 1.000 en todos los niveles).",
        "La normalización por temperatura basal (T − T_basal) no mejora la correlación con el dolor.",
        "La capacidad diagnóstica de la termografía es inferior al nivel del azar para la detección "
        "de dolor (AUC_global = 0.482 &lt; 0.50).",
        "Existe una disociación espacial sistemática entre la región más caliente (R1/R2) y la más "
        "dolorosa (R3), lo que invalida la termografía regional como herramienta de localización "
        "del dolor masticatorio.",
        "Sexo y edad son confundidores estadísticamente significativos (LRT p=0.018) que deben "
        "incluirse en modelos futuros.",
        "Existe una asimetría derecha poblacional robusta (+0.46 a +0.75°C) independiente del estado "
        "de dolor.",
    ]
    for i, c in enumerate(conclusiones, 1):
        story.append(Paragraph(f'{i}. {c}', styles['numbered']))

    story.append(Spacer(1, 4*mm))

    # Final hypothesis summary table
    story += section_header('Tabla resumen final de hipótesis', styles)
    final_data = [
        ['Hipótesis', 'Variable térmica', 'Método', 'Resultado', 'Conclusión'],
        ['H1: ΔT ↔ dolor', 'ΔT_regional (crudo)', 'Spearman + Bonferroni (56 tests)', 'ρ=−0.013, p_bonf=1.000', 'No se rechaza H0'],
        ['H1_alt: ΔT_norm ↔ dolor', 'ΔT_normalizado', 'Spearman + Bonferroni', 'ρ=−0.024, p_bonf=1.000', 'No se rechaza H0'],
        ['H2: Concordancia espacial', 'Región más caliente', 'Kappa de Cohen', 'κ=−0.008, 58.1%', 'No se rechaza H0'],
        ['H3: Utilidad diagnóstica', 'ΔT global', 'ROC/AUC', 'AUC=0.482 (&lt;0.5)', 'No se rechaza H0'],
        ['H4: ΔT_max paciente', 'ΔT_max por paciente', 'Mann-Whitney U + ROC', 'p=0.900, AUC=0.507', 'No se rechaza H0'],
        ['H_asim: Asimetría derecha', 'ΔT_R1–R4', 'Wilcoxon one-sample', 'p&lt;0.0001 todos', 'Se rechaza H0 ★'],
        ['H_conf: Sexo+edad importan', '—', 'LRT Modelo C vs A', 'χ²=8.04, p=0.018', 'Se rechaza H0 ★'],
    ]
    f_ws = [col_width*0.17, col_width*0.17, col_width*0.22, col_width*0.22, col_width*0.22]
    f_tbl = Table(final_data, colWidths=f_ws)
    f_tbl.setStyle(base_table_style(has_header=True, alt_rows=True, num_rows=len(final_data)))
    # Color conclusions
    for i in range(1, len(final_data)):
        concl = final_data[i][4]
        if 'rechaza H0 ★' in concl:
            f_tbl.setStyle(TableStyle([('TEXTCOLOR', (4, i), (4, i), GREEN_OK),
                                        ('FONTNAME', (4, i), (4, i), 'Times-Bold')]))
        elif 'No se rechaza' in concl:
            f_tbl.setStyle(TableStyle([('TEXTCOLOR', (4, i), (4, i), RED_SOFT)]))
    story.append(f_tbl)
    story.append(Spacer(1, 5*mm))

    # Recommendations
    story += section_header('6. RECOMENDACIONES PARA TRABAJO FUTURO', styles)
    recomendaciones = [
        ("<b>Aumentar N:</b> Se necesitan N ≥ 30–40 pacientes Con Dolor para análisis con potencia "
         "adecuada. Target: N=100 total."),
        ("<b>Análisis por subgrupo anatómico:</b> Masetero R3 específicamente (el único punto con "
         "ρ ≈ −0.39 antes de Bonferroni). Un estudio dirigido a ese músculo específico es más prometedor."),
        ("<b>Ajuste por ΔT basal poblacional:</b> Centrar ΔT en la media poblacional (+0.55°C) antes "
         "de los análisis."),
        "<b>Incluir sexo y edad en todos los modelos</b> como covariables.",
        ("<b>Re-extraer coordenadas de píxel de primeras imágenes</b> para habilitar el análisis de "
         "temperatura puntual pre-palpación."),
        ("<b>Diseño longitudinal:</b> Comparar temperatura en el mismo paciente antes vs. durante vs. "
         "después de un episodio de dolor agudo."),
    ]
    for i, rec in enumerate(recomendaciones, 1):
        story.append(Paragraph(f'{i}. {rec}', styles['numbered']))

    story.append(PageBreak())
    return story


# ─────────────────────────────────────────────
# PAGE — FIGURA RESUMEN (Section 7)
# ─────────────────────────────────────────────
def build_figura_resumen(styles, col_width):
    import os
    story = []

    story += section_header('7. FIGURA RESUMEN', styles)
    story.append(Paragraph(
        "Las siguientes figuras sintetizan los principales hallazgos del análisis termográfico. "
        "Cada panel corresponde a una dimensión analítica independiente: "
        "rendimiento diagnóstico ROC, correlación Spearman ΔT–intensidad, "
        "mapa de correlaciones por región, asimetría térmica por grupos, "
        "forest plot del modelo logístico y resumen comparativo AUC.",
        styles['body']
    ))
    story.append(Spacer(1, 4 * mm))

    BASE = os.path.dirname(os.path.abspath(__file__))
    RESDIR = os.path.join(BASE, '../resultados')

    VIZ_FIGURES = [
        ('viz_01_roc_curvas.png',
         'Fig. 1. Curvas ROC por subgrupo anatómico (izq.) y comparación AUC-ROC vs AUC-PR (der.). '
         'R2 (Masetero) presenta el mayor AUC-ROC (0.661); el valor global es 0.477 (<0.5).'),
        ('viz_02_scatter_dt_intensidad.png',
         'Fig. 2. Dispersión ΔT máximo vs intensidad de dolor máxima (nivel paciente, N=43). '
         'Panel derecho: ΔT por región anatómica. ρ Spearman global ≈ −0.018 (n.s.).'),
        ('viz_03_heatmap_correlaciones.png',
         'Fig. 3. Mapa de métricas de asociación ΔT–dolor por subgrupo. '
         'Ningún p (Bonferroni) alcanza significación estadística (p_bonf = 1.000).'),
        ('viz_04_boxplot_dt_grupos.png',
         'Fig. 4. Distribución de ΔT por región (R1–R4): Con Dolor (n=26) vs Sin Dolor (n=17). '
         'Mann-Whitney U. Ninguna diferencia alcanza p<0.05.'),
        ('viz_05_forest_plot_logit.png',
         'Fig. 5. Forest plot del Modelo Logístico A (temp_max + región + lado, SE robusto por cluster). '
         'Ningún predictor alcanza p<0.05; OR de temperatura cercano a 1.'),
        ('viz_06_auc_barplot.png',
         'Fig. 6. AUC-ROC y AUC-PR por subgrupo (izq.) y diagrama sensibilidad vs especificidad '
         'en el umbral ΔT* óptimo según índice de Youden J (der.).'),
    ]

    # Render figures two per row
    IMG_W = col_width * 0.94        # slightly narrower than full width
    IMG_H = IMG_W * 0.42            # landscape aspect ratio

    caption_style = styles.get('caption', styles['body'])

    for fig_file, caption_text in VIZ_FIGURES:
        fig_path = os.path.join(RESDIR, fig_file)
        if os.path.exists(fig_path):
            try:
                img = Image(fig_path, width=IMG_W, height=IMG_H)
                story.append(img)
                story.append(Spacer(1, 2 * mm))
                story.append(Paragraph(caption_text, caption_style))
                story.append(Spacer(1, 6 * mm))
            except Exception as e:
                story.append(Paragraph(
                    f'[Error al cargar {fig_file}: {e}]', styles['body']))
        else:
            story.append(Paragraph(
                f'[Figura no encontrada: {fig_file}]', styles['body']))

    story.append(PageBreak())
    return story


# ─────────────────────────────────────────────
# PAGE 10 — DESIGN GUIDE
# ─────────────────────────────────────────────
def build_design_guide(styles, col_width):
    story = []

    story += section_header('GUÍA DE DISEÑO DEL REPORTE', styles)
    story.append(Paragraph(
        "Esta página documenta el sistema de diseño utilizado para este reporte científico, "
        "para facilitar su reproducción y actualización.",
        styles['body']
    ))

    # Color palette with swatches
    story += subsection_header('Paleta de colores', styles)

    colors_info = [
        (NAVY,     '#0D2B4E', 'Navy',      'Títulos, encabezados de sección, fondos de cubierta'),
        (TEAL,     '#1A7A8A', 'Teal',      'Acentos de sección, encabezados de tabla, reglas'),
        (SLATE,    '#4A5568', 'Slate Gray','Texto del cuerpo'),
        (LIGHT_BG, '#F7FAFC', 'Light BG', 'Filas alternadas de tabla, cajas de llamada'),
        (WHITE,    '#FFFFFF', 'White',     'Fondo de página, texto sobre fondos oscuros'),
        (GOLD,     '#D4A017', 'Gold',      'Resaltados, hallazgos clave, estrellas'),
        (RED_SOFT, '#C53030', 'Soft Red',  'Resultados no significativos, valores negativos'),
        (GREEN_OK, '#276749', 'Green',     'Resultados significativos, valores positivos'),
        (BORDER,   '#CBD5E0', 'Border',    'Líneas de tabla, divisores'),
    ]

    swatch_data = [['Muestra', 'Hex', 'Nombre', 'Uso']]
    for c, hex_val, name, usage in colors_info:
        swatch_cell = Table([['  ']], colWidths=[14*mm], rowHeights=[7*mm])
        swatch_cell.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), c),
            ('BOX', (0, 0), (-1, -1), 0.5, BORDER),
            ('TOPPADDING', (0, 0), (-1, -1), 0),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ]))
        swatch_data.append([swatch_cell, hex_val, name, usage])

    sw_ws = [16*mm, col_width*0.13, col_width*0.15, col_width*0.55]
    sw_tbl = Table(swatch_data, colWidths=sw_ws)
    sw_tbl.setStyle(base_table_style(has_header=True, alt_rows=True, num_rows=len(swatch_data)))
    story.append(sw_tbl)
    story.append(Spacer(1, 4*mm))

    # Font guide
    story += subsection_header('Sistema tipográfico', styles)
    font_data = [
        ['Elemento', 'Fuente', 'Tamaño', 'Color'],
        ['Título de portada', 'Helvetica-Bold', '22pt', 'Blanco sobre Navy'],
        ['Encabezado de sección', 'Helvetica-Bold', '13pt', 'Teal'],
        ['Subsección', 'Helvetica-Bold', '11pt', 'Navy'],
        ['Texto del cuerpo', 'Times-Roman', '10pt', 'Slate Gray'],
        ['Leyenda de tabla/figura', 'Times-Italic', '9pt', 'Slate Gray'],
        ['Encabezado de tabla', 'Helvetica-Bold', '9pt', 'Blanco sobre Teal'],
        ['Cuerpo de tabla', 'Times-Roman', '9pt', 'Slate Gray'],
        ['Notas al pie', 'Times-Italic', '8pt', 'Slate Gray'],
    ]
    ft_ws = [col_width*0.28, col_width*0.22, col_width*0.12, col_width*0.38]
    ft_tbl = Table(font_data, colWidths=ft_ws)
    ft_tbl.setStyle(base_table_style(has_header=True, alt_rows=True, num_rows=len(font_data)))
    story.append(ft_tbl)
    story.append(Spacer(1, 4*mm))

    # Layout description
    story += subsection_header('Cuadrícula de diseño', styles)
    layout_items = [
        'Tamaño de página: A4 (210 × 297 mm)',
        'Márgenes: izquierdo=20mm, derecho=20mm, superior=22mm, inferior=22mm',
        'Ancho de columna útil: 170mm',
        'Diseño de dos columnas para resultados: se usa Table con 2 cols para contenido lado a lado',
        'Espaciado de sección: 12pt antes, 6pt después de cada encabezado de sección',
        'Espaciado de párrafo: 4pt entre párrafos',
    ]
    for item in layout_items:
        story.append(Paragraph(f'• {item}', styles['bullet']))

    story.append(Spacer(1, 4*mm))

    # Visualization recommendations
    story += subsection_header('Recomendaciones de visualización', styles)
    viz_data = [
        ['Tipo de visualización', 'Descripción técnica'],
        ['Curvas ROC', 'Graficadas con bandas de confianza; AUC anotado; línea de referencia AUC=0.5'],
        ['Diagramas de dispersión', 'Scatter plot con línea de regresión LOESS; estratificado por sexo'],
        ['Mapas de calor (heatmaps)', 'Matriz 4×4 de ρ (región × músculo); escala divergente azul–rojo'],
        ['Boxplots', 'ΔT por grupos (Con/Sin Dolor), estratificado por sexo; con puntos individuales'],
        ['Gráficos de barras', 'ΔT por región y lado; con barras de error (±1 DE)'],
        ['Gráficos de violín', 'Distribución de temperatura por ROI, grupo y sexo'],
    ]
    viz_ws = [col_width*0.28, col_width*0.72]
    viz_tbl = Table(viz_data, colWidths=viz_ws)
    viz_tbl.setStyle(base_table_style(has_header=True, alt_rows=True, num_rows=len(viz_data)))
    story.append(viz_tbl)

    story.append(Spacer(1, 6*mm))
    story.append(HRFlowable(width='100%', thickness=1, color=BORDER))
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph(
        "Reporte generado mediante ReportLab 4.x (Python). Todos los análisis estadísticos realizados "
        "con Python 3.x (scipy, statsmodels, sklearn). Datos clínicos: N=43 pacientes. "
        "Análisis computacional, Abril 2026.",
        styles['caption']
    ))

    return story


# ─────────────────────────────────────────────
# MAIN BUILD FUNCTION
# ─────────────────────────────────────────────
def build_report(output_path):
    left_margin = 20 * mm
    right_margin = 20 * mm
    top_margin = 22 * mm
    bottom_margin = 22 * mm

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=left_margin,
        rightMargin=right_margin,
        topMargin=top_margin,
        bottomMargin=bottom_margin,
        title='Reporte Científico: Termografía y TTM',
        author='Análisis computacional',
        subject='Termografía Infrarroja y Trastornos Temporomandibulares',
    )

    page_width = A4[0] - left_margin - right_margin  # usable width
    styles = make_styles()

    story = []
    story += build_cover(styles, page_width)
    story += build_executive_summary(styles, page_width)
    story += build_intro_methods(styles, page_width)
    story += build_results(styles, page_width)
    story += build_discussion(styles, page_width)
    story += build_conclusions(styles, page_width)
    story += build_figura_resumen(styles, page_width)
    story += build_design_guide(styles, page_width)

    doc.build(story)
    print(f"PDF generado exitosamente: {output_path}")


if __name__ == '__main__':
    import os
    output = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '../reporte_analisis_complementario_imagenes_termograficas.pdf'
    )
    build_report(output)
