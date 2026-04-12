# Revisión Final del Proyecto — Termografía y TTM

Revisión exhaustiva de todos los scripts, datos y reportes del proyecto. Muestra definitiva: **N=43 pacientes únicos** (26 Con Dolor, 17 Sin Dolor), **2,295 observaciones pareadas**, **108 puntos dolorosos (4.7%)**.

---

## 🗂️ Estructura del Proyecto

### Scripts de Análisis
| Script | Propósito | Salidas |
|---|---|---|
| `analisis_termografia.py` | Análisis principal: ΔT crudo, correlación, ROC, regresión logística | `resultados/` |
| `analisis_termografia_complementario.py` | Análisis complementario: temperatura normalizada, confundidores demográficos, 4 modelos | `resultado_complementario/` |

### Scripts de Reportes PDF
| Script | Genera |
|---|---|
| `generar_reporte_publicacion.py` | `reporte_analisis_imagenes_termograficas.pdf` |
| `generar_reporte_cientifico.py` | `reporte_analisis_complementario_imagenes_termograficas.pdf` |
| `generar_reporte_interpretacion.py` | `reporte_interpretacion_termografia_ttm1.pdf` |

### Scripts de Visualización
| Script | Genera |
|---|---|
| `generar_visualizaciones_principales.py` | 6 PNGs en `resultados/` (viz_01 a viz_06) |
| `generar_visualizaciones_alterno.py` | 6 PNGs en `resultado_complementario/` (viz_A1 a viz_A6, sufijo `_complementario`) |

### Datos de Entrada
| Archivo | Descripción |
|---|---|
| `datos_finales_termografia_procesados_todas_fotos.csv` | Fuente única. N=43 tras `drop_duplicates(subset=['numero de muestra'])` |

---

## 🧪 Dualidad Metodológica: ¿Por qué dos análisis?

El proyecto cuenta con dos motores de análisis complementarios:

### 1. Análisis Principal (`analisis_termografia.py`)
- **Foco:** Rigor clínico y asimetría térmica regional absoluta
- **Variables:** ΔT crudo = T_derecha − T_izquierda por ROI; temperatura máxima regional
- **Propósito:** Validar si la diferencia térmica bilateral es un biomarcador directo de dolor
- **Modelos:** Modelo A (temp_max + región + lado) vs Modelo B (+ temp_punto de segunda imagen)
- **Salidas:** `resultados/` — 14 CSVs + 6 PNGs de publicación

### 2. Análisis Complementario (`analisis_termografia_complementario.py`)
- **Foco:** Sensibilidad biológica y control de variables confusoras
- **Variables:** ΔT normalizado = (T_facial − T_basal)_derecha − (T_facial − T_basal)_izquierda; sexo; edad
- **Propósito:** Identificar señales térmicas enmascaradas por variabilidad inter-sujeto y asimetría poblacional basal
- **Modelos:** A (base), C (A + sexo + edad ★), D (T_norm), E (D + sexo + edad)
- **Salidas:** `resultado_complementario/` — 11 CSVs + 6 PNGs de publicación (sufijo `_complementario`)

---

## ✅ Validación Estadística y de Rigor

### 1. Control de Independencia (Clúster)
Ambos scripts usan errores estándar robustos (`cov_type='cluster'`, sandwich estimator) con el ID del paciente como clúster. Esto es vital: las 2,295 observaciones no son independientes — cada paciente aporta múltiples puntos de palpación.

### 2. Robustez en Correlaciones
- **Spearman** (no paramétrico, adecuado para escala ordinal de dolor 0–10)
- **Prueba de permutación por bloques** en ambos scripts: `spearman_permutation_test()` reordena bloques completos de paciente, respetando la estructura de clúster (n=1,000 permutaciones)
- **Corrección de Bonferroni** aplicada sobre las 56 pruebas de correlación simultáneas

### 3. Modelos Multivariados
- **Modelo A:** ΔT_regional + región + lado (referencia)
- **Modelo C:** A + sexo + edad — LRT χ²=8.04, p=0.018 ★ → sexo y edad son confundidores significativos
- **Modelo B:** A + temp_punto (segunda imagen) — LRT p=0.451, ΔAUC≈0.0002 → no aporta
- Pesos de clase balanceados (`class_weight='balanced'`) para compensar el desequilibrio severo (4.7% positivos)

### 4. Asimetría Derecha Sistemática (Hallazgo no anticipado)
Validado con Wilcoxon one-sample vs 0, todas las regiones p<0.0001:

| Región | ΔT medio | % Der > Izq |
|---|---|---|
| R1 (Temporal) | +0.751°C | 97.8% |
| R2 (Esternoc./Maset. sup.) | +0.520°C | 93.3% |
| R3 (ATM/Masetero) | +0.469°C | 91.1% |
| R4 (Masetero inf.) | +0.464°C | 97.8% |

El valor esperado bajo H0 no es ΔT=0 sino ΔT≈+0.55°C (promedio poblacional). Los análisis futuros deben centrar ΔT en esta media.

---

## 📊 Hallazgos Clave Validados

| Hipótesis | Método | Resultado | Decisión |
|---|---|---|---|
| H1: ΔT ↔ intensidad dolor | Spearman + permutación + Bonferroni (56 pruebas) | ρ=−0.013, p_bonf=1.000 | No rechazar H0 |
| H1_alt: ΔT_norm ↔ dolor | Spearman + Bonferroni | ρ=−0.024, p_bonf=1.000 | No rechazar H0 |
| H2: Concordancia espacial | Kappa de Cohen | κ=−0.008, 58.1% concordancia | No rechazar H0 |
| H3: Utilidad diagnóstica | AUC-ROC global | AUC=0.482 (<0.5) | No rechazar H0 |
| H4: ΔT_max paciente | Mann-Whitney U + ROC | p=0.900, AUC=0.507 | No rechazar H0 |
| H_asim: Asimetría derecha | Wilcoxon one-sample | p<0.0001 en las 4 ROI | **Se rechaza H0 ★** |
| H_conf: Sexo+edad importan | LRT Modelo C vs A | χ²=8.04, p=0.018 | **Se rechaza H0 ★** |

**Mejor AUC diagnóstico:** R2 (Esternocleidomastoideo/Masetero superior) — AUC-ROC=0.661, pero Youden J=0.343 (insuficiente para uso clínico; se requiere J≥0.60).

**Hallazgo contraintuitivo:** Los pacientes Con Dolor muestran temperatura media **menor** que Sin Dolor en todas las regiones (ΔT negativo), compatible con vasoconstricción periférica asociada a dolor crónico, no con hipertermia inflamatoria.

---

## 🗂️ Directorios de Resultados

### `resultados/` — Análisis Principal
| Archivo | Contenido |
|---|---|
| `correlaciones_globales.csv` | ρ Spearman + p_orig + p_bonf por nivel/subgrupo |
| `resultados_roc.csv` | AUC-ROC, AUC-PR, T*, sens, spec, Youden J por subgrupo |
| `resultados_logit_cluster.csv` | OR, IC95%, p-valor del Modelo A |
| `comparacion_modelos_a_vs_b_temp_punto.csv` | LRT, AIC, AUC — Modelo A vs B |
| `resultados_paciente_n45.csv` | ΔT y dolor por paciente (N=43, nombre histórico) |
| `temperatura_por_region_grupo.csv` | T media por región y grupo |
| `concordancia_kappa.csv` | κ de Cohen, concordancia espacial |
| `top_puntos_dolorosos.csv` | Top-10 puntos anatómicos más dolorosos |
| `viz_01_roc_curvas.png` … `viz_06_auc_barplot.png` | 6 figuras de publicación (300 DPI) |

### `resultado_complementario/` — Análisis Complementario
| Archivo | Contenido |
|---|---|
| `asimetria_derecha_basal_complementario.csv` | ΔT medio y Wilcoxon por región |
| `correlaciones_normalizadas_complementario.csv` | ρ Spearman ΔT_norm vs dolor |
| `correlaciones_intensidad_continua_complementario.csv` | ρ Spearman ΔT vs intensidad continua |
| `resultados_roc_normalizados_complementario.csv` | AUC crudo vs normalizado por región |
| `resultados_logit_extendido_complementario.csv` | OR de los 4 modelos (A, C, D, E) |
| `comparacion_modelos_completa_complementario.csv` | LRT, AIC, AUC — comparación A/C/D/E |
| `analisis_subgrupo_masetero_r3_complementario.csv` | Subgrupo Masetero R3 específico |
| `resultados_paciente_n45_alterno_complementario.csv` | ΔT_norm y dolor por paciente |
| `efectos_sexo_edad_temperatura_complementario.csv` | Mann-Whitney sexo vs T; Spearman edad vs T |
| `viz_A1_asimetria_derecha_complementario.png` … `viz_A6_forest_modelos_A_C_complementario.png` | 6 figuras de publicación (300 DPI) |

---

## 📄 Reportes PDF Generados

| Archivo PDF | Script generador | Contenido |
|---|---|---|
| `reporte_analisis_imagenes_termograficas.pdf` | `generar_reporte_publicacion.py` | Análisis principal integral. Estilo revista médica con header/footer y numeración. 16 páginas, ~2.2 MB |
| `reporte_analisis_complementario_imagenes_termograficas.pdf` | `generar_reporte_cientifico.py` | Análisis complementario: temperatura normalizada, confundidores, 4 modelos. 16 páginas, ~2.1 MB |
| `reporte_interpretacion_termografia_ttm1.pdf` | `generar_reporte_interpretacion.py` | Reporte técnico detallado con advertencias metodológicas |

Ambos reportes principales incluyen:
- Portada con 3 metric cards clave
- Resumen ejecutivo (pregunta / hallazgos / implicaciones clínicas)
- Secciones IMRaD: Introducción, Métodos, Resultados, Discusión, Conclusiones
- **Sección 7: Figura Resumen** — las 6 figuras de publicación embebidas con leyendas
- Guía de diseño del sistema visual (paleta, tipografía, cuadrícula)

---

## 🔍 Manejo de Datos Faltantes (NaN)

### Temperaturas regionales (`temp_max`, `delta_t`)
**Política: No Imputación — exclusión del registro.**
Si el valor de temperatura regional es NaN, la observación se descarta con `continue` durante la creación del dataset pareado. No se convierte NaN a 0 (que sería un valor físico real: congelación).

### Temperatura puntual (`temp_punto`)
**Política: Preservación de NaN — análisis segregado.**
Se mantiene como `np.nan` y se filtra con `.dropna(subset=['temp_punto'])` antes de cada análisis que la requiera. La regresión logística de statsmodels también excluye automáticamente filas con NaN.

### Dolor (`intensidad`, `tiene_dolor`)
**Política: Cero por defecto.**
NaN en columnas de dolor se imputa como 0 (ausencia de dolor confirmada por diseño del instrumento).

### Conversión numérica
Se usa `pd.to_numeric(..., errors='coerce')` en todas las columnas térmicas para convertir cadenas de texto erróneas en NaN en lugar de generar errores de ejecución.

---

## 🛡️ Verificación de Consistencia entre Reportes

| Variable | `reporte_analisis_imagenes_termograficas.pdf` | `reporte_analisis_complementario_imagenes_termograficas.pdf` |
|---|---|---|
| N total | 43 | 43 |
| Con Dolor / Sin Dolor | 26 / 17 | 26 / 17 |
| Sexo femenino | 29 (67.4%) | 29 (67.4%) ✓ |
| Diagnóstico TTM | 8 (18.6%) | 8 (18.6%) ✓ |
| Edad mediana | 27 años | — |
| AUC global | 0.482 | 0.482 |
| Mejor AUC (modelo) | 0.642 (Modelo C) | 0.642 (Modelo C) |

---

*Última actualización: Abril 2026 — N=43 pacientes únicos, duplicados eliminados del CSV fuente.*
