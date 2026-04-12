# 🌡️ Estudio de Termografía Infrarroja y Trastornos Temporomandibulares (TTM)

## 📖 Overview (Visión General)
Este proyecto consiste en un pipeline completo de análisis de datos y modelado estadístico diseñado para evaluar la relación clínica entre la **asimetría térmica facial** (medida mediante termografía infrarroja pre-palpación) y la presencia e intensidad de **dolor orofacial** reportado por pacientes durante el examen clínico de palpación muscular.

El problema central que busca resolver es determinar empíricamente si la termografía puede utilizarse como un biomarcador objetivo o herramienta diagnóstica complementaria para los Trastornos Temporomandibulares (TTM), lidiando computacionalmente con un alto desbalance de clases (solo 4.7% de puntos reportan dolor) y controlando por variables confusoras biológicas (sexo, edad y asimetría térmica poblacional natural).

## 🏗️ Arquitectura y Workflow

El sistema procesa la información de forma secuencial a lo largo de 4 etapas principales (pipeline end-to-end):

1. **Ingesta y Limpieza de Datos (`proceso_de_datos.py`)**
   - Lectura de archivos Excel crudos con mediciones termográficas (temperaturas regionales y puntuales de las ROIs).
   - Limpieza estructurada de datos (conversión a variables numéricas, manejo de strings anómalos y extracción de identificadores).
2. **Fusión y Mapeo Espacial (`unificar_dolor_termografia.py`, `extraer_codigos_dolor.py`)**
   - Mapeo espacial entre las regiones térmicas y los puntos anatómicos específicos de dolor clínico evaluados en los pacientes (escala 0-10).
   - Generación de un conjunto de datos pareado consolidado para cada paciente.
3. **Modelado y Análisis Estadístico (`analisis_termografia.py`, `analisis_termografia_complementario.py`)**
   - **Estadística Descriptiva e Inferencial:** Pruebas de Wilcoxon, Mann-Whitney U, y pruebas de correlación no paramétricas (Spearman) con remuestreo de permutaciones por bloques y corrección de Bonferroni.
   - **Machine Learning (Evaluación Diagnóstica):** Análisis de capacidad discriminativa calculando curvas ROC (AUC), Precision-Recall y el Índice de Youden.
   - **Regresión Multivariada:** Modelos de regresión logística robustos (compensando errores estándar por clúster) y ponderados para tratar el desbalance de clases, comparando predictores de área, puntuales y confusores (Modelos A, B, C, D, E).
4. **Visualización y Reportes Automáticos (`generar_visualizaciones_*.py`, `generar_reporte_*.py`)**
   - Generación programática de figuras de publicación científica (alta resolución, gráficos Forest Plot, curvas ROC y Boxplots).
   - Compilación automatizada de los hallazgos en reportes ejecutivos en formato PDF estructurados bajo el estándar científico IMRaD (Introducción, Métodos, Resultados y Discusión).

## ⚙️ Requisitos e Instalación

**Dependencias Principales:**
- Python 3.8+
- `pandas`, `numpy`, `scipy` (Manipulación de datos y estadística base)
- `scikit-learn` (Métricas ROC/PR, Machine Learning)
- `statsmodels` (Regresión logística multivariante robusta por clústeres)
- `matplotlib` (Visualización estática 2D y generación del pipeline de reportes PDF)

**Instrucciones de Instalación:**
```bash
# Clonar el repositorio
git clone <URL-del-repositorio>
cd Estudio_Termografia_y_Trastornos_Temporomandibulares

# Crear y activar un entorno virtual aislado (recomendado)
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias científicas de Python
pip install pandas numpy scipy scikit-learn statsmodels matplotlib
```

## 🚀 Uso (Comandos paso a paso)

Para ejecutar el pipeline en su orden de dependencia funcional:

**1. Procesamiento Inicial y Fusión de Datos:**
```bash
python herramientas/proceso_de_datos.py
python herramientas/unificar_dolor_termografia.py
```
*(Procesa los datos crudos del directorio `datos/` y genera el dataset analítico maestro `datos_finales_termografia_procesados_todas_fotos.csv`).*

**2. Ejecución del Análisis Estadístico Principal:**
```bash
python herramientas/analisis_termografia.py
```
*(Evalúa el indicador de asimetría térmica crudo ΔT. Produce múltiples tablas de contingencia, correlación CSVs e imágenes temporales en la carpeta `resultados/`).*

**3. Ejecución del Análisis Complementario:**
```bash
python herramientas/analisis_termografia_complementario.py
```
*(Evalúa el ΔT normalizado ajustando variables como sexo y edad. Produce reportes paralelos en la carpeta `resultado_complementario/`).*

**4. Generación (o regeneración) de Gráficos:**
```bash
python herramientas/generar_visualizaciones_principales.py
python herramientas/generar_visualizaciones_alterno.py
```

**5. Compilación y Construcción de Reportes PDF:**
```bash
python herramientas/generar_reporte_analisis_imagenes_termograficas.py
python herramientas/generar_reporte_analisis_complementario_imagenes_termograficas.py
python herramientas/generar_reporte_interpretacion.py
```
*(Transpila todos los análisis, tablas y gráficos estáticos para convertirlos en los informes PDF consolidados en sus respectivas carpetas).*

## 📁 Estructura del Proyecto

```text
.
├── datos/                                 # Datos brutos, diccionarios de variables y datos preprocesados
│   ├── datos_finales_termografia_procesados_todas_fotos.csv  # Base maestra consolidada final
│   └── respuestas_formulario_extendido.csv                   # Resultados tabulados de clínica del dolor
├── herramientas/                          # Código fuente del motor analítico (Scripts Python)
│   ├── analisis_termografia.py            # Motor principal estadístico (ΔT)
│   ├── analisis_termografia_complementario.py # Modelos extendidos (edad, sexo, temperatura basal)
│   ├── proceso_de_datos.py                # Pipeline de extracción (ETL)
│   ├── unificar_dolor_termografia.py      # Script de cruce relacional entre dolor y regiones térmicas
│   └── generar_reporte_*.py               # Rutinas para maquetación y generación de PDF's
├── resultados/                            # Salidas y métricas del Análisis Principal (CSVs, PNGs y PDF)
├── resultado_complementario/              # Salidas del Análisis Complementario
├── informacion_general.md                 # Detalles metodológicos, reglas de imputación y control de calidad
├── objetivos_del_estudio_con_hipotesis_estadisticas.md # Definición formal de las hipótesis clínicas (H0/H1)
└── README.md                              # Este documento
```

## 📊 Outputs / Resultados Esperados

Al finalizar el pipeline, el sistema poblara automáticamente los directorios `resultados/` y `resultado_complementario/` con los siguientes artefactos:
- **Métricas Diagnósticas (CSVs):** Documentos como `resultados_roc.csv` o `resultados_logit_cluster.csv` exponiendo métricas de OR (Odds Ratio), AUC y p-valores.
- **Correlaciones de Variables (CSVs):** Detalle exhaustivo por test estadístico (Spearman, concordancia de Kappa) como `correlaciones_globales.csv`.
- **Artefactos Visuales:** Imágenes `.png` listas para su publicación (300 DPI) que incluyen histogramas, forest plots de regresiones logísticas, barplots de AUC y scatter plots cruzados.
- **Dossier Ejecutivo PDF:** Tres archivos consolidados de hasta ~16 páginas con formato IMRaD, tablas formateadas, advertencias metodológicas, visualizaciones embebidas e insight de conclusiones médicas.

## ⚠️ Limitaciones Conocidas y Notas Metodológicas

* **Desbalance Severo de Clases (Skewness):** Solo el ~4.7% de los 2,295 puntos evaluados en la muestra poblacional exhibieron dolor clínico comprobable. Si bien se utilizaron ponderaciones algorítmicas robustas (`class_weight='balanced'`), esto impone restricciones prácticas al rendimiento del modelo (limitando su AUC global máximo en un rango empírico < 0.70).
* **Manejo de Valores Nulos (NaN):** **No se imputan datos térmicos.** Para evitar introducir temperaturas artificiales falsas, cualquier valor térmico faltante desencadena la exclusión automática (`.dropna()`) del registro para el análisis de dicho punto. Por su parte, los registros faltantes en la intensidad de dolor se imputaron conservadoramente como `0`.
* **Clusters Estadísticos y Dependencia:** Las observaciones (múltiples puntos orofaciales) están agrupadas dentro del mismo paciente, lo cual viola el principio de independencia estadística directa. El código corrige esto empleando estimadores "sandwich" mediante el argumento `cov_type='cluster'` utilizando el ID del paciente como agrupador.
* **Sesgo Basal Descubierto:** Se documentó un hallazgo lateral de asimetría térmica poblacional biológica esperada sistemática con predominancia al hemisferio derecho facial (+0.46°C a +0.75°C). Cualquier análisis de calor predictivo futuro del codebase debería tener esto en cuenta y usar desviaciones corregidas.