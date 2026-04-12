# Objetivos del Estudio e Hipótesis Estadísticas (Actualizado)


## A. Propósito del Análisis
Evaluar la relación entre la asimetría térmica regional pre-palpación (obtenida de las primeras imágenes termográficas) y la presencia e intensidad de dolor reportado durante el examen clínico de palpación muscular.

## B. Agrupación Principal
La variable de agrupación principal para el análisis es la **presencia de dolor** reportado por el propio paciente (Con Dolor vs. Sin Dolor) durante el examen clínico, independientemente del diagnóstico clínico inicial de TTM. Esto permite una correlación directa entre la fisiología térmica y la respuesta clínica inmediata.

## C. Objetivos

### 1. Objetivo General
Correlacionar la asimetría térmica regional (ΔT) pre-palpación con la intensidad y localización de los puntos de dolor reportados durante la palpación muscular en pacientes orofaciales.

### 2. Objetivos Específicos
a) **Identificación de Zonas**: Determinar las regiones de mayor asimetría térmica (ΔT) y los puntos anatómicos con mayor intensidad de dolor clínica.
b) **Correlación Estadística**: Evaluar la relación entre la asimetría térmica (ΔT) y la intensidad del dolor (0-10) mediante pruebas de correlación no paramétricas robustas.
c) **Utilidad Diagnóstica**: Cuantificar la capacidad discriminativa de la termografía (ΔT) para detectar la presencia de dolor mediante análisis de curvas ROC y Precision-Recall.
d) **Modelado Multivariable**: Estimar el efecto de la temperatura máxima regional sobre la probabilidad de dolor, ajustando por región y lado, mediante regresión logística robusta.
e) **Comparación de Modelos**: Evaluar si la temperatura puntual (píxel específico) ofrece una ventaja diagnóstica superior a la temperatura máxima regional.
f) **Análisis a Nivel de Paciente (N=45)**: Comparar la asimetría térmica máxima entre pacientes con y sin dolor para predecir la presencia de sintomatología global.

## D. Variables

### 1. Variable Térmica Principal (Predictora)
*   **Asimetría Térmica Regional (ΔT)**: Diferencia absoluta de temperatura entre el lado derecho e izquierdo (|T_der − T_izq|) en las regiones ROI (R1-R4) tomadas de las **PRIMERAS** imágenes (pre-palpación).
*   **Temperatura Máxima Regional**: Valor máximo en cada ROI.
*   **Temperatura Puntual (p1-p7)**: Temperatura mapeada exactamente en el píxel correspondiente al punto de palpación.

### 2. Variable de Respuesta (Clínica)
*   **Intensidad de Dolor**: Escala de 0 a 10 reportada por el paciente en puntos específicos (ATM, masetero, temporal, esternocleidomastoideo).
*   **Presencia de Dolor**: Variable binaria (1: intensidad > 0, 0: intensidad = 0).

## E. Hipótesis Estadísticas

### Hipótesis 1 — Correlación Asimetría ↔ Dolor (Obj. b)
*   **H₀**: No existe correlación significativa entre la asimetría térmica (ΔT) y la intensidad del dolor.
*   **H₁**: Existe una correlación significativa (positiva o negativa) entre ΔT y la intensidad del dolor.
*   **Prueba**: Coeficiente de Spearman con **prueba de permutación por bloques** (para respetar la dependencia intra-sujeto) y **corrección de Bonferroni** para comparaciones múltiples.

### Hipótesis 2 — Concordancia Espacial (Obj. a)
*   **H₀**: La región más caliente no coincide con la región más dolorosa.
*   **H₁**: Existe concordancia significativa entre la localización de la mayor asimetría térmica y la zona de mayor dolor reportado.
*   **Prueba**: Kappa de Cohen y porcentaje de concordancia directa.

### Hipótesis 3 — Capacidad Diagnóstica y Predictiva (Obj. c y d)
*   **H₀**: La asimetría térmica no predice la presencia de dolor (AUC ≈ 0.5) y sus coeficientes en el modelo logístico no son significativos.
*   **H₁**: La asimetría térmica permite discriminar entre puntos con y sin dolor (AUC > 0.5), y es posible establecer un **umbral térmico (T*)** óptimo mediante el Índice de Youden.
*   **Prueba**: Curvas ROC, AUC Precision-Recall y **Regresión Logística con errores estándar robustos por clúster (paciente)** y pesos balanceados.

### Hipótesis 4 — Diferencia Inter-grupal a Nivel de Paciente (Obj. f)
*   **H₀**: No hay diferencias en la asimetría térmica máxima entre el grupo Con Dolor y el grupo Sin Dolor.
*   **H₁**: Los pacientes que reportan dolor presentan asimetrías térmicas máximas significativamente mayores que aquellos sin dolor.
*   **Prueba**: Mann-Whitney U (bilateral) y análisis ROC a nivel de paciente (N=45).

---
*Nota: Este documento ha sido actualizado para reflejar la metodología implementada en `analisis_termografia.py`, asegurando la coherencia entre los objetivos del estudio y el procesamiento estadístico real.*
