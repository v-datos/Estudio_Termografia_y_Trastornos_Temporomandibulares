# Diccionario de Datos: Termografía y Trastornos Temporomandibulares

Este documento describe las variables contenidas en el archivo `datos_finales_termografia_procesados_todas_fotos.csv`, utilizado para el análisis de la relación entre la temperatura facial y los trastornos temporomandibulares (TTM).

## 1. Estructura de las Variables

El archivo contiene datos demográficos, clínicos y termográficos. Muchas variables siguen un esquema de nomenclatura sistemático para facilitar el procesamiento automatizado.

### Prefijos y Sufijos Comunes:
- **`{Lado}`**: `derecha` o `izquierda` (o `derecho` e `izquierdo` según el contexto del músculo).
- **`{Momento}`**:
    - `_primera`: Datos obtenidos en la sesión de fotos inicial (basal).
    - `_segunda`: Datos obtenidos en la sesión de fotos final (seguimiento).
- **`ROI (R1-R4)`**: Regiones de Interés anatómicas.
- **`Punto (P1-P7)`**: Ubicaciones térmicas específicas dentro de las regiones.

---

## 2. Definiciones de Regiones de Interés (ROI)

| Sigla | Región Anatómica Asociada |
| :--- | :--- |
| **R1** | Región Temporal (Músculo Temporal Anterior) |
| **R2** | Región Maseterina Superior / Esternocleidomastoideo (ECM) |
| **R3** | Articulación Temporomandibular (ATM) / Región Maseterina Media |
| **R4** | Región Maseterina Inferior / Ángulo Mandibular |

---

## 3. Categorías de Variables

### 3.1. Datos Demográficos y Clínicos Base
| Variable | Descripción | Valores / Tipo |
| :--- | :--- | :--- |
| `numero de muestra` | Identificador único del paciente en el estudio. | Numérico |
| `nombre` | Nombre completo del paciente. | Texto |
| `cedula` | Documento de identidad. | Numérico |
| `sexo` | Género biológico del participante. | Femenino / Masculino |
| `ocupacion` | Actividad laboral o académica principal. | Texto |
| `edad` | Edad del participante en años. | Numérico |
| `diagnosticado con ttm` | Indica si el paciente fue diagnosticado con Trastorno Temporomandibular. | Si / No |

### 3.2. Condiciones Ambientales y Basales
| Variable | Descripción |
| :--- | :--- |
| `temperatura ambiente_{momento}` | Temperatura de la sala donde se realizó la termografía (°C). |
| `temperatura basal_{momento}` | Temperatura corporal del paciente (°C). |

### 3.3. Termografía Regional (ROI)
Variables con el formato: `{ROI}: temperatura {métrica} {lado}_{momento}`
- **Métricas**: `maxima`, `minima`, `media`.
- **Ejemplo**: `r3: temperatura maxima derecha_primera`.
- **Nota**: `imagen: temperatura ...` se refiere a la temperatura capturada en el plano general de la termografía facial.

### 3.4. Diferenciales Térmicos (Asimetría)
Variables que miden la diferencia absoluta de temperatura entre el lado derecho e izquierdo.
- **`delta_t_{ROI}_{momento}`**: Diferencia de temperatura regional (Ej: `delta_t_r1_primera`).
- **`delta_t_global_media_{momento}`**: Promedio de las asimetrías de todas las regiones en ese momento.

### 3.5. Temperaturas Puntuales Específicas
Variables con el formato: `{Punto} {lado}_{momento}`
- Miden la temperatura en coordenadas fijas predefinidas (P1 a P7).
- **Ejemplo**: `p7 derecha_segunda`.

### 3.6. Registros de Dolor (Puntos de Palpación)
Variables con el formato: `{músculo}_{lado}_{localización}`
- Representan la intensidad del dolor reportada por el paciente en puntos específicos.
- **Músculos**: `atm_anterior`, `atm_posterior`, `esternocleidomastoideo`, `masetero`, `temporal`.
- **Localización**: Indica la región (R) y el punto (P) asociado (Ej: `masetero_derecho_r3p1`).
- **Valores**: 0 (sin dolor) o intensidad (1-10). Los valores `NaN` se interpretan como `0`.

### 3.7. Temperaturas Asociadas a Puntos de Dolor
Variables con el formato: `temp_punto__{músculo}_{lado}_{localización}`
- Contienen el valor térmico (°C) medido exactamente en la ubicación donde el paciente reportó dolor.
- **Ejemplo**: `temp_punto__temporal_derecho_r1p1`.

---

## 4. Tratamiento de Valores Faltantes (NaN)

Según el rigor metodológico del estudio:
- **Temperaturas**: Los valores faltantes (`NaN`) **no se imputan**. Si falta un dato de temperatura, ese registro se excluye del análisis estadístico específico para evitar sesgos.
- **Dolor**: Los valores faltantes (`NaN`) se asumen como `0` (ausencia de dolor), ya que el diseño del estudio solo registra los puntos con sensibilidad positiva.
