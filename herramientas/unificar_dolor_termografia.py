"""
Unificación de datos termográficos con columnas de dolor extraídas del formulario.

Este script integra en un único dataset consolidado las variables termográficas procesadas 
(provenientes de las tomas primera y segunda) con las variables de dolor derivadas 
del formulario de pacientes. La unión se realiza mediante la cédula de identidad 
normalizada para asegurar la integridad de los datos.

Propósito
---------
1. Consolidar variables termográficas (primeras y segundas fotos) en un solo archivo.
2. Integrar variables de dolor específicas (mapeo anatómico rNpN).
3. Realizar un mapeo automático de temperatura puntual para cada punto de dolor reportado,
   utilizando exclusivamente los datos de la SEGUNDA toma (sufijo '_segunda').
4. Estandarizar tipos de datos e imputar valores nulos en escalas de dolor (0-10).

Flujo de Trabajo
----------------
1. Carga de 'datos/datos_termografia_procesados_todas_fotos.csv' y 'respuestas_formulario_extendido.csv'.
2. Normalización de cédulas y nombres de columnas.
3. Unión (Left Join) preservando todos los registros termográficos.
4. Mapeo de temperatura puntual: por cada columna de dolor (ej: masetero_derecha_r2p1), 
   se busca su temperatura correspondiente (ej: p1 derecha_segunda).
5. Imputación de nulos: las escalas de dolor se llevan a entero (0-10) con 0 para nulos, 
   mientras que las temperaturas mantienen NaN.

Entrada
-------
- `datos_termografia_procesados_todas_fotos.csv`: Dataset base con columnas '_primera' y '_segunda'.
- `respuestas_formulario_extendido.csv`: Formulario con escalas de dolor por punto anatómico.

Salida
------
- `datos_finales_termografia_procesados_todas_fotos.csv`: Dataset final enriquecido.
"""

import pandas as pd
import re

from herramientas.constantes import PAIN_COL_REGEX


# -------------------------------------------------------------------
# Configuración
# -------------------------------------------------------------------
TERMO_PATH = 'datos/datos_termografia_procesados_todas_fotos.csv'
DOLOR_PATH = 'datos/respuestas_formulario_extendido.csv'
OUTPUT_PATH = 'datos/datos_finales_termografia_procesados_todas_fotos.csv'

# Sufijo que identifica las columnas de la segunda toma en el dataset fusionado.
# Las columnas '_primera' se conservan en el output pero no participan en el
# mapeo térmico ni en el tipado de puntos.
SUFIJO_SEGUNDA = '_segunda'

# Claves de unión (nombres tal como existen en cada archivo)
COL_CEDULA_TERMO = 'cedula'
COL_CEDULA_DOLOR = 'cédula'

# Columnas no-dolor del archivo extendido (se excluyen al integrar "códigos de dolor")
COLS_NO_DOLOR = [
    'numero de muestra', 'marca temporal', 'fecha', 'nombre', 'apellido',
    'edad', 'sexo', 'cédula', 'temperatura basal'
]


def normalizar_cedula(val):
    """
    Convierte una cédula a un string canónico para garantizar una unión exitosa.

    Aplica limpieza para manejar inconsistencias comunes como:
    - Valores nulos (NaN).
    - Cédulas leídas como números flotantes (ej. 12345678.0).
    - Espacios en blanco accidentales.

    Args:
        val: El valor original de la cédula (float, int o str).

    Returns:
        str: Cédula normalizada como cadena de texto sin decimales ni espacios.
    """
    if pd.isna(val):
        return ""
    try:
        return str(int(float(val))).strip()
    except Exception:
        return str(val).strip().split('.')[0]


def normalizar_nombres_columnas(df):
    """
    Estandariza los nombres de las columnas del DataFrame.

    Elimina espacios en blanco y convierte todo a minúsculas para evitar 
    errores de coincidencia causados por diferencias de formato.

    Args:
        df (pd.DataFrame): DataFrame original.

    Returns:
        pd.DataFrame: Copia del DataFrame con columnas normalizadas.
    """
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def cargar_archivos():
    """
    Carga los archivos CSV de termografía y dolor desde el sistema de archivos.

    Realiza una normalización inmediata de los nombres de las columnas tras la carga.

    Returns:
        tuple: (df_termo, df_dolor) como DataFrames de pandas.

    Raises:
        FileNotFoundError: Si alguno de los archivos configurados no existe.
    """
    df_termo = pd.read_csv(TERMO_PATH)
    df_dolor = pd.read_csv(DOLOR_PATH)
    df_termo = normalizar_nombres_columnas(df_termo)
    df_dolor = normalizar_nombres_columnas(df_dolor)
    print(f"Archivos cargados. Termo: {len(df_termo)} registros, Dolor: {len(df_dolor)} registros.")
    return df_termo, df_dolor


def seleccionar_columnas_dolor(df_dolor):
    """
    Identifica y selecciona las columnas que contienen datos de dolor.

    Excluye las columnas demográficas y administrativas definidas en COLS_NO_DOLOR.

    Args:
        df_dolor (pd.DataFrame): DataFrame del formulario extendido.

    Returns:
        list: Lista de nombres de columnas que representan puntos de dolor.
    """
    return [c for c in df_dolor.columns if c not in COLS_NO_DOLOR]


def es_columna_dolor_anatomico(col):
    """
    Verifica si una columna sigue el patrón de nomenclatura de dolor anatómico.

    Utiliza una expresión regular para validar el formato {musculo}_{lado}_rNpN.

    Args:
        col (str): Nombre de la columna a validar.

    Returns:
        bool: True si coincide con el patrón anatómico definido.
    """
    if not isinstance(col, str):
        return False
    return bool(PAIN_COL_REGEX.match(col.strip().lower()))


def unificar_datos(df_termo, df_dolor):
    """
    Realiza la integración de los dos conjuntos de datos mediante un Left Join.

    Este proceso asegura que se mantengan todos los registros de termografía, 
    añadiendo la información de dolor correspondiente donde esté disponible. 
    Maneja la limpieza de columnas previas para permitir ejecuciones repetidas (idempotencia).

    Args:
        df_termo (pd.DataFrame): Datos termográficos procesados.
        df_dolor (pd.DataFrame): Respuestas del formulario de dolor.

    Returns:
        tuple: (df_final, cols_dolor_especificas) con el resultado y la lista de columnas unidas.
    """
    # 1) Normalizar la columna clave (cédula) en ambos orígenes para asegurar el matching
    df_termo[COL_CEDULA_TERMO] = df_termo[COL_CEDULA_TERMO].apply(normalizar_cedula)
    df_dolor[COL_CEDULA_DOLOR] = df_dolor[COL_CEDULA_DOLOR].apply(normalizar_cedula)

    print(f"Cédula normalizada muestra 1 (Termo): {df_termo[COL_CEDULA_TERMO].iloc[0]}")
    print(f"Cédula normalizada muestra 1 (Dolor): {df_dolor[COL_CEDULA_DOLOR].iloc[0]}")

    # 2) Determinar columnas específicas de dolor
    cols_dolor_especificas = seleccionar_columnas_dolor(df_dolor)
    print(f"Se unirán {len(cols_dolor_especificas)} columnas de dolor específicas.")
    df_dolor_subset = df_dolor[[COL_CEDULA_DOLOR] + cols_dolor_especificas]

    # 3) Limpieza preventiva para evitar sufijos _x/_y en ejecuciones repetidas
    cols_temp_mapeadas_previas = [c for c in df_termo.columns if str(c).startswith('temp_punto__')]
    cols_a_remover = [c for c in cols_dolor_especificas if c in df_termo.columns] + cols_temp_mapeadas_previas
    df_termo_limpio = df_termo.drop(columns=cols_a_remover)

    # 4) Unión principal
    df_final = pd.merge(
        df_termo_limpio,
        df_dolor_subset,
        left_on=COL_CEDULA_TERMO,
        right_on=COL_CEDULA_DOLOR,
        how='left'
    )

    # Eliminar columna duplicada de cédula proveniente del dataframe de dolor
    if COL_CEDULA_DOLOR in df_final.columns and COL_CEDULA_TERMO in df_final.columns:
        df_final = df_final.drop(columns=[COL_CEDULA_DOLOR])

    return df_final, cols_dolor_especificas


def columna_temp_punto_desde_col_dolor(col_dolor, sufijo=SUFIJO_SEGUNDA):
    """
    Determina la columna de temperatura (pN lado{sufijo}) asociada a un punto de dolor.

    Esta función establece el puente entre la evaluación subjetiva (dolor) y la
    medición objetiva (termografía). Por defecto utiliza la SEGUNDA toma,
    basándose en el protocolo del estudio que asocia el reporte de dolor del
    formulario con el estado térmico de la segunda medición.

    Correspondencias del estudio (Region/Punto):
    - Temporal (r1): p1, p2, p3
    - Esternocleidomastoideo (r2): p1 a p6
    - ATM (r3): p1, p2, p3, p4, p6
    - Masetero (r2): p1 a p4
    - Masetero (r3): p1 a p6
    - Masetero (r4): p1 a p5

    Args:
        col_dolor (str): Nombre de la columna de dolor (ej. 'masetero_derecha_r3p2').
        sufijo (str): Sufijo de toma a usar (por defecto '_segunda').

    Returns:
        str or None: Nombre de la columna térmica (ej. 'p2 derecha_segunda') o None si no hay mapeo.
    """
    if not isinstance(col_dolor, str):
        return None

    # 1) Extraer la región (rN) y el punto (pN) usando expresiones regulares
    m = re.search(r'(r\d+)p(\d+)', col_dolor)
    if not m:
        return None
    region = m.group(1)
    punto = f"p{m.group(2)}"

    # 2) Identificar el músculo y la lateralidad (derecha/izquierda)
    musculo = col_dolor.split('_')[0]
    lado_match = re.search(r'derech[oa]|izquierd[oa]', col_dolor)
    if not lado_match:
        return None
    lado_raw = lado_match.group(0)
    lado = 'derecha' if lado_raw.startswith('derech') else 'izquierda'

    # 3) Definir el conjunto de correspondencias válidas según el protocolo del estudio
    correspondencias = {
        ('temporal', 'r1', 'p1'), ('temporal', 'r1', 'p2'), ('temporal', 'r1', 'p3'),
        ('esternocleidomastoideo', 'r2', 'p1'), ('esternocleidomastoideo', 'r2', 'p2'),
        ('esternocleidomastoideo', 'r2', 'p3'), ('esternocleidomastoideo', 'r2', 'p4'),
        ('esternocleidomastoideo', 'r2', 'p5'), ('esternocleidomastoideo', 'r2', 'p6'),
        ('atm', 'r3', 'p1'), ('atm', 'r3', 'p2'), ('atm', 'r3', 'p3'),
        ('atm', 'r3', 'p4'), ('atm', 'r3', 'p6'),
        ('masetero', 'r2', 'p1'), ('masetero', 'r2', 'p2'), ('masetero', 'r2', 'p3'), ('masetero', 'r2', 'p4'),
        ('masetero', 'r3', 'p1'), ('masetero', 'r3', 'p2'), ('masetero', 'r3', 'p3'),
        ('masetero', 'r3', 'p4'), ('masetero', 'r3', 'p5'), ('masetero', 'r3', 'p6'),
        ('masetero', 'r4', 'p1'), ('masetero', 'r4', 'p2'), ('masetero', 'r4', 'p3'),
        ('masetero', 'r4', 'p4'), ('masetero', 'r4', 'p5'),
    }

    # 4) Verificar si la combinación existe y retornar el nombre de la columna térmica
    if (musculo, region, punto) not in correspondencias:
        return None
    return f"{punto} {lado}{sufijo}"


def agregar_temperatura_puntual_mapeada(df, sufijo=SUFIJO_SEGUNDA):
    """
    Crea nuevas columnas de temperatura vinculadas directamente a cada punto de dolor.

    Para cada columna de dolor válida encontrada, genera una columna hermana
    prefijada con 'temp_punto__' que contiene el valor de temperatura del
    punto anatómico correspondiente.

    Args:
        df (pd.DataFrame): DataFrame unificado.
        sufijo (str): Sufijo de toma a usar para el mapeo térmico (por defecto '_segunda').

    Returns:
        pd.DataFrame: DataFrame con las nuevas columnas mapeadas.
    """
    df = df.copy()
    cols_dolor = [c for c in df.columns if es_columna_dolor_anatomico(c)]

    creadas = 0
    for col_dolor in cols_dolor:
        col_temp = columna_temp_punto_desde_col_dolor(col_dolor, sufijo=sufijo)
        if not col_temp or col_temp not in df.columns:
            continue
        nueva = f"temp_punto__{col_dolor}"
        df[nueva] = pd.to_numeric(df[col_temp], errors='coerce')
        creadas += 1

    print(f"Columnas de temperatura puntual mapeada creadas: {creadas}")
    return df


def imputar_y_tipar_columnas_dolor(df, sufijo=SUFIJO_SEGUNDA):
    """
    Aplica reglas de integridad de datos y tipado para las variables de dolor y térmicas.

    Reglas aplicadas:
    1. Temperaturas (p1..p7{sufijo}): Se aseguran como numéricas. Los valores faltantes 
       se mantienen como NaN para no sesgar promedios o análisis térmicos.
    2. Escalas de Dolor (desde atm_... hasta el final): Se asume que si no hay reporte, 
       el dolor es 0. Se convierten a entero (int64) para facilitar modelos estadísticos.

    Args:
        df (pd.DataFrame): DataFrame unificado.
        sufijo (str): Sufijo de las columnas térmicas a procesar.

    Returns:
        pd.DataFrame: DataFrame con tipos e imputaciones corregidas.
    """
    df = df.copy()

    # 1) p1..p7 derecha/izquierda{sufijo} son TEMPERATURA:
    # mantener faltantes como NaN (no imputar 0).
    cols_puntos = (
        [f"p{i} derecha{sufijo}" for i in range(1, 8)] +
        [f"p{i} izquierda{sufijo}" for i in range(1, 8)]
    )
    cols_puntos_presentes = [c for c in cols_puntos if c in df.columns]
    if cols_puntos_presentes:
        df[cols_puntos_presentes] = (
            df[cols_puntos_presentes]
            .apply(pd.to_numeric, errors='coerce')
        )

    # 2) Columnas de dolor anatómico -> escala 0..10 (int, nulos a 0)
    # Se identifican por contenido (regex anatómico), no por posición en el DataFrame,
    # evitando que un reordenamiento de columnas aplique el tipado a variables incorrectas.
    cols_dolor_solo = [
        c for c in df.columns
        if es_columna_dolor_anatomico(c) and not str(c).startswith('temp_punto__')
    ]
    if cols_dolor_solo:
        df[cols_dolor_solo] = (
            df[cols_dolor_solo]
            .apply(pd.to_numeric, errors='coerce')
            .fillna(0)
            .astype('int64')
        )

    return df


def procesar_unificacion(termo_path, output_path, sufijo_mapeo):
    """
    Orquestador de unificación para un dataset específico (primeras o segundas fotos).
    """
    print(f"\n--- Procesando Unificación: {output_path} (Sufijo: {sufijo_mapeo}) ---")
    try:
        df_termo = pd.read_csv(termo_path)
        df_dolor = pd.read_csv(DOLOR_PATH)
        df_termo = normalizar_nombres_columnas(df_termo)
        df_dolor = normalizar_nombres_columnas(df_dolor)
    except Exception as e:
        print(f"Error al cargar archivos para {output_path}: {e}")
        return

    # 1. Unificar
    df_unificado, cols_dolor = unificar_datos(df_termo, df_dolor)

    # 2. Tipar dolor (0 si NaN)
    df_unificado = imputar_y_tipar_columnas_dolor(df_unificado, sufijo=sufijo_mapeo)

    # 3. Mapear temperaturas puntuales (sufijo pasado explícitamente, sin mutar global)
    df_unificado = agregar_temperatura_puntual_mapeada(df_unificado, sufijo=sufijo_mapeo)

    # 4. Guardar
    df_unificado.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Resultado guardado en: {output_path}")


def main():
    """
    Punto de entrada principal para el proceso de unificación.

    Genera datasets finales para:
    1. Primeras fotos (pre-palpación)
    2. Segundas fotos (post-palpación)
    3. Dataset combinado (con ambas tomas)
    """
    # A) Primeras Fotos
    procesar_unificacion(
        'datos/datos_termografia_procesados_primeras_fotos.csv',
        'datos/datos_finales_termografia_procesados_primeras_fotos.csv',
        sufijo_mapeo='' # No tienen sufijo en su propio archivo
    )

    # B) Segundas Fotos
    procesar_unificacion(
        'datos/datos_termografia_procesados_segundas_fotos.csv',
        'datos/datos_finales_termografia_procesados_segundas_fotos.csv',
        sufijo_mapeo='' # No tienen sufijo en su propio archivo
    )

    # C) Todas las Fotos (Merged)
    procesar_unificacion(
        'datos/datos_termografia_procesados_todas_fotos.csv',
        'datos/datos_finales_termografia_procesados_todas_fotos.csv',
        sufijo_mapeo='_segunda' # Mapeo por defecto a la segunda toma como antes
    )


if __name__ == "__main__":
    main()
