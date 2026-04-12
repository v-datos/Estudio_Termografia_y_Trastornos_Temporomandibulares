"""
================================================================================
PIPELINE DE PROCESAMIENTO — TERMOGRAFÍA INFRARROJA
================================================================================
Propósito:
  1) Leer un archivo Excel con múltiples pestañas (una por paciente).
  2) Estandarizar nombres de variables clínicas y térmicas.
  3) Aplanar pares clave-valor a un DataFrame tabular único.
  4) Calcular métricas de asimetría térmica (Delta T por ROI y global).
  5) Exportar CSV y XLSX listos para análisis estadístico.

Guía rápida de uso:
  1) Asegurar que el archivo Excel de entrada esté en la raíz:
       'datos/DATOS DE LAS SEGUNDAS FOTOS TODOS.xlsx'
  2) Ejecutar:
       python proceso_de_datos.py
  3) Revisar salidas:
       - datos_termografia_procesados_segundas_fotos.csv
       - datos_termografia_procesados_segundas_fotos.xlsx

Lógica de procesamiento:
  - Normalización de texto: Elimina tildes, puntos, y estandariza 'derecha/izquierda'.
  - Detección de lateralidad: Propaga la lateralidad desde encabezados de sección.
  - Soporte de puntos térmicos: Procesa p1 a p7 (puntos específicos de interés).
  - Cálculo de Delta T: Abs(Media Derecha - Media Izquierda).
================================================================================
"""

import re

import pandas as pd
import numpy as np
from pathlib import Path
from difflib import get_close_matches

# Coincide con símbolos de unidad Celsius: ℃, °C, °, o una 'C' sola al final
# de un número (p. ej. "34.9C"). No elimina 'C' de texto arbitrario.
_CELSIUS_RE = re.compile(r'[℃°]\s*[Cc]?|(?<=[\d.])\s*[Cc]$')


def limpiar_valor_numerico(valor):
    """
    Intenta convertir un valor a float cuando representa temperatura/numérico.

    Estrategia:
    - Elimina símbolos de unidad Celsius (℃, °C, °, C tras dígito) con regex.
    - Elimina comillas y espacios residuales.
    - Cambia coma decimal por punto.
    - Si la conversión falla, retorna el valor *original* sin modificar.

    Parameters
    ----------
    valor : Any
        Valor original leído desde el Excel.

    Returns
    -------
    float | str | np.nan
        Float si es convertible, str original si no es numérico, NaN si falta.
    """
    if pd.isna(valor):
        return np.nan

    valor_original = str(valor)
    valor_str = _CELSIUS_RE.sub('', valor_original)
    valor_str = valor_str.replace('"', '').replace("'", '').replace(' ', '')
    valor_str = valor_str.replace(',', '.')

    try:
        return float(valor_str)
    except ValueError:
        return valor_original  # Retorna el texto original sin mutilar (ej. "Femenino")

def mostrar_nulos(df, columna):
    """
    Utilidad de auditoría para inspeccionar nulos en una columna puntual.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a inspeccionar.
    columna : str
        Nombre de la columna objetivo.

    Returns
    -------
    pd.DataFrame | None
        Subconjunto de filas con NaN en la columna; None si la columna no existe.
    """
    if columna not in df.columns:
        print(f"Error: La columna '{columna}' no existe en el DataFrame.")
        return None

    nulos = df[df[columna].isna()]

    if nulos.empty:
        print(f"No se encontraron valores nulos en la columna '{columna}'.")
    else:
        print(f"Se encontraron {len(nulos)} registros con valores nulos en '{columna}':")
        print(nulos)

    return nulos

def normalizar_clave(texto):
    """
    Normaliza etiquetas de variables para que claves equivalentes queden uniformes.

    Incluye:
    - Minúsculas y espacios normalizados.
    - Remoción de tildes.
    - Normalización de variaciones de max/min.
    - Armonización de género en lateridad (derecho -> derecha, etc.).
    - Corrección de typos frecuentes observados en el dataset.

    Esta normalización es crítica para poder combinar hojas heterogéneas
    en un único esquema de columnas.
    """
    if not isinstance(texto, str):
        return str(texto)

    # Diccionario de tildes
    tildes = {'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u', 'ü': 'u'}
    
    res = texto.lower()
    # Reemplazar múltiples espacios por uno solo
    res = ' '.join(res.split())
    
    # Normalizar variaciones de palabras clave (quitar puntos y tildes antes)
    res = res.replace('max.', 'maxima').replace('min.', 'minima')
    res = res.replace('máx', 'max').replace('mín', 'min')
    res = res.replace('maximo', 'max').replace('minimo', 'min')
    res = res.replace('máximo', 'max').replace('mínimo', 'min')
    
    for t, n in tildes.items():
        res = res.replace(t, n)
    
    # Normalizar variaciones de palabras clave
    res = res.replace('maxima', 'max').replace('máxima', 'max') # Primero a corto para luego a largo
    res = res.replace('minima', 'min').replace('mínima', 'min')
    
    # Ahora pasar todo a la forma larga estándar
    res = res.replace('max', 'maxima').replace('min', 'minima')
    
    # Manejar género
    res = res.replace('derecho', 'derecha').replace('izquierdo', 'izquierda')
    
    # Manejar errores de pegado de palabras (ej. máxderecha)
    res = res.replace('maximaderecha', 'maxima derecha').replace('minimaderecha', 'minima derecha')
    
    # Corregir typos específicos
    if 'temperratura' in res:
        res = res.replace('temperratura', 'temperatura')
    if 'izaquierda' in res:
        res = res.replace('izaquierda', 'izquierda')
    if 'izquierdad' in res:
        res = res.replace('izquierdad', 'izquierda')
    if 'izuierda' in res:
        res = res.replace('izuierda', 'izquierda')
    if 'derechad' in res:
        res = res.replace('derechad', 'derecha')
        
    # Limpiar basura incrustada en claves (ej. imagen: temperatura max.," 34,9℃")
    if ',"' in res:
        res = res.split(',"')[0].strip()
        
    # Normalizar "max." a "maxima" para consistencia con el resto del pipeline
    res = res.replace('max.', 'maxima').replace('min.', 'minima')
    res = res.replace('maximo', 'maxima').replace('minimo', 'minima')
    res = res.replace('mínimo', 'minima').replace('máximo', 'maxima')

    return res


def resolver_ruta_excel(nombre_archivo):
    """
    Resuelve y valida la ruta del Excel usando el directorio del script.

    Si el archivo no existe, lanza un error claro con sugerencias de nombres
    similares y el listado de archivos .xlsx disponibles.
    """
    ruta = Path(nombre_archivo)
    if ruta.exists():
        return ruta

    base_dir = Path(__file__).resolve().parent
    ruta_base = base_dir / nombre_archivo
    if ruta_base.exists():
        return ruta_base

    archivos_xlsx = sorted(p.name for p in base_dir.glob("*.xlsx"))
    sugerencias = get_close_matches(nombre_archivo, archivos_xlsx, n=3, cutoff=0.45)

    mensaje = [
        f"No se encontró el archivo Excel: '{nombre_archivo}'.",
        f"Directorio buscado: {base_dir}",
    ]
    if sugerencias:
        mensaje.append(f"Sugerencias: {', '.join(sugerencias)}")
    if archivos_xlsx:
        mensaje.append(f"Archivos .xlsx disponibles: {', '.join(archivos_xlsx)}")

    raise FileNotFoundError(" ".join(mensaje))


def es_columna_p_termica(columna):
    """
    Indica si una columna corresponde a puntos térmicos p1..p7 por lado.

    Estas columnas representan temperatura en puntos anatómicos específicos
    (p1 a p7) para los lados derecha e izquierda, y deben conservar NaN
    cuando falta el dato (nunca imputar 0 por defecto).

    Parameters
    ----------
    columna : Any
        Nombre de columna a evaluar.

    Returns
    -------
    bool
        True si la columna sigue el patrón 'pN derecha/izquierda' con N en 1..7.
    """
    if not isinstance(columna, str):
        return False
    partes = columna.strip().split()
    if len(partes) != 2:
        return False
    punto, lado = partes
    return punto in {'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7'} and lado in {'derecha', 'izquierda'}

def procesar_excel_termografia(ruta_archivo, incluir_puntos_termicos=True):
    """
    Procesa el Excel crudo y devuelve un DataFrame maestro listo para análisis.

    Flujo:
    1) Carga todas las pestañas.
    2) Aplana pares clave-valor por pestaña.
    3) Calcula delta térmico regional y global.
    4) Filtra columnas de interés y aplica tipos finales.

    Parameters
    ----------
    ruta_archivo : str
        Ruta al archivo Excel origen.
    incluir_puntos_termicos : bool, optional
        Si es True, procesa y extrae puntos p1 a p7 (puntos específicos).
        Por defecto True.

    Returns
    -------
    pd.DataFrame
        DataFrame tabular limpio, con métricas delta y columnas seleccionadas.
    """
    # Cargar todas las pestañas en un diccionario: {nombre_hoja: DataFrame}
    hojas_excel = pd.read_excel(ruta_archivo, sheet_name=None, header=None)

    datos_aplanados = []

    for nombre_hoja, df in hojas_excel.items():
        # Se asume formato mínimo de dos columnas: [clave, valor].
        # Si no cumple, la pestaña se omite.
        if df.shape[1] < 2:
            print(f"Advertencia: Saltando pestaña '{nombre_hoja}' porque no tiene el formato esperado (mínimo 2 columnas).")
            continue

        # Eliminamos filas completamente vacías
        df = df.dropna(how='all')

        muestra_dict = {
            'ID_Pestaña': nombre_hoja
        }

        # Iterar filas para extraer pares clave-valor.
        # seccion_actual permite heredar la lateralidad cuando no viene en la clave.
        seccion_actual = ""  # " derecha" o " izquierda"
        for index, row in df.iterrows():
            # Asegurarse de que tenemos acceso a la columna 0 y 1
            if len(row) < 2:
                continue

            clave = str(row[0]).strip()
            valor = row[1]

            clave_norm_orig = normalizar_clave(clave)
            
            # Detectar encabezados de sección que marcan lateralidad.
            if clave_norm_orig in ['derecha', 'derecho']:
                seccion_actual = " derecha"
                continue
            elif clave_norm_orig in ['izquierda', 'izquierdo']:
                seccion_actual = " izquierda"
                continue
            elif clave_norm_orig in ['resultados de medida de temperatura derecha']:
                seccion_actual = " derecha"
                continue
            elif clave_norm_orig in ['resultados de medida de temperatura izquierda']:
                seccion_actual = " izquierda"
                continue

            # Ignorar filas vacías y encabezados no informativos.
            if pd.isna(valor) or clave_norm_orig in ['resultados de medida de temperatura']:
                continue

            clave_normalizada = normalizar_clave(clave)
            
            # Si la clave no trae lado explícito, heredarlo desde la sección.
            if seccion_actual and "derecha" not in clave_normalizada and "izquierda" not in clave_normalizada:
                # Solo añadir a claves de temperatura/imagen/regiones/puntos de dolor
                palabras_clave = ['imagen', 'r1', 'r2', 'r3', 'r4', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'temperatura']
                if any(p in clave_normalizada for p in palabras_clave):
                    clave_normalizada += seccion_actual
            
            # ---------------------------------------------------------
            # Filtro opcional de puntos térmicos (p1..p7)
            # ---------------------------------------------------------
            # SIEMPRE incluirlos en el diccionario interno si existen en el Excel.
            # El filtrado se hará solo al final en la selección de columnas.
            # if not incluir_puntos_termicos and es_columna_p_termica(clave_normalizada):
            #    continue

            # Limpiar valor (numérico cuando aplica).
            valor_limpio = limpiar_valor_numerico(valor)
            muestra_dict[clave_normalizada] = valor_limpio

        datos_aplanados.append(muestra_dict)

    # Convertir lista de registros por pestaña en tabla maestra.
    df_maestro = pd.DataFrame(datos_aplanados)

    # ---------------------------------------------------------
    # PASO 1b: Ingeniería de características (Delta T)
    # ---------------------------------------------------------
    # Calculamos la asimetría térmica (Diferencia absoluta entre Derecha e Izquierda)
    # para las temperaturas MEDIAS de cada Región de Interés (R1, R2, R3, R4)

    regiones = ['r1', 'r2', 'r3', 'r4']

    for r in regiones:
        col_der = f'{r}: temperatura media derecha'
        col_izq = f'{r}: temperatura media izquierda'

        # Solo calcular si las columnas existen en el DataFrame extraído
        if col_der in df_maestro.columns and col_izq in df_maestro.columns:
            nombre_nueva_col = f'delta_t_{r}'
            
            # Asegurarse de que las columnas sean numéricas (forzando errores a NaN)
            df_maestro[col_der] = pd.to_numeric(df_maestro[col_der], errors='coerce')
            df_maestro[col_izq] = pd.to_numeric(df_maestro[col_izq], errors='coerce')

            # Diferencia absoluta: | Temp_Derecha - Temp_Izquierda |
            df_maestro[nombre_nueva_col] = abs(df_maestro[col_der] - df_maestro[col_izq])

            # Redondear a 1 decimal como es estándar en termografía
            df_maestro[nombre_nueva_col] = df_maestro[nombre_nueva_col].round(1)

    # Delta T global = promedio de deltas regionales disponibles por muestra.
    cols_delta = [f'delta_t_{r}' for r in regiones if f'delta_t_{r}' in df_maestro.columns]
    if cols_delta:
        df_maestro['delta_t_global_media'] = df_maestro[cols_delta].mean(axis=1).round(2)

    # ---------------------------------------------------------
    # PASO 1c: Selección de columnas requeridas para análisis downstream
    # ---------------------------------------------------------
    # Nota: Las columnas p1..p7 se incluyen solo si se detectaron en al menos una pestaña
    # del archivo Excel fuente. Si no aparecen en el Excel, no se añadirán al DataFrame.
    # Si incluir_puntos_termicos es False, se excluyen explícitamente.
    columnas_deseadas = [
        'numero de muestra', 'nombre', 'cedula', 'sexo', 'ocupacion', 'edad',
        'temperatura ambiente', 'temperatura basal',
        'diagnosticado con ttm',
        'imagen: temperatura maxima derecha', 'imagen: temperatura minima derecha', 'imagen: temperatura media derecha',
        'r2: temperatura maxima derecha', 'r2: temperatura minima derecha', 'r2: temperatura media derecha',
        'r3: temperatura maxima derecha', 'r3: temperatura minima derecha', 'r3: temperatura media derecha',
        'r4: temperatura maxima derecha', 'r4: temperatura minima derecha', 'r4: temperatura media derecha',
        'r1: temperatura maxima derecha', 'r1: temperatura minima derecha', 'r1: temperatura media derecha',
        'imagen: temperatura maxima izquierda', 'imagen: temperatura minima izquierda', 'imagen: temperatura media izquierda',
        'r2: temperatura maxima izquierda', 'r2: temperatura minima izquierda', 'r2: temperatura media izquierda',
        'r3: temperatura maxima izquierda', 'r3: temperatura minima izquierda', 'r3: temperatura media izquierda',
        'r4: temperatura maxima izquierda', 'r4: temperatura minima izquierda', 'r4: temperatura media izquierda',
        'r1: temperatura maxima izquierda', 'r1: temperatura minima izquierda', 'r1: temperatura media izquierda'
    ]

    if incluir_puntos_termicos:
        columnas_deseadas.extend([
            'p1 derecha', 'p2 derecha', 'p3 derecha', 'p4 derecha', 'p5 derecha', 'p6 derecha', 'p7 derecha',
            'p1 izquierda', 'p2 izquierda', 'p3 izquierda', 'p4 izquierda', 'p5 izquierda', 'p6 izquierda', 'p7 izquierda',
        ])

    # Añadir columnas de Delta T al set final.
    columnas_finales = [col for col in columnas_deseadas if col in df_maestro.columns]
    columnas_finales.extend(cols_delta)
    if 'delta_t_global_media' in df_maestro.columns:
        columnas_finales.append('delta_t_global_media')

    # Filtrar DataFrame final con esquema objetivo.
    df_maestro = df_maestro[columnas_finales]

    # p1..p6 son temperaturas puntuales:
    # convertir a numérico y mantener faltantes como NaN (no usar 0).
    for col in df_maestro.columns:
        if es_columna_p_termica(col):
            df_maestro[col] = pd.to_numeric(df_maestro[col], errors='coerce')

    # Tipado final: identificadores y edad como enteros.
    # Int64 de pandas permite nulos sin convertir la columna a float.
    for col_int in ['numero de muestra', 'cedula', 'edad']:
        if col_int in df_maestro.columns:
            df_maestro[col_int] = pd.to_numeric(df_maestro[col_int], errors='coerce').astype('Int64')



    return df_maestro


def main():
    """
    Punto de entrada del pipeline de procesamiento termográfico.

    Ejecuta tres pasos en secuencia:

    1. **Primeras fotos** — procesa 'datos/DATOS TERMICOS PRIMERAS FOTOS TODOS.xlsx'
       y exporta:
         - datos_termografia_procesados_primeras_fotos.csv
         - datos_termografia_procesados_primeras_fotos.xlsx

    2. **Segundas fotos** — procesa 'datos/DATOS DE LAS SEGUNDAS FOTOS TODOS.xlsx'
       y exporta:
         - datos_termografia_procesados_segundas_fotos.csv
         - datos_termografia_procesados_segundas_fotos.xlsx

    3. **Dataset combinado** — une ambos DataFrames con merge outer sobre los
       identificadores comunes (numero de muestra, cedula, nombre, sexo, edad,
       ocupacion, diagnosticado con ttm); las columnas clínicas y térmicas
       reciben sufijos '_primera' / '_segunda' para distinguir la toma:
         - datos_termografia_procesados_todas_fotos.csv
    """
    # 1. Procesar PRIMERAS fotos
    print("Procesando primeras fotos (puntos p1-p7 omitidos por protocolo)...")
    ruta_p = resolver_ruta_excel('datos/DATOS TERMICOS PRIMERAS FOTOS TODOS.xlsx')
    df_p = procesar_excel_termografia(ruta_p, incluir_puntos_termicos=False)
    df_p.to_csv('datos/datos_termografia_procesados_primeras_fotos.csv', index=False, encoding='utf-8-sig')
    df_p.to_excel('datos/datos_termografia_procesados_primeras_fotos.xlsx', index=False)

    # 2. Procesar SEGUNDAS fotos
    print("Procesando segundas fotos (puntos p1-p7 incluidos)...")
    ruta_s = resolver_ruta_excel('datos/DATOS DE LAS SEGUNDAS FOTOS TODOS.xlsx')
    df_s = procesar_excel_termografia(ruta_s, incluir_puntos_termicos=True)
    df_s.to_csv('datos/datos_termografia_procesados_segundas_fotos.csv', index=False, encoding='utf-8-sig')
    df_s.to_excel('datos/datos_termografia_procesados_segundas_fotos.xlsx', index=False)

    # 3. Crear MERGED dataset (Todas las fotos)
    # Agregamos sufijos para diferenciar columnas si es necesario, 
    # aunque normalmente se analizan por separado o apiladas.
    # En este caso, el usuario pide un "merged dataset".
    
    # Identificadores base para el merge
    id_cols = ['numero de muestra', 'cedula', 'nombre', 'sexo', 'edad', 'ocupacion', 'diagnosticado con ttm']
    id_cols_existentes = [c for c in id_cols if c in df_p.columns and c in df_s.columns]

    # Para asegurar que TODAS las columnas que no son ID tengan sufijo, incluso si no son comunes,
    # las renombramos manualmente antes del merge (pandas suffixes solo aplica a col. comunes).
    cols_a_renombrar_p = {c: f"{c}_primera" for c in df_p.columns if c not in id_cols_existentes}
    cols_a_renombrar_s = {c: f"{c}_segunda" for c in df_s.columns if c not in id_cols_existentes}
    
    df_p_renombrado = df_p.rename(columns=cols_a_renombrar_p)
    df_s_renombrado = df_s.rename(columns=cols_a_renombrar_s)

    print("Generando dataset combinado (merged)...")
    # Realizamos un merge exterior para no perder pacientes que solo estén en una toma
    df_merged = pd.merge(
        df_p_renombrado, 
        df_s_renombrado, 
        on=id_cols_existentes, 
        how='outer'
    )

    salida_todas = 'datos/datos_termografia_procesados_todas_fotos.csv'
    df_merged.to_csv(salida_todas, index=False, encoding='utf-8-sig')

    print("\nProcesamiento finalizado con éxito.")
    print(f"Archivos de primeras fotos: datos_termografia_procesados_primeras_fotos.csv/.xlsx")
    print(f"Archivos de segundas fotos: datos_termografia_procesados_segundas_fotos.csv/.xlsx")
    print(f"Archivo combinado: {salida_todas}")
    print(f"Total registros combinados: {len(df_merged)}")

    # Utilidad opcional:
    # mostrar_nulos(df_final, 'edad')


if __name__ == "__main__":
    main()
