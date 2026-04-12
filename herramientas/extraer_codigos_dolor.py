"""
Extracción y expansión de códigos de dolor desde respuestas de formulario.

Objetivo
--------
Convertir celdas con formato semiestructurado (por ejemplo:
    "8 (R3P2) (R4P3P4)")
en columnas analíticas explícitas con la forma:
    <NOMBRE_NORMALIZADO>_R#P#

Ejemplo:
    "ATM ANTERIOR DERECHA [Dolor]" + "8 (R3P2)"
    -> columna "ATM_ANTERIOR_DERECHA_R3P2" con valor 8.

Salida principal
----------------
- respuestas_formulario_extendido.csv
  Incluye columnas demográficas clave + columnas específicas extraídas.
"""

import re
import unicodedata

import pandas as pd

# -------------------------------------------------------------------
# Configuración de entrada/salida
# -------------------------------------------------------------------
INPUT_PATH = 'datos/respuestas_de_formulario.csv'
OUTPUT_PATH = 'datos/respuestas_formulario_extendido.csv'


def normalizar_nombre_columna(nombre):
    """
    Normaliza un nombre de columna para usarlo como prefijo analítico.

    Reglas:
    - Elimina etiquetas entre corchetes: [Dolor], [No], etc.
    - Quita acentos y caracteres especiales.
    - Convierte a MAYÚSCULAS y reemplaza separadores por '_'.

    Ejemplo
    -------
    'ATM ANTERIOR DERECHA [Dolor]' -> 'ATM_ANTERIOR_DERECHA'
    """
    # 1) Eliminar segmentos entre corchetes
    res = re.sub(r'\[.*?\]', '', nombre)

    # 2) Quitar acentos mediante normalización Unicode
    res = unicodedata.normalize('NFKD', res).encode('ASCII', 'ignore').decode('ASCII')

    # 3) Estandarizar a A-Z0-9_ para nombres de columnas robustos
    res = re.sub(r'[^A-Z0-9]+', '_', res.strip().upper())

    # 4) Limpiar '_' residuales en extremos
    return res.strip('_')


def separar_codigos_compuestos(codigo):
    """
    Divide códigos compuestos de punto en códigos unitarios.

    Ejemplo
    -------
    'R3P5P6' -> ['R3P5', 'R3P6']

    Si no es compuesto, retorna una lista con el código original.
    """
    codigo = codigo.strip().upper()

    # Detectar patrón compuesto: R#P#P#...
    if re.match(r'R\d+P\d+P\d+', codigo):
        # Extraer región base (R#)
        region_match = re.match(r'(R\d+)', codigo)
        if region_match:
            region = region_match.group(1)
            # Extraer cada P#
            puntos = re.findall(r'P(\d+)', codigo)
            return [f"{region}P{p}" for p in puntos]
    return [codigo]


def procesar_celda(valor, nombre_col):
    """
    Extrae pares (columna_final, valor) a partir de una celda del formulario.

    Formato esperado de celda:
    - Valor base + uno o más códigos entre paréntesis.
      Ejemplo: '8 (R3P2) (R4P3P4)'

    Retorna
    -------
    list[tuple[str, str]]
        Pares (nombre_columna_expandida, valor_numérico_base como string).
    """
    if pd.isna(valor) or not isinstance(valor, str):
        return []

    # 1) Extraer valor base al inicio (0-10 típicamente)
    val_match = re.search(r'^(\d+)', valor.strip())
    if not val_match:
        return []

    val_numerico = val_match.group(1)

    # 2) Extraer todos los bloques (...) de códigos
    bloques_codigos = re.findall(r'\(([^)]+)\)', valor)

    resultados = []
    prefix = normalizar_nombre_columna(nombre_col)

    for bloque in bloques_codigos:
        # 3) Expandir códigos compuestos (R4P3P4 -> R4P3, R4P4)
        codigos_unitarios = separar_codigos_compuestos(bloque)
        for cod in codigos_unitarios:
            col_final = f"{prefix}_{cod}"
            resultados.append((col_final, val_numerico))

    return resultados


def extraer_filas(df):
    """
    Aplica procesar_celda a cada celda del DataFrame y devuelve una lista
    de dicts con las columnas expandidas por fila.
    """
    nuevos_datos = []
    for _, row in df.iterrows():
        fila_extraida = {}
        for col in df.columns:
            pares_extraidos = procesar_celda(row[col], col)
            for col_final, val in pares_extraidos:
                # Consolidación por fila:
                # si la misma columna expandida aparece varias veces, conservar el valor máximo.
                if col_final in fila_extraida:
                    try:
                        prev_v = float(fila_extraida[col_final])
                        new_v = float(val)
                        fila_extraida[col_final] = str(max(prev_v, new_v))
                    except (ValueError, TypeError):
                        fila_extraida[col_final] = val
                else:
                    fila_extraida[col_final] = val
        nuevos_datos.append(fila_extraida)
    return nuevos_datos


def main():
    # 1) Carga de datos crudos del formulario
    try:
        df = pd.read_csv(INPUT_PATH)
        print(f"Archivo cargado: '{INPUT_PATH}'. Filas: {len(df)}, Columnas: {len(df.columns)}")
    except Exception as e:
        print(f"Error al cargar el archivo: {e}")
        return

    # 2) Extracción sistemática fila a fila
    nuevos_datos = extraer_filas(df)

    # 3) Crear DataFrame expandido y unir con datos originales
    df_extraido = pd.DataFrame(nuevos_datos)

    if not df_extraido.empty:
        # Orden alfabético para inspección y trazabilidad.
        df_extraido = df_extraido.reindex(sorted(df_extraido.columns), axis=1)

        print(f"Se detectaron {len(df_extraido.columns)} nuevas columnas específicas.")

        # Unión horizontal: datos originales + variables expandidas.
        df_final = pd.concat([df, df_extraido], axis=1)

        # 4) Filtrado y renombramiento final para dataset de análisis
        print("\nIniciando filtrado y renombramiento final...")

        # 4.1 Renombrar temperatura basal a etiqueta estándar del proyecto
        col_temp_original = 'Temperatura corporal (en grados celsius)'
        if col_temp_original in df_final.columns:
            df_final = df_final.rename(columns={col_temp_original: 'temperatura basal'})
            print(f"Columna '{col_temp_original}' renombrada a 'temperatura basal'.")
        else:
            # Puede no existir según versión del formulario.
            print(f"Advertencia: No se encontró la columna '{col_temp_original}'.")

        # 4.2 Columnas demográficas base a conservar en la salida
        cols_demo_base = [
            'Marca temporal', 'Fecha', 'Nombre', 'Apellido',
            'Edad', 'Sexo', 'Cédula ', 'temperatura basal'
        ]

        # Compatibilidad con formularios donde 'numero de muestra' llega como 'Column 67'
        if 'Column 67' in df_final.columns:
            df_final = df_final.rename(columns={'Column 67': 'numero de muestra'})
            print("Columna 'Column 67' renombrada a 'numero de muestra'.")

        # Lista final demográfica (siempre intenta incluir numero de muestra primero)
        cols_demo = ['numero de muestra'] + cols_demo_base

        # Respetar solo columnas realmente presentes
        cols_demo_final = [c for c in cols_demo if c in df_final.columns]

        # 4.3 Columnas derivadas de códigos de dolor
        cols_nuevas = list(df_extraido.columns)

        # 4.4 Esquema final de salida
        cols_a_mantener = cols_demo_final + cols_nuevas

        df_filtrado = df_final[cols_a_mantener]
        print(f"Filtrado completado: {len(df_filtrado.columns)} columnas conservadas.")

        # 5) Guardar resultado final para integración con termografía
        df_filtrado.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
        print(f"\nProceso completado. Archivo final guardado como '{OUTPUT_PATH}'.")

        # 6) Verificación rápida de una fila de control
        print("\nVerificación final de la Fila 11 (Índice 10) en archivo filtrado:")
        print(df_filtrado.loc[10, ['Nombre', 'Apellido', 'numero de muestra', 'temperatura basal']].dropna())
    else:
        print("No se encontraron códigos con el formato esperado en el archivo.")


if __name__ == "__main__":
    main()
