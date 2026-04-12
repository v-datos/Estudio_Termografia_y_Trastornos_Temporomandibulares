"""
constantes.py — Constantes compartidas entre scripts del proyecto.

Centraliza definiciones que de otro modo se duplicarían en cada módulo,
eliminando el riesgo de divergencia cuando cambie la nomenclatura anatómica.
"""

import re

# Expresión regular que identifica columnas de dolor anatómico en el formato:
#   {musculo}_{lado}_r{N}p{N}
# Músculos reconocidos: atm (con variantes anterior/posterior),
#   esternocleidomastoideo, masetero, temporal.
# Lados: derecha/derecho, izquierda/izquierdo.
PAIN_COL_REGEX = re.compile(
    r'^(atm(?:_(?:anterior|posterior))?|esternocleidomastoideo|masetero|temporal)'
    r'_(derech[oa]|izquierd[oa])_r\d+p\d+$'
)
