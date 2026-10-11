"""Extracción de texto de documentos regulatorios y división por artículo.

Este módulo no llama a AWS ni a la base de datos: recibe texto y devuelve
fragmentos. Así se puede probar con ejemplos escritos a mano.
"""

from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from docx import Document

# "Artículo 5.", "ARTÍCULO 5.-", "Articulo 5:", "Artículo 5º.-", "Artículo 5 bis."
# Solo cuenta al inicio del párrafo: "según el artículo 5" dentro de un texto no
# abre un artículo nuevo.
PATRON_ARTICULO = re.compile(
    r"^\s*art[íi]culo\s+(\d+)\s*[º°]?\s*(bis|ter)?\b", re.IGNORECASE
)
# "Transitorio I", "TRANSITORIO II.-", "Transitorio único"
PATRON_TRANSITORIO = re.compile(
    r"^\s*transitorio\s+(único|unico|[IVXLC]+|\d+)\b", re.IGNORECASE
)
# "CAPÍTULO II", "Sección III", "TÍTULO I": encabezados de estructura, no son
# contenido de ningún artículo.
PATRON_ESTRUCTURA = re.compile(
    r"^\s*(cap[íi]tulo|secci[óo]n|t[íi]tulo)\s+\S+", re.IGNORECASE
)

# Titan Text Embeddings V2 acepta hasta 8192 tokens (50 000 caracteres). Este
# umbral es más bajo a propósito: un fragmento muy largo mezcla temas y su vector
# se "diluye". Es un valor provisional que revisaremos con métricas en la Fase 3.
MAX_CARACTERES_RECOMENDADO = 8000


@dataclass
class Articulo:
    etiqueta: str  # "Art. 5", "Art. 5 bis", "Transitorio I"
    texto: str  # texto completo, incluido el encabezado del artículo


@dataclass
class Division:
    preambulo: list[str] = field(default_factory=list)
    articulos: list[Articulo] = field(default_factory=list)


def extraer_parrafos_docx(origen: Path | io.BytesIO) -> list[str]:
    """Devuelve los párrafos no vacíos del cuerpo del documento, en orden.

    Limitaciones conocidas: python-docx no incluye aquí el texto de las tablas
    ni la numeración automática de Word (listas numeradas).
    """
    documento = Document(origen if isinstance(origen, io.BytesIO) else str(origen))
    parrafos = (p.text.strip() for p in documento.paragraphs)
    return [p for p in parrafos if p]


def parrafos_del_cuerpo(ruta: Path) -> tuple[str, list[str]]:
    """Elige la parte del DOCX que contiene el articulado y devuelve (nombre, párrafos).

    Algunos DOCX oficiales (como los de SUGESE) son un contenedor: la página
    principal solo tiene íconos, y los considerandos, el reglamento y el control
    de cambios vienen como documentos Word incrustados en word/embeddings/.
    Se elige la parte con más encabezados "Artículo N".
    """
    candidatos = [("documento principal", extraer_parrafos_docx(ruta))]
    with zipfile.ZipFile(ruta) as paquete:
        for nombre in sorted(paquete.namelist()):
            if nombre.startswith("word/embeddings/") and nombre.endswith(".docx"):
                datos = io.BytesIO(paquete.read(nombre))
                candidatos.append((nombre, extraer_parrafos_docx(datos)))

    def cantidad_de_articulos(candidato: tuple[str, list[str]]) -> int:
        return sum(1 for p in candidato[1] if PATRON_ARTICULO.match(p))

    return max(candidatos, key=cantidad_de_articulos)


def etiqueta_de(parrafo: str) -> str | None:
    """Si el párrafo abre un artículo o un transitorio, devuelve su etiqueta."""
    if m := PATRON_ARTICULO.match(parrafo):
        numero, sufijo = m.group(1), m.group(2)
        return f"Art. {numero}" + (f" {sufijo.lower()}" if sufijo else "")
    if m := PATRON_TRANSITORIO.match(parrafo):
        valor = m.group(1)
        return (
            "Transitorio único"
            if valor.lower() in ("único", "unico")
            else f"Transitorio {valor.upper()}"
        )
    return None


def dividir_por_articulo(parrafos: list[str]) -> Division:
    """Agrupa los párrafos en artículos.

    - Lo que aparece antes del primer artículo (considerandos, acuerdo de
      aprobación) queda en `preambulo` y no se carga.
    - Los encabezados de título, capítulo o sección, y todo lo que los sigue
      hasta el próximo artículo (su nombre, p. ej. "Disposiciones generales"),
      se descartan para no pegarse al final del artículo anterior.
    """
    division = Division()
    actual: list[str] | None = None
    etiqueta_actual = ""
    en_encabezado = False

    def cerrar() -> None:
        if actual:
            division.articulos.append(Articulo(etiqueta_actual, "\n".join(actual)))

    for parrafo in parrafos:
        if PATRON_ESTRUCTURA.match(parrafo):
            en_encabezado = True
            continue

        if etiqueta := etiqueta_de(parrafo):
            cerrar()
            etiqueta_actual, actual = etiqueta, [parrafo]
            en_encabezado = False
        elif en_encabezado:
            continue  # nombre del título o capítulo
        elif actual is None:
            division.preambulo.append(parrafo)
        else:
            actual.append(parrafo)

    cerrar()
    return division


def validar(articulos: list[Articulo]) -> list[str]:
    """Devuelve avisos sobre la división: no corrige nada, solo informa."""
    avisos: list[str] = []
    vistos: set[str] = set()
    for a in articulos:
        if a.etiqueta in vistos:
            avisos.append(f"Etiqueta repetida: {a.etiqueta}")
        vistos.add(a.etiqueta)
        if len(a.texto) > MAX_CARACTERES_RECOMENDADO:
            avisos.append(
                f"{a.etiqueta} tiene {len(a.texto)} caracteres (> {MAX_CARACTERES_RECOMENDADO})"
            )

    # Los artículos principales deberían ir 1, 2, 3... sin saltos.
    numeros = [
        int(a.etiqueta.split()[1])
        for a in articulos
        if a.etiqueta.startswith("Art. ") and len(a.etiqueta.split()) == 2
    ]
    for anterior, siguiente in zip(numeros, numeros[1:]):
        if siguiente != anterior + 1:
            avisos.append(
                f"Salto en la numeración: Art. {anterior} seguido de Art. {siguiente}"
            )
    return avisos
