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
# "ANEXO I", "Anexo 2", "Anexo único": sin esto, el anexo queda pegado al
# último artículo o transitorio.
PATRON_ANEXO = re.compile(r"^\s*anexo\s+(único|unico|[IVXLC]+|\d+)\b", re.IGNORECASE)
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
    """Si el párrafo abre un artículo, un transitorio o un anexo, devuelve su etiqueta."""
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
    if m := PATRON_ANEXO.match(parrafo):
        valor = m.group(1)
        return "Anexo único" if valor.lower() in ("único", "unico") else f"Anexo {valor.upper()}"
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


# "DEROGADO: Mediante artículo 11...", "Derogado por...", "(Así derogado...)".
# También aparece cuando solo se derogó un inciso: por eso se usa para informar,
# no para excluir. Decidir si el artículo completo está derogado requiere leerlo.
PATRON_DEROGACION = re.compile(r"\bderogad[oa]s?\b", re.IGNORECASE)


def menciones_de_derogacion(articulos: list[Articulo]) -> list[str]:
    """Etiquetas de los artículos cuyo texto menciona una derogación."""
    return [a.etiqueta for a in articulos if PATRON_DEROGACION.search(a.texto)]


def excluir_derogados(articulos: list[Articulo], derogados: set[str] | frozenset[str]) -> list[Articulo]:
    """Quita los artículos que una persona confirmó como derogados leyendo el texto.

    El texto consolidado conserva el número y el contenido de un artículo
    derogado, con una nota al final. Si se carga, la búsqueda lo devuelve como
    vigente. Lanza ValueError si alguna etiqueta no existe en el documento, para
    que un error de escritura en el catálogo no deje un derogado cargado sin aviso.
    """
    faltan = set(derogados) - {a.etiqueta for a in articulos}
    if faltan:
        raise ValueError(f"No existen en el documento: {sorted(faltan)}")
    return [a for a in articulos if a.etiqueta not in derogados]

# Reemplaza los párrafos derogados dentro de un artículo vigente. Dice "derogado"
# a propósito: si alguien pregunta por esa figura, el modelo ve que existió y ya
# no rige, en lugar de un hueco en la numeración de los incisos.
MARCA_PARRAFO_DEROGADO = "[Párrafo derogado: su texto no se carga]"


def quitar_parrafos_derogados(
    articulos: list[Articulo], parrafos: dict[str, frozenset[int]]
) -> list[Articulo]:
    """Reemplaza por una marca los párrafos que una persona confirmó como derogados.

    `parrafos` asocia la etiqueta de un artículo vigente con los números de sus
    párrafos derogados, contando desde 1 (el párrafo 1 es el encabezado del
    artículo y no se puede quitar). Los párrafos seguidos se reemplazan por una
    sola marca. Lanza ValueError si una etiqueta no existe, si un número está
    fuera del artículo o si ninguno de los párrafos quitados menciona una
    derogación: así un número mal escrito no borra texto vigente sin aviso.
    """
    faltan = set(parrafos) - {a.etiqueta for a in articulos}
    if faltan:
        raise ValueError(f"No existen en el documento: {sorted(faltan)}")

    resultado = []
    for a in articulos:
        quitar = parrafos.get(a.etiqueta)
        if not quitar:
            resultado.append(a)
            continue
        lineas = a.texto.split("\n")
        fuera = sorted(n for n in quitar if not 2 <= n <= len(lineas))
        if fuera:
            raise ValueError(f"{a.etiqueta} tiene {len(lineas)} párrafos; no se pueden quitar {fuera}")
        if not any(PATRON_DEROGACION.search(lineas[n - 1]) for n in quitar):
            raise ValueError(f"{a.etiqueta}: ninguno de los párrafos {sorted(quitar)} menciona una derogación")
        nuevas: list[str] = []
        for numero, linea in enumerate(lineas, start=1):
            if numero not in quitar:
                nuevas.append(linea)
            elif not nuevas or nuevas[-1] != MARCA_PARRAFO_DEROGADO:
                nuevas.append(MARCA_PARRAFO_DEROGADO)
        resultado.append(Articulo(a.etiqueta, "\n".join(nuevas)))
    return resultado
