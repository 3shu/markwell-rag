"""Preguntas de evaluación y métricas de recuperación (hit rate y MRR).

Este módulo no llama a AWS ni a la base de datos: recibe resultados de búsqueda
y calcula métricas. Así se puede probar con ejemplos escritos a mano.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from markwell_rag.rag import clave_articulo
from markwell_rag.repositorio import Resultado


@dataclass(frozen=True)
class ArticuloEsperado:
    norma: str     # código exacto, como está en la base: "SUGESE 06-13"
    articulo: str  # "Art. 7", "Artículo 7" o "Transitorio I": se normaliza al comparar


@dataclass
class PreguntaEval:
    id: str
    pregunta: str
    fecha: date
    esperados: list[ArticuloEsperado] = field(default_factory=list)
    nota: str = ""

    @property
    def respondible(self) -> bool:
        """Sin artículos esperados, la normativa cargada no responde la pregunta."""
        return bool(self.esperados)


def cargar_preguntas(ruta: Path) -> list[PreguntaEval]:
    """Lee un archivo JSON Lines: una pregunta por línea; las líneas vacías se ignoran."""
    preguntas: list[PreguntaEval] = []
    ids: set[str] = set()
    for numero, linea in enumerate(ruta.read_text(encoding="utf-8").splitlines(), start=1):
        if not linea.strip():
            continue
        try:
            datos = json.loads(linea)
            pregunta = PreguntaEval(
                id=datos["id"],
                pregunta=datos["pregunta"],
                fecha=date.fromisoformat(datos["fecha"]),
                esperados=[ArticuloEsperado(**e) for e in datos.get("esperados", [])],
                nota=datos.get("nota", ""),
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            # El número de línea ahorra buscar el error a ojo en un archivo largo.
            raise ValueError(f"{ruta}, línea {numero}: {e}") from e
        if pregunta.id in ids:
            raise ValueError(f"{ruta}, línea {numero}: id repetido {pregunta.id}")
        ids.add(pregunta.id)
        preguntas.append(pregunta)
    return preguntas


def posicion_del_primer_acierto(esperados: list[ArticuloEsperado], recuperados: list[Resultado]) -> int | None:
    """Posición (desde 1) del primer fragmento recuperado que es uno de los esperados.

    Coinciden la norma (exacta) y el artículo (normalizado): el mismo número de
    artículo en otra norma no cuenta. None si ninguno aparece.
    """
    claves = {(e.norma, clave_articulo(e.articulo)) for e in esperados}
    for posicion, r in enumerate(recuperados, start=1):
        if (r.codigo, clave_articulo(r.articulo)) in claves:
            return posicion
    return None


def hit_rate(posiciones: list[int | None], k: int) -> float:
    """Proporción de preguntas con al menos un acierto entre los primeros k."""
    if not posiciones:
        raise ValueError("No hay preguntas para medir")
    return sum(1 for p in posiciones if p is not None and p <= k) / len(posiciones)


def mrr(posiciones: list[int | None]) -> float:
    """Mean Reciprocal Rank: promedio de 1/posición; una pregunta sin acierto suma 0."""
    if not posiciones:
        raise ValueError("No hay preguntas para medir")
    return sum(1 / p for p in posiciones if p is not None) / len(posiciones)
