"""Preguntas de evaluación y métricas de recuperación (hit rate y MRR).

Este módulo no llama a AWS ni a la base de datos: recibe resultados de búsqueda
y calcula métricas. Así se puede probar con ejemplos escritos a mano.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from markwell_rag.rag import RespuestaRAG, clave_articulo
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


@dataclass
class Fidelidad:
    """Lo que se puede medir en código de una respuesta del asistente, sin juez ni lectura humana."""

    id: str
    respondible: bool
    sustentada: bool          # lo que declaró el modelo
    citas: int
    citas_verificadas: int    # citas cuya norma y artículo estaban en el contexto
    cita_esperada: bool       # alguna cita verificada es uno de los artículos esperados
    contexto_esperado: bool   # algún artículo esperado llegó al contexto del modelo

    @property
    def alertas(self) -> list[str]:
        """Motivos para leer la respuesta a mano. Lista vacía: nada que revisar."""
        alertas = []
        if not self.respondible and self.sustentada:
            alertas.append("control respondido como sustentado")
        if self.citas_verificadas < self.citas:
            alertas.append("citas no verificadas")
        if self.respondible and not self.sustentada:
            alertas.append(
                "sin sustento con el artículo esperado en el contexto"
                if self.contexto_esperado
                else "sin sustento: el artículo esperado no llegó al contexto"
            )
        if self.respondible and self.sustentada and not self.cita_esperada:
            alertas.append("no cita ningún artículo esperado")
        return alertas


def medir_fidelidad(pregunta: PreguntaEval, r: RespuestaRAG) -> Fidelidad:
    verificadas = [c.resultado for c in r.citas if c.verificada]
    return Fidelidad(
        id=pregunta.id,
        respondible=pregunta.respondible,
        sustentada=r.respuesta.sustentada,
        citas=len(r.citas),
        citas_verificadas=len(verificadas),
        cita_esperada=posicion_del_primer_acierto(pregunta.esperados, verificadas) is not None,
        contexto_esperado=posicion_del_primer_acierto(pregunta.esperados, r.recuperados) is not None,
    )


def _proporcion(parte: int, total: int) -> float | None:
    return parte / total if total else None


def resumir_fidelidad(mediciones: list[Fidelidad]) -> dict[str, float | None]:
    """Métricas agregadas; None cuando no hay casos para calcularla."""
    respondibles = [m for m in mediciones if m.respondible]
    controles = [m for m in mediciones if not m.respondible]
    return {
        "Sustentadas (respondibles)": _proporcion(sum(m.sustentada for m in respondibles), len(respondibles)),
        "Citan un artículo esperado": _proporcion(sum(m.cita_esperada for m in respondibles), len(respondibles)),
        "Controles con sustentada=false": _proporcion(sum(not m.sustentada for m in controles), len(controles)),
        "Citas verificadas": _proporcion(
            sum(m.citas_verificadas for m in mediciones), sum(m.citas for m in mediciones)
        ),
    }
