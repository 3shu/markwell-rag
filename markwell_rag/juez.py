"""Juez de fidelidad: un modelo revisa si cada afirmación de una respuesta está en los fragmentos.

Las métricas de evaluacion.py prueban que una cita existe en el contexto, no que
respalde lo que dice la respuesta (sesión 13, P25). El juez lee la respuesta y
los fragmentos que recibió el asistente, y clasifica la respuesta como fiel,
parcial o no fiel. Antes de usarlo como métrica se compara con las etiquetas
manuales de eval/etiquetas_fidelidad.jsonl.
"""

from dataclasses import dataclass
from datetime import date
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field, model_validator

PROMPT_JUEZ = """Eres un evaluador de fidelidad de las respuestas de un asistente de normativa de seguros de Costa Rica.
Recibes los fragmentos de normativa que recibió el asistente, la pregunta y la respuesta del asistente.
No evalúas si la respuesta es útil ni si conoces otra norma: solo si lo que afirma está respaldado por los fragmentos. No uses conocimiento propio.

Procedimiento:
1. Divide la respuesta en afirmaciones verificables: plazos, obligaciones, sujetos, definiciones, montos y citas. Ignora el aviso legal y las frases de cortesía.
2. Para cada afirmación indica si un fragmento la respalda: "sí" (lo dice), "parcial" (lo dice solo en parte o con otro alcance) o "no" (no lo dice o dice lo contrario), y cuál fragmento.
3. Un texto que el propio fragmento marca como derogado no respalda ninguna afirmación, aunque la respuesta no diga "vigente". Solo está respaldada la afirmación que dice expresamente que ese texto fue derogado.
4. Si la respuesta dice que la normativa no regula algo, está respaldada cuando ningún fragmento lo regula.
5. Una afirmación atribuida a un documento, anexo o artículo que no está entre los fragmentos no está respaldada.
6. Indica si la respuesta se contradice: dos afirmaciones incompatibles entre sí. Una regla general y una regla específica que el mismo fragmento establece no son una contradicción.

Veredicto:
- "fiel": todas las afirmaciones tienen respaldo "sí" y la respuesta no se contradice.
- "parcial": la afirmación central está respaldada, pero alguna otra no o solo en parte, o la respuesta se contradice.
- "no fiel": la afirmación central no está respaldada o contradice los fragmentos."""

Respaldo = Literal["sí", "parcial", "no"]
Veredicto = Literal["fiel", "parcial", "no fiel"]


class Afirmacion(BaseModel):
    texto: str = Field(description="La afirmación, en una frase")
    respaldo: Respaldo
    fragmento: str | None = Field(
        default=None, description="Norma y artículo que la respalda, por ejemplo 'SUGESE 06-13 Art. 7'; null si ninguno"
    )


class EvaluacionJuez(BaseModel):
    # Las afirmaciones van antes del veredicto: el modelo llena el esquema en orden.
    afirmaciones: list[Afirmacion] = Field(min_length=1)
    contradiccion: bool = Field(description="true si la respuesta contiene dos afirmaciones incompatibles entre sí")
    veredicto: Veredicto
    justificacion: str = Field(description="Una o dos frases que explican el veredicto")

    @model_validator(mode="after")
    def veredicto_coherente(self) -> "EvaluacionJuez":
        todas_si = all(a.respaldo == "sí" for a in self.afirmaciones)
        if todas_si and not self.contradiccion and self.veredicto != "fiel":
            raise ValueError("todas las afirmaciones tienen respaldo 'sí' y no hay contradicción, pero el veredicto no es 'fiel'")
        if self.veredicto == "fiel" and (not todas_si or self.contradiccion):
            raise ValueError("veredicto 'fiel' con afirmaciones sin respaldo completo o con contradicción")
        return self


@dataclass
class FragmentoJuzgado:
    norma: str
    articulo: str
    vigente_desde: date
    texto: str


class Evaluador(Protocol):
    model_id: str

    def consultar_estructurado(self, pregunta: str, esquema: Any) -> tuple[Any, Any]: ...


def armar_consulta_juez(pregunta: str, respuesta: str, fragmentos: list[FragmentoJuzgado]) -> str:
    bloques = "\n".join(
        f'<fragmento norma="{f.norma}" articulo="{f.articulo}" vigente_desde="{f.vigente_desde.isoformat()}">\n'
        f"{f.texto}\n</fragmento>"
        for f in fragmentos
    )
    return (
        f"<fragmentos>\n{bloques}\n</fragmentos>\n\n"
        f"<pregunta>{pregunta}</pregunta>\n\n"
        f"<respuesta>{respuesta}</respuesta>\n\n"
        "Evalúa la fidelidad de la respuesta con el procedimiento indicado."
    )


def juzgar(evaluador: Evaluador, pregunta: str, respuesta: str, fragmentos: list[FragmentoJuzgado]):
    """Devuelve (EvaluacionJuez, metadatos de la llamada). SalidaInvalidaError si no cumple el esquema."""
    return evaluador.consultar_estructurado(armar_consulta_juez(pregunta, respuesta, fragmentos), EvaluacionJuez)


def resumir_concordancia(pares: list[tuple[str, str]]) -> dict[str, float | None]:
    """Compara (etiqueta manual, veredicto del juez). None cuando no hay casos."""
    no_fiel = [juez for manual, juez in pares if manual == "no fiel"]
    return {
        "Coincidencia exacta": sum(m == j for m, j in pares) / len(pares) if pares else None,
        "No fiel detectados": sum(j == "no fiel" for j in no_fiel) / len(no_fiel) if no_fiel else None,
        "Fiel cuando no lo era": (
            sum(j == "fiel" for m, j in pares if m != "fiel") / sum(m != "fiel" for m, _ in pares)
            if any(m != "fiel" for m, _ in pares) else None
        ),
    }
