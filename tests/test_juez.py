"""Pruebas del juez de fidelidad con un evaluador falso (sin AWS)."""

from datetime import date

import pytest
from pydantic import ValidationError

from markwell_rag.juez import (
    EvaluacionJuez,
    FragmentoJuzgado,
    armar_consulta_juez,
    juzgar,
    resumir_concordancia,
)

FRAGMENTO = FragmentoJuzgado("SUGESE 06-13", "Art. 7", date(2020, 9, 1), "Artículo 7. Plazo de resolución.\nTreinta días naturales.")


class EvaluadorFalso:
    model_id = "falso"

    def __init__(self, datos):
        self.datos = datos
        self.consultas = []

    def consultar_estructurado(self, pregunta, esquema):
        self.consultas.append(pregunta)
        return esquema.model_validate(self.datos), None


def test_juzgar_envia_fragmentos_pregunta_y_respuesta():
    evaluador = EvaluadorFalso({
        "afirmaciones": [{"texto": "El plazo es de 30 días naturales", "respaldo": "sí", "fragmento": "SUGESE 06-13 Art. 7"}],
        "contradiccion": False, "veredicto": "fiel",
        "justificacion": "El Art. 7 fija el plazo.",
    })

    evaluacion, _ = juzgar(evaluador, "¿Plazo para resolver?", "Treinta días naturales.", [FRAGMENTO])

    assert evaluacion.veredicto == "fiel"
    consulta = evaluador.consultas[0]
    assert '<fragmento norma="SUGESE 06-13" articulo="Art. 7" vigente_desde="2020-09-01">' in consulta
    assert consulta.index("<fragmentos>") < consulta.index("<pregunta>") < consulta.index("<respuesta>")


def test_evaluacion_rechaza_un_veredicto_incoherente():
    with pytest.raises(ValidationError, match="veredicto 'fiel'"):
        EvaluacionJuez.model_validate({
            "afirmaciones": [{"texto": "Rige el inciso n", "respaldo": "no"}],
            "contradiccion": False, "veredicto": "fiel",
            "justificacion": "-",
        })
    # Todo respaldado, pero contradictorio: "parcial" es válido (P26).
    assert EvaluacionJuez.model_validate({
        "afirmaciones": [{"texto": "Treinta días", "respaldo": "sí"}],
        "contradiccion": True,
        "veredicto": "parcial",
        "justificacion": "-",
    }).veredicto == "parcial"
    with pytest.raises(ValidationError, match="no es 'fiel'"):
        EvaluacionJuez.model_validate({
            "afirmaciones": [{"texto": "Treinta días", "respaldo": "sí"}],
            "contradiccion": False, "veredicto": "parcial",
            "justificacion": "-",
        })


def test_resumir_concordancia():
    pares = [("fiel", "fiel"), ("parcial", "fiel"), ("no fiel", "no fiel"), ("parcial", "parcial")]

    resumen = resumir_concordancia(pares)

    assert resumen["Coincidencia exacta"] == 0.75
    assert resumen["No fiel detectados"] == 1.0
    assert resumen["Fiel cuando no lo era"] == pytest.approx(1 / 3)
    assert resumir_concordancia([("fiel", "fiel")])["No fiel detectados"] is None
