"""Pruebas de la carga de preguntas y de las métricas de recuperación (sin AWS ni base de datos)."""

from datetime import date

import pytest

from markwell_rag.esquemas import Fuente, RespuestaRegulatoria
from markwell_rag.evaluacion import (
    ArticuloEsperado,
    PreguntaEval,
    cargar_preguntas,
    hit_rate,
    medir_fidelidad,
    mrr,
    posicion_del_primer_acierto,
    resumir_fidelidad,
)
from markwell_rag.rag import Cita, RespuestaRAG
from markwell_rag.repositorio import Resultado


def resultado(codigo: str, articulo: str) -> Resultado:
    return Resultado(
        fragmento_id=1, entidad="SUGESE", codigo=codigo, titulo="Reglamento de prueba",
        articulo=articulo, texto="Texto.", vigente_desde=date(2020, 9, 1), vigente_hasta=None,
        fuente_url="https://example.com", distancia=0.4,
    )


def test_cargar_preguntas_lee_jsonl(tmp_path):
    archivo = tmp_path / "preguntas.jsonl"
    archivo.write_text(
        '{"id": "P01", "pregunta": "¿Plazo?", "fecha": "2026-10-17", '
        '"esperados": [{"norma": "SUGESE 06-13", "articulo": "Art. 7"}]}\n'
        "\n"
        '{"id": "P02", "pregunta": "¿Capital mínimo?", "fecha": "2026-10-17", "esperados": [], '
        '"nota": "fuera de la normativa cargada"}\n',
        encoding="utf-8",
    )

    p1, p2 = cargar_preguntas(archivo)

    assert p1.fecha == date(2026, 10, 17)
    assert p1.esperados == [ArticuloEsperado("SUGESE 06-13", "Art. 7")]
    assert p1.respondible
    assert not p2.respondible
    assert p2.nota == "fuera de la normativa cargada"


def test_cargar_preguntas_indica_la_linea_del_error(tmp_path):
    archivo = tmp_path / "preguntas.jsonl"
    archivo.write_text(
        '{"id": "P01", "pregunta": "¿A?", "fecha": "2026-10-17"}\n'
        '{"id": "P01", "pregunta": "¿B?", "fecha": "2026-10-17"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="línea 2: id repetido P01"):
        cargar_preguntas(archivo)


def test_posicion_acepta_variantes_y_exige_la_norma():
    recuperados = [resultado("SUGESE 06-13", "Art. 7"), resultado("SUGESE 03-10", "Art. 7")]

    # El mismo número de artículo en otra norma no es un acierto.
    assert posicion_del_primer_acierto([ArticuloEsperado("SUGESE 03-10", "Artículo 7")], recuperados) == 2
    assert posicion_del_primer_acierto([ArticuloEsperado("SUGESE 03-10", "Art. 9")], recuperados) is None


def test_hit_rate_y_mrr():
    posiciones = [1, 3, None, 2]

    assert hit_rate(posiciones, 1) == 0.25
    assert hit_rate(posiciones, 3) == 0.75
    assert mrr(posiciones) == pytest.approx((1 + 1 / 3 + 0 + 1 / 2) / 4)


FECHA = date(2026, 10, 24)


def respuesta_rag(recuperados: list[Resultado], citas: list[tuple[str, str, Resultado | None]]) -> RespuestaRAG:
    """citas: (norma, artículo, fragmento recuperado o None si no estaba en el contexto)."""
    fuentes = [Fuente(entidad="SUGESE", norma=n, articulo=a) for n, a, _ in citas]
    respuesta = RespuestaRegulatoria(
        respuesta="Texto." if fuentes else "Los documentos no responden la pregunta.",
        sustentada=bool(fuentes),
        fuentes=fuentes,
    )
    return RespuestaRAG(
        "¿Pregunta?", FECHA, respuesta, recuperados,
        citas=[Cita(f, r) for f, (_, _, r) in zip(fuentes, citas)],
    )


def test_fidelidad_de_una_respuesta_correcta():
    art7 = resultado("SUGESE 06-13", "Art. 7")
    pregunta = PreguntaEval("P01", "¿Plazo?", FECHA, [ArticuloEsperado("SUGESE 06-13", "Art. 7")])

    m = medir_fidelidad(pregunta, respuesta_rag([resultado("SUGESE 06-13", "Art. 3"), art7], [("SUGESE 06-13", "Art. 7", art7)]))

    assert (m.sustentada, m.citas, m.citas_verificadas, m.cita_esperada, m.contexto_esperado) == (True, 1, 1, True, True)
    assert m.alertas == []


def test_fidelidad_detecta_cita_no_verificada_y_articulo_equivocado():
    art3 = resultado("SUGESE 06-13", "Art. 3")
    pregunta = PreguntaEval("P01", "¿Plazo?", FECHA, [ArticuloEsperado("SUGESE 06-13", "Art. 7")])
    r = respuesta_rag([art3, resultado("SUGESE 06-13", "Art. 7")], [("SUGESE 06-13", "Art. 3", art3), ("SUGESE 06-13", "Art. 25", None)])

    m = medir_fidelidad(pregunta, r)

    assert (m.citas, m.citas_verificadas, m.cita_esperada, m.contexto_esperado) == (2, 1, False, True)
    assert m.alertas == ["citas no verificadas", "no cita ningún artículo esperado"]


def test_fidelidad_de_controles_y_resumen():
    contexto = [resultado("SUGESE 06-13", "Art. 3")]
    control = PreguntaEval("P10", "¿Capital mínimo?", FECHA)
    abstencion = medir_fidelidad(control, respuesta_rag(contexto, []))
    invento = medir_fidelidad(control, respuesta_rag(contexto, [("SUGESE 06-13", "Art. 3", contexto[0])]))
    sin_contexto = medir_fidelidad(
        PreguntaEval("P01", "¿Plazo?", FECHA, [ArticuloEsperado("SUGESE 06-13", "Art. 7")]),
        respuesta_rag(contexto, []),
    )

    assert abstencion.alertas == []
    assert invento.alertas == ["control respondido como sustentado"]
    assert sin_contexto.alertas == ["sin sustento: el artículo esperado no llegó al contexto"]
    assert resumir_fidelidad([abstencion, invento, sin_contexto]) == {
        "Sustentadas (respondibles)": 0.0,
        "Citan un artículo esperado": 0.0,
        "Controles con sustentada=false": 0.5,
        "Citas verificadas": 1.0,
    }
    assert resumir_fidelidad([abstencion])["Citas verificadas"] is None  # sin citas no hay proporción
