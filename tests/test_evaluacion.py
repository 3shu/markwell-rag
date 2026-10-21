"""Pruebas de la carga de preguntas y de las métricas de recuperación (sin AWS ni base de datos)."""

from datetime import date

import pytest

from markwell_rag.evaluacion import (
    ArticuloEsperado,
    cargar_preguntas,
    hit_rate,
    mrr,
    posicion_del_primer_acierto,
)
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
