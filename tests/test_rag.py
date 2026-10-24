"""Pruebas unitarias del flujo RAG con objetos falsos (sin AWS ni base de datos)."""

from datetime import date

import pytest

from markwell_rag.embeddings import Embedding
from markwell_rag.esquemas import Fuente, RespuestaRegulatoria
from markwell_rag.llm import Respuesta
from markwell_rag.rag import (
    AVISO_LEGAL,
    Asistente,
    clave_articulo,
    formatear_contexto,
    texto_final,
    verificar_citas,
)
from markwell_rag.repositorio import Resultado
from markwell_rag.reranking import Reordenado


def resultado(
    articulo: str, texto: str = "Texto del artículo.", hasta: date | None = None
) -> Resultado:
    return Resultado(
        fragmento_id=1,
        entidad="SUGESE",
        codigo="SUGESE 06-13",
        titulo="Reglamento de prueba",
        articulo=articulo,
        texto=texto,
        vigente_desde=date(2020, 9, 1),
        vigente_hasta=hasta,
        fuente_url="https://example.com/reglamento",
        distancia=0.4,
    )


def sustentada(*fuentes: Fuente) -> RespuestaRegulatoria:
    return RespuestaRegulatoria(
        respuesta="El plazo es de diez días.", sustentada=True, fuentes=list(fuentes)
    )


class EmbeddingsFalso:
    model_id = "amazon.titan-embed-text-v2:0"

    def vectorizar(self, texto: str) -> Embedding:
        return Embedding(vector=[0.1, 0.2, 0.3], tokens=8)


class LLMFalso:
    def __init__(self, respuesta: RespuestaRegulatoria) -> None:
        self.respuesta = respuesta
        self.consultas: list[str] = []

    def consultar_estructurado(self, pregunta, esquema):
        self.consultas.append(pregunta)
        meta = Respuesta(
            texto="",
            modelo="modelo-falso",
            tokens_entrada=1000,
            tokens_salida=100,
            costo_usd=0.0015,
            latencia_ms=10,
            stop_reason="tool_use",
        )
        return self.respuesta, meta


def test_el_contexto_incluye_los_datos_para_citar():
    contexto = formatear_contexto(
        [resultado("Art. 7", "Artículo 7. Plazo de respuesta.")]
    )

    assert '<documento id="1">' in contexto
    assert "Norma: SUGESE 06-13" in contexto
    assert "Artículo: Art. 7" in contexto
    assert "Vigente desde: 2020-09-01" in contexto
    assert "Vigente hasta: sin fecha de fin" in contexto
    assert "Artículo 7. Plazo de respuesta." in contexto


def test_clave_articulo_acepta_variantes():
    assert (
        clave_articulo("Art. 25")
        == clave_articulo("Artículo 25")
        == clave_articulo("25")
        == "25"
    )
    assert clave_articulo("Art. 5 bis") == "5 bis"
    assert clave_articulo("Transitorio I") == "transitorio i"


def test_cita_verificada_toma_la_vigencia_de_la_base():
    # El modelo escribió mal la fecha; la base de datos manda.
    fuente = Fuente(
        entidad="SUGESE",
        norma="Acuerdo SUGESE 06-13",
        articulo="Artículo 7",
        vigente_desde=date(2013, 7, 31),
    )

    citas = verificar_citas(
        sustentada(fuente), [resultado("Art. 6"), resultado("Art. 7")]
    )

    assert citas[0].verificada
    assert citas[0].resultado.articulo == "Art. 7"
    assert citas[0].fuente.vigente_desde == date(2020, 9, 1)


def test_cita_que_no_estaba_en_el_contexto_no_se_verifica():
    fuente = Fuente(entidad="SUGESE", norma="SUGESE 06-13", articulo="Art. 25")

    citas = verificar_citas(sustentada(fuente), [resultado("Art. 7")])

    assert not citas[0].verificada


def test_sin_resultados_no_llama_al_modelo():
    llm = LLMFalso(
        sustentada(Fuente(entidad="SUGESE", norma="SUGESE 06-13", articulo="Art. 7"))
    )
    asistente = Asistente(
        EmbeddingsFalso(), llm, recuperar=lambda vector, modelo, fecha, k: []
    )

    r = asistente.responder("¿Plazo?", date(2010, 1, 1))

    assert llm.consultas == []
    assert r.respuesta.sustentada is False
    assert "2010-01-01" in r.respuesta.respuesta
    assert r.costo_usd == pytest.approx(8 * 0.02 / 1_000_000)


def test_responder_recupera_genera_y_verifica():
    llamadas = []

    def recuperar(vector, modelo, fecha, k):
        llamadas.append((modelo, fecha, k))
        return [
            resultado(
                "Art. 7", "Artículo 7. La entidad responderá en diez días hábiles."
            )
        ]

    llm = LLMFalso(
        sustentada(Fuente(entidad="SUGESE", norma="SUGESE 06-13", articulo="Art. 7"))
    )
    asistente = Asistente(EmbeddingsFalso(), llm, recuperar, k=3)

    r = asistente.responder("¿En qué plazo responden un reclamo?", date(2026, 10, 16))

    assert llamadas == [("amazon.titan-embed-text-v2:0", date(2026, 10, 16), 3)]
    consulta = llm.consultas[0]
    assert "Fecha de consulta: 2026-10-16" in consulta
    assert "diez días hábiles" in consulta
    assert consulta.rstrip().endswith("Pregunta: ¿En qué plazo responden un reclamo?")
    assert r.citas_no_verificadas == []
    assert r.costo_usd == pytest.approx(8 * 0.02 / 1_000_000 + 0.0015)


def test_texto_final_agrega_el_aviso_una_sola_vez():
    sin_aviso = RespuestaRegulatoria(respuesta="No hay sustento.", sustentada=False)
    con_aviso = RespuestaRegulatoria(
        respuesta=f"No hay sustento.\n{AVISO_LEGAL}", sustentada=False
    )

    assert texto_final(sin_aviso) == f"No hay sustento.\n\n{AVISO_LEGAL}"
    assert texto_final(con_aviso) == texto_final(sin_aviso)
class ReordenadorFalso:
    """Invierte el orden de los candidatos: así se nota si el asistente usó su orden."""

    model_id = "reranker-falso"

    def __init__(self) -> None:
        self.llamadas: list[tuple[str, list[str], int]] = []

    def reordenar(self, pregunta, candidatos, top_n):
        self.llamadas.append((pregunta, [c.articulo for c in candidatos], top_n))
        invertidos = list(reversed(candidatos))[:top_n]
        return [Reordenado(c, 1.0 - i / 10) for i, c in enumerate(invertidos)]

    def costo_estimado(self, documentos):
        return 0.002


def test_con_reranking_trae_candidatos_y_pasa_k_al_modelo():
    pedidos = []

    def recuperar(vector, modelo, fecha, k):
        pedidos.append(k)
        return [resultado(f"Art. {n}", f"Texto del artículo {n}.") for n in range(1, 5)]

    llm = LLMFalso(sustentada(Fuente(entidad="SUGESE", norma="SUGESE 06-13", articulo="Art. 4")))
    reordenador = ReordenadorFalso()
    asistente = Asistente(EmbeddingsFalso(), llm, recuperar, k=2, reordenador=reordenador, candidatos=4)

    r = asistente.responder("¿Plazo?", date(2026, 10, 23))

    assert pedidos == [4]  # la búsqueda trae los candidatos, no k
    assert reordenador.llamadas == [("¿Plazo?", ["Art. 1", "Art. 2", "Art. 3", "Art. 4"], 2)]
    assert [x.articulo for x in r.recuperados] == ["Art. 4", "Art. 3"]
    consulta = llm.consultas[0]
    assert "Texto del artículo 4." in consulta and "Texto del artículo 1." not in consulta
    assert r.citas_no_verificadas == []
    assert r.reranking.candidatos == 4
    assert r.reranking.relevancias == [1.0, 0.9]
    assert r.costo_usd == pytest.approx(8 * 0.02 / 1_000_000 + 0.0015 + 0.002)


def test_sin_candidatos_no_llama_al_reranker():
    reordenador = ReordenadorFalso()
    llm = LLMFalso(sustentada(Fuente(entidad="SUGESE", norma="SUGESE 06-13", articulo="Art. 7")))
    asistente = Asistente(EmbeddingsFalso(), llm, lambda vector, modelo, fecha, k: [], reordenador=reordenador)

    r = asistente.responder("¿Plazo?", date(2010, 1, 1))

    assert reordenador.llamadas == []
    assert llm.consultas == []
    assert r.reranking is None


def test_candidatos_menor_que_k_es_un_error():
    with pytest.raises(ValueError):
        Asistente(EmbeddingsFalso(), LLMFalso(None), lambda *a: [], k=5, reordenador=ReordenadorFalso(), candidatos=3)
