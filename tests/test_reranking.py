"""Pruebas del cliente de re-ranking con respuestas simuladas (no llaman a AWS)."""

from datetime import date

import boto3
import pytest
from botocore.stub import Stubber

from markwell_rag import bedrock
from markwell_rag.bedrock import ModeloNoDisponibleError
from markwell_rag.repositorio import Resultado
from markwell_rag.reranking import ClienteReranking, texto_para_reranking

MODELO = "cohere.rerank-v3-5:0"


def resultado(articulo: str, texto: str) -> Resultado:
    return Resultado(
        fragmento_id=1,
        entidad="SUGESE",
        codigo="SUGESE 03-10",
        titulo="Reglamento de prueba",
        articulo=articulo,
        texto=texto,
        vigente_desde=date(2021, 10, 27),
        vigente_hasta=None,
        fuente_url="https://example.com",
        distancia=0.4,
    )


@pytest.fixture
def esperas(monkeypatch):
    registro: list[float] = []
    monkeypatch.setattr(bedrock.time, "sleep", registro.append)
    return registro


@pytest.fixture
def stubber():
    # Otro servicio: la API Rerank vive en bedrock-agent-runtime, no en bedrock-runtime.
    client = boto3.client(
        "bedrock-agent-runtime",
        region_name="us-east-1",
        aws_access_key_id="prueba",
        aws_secret_access_key="prueba",
    )
    with Stubber(client) as s:
        yield s
        s.assert_no_pending_responses()


def test_reordenar_envia_la_peticion_y_respeta_el_indice(stubber):
    candidatos = [
        resultado("Art. 3", "Definiciones."),
        resultado("Art. 9", "Requisitos del agente."),
        resultado("Art. 10", "Responsabilidad."),
    ]
    stubber.add_response(
        "rerank",
        {
            "results": [
                {"index": 1, "relevanceScore": 0.91},
                {"index": 0, "relevanceScore": 0.12},
            ]
        },
        expected_params={
            "queries": [
                {"type": "TEXT", "textQuery": {"text": "¿Requisitos para ser agente?"}}
            ],
            "sources": [
                {
                    "type": "INLINE",
                    "inlineDocumentSource": {
                        "type": "TEXT",
                        "textDocument": {"text": texto_para_reranking(c)},
                    },
                }
                for c in candidatos
            ],
            "rerankingConfiguration": {
                "type": "BEDROCK_RERANKING_MODEL",
                "bedrockRerankingConfiguration": {
                    "numberOfResults": 2,
                    "modelConfiguration": {
                        "modelArn": f"arn:aws:bedrock:us-east-1::foundation-model/{MODELO}"
                    },
                },
            },
        },
    )

    reordenados = ClienteReranking(model_id=MODELO, client=stubber.client).reordenar(
        "¿Requisitos para ser agente?", candidatos, top_n=2
    )

    assert [r.resultado.articulo for r in reordenados] == ["Art. 9", "Art. 3"]
    assert reordenados[0].relevancia == 0.91
    assert (
        texto_para_reranking(candidatos[1])
        == "SUGESE 03-10 (Reglamento de prueba), Art. 9\nRequisitos del agente."
    )


def test_sin_candidatos_no_llama_a_bedrock(stubber):
    # El stubber no tiene respuestas: si el cliente llamara a rerank, la prueba fallaría.
    assert (
        ClienteReranking(model_id=MODELO, client=stubber.client).reordenar(
            "¿Algo?", [], top_n=5
        )
        == []
    )


def test_access_denied_en_reranking(stubber, esperas):
    stubber.add_client_error("rerank", "AccessDeniedException", "Sin acceso", 403)

    with pytest.raises(ModeloNoDisponibleError):
        ClienteReranking(model_id=MODELO, client=stubber.client).reordenar(
            "¿Algo?", [resultado("Art. 1", "x")], top_n=1
        )

    assert esperas == []
