"""Pruebas del cliente de embeddings con respuestas simuladas (no llaman a AWS)."""

import io
import json

import boto3
import pytest
from botocore.response import StreamingBody
from botocore.stub import Stubber

from markwell_rag import bedrock
from markwell_rag.bedrock import BedrockError, ModeloNoDisponibleError
from markwell_rag.embeddings import ClienteEmbeddings

MODELO = "amazon.titan-embed-text-v2:0"


def cuerpo_titan(vector: list[float], tokens: int = 5) -> StreamingBody:
    """Arma el cuerpo de respuesta con la misma forma que devuelve Titan V2."""
    datos = json.dumps({"embedding": vector, "inputTextTokenCount": tokens}).encode()
    return StreamingBody(io.BytesIO(datos), len(datos))


class ClienteFalso:
    """Registra cada petición para poder inspeccionar el cuerpo enviado."""

    def __init__(self) -> None:
        self.peticiones: list[dict] = []

    def invoke_model(self, **peticion):
        self.peticiones.append(peticion)
        n = len(self.peticiones)
        # Cada llamada devuelve un vector distinto para comprobar el orden.
        return {"body": cuerpo_titan([float(n), 0.0], tokens=n)}


# Mismas fixtures que en test_llm.py. Si aparece un tercer archivo de pruebas,
# las movemos a tests/conftest.py.
@pytest.fixture
def esperas(monkeypatch):
    registro: list[float] = []
    monkeypatch.setattr(bedrock.time, "sleep", registro.append)
    return registro


@pytest.fixture
def stubber(monkeypatch):
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    client = boto3.client(
        "bedrock-runtime",
        region_name="us-east-1",
        aws_access_key_id="prueba",
        aws_secret_access_key="prueba",
    )
    with Stubber(client) as s:
        yield s
        s.assert_no_pending_responses()


def test_vectorizar_envia_parametros_de_titan():
    falso = ClienteFalso()

    e = ClienteEmbeddings(model_id=MODELO, dimensiones=512, client=falso).vectorizar(
        "Plazo de pago"
    )

    peticion = falso.peticiones[0]
    assert peticion["modelId"] == MODELO
    assert json.loads(peticion["body"]) == {
        "inputText": "Plazo de pago",
        "dimensions": 512,
        "normalize": True,
    }
    assert e.vector == [1.0, 0.0]
    assert e.tokens == 1


def test_texto_vacio_no_llama_a_bedrock():
    falso = ClienteFalso()

    with pytest.raises(ValueError, match="vacío"):
        ClienteEmbeddings(model_id=MODELO, client=falso).vectorizar("   ")

    assert falso.peticiones == []


def test_dimensiones_invalidas_se_rechazan():
    with pytest.raises(ValueError, match="dimensiones"):
        ClienteEmbeddings(model_id=MODELO, dimensiones=768, client=ClienteFalso())


def test_lote_conserva_el_orden():
    falso = ClienteFalso()

    resultado = ClienteEmbeddings(model_id=MODELO, client=falso).vectorizar_lote(
        ["a", "b", "c"]
    )

    assert [e.vector[0] for e in resultado] == [1.0, 2.0, 3.0]
    assert [json.loads(p["body"])["inputText"] for p in falso.peticiones] == [
        "a",
        "b",
        "c",
    ]


def test_reintenta_throttling_en_embeddings(stubber, esperas):
    stubber.add_client_error(
        "invoke_model", "ThrottlingException", "Too many requests", 429
    )
    stubber.add_response(
        "invoke_model",
        {"body": cuerpo_titan([0.6, 0.8]), "contentType": "application/json"},
    )

    e = ClienteEmbeddings(model_id=MODELO, client=stubber.client).vectorizar("Hola")

    assert e.vector == [0.6, 0.8]
    assert len(esperas) == 1


def test_access_denied_en_embeddings(stubber, esperas):
    stubber.add_client_error("invoke_model", "AccessDeniedException", "Sin acceso", 403)

    with pytest.raises(ModeloNoDisponibleError):
        ClienteEmbeddings(model_id=MODELO, client=stubber.client).vectorizar("Hola")

    assert esperas == []


def test_error_permanente_no_se_reintenta(stubber, esperas):
    stubber.add_client_error(
        "invoke_model", "ValidationException", "Entrada inválida", 400
    )

    with pytest.raises(BedrockError, match="ValidationException"):
        ClienteEmbeddings(model_id=MODELO, client=stubber.client).vectorizar("Hola")

    assert esperas == []
