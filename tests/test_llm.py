"""Pruebas del cliente de Bedrock con respuestas simuladas (no llaman a AWS)."""

import boto3
import pytest
from botocore.stub import ANY, Stubber

from markwell_rag import llm
from markwell_rag.esquemas import RespuestaRegulatoria
from markwell_rag.llm import (
    BedrockError,
    ClienteBedrock,
    ModeloNoDisponibleError,
    SalidaInvalidaError,
)

MODELO = "us.anthropic.claude-haiku-4-5-20251001-v1:0"


def respuesta_converse(
    contenido: list, entrada: int = 10, salida: int = 5, stop: str = "end_turn"
) -> dict:
    """Arma una respuesta con la misma forma que devuelve la Converse API."""
    return {
        "output": {"message": {"role": "assistant", "content": contenido}},
        "stopReason": stop,
        "usage": {
            "inputTokens": entrada,
            "outputTokens": salida,
            "totalTokens": entrada + salida,
        },
        "metrics": {"latencyMs": 100},
    }


@pytest.fixture
def esperas(monkeypatch):
    """Sustituye time.sleep: las pruebas no esperan y registran cada espera."""
    registro: list[float] = []
    monkeypatch.setattr(llm.time, "sleep", registro.append)
    return registro


@pytest.fixture
def stubber(monkeypatch):
    """Cliente real de boto3 con credenciales falsas; Stubber intercepta cada llamada."""
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


def crear_cliente(stubber: Stubber, **kwargs) -> ClienteBedrock:
    return ClienteBedrock(model_id=MODELO, client=stubber.client, **kwargs)


def test_consultar_envia_system_prompt_y_calcula_costo(stubber):
    stubber.add_response(
        "converse",
        respuesta_converse([{"text": "Hola"}], entrada=1000, salida=200),
        expected_params={
            "modelId": MODELO,
            "system": [{"text": llm.SYSTEM_PROMPT}],
            "messages": ANY,
            "inferenceConfig": ANY,
        },
    )

    r = crear_cliente(stubber).consultar("¿Qué es SUGESE?")

    assert r.texto == "Hola"
    assert (r.tokens_entrada, r.tokens_salida) == (1000, 200)
    # 1000 * 1 USD/M + 200 * 5 USD/M = 0,002 USD
    assert r.costo_usd == pytest.approx(0.002)


def test_reintenta_throttling_y_luego_responde(stubber, esperas):
    stubber.add_client_error(
        "converse", "ThrottlingException", "Too many requests", 429
    )
    stubber.add_client_error(
        "converse", "ThrottlingException", "Too many requests", 429
    )
    stubber.add_response("converse", respuesta_converse([{"text": "OK"}]))

    r = crear_cliente(stubber).consultar("Hola")

    assert r.texto == "OK"
    assert len(esperas) == 2
    # Jitter completo: intento 1 espera entre 0 y 1 s; intento 2, entre 0 y 2 s.
    assert 0 <= esperas[0] <= 1
    assert 0 <= esperas[1] <= 2


def test_no_reintenta_errores_permanentes(stubber, esperas):
    stubber.add_client_error("converse", "ValidationException", "Modelo inválido", 400)

    with pytest.raises(BedrockError, match="ValidationException"):
        crear_cliente(stubber).consultar("Hola")

    assert esperas == []


def test_access_denied_se_traduce_a_modelo_no_disponible(stubber, esperas):
    stubber.add_client_error("converse", "AccessDeniedException", "Sin acceso", 403)

    with pytest.raises(ModeloNoDisponibleError):
        crear_cliente(stubber).consultar("Hola")

    assert esperas == []


def test_falla_al_agotar_los_intentos(stubber, esperas):
    for _ in range(3):
        stubber.add_client_error(
            "converse", "ServiceUnavailableException", "No disponible", 503
        )

    with pytest.raises(BedrockError, match="tras 3 intentos"):
        crear_cliente(stubber, max_intentos=3).consultar("Hola")

    # Tres intentos implican solo dos esperas: tras el último no se espera.
    assert len(esperas) == 2


class ClienteFalsoStream:
    """Stubber no simula bien los event streams; un objeto falso basta."""

    def converse_stream(self, **peticion):
        return {
            "stream": iter(
                [
                    {"messageStart": {"role": "assistant"}},
                    {
                        "contentBlockDelta": {
                            "delta": {"text": "Hola "},
                            "contentBlockIndex": 0,
                        }
                    },
                    {
                        "contentBlockDelta": {
                            "delta": {"text": "Mario"},
                            "contentBlockIndex": 0,
                        }
                    },
                    {"messageStop": {"stopReason": "end_turn"}},
                    {
                        "metadata": {
                            "usage": {
                                "inputTokens": 8,
                                "outputTokens": 2,
                                "totalTokens": 10,
                            }
                        }
                    },
                ]
            )
        }


def test_stream_entrega_fragmentos_en_orden():
    recibidos: list[str] = []
    cliente = ClienteBedrock(model_id=MODELO, client=ClienteFalsoStream())

    r = cliente.consultar_stream("Hola", al_recibir=recibidos.append)

    assert recibidos == ["Hola ", "Mario"]
    assert r.texto == "Hola Mario"
    assert r.tokens_salida == 2
    assert r.stop_reason == "end_turn"


def test_consultar_estructurado_valida_la_salida(stubber):
    datos = {
        "respuesta": "El plazo es de 30 días naturales.",
        "sustentada": True,
        "fuentes": [
            {
                "entidad": "SUGESE",
                "norma": "PRUEBA-001",
                "articulo": "4",
                "vigente_desde": "2026-01-15",
            }
        ],
    }
    stubber.add_response(
        "converse",
        respuesta_converse(
            [
                {
                    "toolUse": {
                        "toolUseId": "t1",
                        "name": llm.HERRAMIENTA_SALIDA,
                        "input": datos,
                    }
                }
            ],
            stop="tool_use",
        ),
    )

    r, meta = crear_cliente(stubber).consultar_estructurado(
        "¿Plazo?", RespuestaRegulatoria
    )

    assert r.sustentada is True
    assert r.fuentes[0].vigente_desde.isoformat() == "2026-01-15"
    assert meta.stop_reason == "tool_use"


def test_consultar_estructurado_rechaza_salida_incoherente(stubber):
    # El modelo dice que hay sustento, pero no cita ninguna fuente.
    datos = {"respuesta": "El plazo es de 30 días.", "sustentada": True, "fuentes": []}
    stubber.add_response(
        "converse",
        respuesta_converse(
            [
                {
                    "toolUse": {
                        "toolUseId": "t1",
                        "name": llm.HERRAMIENTA_SALIDA,
                        "input": datos,
                    }
                }
            ],
            stop="tool_use",
        ),
    )

    with pytest.raises(SalidaInvalidaError, match="al menos una fuente"):
        crear_cliente(stubber).consultar_estructurado("¿Plazo?", RespuestaRegulatoria)
