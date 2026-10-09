"""Cliente reutilizable de Amazon Bedrock basado en la Converse API."""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, TypeVar

from botocore.exceptions import EventStreamError
from pydantic import BaseModel, ValidationError

# Se reexportan BedrockError y ModeloNoDisponibleError para que el código existente
# (pruebas y scripts) pueda seguir importándolos desde markwell_rag.llm.
from markwell_rag.bedrock import (
    BedrockError,
    ModeloNoDisponibleError,
    con_reintentos,
    crear_cliente_runtime,
)

logger = logging.getLogger("markwell_rag.llm")

SYSTEM_PROMPT = """Eres un asistente de consulta regulatoria para el sector asegurador de Costa Rica.
Respondes en español, de forma clara, profesional y breve.

Reglas:
1. Responde solo con base en los documentos que se incluyan en la consulta. Si no hay documentos o no contienen la respuesta, dilo de forma explícita y no inventes normas, artículos, plazos ni números de circular.
2. Cuando cites una norma, indica la entidad emisora, el número de norma o circular, el artículo y su fecha de vigencia.
3. Escribe en texto plano. No uses Markdown: ni encabezados, ni negritas, ni cursivas, ni viñetas con asteriscos o guiones. Si necesitas enumerar, usa números seguidos de punto.
4. Termina siempre con esta línea exacta:
Aviso: esta respuesta es de referencia y no constituye asesoría legal."""

# USD por millón de tokens. Precio de lista de Anthropic para Haiku 4.5;
# verificar el precio vigente en https://aws.amazon.com/bedrock/pricing/
PRECIOS_POR_MILLON = {
    "us.anthropic.claude-haiku-4-5-20251001-v1:0": {"entrada": 1.00, "salida": 5.00},
}

# Nombre de la herramienta que el modelo debe usar para entregar la salida estructurada.
HERRAMIENTA_SALIDA = "entregar_respuesta"

T = TypeVar("T", bound=BaseModel)


class SalidaInvalidaError(BedrockError):
    """El modelo devolvió una salida que no cumple el esquema solicitado."""


@dataclass
class Respuesta:
    texto: str
    modelo: str
    tokens_entrada: int
    tokens_salida: int
    costo_usd: float | None
    latencia_ms: int
    stop_reason: str | None


class ClienteBedrock:
    def __init__(
        self,
        model_id: str | None = None,
        system_prompt: str = SYSTEM_PROMPT,
        max_tokens: int = 800,
        temperature: float = 0.2,
        max_intentos: int = 4,
        client: Any = None,
    ) -> None:
        self.model_id = model_id or os.environ["BEDROCK_MODEL_ID"]
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_intentos = max_intentos
        # Las pruebas inyectan un cliente simulado; en uso normal se crea el real.
        self._client = client if client is not None else crear_cliente_runtime()

    def consultar(self, pregunta: str) -> Respuesta:
        """Envía la pregunta y devuelve la respuesta completa."""
        inicio = time.perf_counter()
        r = self._con_reintentos(
            lambda: self._client.converse(**self._peticion(pregunta))
        )
        texto = "".join(
            b["text"] for b in r["output"]["message"]["content"] if "text" in b
        )
        return self._registrar(texto, r["usage"], r["stopReason"], inicio)

    def consultar_stream(
        self,
        pregunta: str,
        al_recibir: Callable[[str], None] = lambda t: print(t, end="", flush=True),
    ) -> Respuesta:
        """Envía la pregunta y entrega el texto por fragmentos a medida que llega."""
        inicio = time.perf_counter()
        r = self._con_reintentos(
            lambda: self._client.converse_stream(**self._peticion(pregunta))
        )

        partes: list[str] = []
        usage: dict = {}
        stop_reason = None
        try:
            for evento in r["stream"]:
                if "contentBlockDelta" in evento:
                    fragmento = evento["contentBlockDelta"]["delta"].get("text", "")
                    partes.append(fragmento)
                    al_recibir(fragmento)
                elif "messageStop" in evento:
                    stop_reason = evento["messageStop"]["stopReason"]
                elif "metadata" in evento:
                    usage = evento["metadata"]["usage"]
        except EventStreamError as e:
            # Si el stream falla a la mitad no reintentamos: el usuario ya vio texto parcial.
            raise BedrockError(f"Error durante el streaming: {e}") from e

        return self._registrar("".join(partes), usage, stop_reason, inicio)

    def consultar_estructurado(
        self, pregunta: str, esquema: type[T]
    ) -> tuple[T, Respuesta]:
        """Obliga al modelo a responder con el esquema indicado y valida el resultado."""
        inicio = time.perf_counter()
        peticion = self._peticion(pregunta)
        peticion["toolConfig"] = {
            "tools": [
                {
                    "toolSpec": {
                        "name": HERRAMIENTA_SALIDA,
                        "description": "Entrega la respuesta final al usuario junto con sus fuentes.",
                        "inputSchema": {"json": esquema.model_json_schema()},
                    }
                }
            ],
            # Forzamos esta herramienta: el modelo no puede responder en texto libre.
            "toolChoice": {"tool": {"name": HERRAMIENTA_SALIDA}},
        }
        r = self._con_reintentos(lambda: self._client.converse(**peticion))
        datos = next(
            (
                b["toolUse"]["input"]
                for b in r["output"]["message"]["content"]
                if "toolUse" in b
            ),
            None,
        )
        # Registramos antes de validar: una salida inválida también consume tokens.
        meta = self._registrar(
            json.dumps(datos, ensure_ascii=False), r["usage"], r["stopReason"], inicio
        )
        if datos is None:
            raise SalidaInvalidaError("El modelo no devolvió la salida estructurada")
        try:
            return esquema.model_validate(datos), meta
        except ValidationError as e:
            raise SalidaInvalidaError(
                f"La salida no cumple {esquema.__name__}: {e}"
            ) from e

    def _peticion(self, pregunta: str) -> dict:
        return {
            "modelId": self.model_id,
            "system": [{"text": self.system_prompt}],
            "messages": [{"role": "user", "content": [{"text": pregunta}]}],
            "inferenceConfig": {
                "maxTokens": self.max_tokens,
                "temperature": self.temperature,
            },
        }

    def _con_reintentos(self, operacion: Callable[[], dict]) -> dict:
        return con_reintentos(operacion, self.model_id, self.max_intentos)

    def _registrar(
        self, texto: str, usage: dict, stop_reason: str | None, inicio: float
    ) -> Respuesta:
        tokens_entrada = usage.get("inputTokens", 0)
        tokens_salida = usage.get("outputTokens", 0)
        precios = PRECIOS_POR_MILLON.get(self.model_id)
        costo = None
        if precios:
            costo = (
                tokens_entrada * precios["entrada"] + tokens_salida * precios["salida"]
            ) / 1_000_000

        respuesta = Respuesta(
            texto=texto,
            modelo=self.model_id,
            tokens_entrada=tokens_entrada,
            tokens_salida=tokens_salida,
            costo_usd=costo,
            latencia_ms=round((time.perf_counter() - inicio) * 1000),
            stop_reason=stop_reason,
        )
        registro = {k: v for k, v in asdict(respuesta).items() if k != "texto"}
        logger.info(json.dumps(registro, ensure_ascii=False))
        return respuesta
