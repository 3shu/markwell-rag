"""Embeddings de texto con Amazon Titan Text Embeddings V2 en Bedrock."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

from markwell_rag.bedrock import con_reintentos, crear_cliente_runtime

logger = logging.getLogger("markwell_rag.embeddings")

MODELO_POR_DEFECTO = "amazon.titan-embed-text-v2:0"

# Titan V2 solo acepta estas dimensiones.
DIMENSIONES_VALIDAS = {256, 512, 1024}

# USD por millón de tokens de entrada (los embeddings no tienen tokens de salida).
# Verificar el precio vigente en https://aws.amazon.com/bedrock/pricing/
PRECIOS_POR_MILLON = {
    "amazon.titan-embed-text-v2:0": 0.02,
}


@dataclass
class Embedding:
    vector: list[float]
    tokens: int


class ClienteEmbeddings:
    def __init__(
        self,
        model_id: str | None = None,
        dimensiones: int = 1024,
        max_intentos: int = 4,
        client: Any = None,
    ) -> None:
        if dimensiones not in DIMENSIONES_VALIDAS:
            raise ValueError(
                f"dimensiones debe ser una de {sorted(DIMENSIONES_VALIDAS)}"
            )
        self.model_id = model_id or os.getenv(
            "BEDROCK_EMBEDDING_MODEL_ID", MODELO_POR_DEFECTO
        )
        self.dimensiones = dimensiones
        self.max_intentos = max_intentos
        self._client = client if client is not None else crear_cliente_runtime()

    def vectorizar(self, texto: str) -> Embedding:
        """Convierte un texto en un vector normalizado (longitud 1)."""
        if not texto.strip():
            # Lo rechazamos aquí: Bedrock también lo rechazaría, pero cobrando una llamada.
            raise ValueError("No se puede vectorizar un texto vacío")

        inicio = time.perf_counter()
        cuerpo = json.dumps(
            {"inputText": texto, "dimensions": self.dimensiones, "normalize": True},
            ensure_ascii=False,
        )
        # Los embeddings no usan la Converse API: se invocan con InvokeModel y un
        # cuerpo JSON propio de cada modelo.
        r = con_reintentos(
            lambda: self._client.invoke_model(
                modelId=self.model_id,
                body=cuerpo,
                contentType="application/json",
                accept="application/json",
            ),
            self.model_id,
            self.max_intentos,
        )
        datos = json.loads(r["body"].read())
        embedding = Embedding(
            vector=datos["embedding"], tokens=datos["inputTextTokenCount"]
        )
        self._registrar(embedding, inicio)
        return embedding

    def vectorizar_lote(self, textos: list[str]) -> list[Embedding]:
        """Vectoriza varios textos en orden. Titan V2 recibe un texto por llamada."""
        return [self.vectorizar(t) for t in textos]

    def _registrar(self, embedding: Embedding, inicio: float) -> None:
        precio = PRECIOS_POR_MILLON.get(self.model_id)
        costo = embedding.tokens * precio / 1_000_000 if precio else None
        registro = {
            "modelo": self.model_id,
            "dimensiones": len(embedding.vector),
            "tokens_entrada": embedding.tokens,
            "costo_usd": costo,
            "latencia_ms": round((time.perf_counter() - inicio) * 1000),
        }
        logger.info(json.dumps(registro, ensure_ascii=False))
