"""Re-ranking de candidatos con un modelo de Bedrock (Cohere Rerank 3.5) usando la API Rerank."""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass
from typing import Any

from markwell_rag.bedrock import con_reintentos, crear_cliente_runtime
from markwell_rag.repositorio import Resultado

logger = logging.getLogger("markwell_rag.reranking")

MODELO_POR_DEFECTO = "cohere.rerank-v3-5:0"

# USD por cada 1000 consultas; una consulta admite hasta 100 documentos.
# Verificar el precio vigente en https://aws.amazon.com/bedrock/pricing/
PRECIOS_POR_MIL_CONSULTAS = {
    "cohere.rerank-v3-5:0": 2.00,
}
DOCUMENTOS_POR_CONSULTA = 100


@dataclass
class Reordenado:
    resultado: Resultado
    relevancia: float  # puntaje del reranker: sirve para ordenar, no es una probabilidad calibrada


def texto_para_reranking(r: Resultado) -> str:
    """Documento que ve el reranker: norma y artículo antes del texto, igual que en el contexto del modelo."""
    encabezado = f"{r.codigo} ({r.titulo})"
    if r.articulo:
        encabezado += f", {r.articulo}"
    return f"{encabezado}\n{r.texto}"


class ClienteReranking:
    def __init__(self, model_id: str | None = None, max_intentos: int = 4, client: Any = None) -> None:
        self.model_id = model_id or os.getenv("BEDROCK_RERANK_MODEL_ID", MODELO_POR_DEFECTO)
        self.max_intentos = max_intentos
        self._client = client if client is not None else crear_cliente_runtime("bedrock-agent-runtime")

    @property
    def model_arn(self) -> str:
        # La API Rerank pide el ARN del modelo, no solo su id.
        return f"arn:aws:bedrock:{self._client.meta.region_name}::foundation-model/{self.model_id}"

    def reordenar(self, pregunta: str, candidatos: list[Resultado], top_n: int) -> list[Reordenado]:
        """Devuelve los top_n candidatos más relevantes para la pregunta, de mayor a menor relevancia."""
        if not candidatos:
            return []
        if len(candidatos) > DOCUMENTOS_POR_CONSULTA:
            # Más de 100 documentos se cobran como varias consultas: lo evitamos a propósito.
            raise ValueError(f"Máximo {DOCUMENTOS_POR_CONSULTA} candidatos por consulta; llegaron {len(candidatos)}")
        top_n = min(top_n, len(candidatos))

        inicio = time.perf_counter()
        r = con_reintentos(
            lambda: self._client.rerank(
                queries=[{"type": "TEXT", "textQuery": {"text": pregunta}}],
                sources=[
                    {
                        "type": "INLINE",
                        "inlineDocumentSource": {"type": "TEXT", "textDocument": {"text": texto_para_reranking(c)}},
                    }
                    for c in candidatos
                ],
                rerankingConfiguration={
                    "type": "BEDROCK_RERANKING_MODEL",
                    "bedrockRerankingConfiguration": {
                        "numberOfResults": top_n,
                        "modelConfiguration": {"modelArn": self.model_arn},
                    },
                },
            ),
            self.model_id,
            self.max_intentos,
        )
        # "index" apunta a la posición del documento en "sources", que es la misma de "candidatos".
        reordenados = [Reordenado(candidatos[x["index"]], x["relevanceScore"]) for x in r["results"]]
        # La API ya los devuelve ordenados; ordenamos igual para no depender de eso.
        reordenados.sort(key=lambda x: x.relevancia, reverse=True)
        self._registrar(len(candidatos), top_n, inicio)
        return reordenados

    def costo_estimado(self, documentos: int) -> float | None:
        """Costo mínimo estimado de reordenar `documentos` candidatos; None si no hay precio conocido.

        Mínimo: Bedrock divide los documentos muy largos y puede cobrar más.
        """
        precio = PRECIOS_POR_MIL_CONSULTAS.get(self.model_id)
        if precio is None:
            return None
        return math.ceil(documentos / DOCUMENTOS_POR_CONSULTA) * precio / 1000

    def _registrar(self, documentos: int, top_n: int, inicio: float) -> None:
        registro = {
            "modelo": self.model_id,
            "documentos": documentos,
            "top_n": top_n,
            "costo_usd": self.costo_estimado(documentos),
            "latencia_ms": round((time.perf_counter() - inicio) * 1000),
        }
        logger.info(json.dumps(registro, ensure_ascii=False))
