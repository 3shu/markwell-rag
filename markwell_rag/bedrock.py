"""Piezas compartidas para invocar Amazon Bedrock: cliente, errores y reintentos."""

from __future__ import annotations

import logging
import os
import random
import time
from collections.abc import Callable
from typing import Any, TypeVar

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("markwell_rag.bedrock")

# Errores transitorios: tiene sentido volver a intentar.
ERRORES_REINTENTABLES = {
    "ThrottlingException",
    "ServiceUnavailableException",
    "InternalServerException",
    "ModelNotReadyException",
}

R = TypeVar("R")


class BedrockError(Exception):
    """Error al invocar Bedrock que no se resolvió con reintentos."""


class ModeloNoDisponibleError(BedrockError):
    """La cuenta no tiene acceso al modelo solicitado."""


def crear_cliente_runtime() -> Any:
    """Crea el cliente bedrock-runtime con el perfil y la región del .env."""
    session = boto3.Session(
        profile_name=os.environ["AWS_PROFILE"],
        region_name=os.environ["AWS_REGION"],
    )
    # Desactivamos los reintentos internos de botocore para controlarlos
    # nosotros y no reintentar dos veces el mismo error.
    config = Config(
        retries={"total_max_attempts": 1, "mode": "standard"},
        connect_timeout=5,
        read_timeout=60,
    )
    return session.client("bedrock-runtime", config=config)


def con_reintentos(operacion: Callable[[], R], model_id: str, max_intentos: int) -> R:
    """Ejecuta la operación con backoff exponencial y jitter completo."""
    for intento in range(1, max_intentos + 1):
        try:
            return operacion()
        except ClientError as e:
            codigo = e.response["Error"]["Code"]
            mensaje = e.response["Error"].get("Message", "")
            if codigo == "AccessDeniedException":
                raise ModeloNoDisponibleError(
                    f"Sin acceso a {model_id}: {mensaje}"
                ) from e
            if codigo not in ERRORES_REINTENTABLES:
                raise BedrockError(f"{codigo}: {mensaje}") from e
            if intento == max_intentos:
                raise BedrockError(
                    f"{codigo} tras {intento} intentos: {mensaje}"
                ) from e

            espera = random.uniform(0, min(20.0, 2 ** (intento - 1)))
            logger.warning(
                "%s en intento %d/%d; nuevo intento en %.2f s",
                codigo,
                intento,
                max_intentos,
                espera,
            )
            time.sleep(espera)
    raise AssertionError("inalcanzable")
