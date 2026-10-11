"""Conexión a PostgreSQL y aplicación del esquema."""

from __future__ import annotations

import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from psycopg.conninfo import make_conninfo

load_dotenv()

RUTA_ESQUEMA = Path(__file__).resolve().parent.parent / "sql" / "001_esquema.sql"


def cadena_conexion() -> str:
    """Arma la cadena de conexión con las variables del .env."""
    # make_conninfo escapa los valores: una contraseña con espacios o comillas
    # no rompe la cadena, como sí pasaría con un f-string.
    return make_conninfo(
        host="localhost",
        port=os.environ["POSTGRES_PORT"],
        dbname=os.environ["POSTGRES_DB"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
    )


def aplicar_esquema(ruta: Path = RUTA_ESQUEMA) -> None:
    """Ejecuta el archivo SQL del esquema y confirma los cambios."""
    # Conexión sin register_vector: la primera vez la extensión todavía no existe.
    with psycopg.connect(cadena_conexion()) as conn:
        conn.execute(ruta.read_text(encoding="utf-8"))


def conectar() -> psycopg.Connection:
    """Abre una conexión que sabe enviar y recibir el tipo vector."""
    conn = psycopg.connect(cadena_conexion())
    register_vector(conn)
    return conn
