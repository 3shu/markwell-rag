"""Acceso a la normativa versionada: inserción de fragmentos y búsqueda vectorial.

Las funciones reciben la conexión y no hacen commit: quien llama decide cuándo
confirmar la transacción. Así una versión se guarda completa o no se guarda.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date

import psycopg
from pgvector import Vector


@dataclass
class FragmentoNuevo:
    texto: str
    vector: list[float]
    tokens: int
    articulo: str | None = None


@dataclass
class Resultado:
    fragmento_id: int
    entidad: str
    codigo: str
    titulo: str
    articulo: str | None
    texto: str
    vigente_desde: date
    vigente_hasta: date | None
    fuente_url: str
    distancia: float


def obtener_o_crear_documento(
    conn: psycopg.Connection, entidad: str, codigo: str, titulo: str
) -> int:
    """Devuelve el id del documento; lo crea si no existe."""
    fila = conn.execute(
        """
        INSERT INTO documento (entidad, codigo, titulo)
        VALUES (%s, %s, %s)
        ON CONFLICT (entidad, codigo) DO UPDATE SET titulo = EXCLUDED.titulo
        RETURNING id
        """,
        (entidad, codigo, titulo),
    ).fetchone()
    return fila[0]


def registrar_version(
    conn: psycopg.Connection,
    documento_id: int,
    numero: int,
    vigente_desde: date,
    vigente_hasta: date | None,
    fuente_url: str,
    modelo_embedding: str,
    fragmentos: list[FragmentoNuevo],
) -> int:
    """Inserta una versión con todos sus fragmentos y devuelve el id de la versión."""
    # El hash permitirá a la ingesta detectar si un documento cambió sin comparar textos.
    hash_contenido = calcular_hash([f.texto for f in fragmentos])

    version_id = conn.execute(
        """
        INSERT INTO version_documento
            (documento_id, numero, vigente_desde, vigente_hasta, fuente_url, hash_contenido)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING id
        """,
        (
            documento_id,
            numero,
            vigente_desde,
            vigente_hasta,
            fuente_url,
            hash_contenido,
        ),
    ).fetchone()[0]

    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO fragmento
                (version_id, orden, articulo, texto, tokens, modelo_embedding, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            [
                (
                    version_id,
                    orden,
                    f.articulo,
                    f.texto,
                    f.tokens,
                    modelo_embedding,
                    Vector(f.vector),
                )
                for orden, f in enumerate(fragmentos, start=1)
            ],
        )
    return version_id


def buscar(
    conn: psycopg.Connection,
    vector: list[float],
    modelo_embedding: str,
    fecha: date,
    k: int = 5,
) -> list[Resultado]:
    """Devuelve los k fragmentos más cercanos entre las versiones vigentes en la fecha."""
    filas = conn.execute(
        """
        SELECT f.id, d.entidad, d.codigo, d.titulo, f.articulo, f.texto,
               v.vigente_desde, v.vigente_hasta, v.fuente_url,
               f.embedding <=> %(vector)s AS distancia
        FROM fragmento f
        JOIN version_documento v ON v.id = f.version_id
        JOIN documento d ON d.id = v.documento_id
        WHERE f.modelo_embedding = %(modelo)s
          AND v.vigente_desde <= %(fecha)s
          AND (v.vigente_hasta IS NULL OR v.vigente_hasta > %(fecha)s)
        ORDER BY f.embedding <=> %(vector)s
        LIMIT %(k)s
        """,
        {"vector": Vector(vector), "modelo": modelo_embedding, "fecha": fecha, "k": k},
    ).fetchall()
    return [Resultado(*fila) for fila in filas]


def calcular_hash(textos: list[str]) -> str:
    """SHA-256 del contenido de una versión; es el mismo que guarda registrar_version."""
    return hashlib.sha256("\n".join(textos).encode("utf-8")).hexdigest()


def hash_de_version(
    conn: psycopg.Connection, documento_id: int, numero: int
) -> str | None:
    """Devuelve el hash de una versión ya cargada, o None si no existe."""
    fila = conn.execute(
        "SELECT hash_contenido FROM version_documento WHERE documento_id = %s AND numero = %s",
        (documento_id, numero),
    ).fetchone()
    return fila[0] if fila else None
