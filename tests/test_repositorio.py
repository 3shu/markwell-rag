"""Pruebas de integración del repositorio contra el PostgreSQL local (no llaman a AWS).

Cada prueba trabaja dentro de una transacción que se revierte al final, así que
no deja datos en la base. Si el contenedor no está levantado, se omiten.
"""

from datetime import date

import psycopg
import pytest

from markwell_rag import db
from markwell_rag.repositorio import (
    FragmentoNuevo,
    buscar,
    obtener_o_crear_documento,
    registrar_version,
)

MODELO = "modelo-de-prueba"
DIMENSIONES = 1024


def vector_base(i: int) -> list[float]:
    """Vector unitario con un 1 en la posición i: ortogonal a los demás."""
    v = [0.0] * DIMENSIONES
    v[i] = 1.0
    return v


def fragmento(texto: str, i: int) -> FragmentoNuevo:
    return FragmentoNuevo(texto=texto, vector=vector_base(i), tokens=3)


@pytest.fixture
def conn():
    try:
        db.aplicar_esquema()
        conexion = db.conectar()
    except psycopg.OperationalError as e:
        pytest.skip(f"PostgreSQL no disponible (¿docker compose up -d?): {e}")
    try:
        yield conexion
    finally:
        conexion.rollback()
        conexion.close()


@pytest.fixture
def documento_id(conn):
    return obtener_o_crear_documento(conn, "PRUEBA", "DOC-1", "Documento de prueba")


def test_ordena_por_distancia_coseno(conn, documento_id):
    registrar_version(
        conn,
        documento_id,
        1,
        date(2020, 1, 1),
        None,
        "https://example.com",
        MODELO,
        [fragmento("uno", 0), fragmento("dos", 1), fragmento("tres", 2)],
    )

    resultados = buscar(conn, vector_base(1), MODELO, date(2026, 1, 1), k=3)

    assert [r.texto for r in resultados][0] == "dos"
    assert resultados[0].distancia == pytest.approx(0.0)
    assert resultados[1].distancia == pytest.approx(1.0)  # vectores ortogonales


def test_filtra_por_vigencia(conn, documento_id):
    registrar_version(
        conn,
        documento_id,
        1,
        date(2020, 1, 1),
        date(2024, 1, 1),
        "https://example.com",
        MODELO,
        [fragmento("versión 1", 0)],
    )
    registrar_version(
        conn,
        documento_id,
        2,
        date(2024, 1, 1),
        None,
        "https://example.com",
        MODELO,
        [fragmento("versión 2", 0)],
    )

    assert [
        r.texto for r in buscar(conn, vector_base(0), MODELO, date(2023, 6, 1))
    ] == ["versión 1"]
    assert [
        r.texto for r in buscar(conn, vector_base(0), MODELO, date(2025, 6, 1))
    ] == ["versión 2"]
    # vigente_hasta es exclusiva: el 1 de enero de 2024 ya rige la versión 2.
    assert [
        r.texto for r in buscar(conn, vector_base(0), MODELO, date(2024, 1, 1))
    ] == ["versión 2"]


def test_rechaza_vigencias_traslapadas(conn, documento_id):
    registrar_version(
        conn,
        documento_id,
        1,
        date(2020, 1, 1),
        None,
        "https://example.com",
        MODELO,
        [fragmento("versión 1", 0)],
    )
    with pytest.raises(psycopg.errors.ExclusionViolation):
        registrar_version(
            conn,
            documento_id,
            2,
            date(2024, 1, 1),
            None,
            "https://example.com",
            MODELO,
            [fragmento("versión 2", 0)],
        )


def test_ignora_vectores_de_otro_modelo(conn, documento_id):
    registrar_version(
        conn,
        documento_id,
        1,
        date(2020, 1, 1),
        None,
        "https://example.com",
        "otro-modelo",
        [fragmento("uno", 0)],
    )
    assert buscar(conn, vector_base(0), MODELO, date(2026, 1, 1)) == []


def test_rechaza_dimensiones_incorrectas(conn, documento_id):
    with pytest.raises(psycopg.errors.DataException, match="1024"):
        registrar_version(
            conn,
            documento_id,
            1,
            date(2020, 1, 1),
            None,
            "https://example.com",
            MODELO,
            [FragmentoNuevo(texto="corto", vector=[1.0, 0.0], tokens=1)],
        )


def test_hash_de_version(conn, documento_id):
    from markwell_rag.repositorio import calcular_hash, hash_de_version

    assert hash_de_version(conn, documento_id, 1) is None
    registrar_version(
        conn,
        documento_id,
        1,
        date(2020, 1, 1),
        None,
        "https://example.com",
        MODELO,
        [fragmento("uno", 0), fragmento("dos", 1)],
    )
    assert hash_de_version(conn, documento_id, 1) == calcular_hash(["uno", "dos"])
