import logging
import sys
from datetime import date

from markwell_rag import db
from markwell_rag.bedrock import BedrockError
from markwell_rag.embeddings import ClienteEmbeddings
from markwell_rag.repositorio import (
    FragmentoNuevo,
    buscar,
    obtener_o_crear_documento,
    registrar_version,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")

PREGUNTA = "¿En cuánto tiempo debe la aseguradora pagar un reclamo?"

# Reglamento FICTICIO escrito para el experimento. No es normativa real.
ENTIDAD = "EJEMPLO"
CODIGO = "REGLAMENTO-FICTICIO-1"
TITULO = "Reglamento ficticio de atención de reclamos"
FUENTE = "https://example.com/reglamento-ficticio"

ART_11 = (
    "Art. 11",
    "El plazo para presentar un reclamo en la oficina de atención al cliente es de cinco días.",
)
ART_12 = (
    "Art. 12",
    "Las primas del seguro de automóviles se calculan según el historial del conductor.",
)

VERSIONES = [
    # (número, vigente_desde, vigente_hasta, [(artículo, texto), ...])
    (
        1,
        date(2019, 1, 1),
        date(2025, 7, 1),
        [
            (
                "Art. 10",
                "La aseguradora debe pagar la indemnización en un plazo máximo de treinta días naturales desde que el asegurado entrega la documentación completa.",
            ),
            ART_11,
            ART_12,
        ],
    ),
    (
        2,
        date(2025, 7, 1),
        None,
        [
            (
                "Art. 10",
                "La aseguradora debe pagar la indemnización en un plazo máximo de quince días hábiles desde que el asegurado entrega la documentación completa.",
            ),
            ART_11,
            ART_12,
        ],
    ),
]

FECHAS = [date.today(), date(2024, 3, 15)]

cliente = ClienteEmbeddings()

try:
    db.aplicar_esquema()
    with db.conectar() as conn:
        # Borramos el ejemplo anterior para poder repetir el script. ON DELETE CASCADE
        # elimina también sus versiones y fragmentos.
        conn.execute("DELETE FROM documento WHERE entidad = %s", (ENTIDAD,))
        documento_id = obtener_o_crear_documento(conn, ENTIDAD, CODIGO, TITULO)

        for numero, desde, hasta, articulos in VERSIONES:
            embeddings = cliente.vectorizar_lote([texto for _, texto in articulos])
            fragmentos = [
                FragmentoNuevo(
                    texto=texto, vector=e.vector, tokens=e.tokens, articulo=articulo
                )
                for (articulo, texto), e in zip(articulos, embeddings)
            ]
            registrar_version(
                conn,
                documento_id,
                numero,
                desde,
                hasta,
                FUENTE,
                cliente.model_id,
                fragmentos,
            )
        # Al salir del bloque "with" sin errores, psycopg hace commit.

    pregunta = cliente.vectorizar(PREGUNTA)
except BedrockError as e:
    print(f"{type(e).__name__}: {e}")
    sys.exit(1)

print(f"\nPregunta: {PREGUNTA}")
with db.conectar() as conn:
    for fecha in FECHAS:
        print(f"\nVigentes al {fecha.isoformat()}:")
        for r in buscar(conn, pregunta.vector, cliente.model_id, fecha, k=3):
            hasta = (
                r.vigente_hasta.isoformat() if r.vigente_hasta else "sin fecha de fin"
            )
            print(
                f"  distancia {r.distancia:.4f} | similitud {1 - r.distancia:.4f} | "
                f"{r.articulo} (vigente {r.vigente_desde.isoformat()} a {hasta})"
            )
            print(f"    {r.texto[:90]}")
