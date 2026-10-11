"""Ingesta del Reglamento de Defensa y Protección al Consumidor de Seguros (SUGESE 06-13).

Uso:
    uv run python ingestar_sugese_06_13.py --revisar   # solo extrae y divide; no llama a AWS
    uv run python ingestar_sugese_06_13.py             # vectoriza y carga en PostgreSQL
"""

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

from markwell_rag.ingesta import dividir_por_articulo, parrafos_del_cuerpo, validar

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")

ENTIDAD = "SUGESE"
CODIGO = "SUGESE 06-13"
TITULO = "Reglamento de Defensa y Protección al Consumidor de Seguros"
FUENTE_URL = (
    "https://www.sugese.fi.cr/content/dam/sugese/marco-legal/reglamento-mercado-seguros/"
    "sugese-06-13-reglamento-defensa-y-proteccion-al-consulmidor-seguros.docx"
)
RUTA = Path("data/fuentes/sugese-06-13.docx")

# Número interno de versión en NUESTRA base (1 = primera que cargamos), no el
# número de reformas que ha tenido el reglamento.
NUMERO_VERSION = 1
# Fecha desde la que rige el texto descargado. Se completa en el paso 4 leyendo
# el encabezado del documento. Mientras sea None, el script no carga nada.
VIGENTE_DESDE: date | None = None

PRECIO_TITAN_POR_MILLON = 0.02  # USD; el mismo valor que usa embeddings.py


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--revisar",
        action="store_true",
        help="solo extraer y dividir, sin AWS ni base de datos",
    )
    args = parser.parse_args()

    parte, parrafos = parrafos_del_cuerpo(RUTA)
    division = dividir_por_articulo(parrafos)
    articulos = division.articulos
    print(f"Parte del DOCX con el articulado: {parte}")
    if not articulos:
        sys.exit(
            "No se encontró ningún artículo. Revisa el formato del documento (paso 4)."
        )

    largo = max(articulos, key=lambda a: len(a.texto))
    print(f"Párrafos de preámbulo (no se cargan): {len(division.preambulo)}")
    print(f"Fragmentos: {len(articulos)}")
    print(f"Primeros: {[a.etiqueta for a in articulos[:3]]}")
    print(f"Últimos:  {[a.etiqueta for a in articulos[-3:]]}")
    print(f"Más largo: {largo.etiqueta} ({len(largo.texto)} caracteres)")
    print(f"Caracteres en total: {sum(len(a.texto) for a in articulos)}")
    avisos = validar(articulos)
    print(f"Avisos ({len(avisos)}):")
    for aviso in avisos:
        print(f"  - {aviso}")

    if args.revisar:
        return
    if VIGENTE_DESDE is None:
        sys.exit(
            "Define VIGENTE_DESDE con la fecha que leíste en el documento antes de cargar."
        )

    # Importamos aquí para que --revisar funcione sin credenciales de AWS ni Docker.
    from markwell_rag import db
    from markwell_rag.bedrock import BedrockError
    from markwell_rag.embeddings import ClienteEmbeddings
    from markwell_rag.repositorio import (
        FragmentoNuevo,
        calcular_hash,
        hash_de_version,
        obtener_o_crear_documento,
        registrar_version,
    )

    hash_nuevo = calcular_hash([a.texto for a in articulos])

    # Antes de pagar por los embeddings, revisamos si esta versión ya está cargada.
    db.aplicar_esquema()
    with db.conectar() as conn:
        documento_id = obtener_o_crear_documento(conn, ENTIDAD, CODIGO, TITULO)
        hash_guardado = hash_de_version(conn, documento_id, NUMERO_VERSION)
    if hash_guardado == hash_nuevo:
        print(
            f"\nLa versión {NUMERO_VERSION} ya está cargada con el mismo contenido. No se hace nada."
        )
        return
    if hash_guardado is not None:
        sys.exit(
            f"\nLa versión {NUMERO_VERSION} ya existe con OTRO contenido. Si el documento cambió, "
            "regístralo como una versión nueva y cierra la vigencia de la anterior."
        )

    cliente = ClienteEmbeddings()
    try:
        embeddings = cliente.vectorizar_lote([a.texto for a in articulos])
    except BedrockError as e:
        sys.exit(f"{type(e).__name__}: {e}")

    fragmentos = [
        FragmentoNuevo(
            texto=a.texto, vector=e.vector, tokens=e.tokens, articulo=a.etiqueta
        )
        for a, e in zip(articulos, embeddings)
    ]
    with db.conectar() as conn:
        version_id = registrar_version(
            conn,
            documento_id,
            NUMERO_VERSION,
            VIGENTE_DESDE,
            None,
            FUENTE_URL,
            cliente.model_id,
            fragmentos,
        )

    tokens = sum(f.tokens for f in fragmentos)
    print(
        f"\nCargada la versión {NUMERO_VERSION} (id {version_id}) con {len(fragmentos)} fragmentos."
    )
    print(
        f"Tokens de entrada: {tokens} | costo estimado: USD {tokens * PRECIO_TITAN_POR_MILLON / 1_000_000:.6f}"
    )


if __name__ == "__main__":
    main()
