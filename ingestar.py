"""Ingesta de normativa desde DOCX: un catálogo de normas y un solo script.

Uso:
    uv run python ingestar.py "SUGESE 03-10" --revisar   # solo extrae y divide; no llama a AWS
    uv run python ingestar.py "SUGESE 03-10"             # vectoriza y carga en PostgreSQL
"""

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from markwell_rag.ingesta import dividir_por_articulo, parrafos_del_cuerpo, validar

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")

PRECIO_TITAN_POR_MILLON = 0.02  # USD; el mismo valor que usa embeddings.py


@dataclass(frozen=True)
class Norma:
    entidad: str
    codigo: str
    titulo: str
    fuente_url: str
    ruta: Path
    # Fecha desde la que rige el texto descargado; la decide una persona leyendo
    # el control de cambios. Mientras sea None, el script no carga nada.
    vigente_desde: date | None
    # Número interno de versión en NUESTRA base (1 = primera que cargamos).
    numero_version: int = 1


CATALOGO = {
    n.codigo: n
    for n in [
        Norma(
            entidad="SUGESE",
            codigo="SUGESE 06-13",
            titulo="Reglamento de Defensa y Protección al Consumidor de Seguros",
            fuente_url=(
                "https://www.sugese.fi.cr/content/dam/sugese/marco-legal/reglamento-mercado-seguros/"
                "sugese-06-13-reglamento-defensa-y-proteccion-al-consulmidor-seguros.docx"
            ),
            ruta=Path("data/fuentes/sugese-06-13.docx"),
            vigente_desde=date(2020, 9, 1),
        ),
        Norma(
            entidad="SUGESE",
            codigo="SUGESE 03-10",
            titulo="Reglamento sobre la Comercialización de Seguros",
            fuente_url=(
                "https://www.sugese.fi.cr/content/dam/sugese/marco-legal/reglamento-mercado-seguros/"
                "reglamento-comercializacion-seguros.docx"
            ),
            ruta=Path("data/fuentes/sugese-03-10.docx"),
            vigente_desde=date(2024, 5, 22),
        ),
    ]
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("codigo", choices=sorted(CATALOGO), help="código de la norma en el catálogo")
    parser.add_argument("--revisar", action="store_true", help="solo extraer y dividir, sin AWS ni base de datos")
    args = parser.parse_args()
    norma = CATALOGO[args.codigo]

    if not norma.ruta.exists():
        sys.exit(f"No existe {norma.ruta}. Descárgalo de {norma.fuente_url}")

    parte, parrafos = parrafos_del_cuerpo(norma.ruta)
    division = dividir_por_articulo(parrafos)
    articulos = division.articulos
    print(f"{norma.codigo}: {norma.titulo}")
    print(f"Parte del DOCX con el articulado: {parte}")
    if not articulos:
        sys.exit("No se encontró ningún artículo. Revisa el formato del documento.")

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
    if norma.vigente_desde is None:
        sys.exit(f"Define vigente_desde de {norma.codigo} en CATALOGO antes de cargar.")

    # Importamos aquí para que --revisar funcione sin credenciales de AWS ni Docker.
    from markwell_rag import db
    from markwell_rag.bedrock import BedrockError
    from markwell_rag.embeddings import ClienteEmbeddings
    from markwell_rag.repositorio import FragmentoNuevo, calcular_hash, obtener_o_crear_documento, registrar_version

    hash_nuevo = calcular_hash([a.texto for a in articulos])

    # Antes de pagar por los embeddings, revisamos si esta versión ya está cargada.
    db.aplicar_esquema()
    with db.conectar() as conn:
        documento_id = obtener_o_crear_documento(conn, norma.entidad, norma.codigo, norma.titulo)
        guardada = conn.execute(
            "SELECT hash_contenido, vigente_desde FROM version_documento WHERE documento_id = %s AND numero = %s",
            (documento_id, norma.numero_version),
        ).fetchone()
    if guardada is not None:
        hash_guardado, desde_guardado = guardada
        if hash_guardado != hash_nuevo:
            sys.exit(
                f"\nLa versión {norma.numero_version} ya existe con OTRO contenido. Si el documento cambió, "
                "regístralo como una versión nueva y cierra la vigencia de la anterior."
            )
        if desde_guardado != norma.vigente_desde:
            sys.exit(
                f"\nLa versión {norma.numero_version} está cargada con vigencia desde {desde_guardado}, "
                f"pero el catálogo dice {norma.vigente_desde}. Corrige una de las dos antes de seguir."
            )
        print(f"\nLa versión {norma.numero_version} ya está cargada con el mismo contenido. No se hace nada.")
        return

    cliente = ClienteEmbeddings()
    try:
        embeddings = cliente.vectorizar_lote([a.texto for a in articulos])
    except BedrockError as e:
        sys.exit(f"{type(e).__name__}: {e}")

    fragmentos = [
        FragmentoNuevo(texto=a.texto, vector=e.vector, tokens=e.tokens, articulo=a.etiqueta)
        for a, e in zip(articulos, embeddings)
    ]
    with db.conectar() as conn:
        version_id = registrar_version(
            conn, documento_id, norma.numero_version, norma.vigente_desde, None,
            norma.fuente_url, cliente.model_id, fragmentos,
        )

    tokens = sum(f.tokens for f in fragmentos)
    print(f"\nCargada la versión {norma.numero_version} (id {version_id}) con {len(fragmentos)} fragmentos.")
    print(f"Tokens de entrada: {tokens} | costo estimado: USD {tokens * PRECIO_TITAN_POR_MILLON / 1_000_000:.6f}")


if __name__ == "__main__":
    main()
