"""Lista los artículos vigentes de una norma cargada, para etiquetar preguntas de evaluación.

Uso:
    uv run python indice_normativa.py "SUGESE 03-10"                      # primera línea de cada artículo
    uv run python indice_normativa.py "SUGESE 03-10" --articulo "Art. 7"  # texto completo de uno
"""

import argparse
import sys
from datetime import date

from markwell_rag import db
from markwell_rag.rag import clave_articulo

parser = argparse.ArgumentParser()
parser.add_argument("codigo", help='código exacto, por ejemplo "SUGESE 06-13"')
parser.add_argument("--articulo", help='muestra el texto completo de ese artículo, por ejemplo "Art. 7"')
parser.add_argument("--fecha", type=date.fromisoformat, default=date.today(), help="AAAA-MM-DD; por defecto, hoy")
args = parser.parse_args()

with db.conectar() as conn:
    filas = conn.execute(
        """
        SELECT f.articulo, f.texto
        FROM fragmento f
        JOIN version_documento v ON v.id = f.version_id
        JOIN documento d ON d.id = v.documento_id
        WHERE d.codigo = %(codigo)s
          AND v.vigente_desde <= %(fecha)s
          AND (v.vigente_hasta IS NULL OR v.vigente_hasta > %(fecha)s)
        ORDER BY f.orden
        """,
        {"codigo": args.codigo, "fecha": args.fecha},
    ).fetchall()

if not filas:
    sys.exit(f"No hay artículos de {args.codigo} vigentes al {args.fecha.isoformat()}.")

if args.articulo:
    buscado = clave_articulo(args.articulo)
    textos = [texto for articulo, texto in filas if clave_articulo(articulo) == buscado]
    if not textos:
        sys.exit(f"{args.codigo} no tiene {args.articulo}.")
    print(textos[0])
else:
    for articulo, texto in filas:
        print(f"{articulo:<16} {texto.splitlines()[0][:110]}")
