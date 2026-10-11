"""Busca en la normativa cargada los fragmentos más cercanos a una pregunta.

Uso:
    uv run python buscar_normativa.py "¿En qué plazo debe responder la aseguradora un reclamo?"
    uv run python buscar_normativa.py "..." --fecha 2024-03-15 --k 3
"""

import argparse
import logging
import sys
from datetime import date

from markwell_rag import db
from markwell_rag.bedrock import BedrockError
from markwell_rag.embeddings import ClienteEmbeddings
from markwell_rag.repositorio import buscar

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")

parser = argparse.ArgumentParser()
parser.add_argument("pregunta")
parser.add_argument(
    "--fecha",
    type=date.fromisoformat,
    default=date.today(),
    help="AAAA-MM-DD; por defecto, hoy",
)
parser.add_argument("--k", type=int, default=5)
args = parser.parse_args()

cliente = ClienteEmbeddings()
try:
    pregunta = cliente.vectorizar(args.pregunta)
except BedrockError as e:
    sys.exit(f"{type(e).__name__}: {e}")

with db.conectar() as conn:
    resultados = buscar(conn, pregunta.vector, cliente.model_id, args.fecha, k=args.k)

print(f"Pregunta: {args.pregunta}")
print(f"Normativa vigente al {args.fecha.isoformat()}\n")
if not resultados:
    print("Sin resultados: no hay normativa cargada vigente en esa fecha.")
for posicion, r in enumerate(resultados, start=1):
    hasta = r.vigente_hasta.isoformat() if r.vigente_hasta else "sin fecha de fin"
    print(f"{posicion}. {r.codigo}, {r.articulo} | similitud {1 - r.distancia:.4f}")
    print(f"   Vigente desde {r.vigente_desde.isoformat()} hasta {hasta}")
    print(f"   Fuente: {r.fuente_url}")
    print(f"   {r.texto[:200].replace(chr(10), ' ')}...\n")
