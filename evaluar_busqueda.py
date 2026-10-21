"""Mide la recuperación con las preguntas de evaluación: hit rate y MRR, sin llamar al modelo.

Uso:
    uv run python evaluar_busqueda.py
    uv run python evaluar_busqueda.py --k 5 --archivo eval/preguntas.jsonl
"""

import argparse
import logging
import sys
from pathlib import Path

from markwell_rag import db
from markwell_rag.bedrock import BedrockError
from markwell_rag.embeddings import PRECIOS_POR_MILLON, ClienteEmbeddings
from markwell_rag.evaluacion import cargar_preguntas, hit_rate, mrr, posicion_del_primer_acierto
from markwell_rag.repositorio import buscar

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")

parser = argparse.ArgumentParser()
parser.add_argument("--archivo", type=Path, default=Path("eval/preguntas.jsonl"))
parser.add_argument("--k", type=int, default=10, help="fragmentos recuperados por pregunta")
args = parser.parse_args()

try:
    preguntas = cargar_preguntas(args.archivo)
except ValueError as e:
    sys.exit(str(e))
respondibles = [p for p in preguntas if p.respondible]
if not respondibles:
    sys.exit(f"Ninguna pregunta tiene artículos esperados: completa 'esperados' en {args.archivo}.")

cliente = ClienteEmbeddings()
try:
    vectores = cliente.vectorizar_lote([p.pregunta for p in respondibles])
except BedrockError as e:
    sys.exit(f"{type(e).__name__}: {e}")

posiciones: list[int | None] = []
with db.conectar() as conn:
    for p, e in zip(respondibles, vectores):
        recuperados = buscar(conn, e.vector, cliente.model_id, p.fecha, k=args.k)
        posicion = posicion_del_primer_acierto(p.esperados, recuperados)
        posiciones.append(posicion)
        primero = f"{recuperados[0].codigo} {recuperados[0].articulo}" if recuperados else "sin resultados"
        esperados = ", ".join(f"{x.norma} {x.articulo}" for x in p.esperados)
        texto_posicion = "no está" if posicion is None else f"posición {posicion}"
        print(f"{p.id}  {texto_posicion:<12} | 1.º recuperado: {primero:<24} | esperado: {esperados}")

excluidas = [p.id for p in preguntas if not p.respondible]
print(f"\nPreguntas medidas: {len(posiciones)} | sin artículo esperado (excluidas): {excluidas}")
for n in (1, 3, 5):
    if n <= args.k:
        print(f"Hit rate@{n}: {hit_rate(posiciones, n):.2f}")
print(f"MRR@{args.k}: {mrr(posiciones):.3f}")

tokens = sum(e.tokens for e in vectores)
precio = PRECIOS_POR_MILLON.get(cliente.model_id)
costo = f"USD {tokens * precio / 1_000_000:.6f}" if precio else "sin precio conocido"
print(f"Tokens de embeddings: {tokens} | costo estimado: {costo}")
