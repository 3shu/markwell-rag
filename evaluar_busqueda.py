"""Mide la recuperación con las preguntas de evaluación: hit rate y MRR, sin llamar al modelo.

Compara dos órdenes sobre los mismos candidatos: el de la búsqueda vectorial y,
si se pide, el del reranker.

Uso:
    uv run python evaluar_busqueda.py                       # solo búsqueda vectorial (línea base)
    uv run python evaluar_busqueda.py --reranking           # vectorial y re-ranking, lado a lado
    uv run python evaluar_busqueda.py --reranking --candidatos 30 --k 10
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
parser.add_argument("--k", type=int, default=10, help="posiciones que se evalúan")
parser.add_argument("--candidatos", type=int, default=20, help="fragmentos que trae la búsqueda vectorial")
parser.add_argument("--reranking", action="store_true", help="reordena los candidatos con el reranker de Bedrock")
args = parser.parse_args()
if args.candidatos < args.k:
    sys.exit("--candidatos debe ser mayor o igual que --k")

try:
    preguntas = cargar_preguntas(args.archivo)
except ValueError as e:
    sys.exit(str(e))
respondibles = [p for p in preguntas if p.respondible]
if not respondibles:
    sys.exit(f"Ninguna pregunta tiene artículos esperados: completa 'esperados' en {args.archivo}.")

cliente = ClienteEmbeddings()
reranker = None
if args.reranking:
    from markwell_rag.reranking import PRECIOS_POR_MIL_CONSULTAS, ClienteReranking

    reranker = ClienteReranking()


def texto(posicion: int | None) -> str:
    return "no está" if posicion is None else str(posicion)


try:
    vectores = cliente.vectorizar_lote([p.pregunta for p in respondibles])
    posiciones_vector: list[int | None] = []
    posiciones_rerank: list[int | None] = []
    with db.conectar() as conn:
        for p, e in zip(respondibles, vectores):
            candidatos = buscar(conn, e.vector, cliente.model_id, p.fecha, k=args.candidatos)
            # Los primeros k del orden vectorial: es exactamente la línea base de la sesión 9.
            vectorial = candidatos[: args.k]
            pos_v = posicion_del_primer_acierto(p.esperados, vectorial)
            posiciones_vector.append(pos_v)
            esperados = ", ".join(f"{x.norma} {x.articulo}" for x in p.esperados)
            if reranker is None:
                primero = f"{vectorial[0].codigo} {vectorial[0].articulo}" if vectorial else "sin resultados"
                print(f"{p.id}  vectorial: {texto(pos_v):<8} | 1.º: {primero:<24} | esperado: {esperados}")
                continue
            reordenados = [r.resultado for r in reranker.reordenar(p.pregunta, candidatos, top_n=args.k)]
            pos_r = posicion_del_primer_acierto(p.esperados, reordenados)
            posiciones_rerank.append(pos_r)
            primero = f"{reordenados[0].codigo} {reordenados[0].articulo}" if reordenados else "sin resultados"
            print(
                f"{p.id}  vectorial: {texto(pos_v):<8} re-ranking: {texto(pos_r):<8} "
                f"| 1.º re-ranking: {primero:<24} | esperado: {esperados}"
            )
except BedrockError as e:
    sys.exit(f"{type(e).__name__}: {e}")

excluidas = [p.id for p in preguntas if not p.respondible]
print(f"\nPreguntas medidas: {len(posiciones_vector)} | sin artículo esperado (excluidas): {excluidas}")
print(f"Candidatos por pregunta: {args.candidatos} | posiciones evaluadas: {args.k}")
columnas = [("vectorial", posiciones_vector)]
if reranker is not None:
    columnas.append(("re-ranking", posiciones_rerank))
print(f"{'':<14}" + "".join(f"{nombre:>12}" for nombre, _ in columnas))
for n in (1, 3, 5):
    if n <= args.k:
        print(f"{f'Hit rate@{n}':<14}" + "".join(f"{hit_rate(pos, n):>12.2f}" for _, pos in columnas))
print(f"{f'MRR@{args.k}':<14}" + "".join(f"{mrr(pos):>12.3f}" for _, pos in columnas))

tokens = sum(e.tokens for e in vectores)
precio = PRECIOS_POR_MILLON.get(cliente.model_id)
costo = f"USD {tokens * precio / 1_000_000:.6f}" if precio else "sin precio conocido"
print(f"\nTokens de embeddings: {tokens} | costo estimado: {costo}")
if reranker is not None:
    precio_rerank = PRECIOS_POR_MIL_CONSULTAS.get(reranker.model_id)
    if precio_rerank:
        # Una consulta por pregunta (hasta 100 candidatos); mínimo estimado.
        print(f"Consultas de re-ranking: {len(respondibles)} | costo estimado: USD {len(respondibles) * precio_rerank / 1000:.4f}")
