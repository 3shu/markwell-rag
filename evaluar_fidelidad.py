"""Mide la fidelidad de las respuestas del asistente con las preguntas de evaluación.

Corre el asistente completo (búsqueda, re-ranking y modelo) sobre cada pregunta
y mide en código: si la respuesta se declara sustentada, cuántas citas se
verifican contra el contexto y si cita alguno de los artículos esperados.

Uso:
    uv run python evaluar_fidelidad.py
    uv run python evaluar_fidelidad.py --ids P10 P15 P20     # solo algunas preguntas
    uv run python evaluar_fidelidad.py --sin-reranking
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from markwell_rag import db
from markwell_rag.bedrock import BedrockError
from markwell_rag.embeddings import ClienteEmbeddings
from markwell_rag.evaluacion import cargar_preguntas, medir_fidelidad, resumir_fidelidad
from markwell_rag.llm import ClienteBedrock, SalidaInvalidaError
from markwell_rag.rag import Asistente
from markwell_rag.repositorio import buscar

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")

parser = argparse.ArgumentParser()
parser.add_argument("--archivo", type=Path, default=Path("eval/preguntas.jsonl"))
parser.add_argument("--ids", nargs="+", help="evalúa solo estas preguntas")
parser.add_argument("--k", type=int, default=5, help="fragmentos que recibe el modelo")
parser.add_argument("--candidatos", type=int, default=10, help="fragmentos que trae la búsqueda vectorial para el reranker")
parser.add_argument("--sin-reranking", action="store_true")
parser.add_argument("--salida", type=Path, help="JSON Lines con el detalle; por defecto eval/resultados/fidelidad-<fecha y hora>.jsonl")
args = parser.parse_args()

try:
    preguntas = cargar_preguntas(args.archivo)
except ValueError as e:
    sys.exit(str(e))
if args.ids:
    faltan = set(args.ids) - {p.id for p in preguntas}
    if faltan:
        sys.exit(f"No existen en {args.archivo}: {sorted(faltan)}")
    preguntas = [p for p in preguntas if p.id in args.ids]

salida = args.salida or Path(f"eval/resultados/fidelidad-{datetime.now():%Y%m%d-%H%M%S}.jsonl")
salida.parent.mkdir(parents=True, exist_ok=True)

reordenador = None
if not args.sin_reranking:
    from markwell_rag.reranking import ClienteReranking

    reordenador = ClienteReranking()


def si_no(valor: bool) -> str:
    return "sí" if valor else "no"


mediciones = []
errores: list[str] = []
revisar = []
costo_total = 0.0
with db.conectar() as conn, salida.open("w", encoding="utf-8") as archivo:
    asistente = Asistente(
        embeddings=ClienteEmbeddings(),
        # temperature=0: menos variación entre corridas. No garantiza respuestas idénticas.
        llm=ClienteBedrock(max_tokens=1500, temperature=0.0),
        recuperar=lambda vector, modelo, fecha, k: buscar(conn, vector, modelo, fecha, k=k),
        k=args.k,
        reordenador=reordenador,
        candidatos=args.candidatos,
    )
    for p in preguntas:
        try:
            r = asistente.responder(p.pregunta, p.fecha)
        except SalidaInvalidaError as e:
            # Una salida que no cumple el esquema es un fallo de la respuesta, no del script: se cuenta y se sigue.
            errores.append(p.id)
            print(f"{p.id}  ERROR salida inválida: {e}")
            archivo.write(json.dumps({"id": p.id, "error": str(e)}, ensure_ascii=False) + "\n")
            continue
        except BedrockError as e:
            sys.exit(f"{p.id}: {type(e).__name__}: {e}")

        m = medir_fidelidad(p, r)
        mediciones.append(m)
        costo_total += r.costo_usd or 0.0
        tipo = "respondible" if p.respondible else "control"
        print(
            f"{p.id}  {tipo:<11} | sustentada: {si_no(m.sustentada):<2} | citas verificadas: {m.citas_verificadas}/{m.citas} "
            f"| cita esperado: {si_no(m.cita_esperada) if p.respondible else '-':<2} "
            f"| esperado en contexto: {si_no(m.contexto_esperado) if p.respondible else '-'}"
        )
        if m.alertas:
            revisar.append((p, r, m))
        archivo.write(json.dumps({
            "id": p.id,
            "pregunta": p.pregunta,
            "fecha": p.fecha.isoformat(),
            "esperados": [f"{e.norma} {e.articulo}" for e in p.esperados],
            "recuperados": [f"{x.codigo} {x.articulo}" for x in r.recuperados],
            "sustentada": m.sustentada,
            "respuesta": r.respuesta.respuesta,
            "citas": [
                {"norma": c.fuente.norma, "articulo": c.fuente.articulo, "verificada": c.verificada}
                for c in r.citas
            ],
            "alertas": m.alertas,
            "costo_usd": r.costo_usd,
        }, ensure_ascii=False) + "\n")

if not mediciones:
    sys.exit("Ninguna pregunta se pudo medir.")

controles = [m.id for m in mediciones if not m.respondible]
print(f"\nPreguntas: {len(preguntas)} | respondibles: {len(mediciones) - len(controles)} | controles: {controles}")
print(f"Salidas inválidas: {len(errores)} {errores if errores else ''}")
for nombre, valor in resumir_fidelidad(mediciones).items():
    print(f"{nombre:<32}{'sin casos' if valor is None else f'{valor:>6.2f}'}")
print(f"\nCosto estimado (embeddings + reranker + modelo): USD {costo_total:.4f}")
print(f"Detalle: {salida}")

if revisar:
    print("\nPara revisar:")
for p, r, m in revisar:
    citas = ", ".join(f"{c.fuente.norma} {c.fuente.articulo}{'' if c.verificada else ' (NO VERIFICADA)'}" for c in r.citas)
    print(f"\n{p.id}: {'; '.join(m.alertas)}")
    print(f"  Pregunta: {p.pregunta}")
    print(f"  Esperados: {', '.join(f'{e.norma} {e.articulo}' for e in p.esperados) or '(control)'}")
    print(f"  Contexto: {', '.join(f'{x.codigo} {x.articulo}' for x in r.recuperados)}")
    print(f"  Citas: {citas or '(ninguna)'}")
    print(f"  Respuesta: {r.respuesta.respuesta}")
