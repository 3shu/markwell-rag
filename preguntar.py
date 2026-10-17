"""Responde una pregunta con la normativa cargada: recupera, genera y verifica las citas.

Uso:
    uv run python preguntar.py "¿En qué plazo debe responder la aseguradora un reclamo?"
    uv run python preguntar.py "..." --fecha 2024-03-15 --k 5 --ver-contexto
"""

import argparse
import logging
import sys
from datetime import date

from markwell_rag import db
from markwell_rag.bedrock import BedrockError
from markwell_rag.embeddings import ClienteEmbeddings
from markwell_rag.llm import ClienteBedrock
from markwell_rag.rag import Asistente, armar_consulta, texto_final
from markwell_rag.repositorio import buscar

parser = argparse.ArgumentParser()
parser.add_argument("pregunta")
parser.add_argument(
    "--fecha",
    type=date.fromisoformat,
    default=date.today(),
    help="AAAA-MM-DD; por defecto, hoy",
)
parser.add_argument("--k", type=int, default=5)
parser.add_argument(
    "--ver-contexto",
    action="store_true",
    help="imprime la consulta completa enviada al modelo",
)
parser.add_argument(
    "--log", action="store_true", help="muestra el registro JSON de cada llamada"
)
args = parser.parse_args()

logging.basicConfig(
    level=logging.INFO if args.log else logging.WARNING,
    format="%(levelname)s %(name)s %(message)s",
)

with db.conectar() as conn:
    asistente = Asistente(
        embeddings=ClienteEmbeddings(),
        # 1500 tokens de salida: la respuesta estructurada con varias fuentes no
        # cabe con holgura en los 800 por defecto.
        llm=ClienteBedrock(max_tokens=1500),
        recuperar=lambda vector, modelo, fecha, k: buscar(
            conn, vector, modelo, fecha, k=k
        ),
        k=args.k,
    )
    try:
        r = asistente.responder(args.pregunta, args.fecha)
    except BedrockError as e:
        sys.exit(f"{type(e).__name__}: {e}")

print(f"Pregunta: {r.pregunta}")
print(f"Normativa vigente al {r.fecha.isoformat()}\n")

print(f"Fragmentos recuperados ({len(r.recuperados)}):")
for posicion, x in enumerate(r.recuperados, start=1):
    print(f"  {posicion}. {x.codigo}, {x.articulo} | similitud {1 - x.distancia:.4f}")

if args.ver_contexto and r.recuperados:
    print("\n--- Consulta enviada al modelo ---")
    print(armar_consulta(r.pregunta, r.fecha, r.recuperados))
    print("--- Fin de la consulta ---")

print("\nRespuesta:")
print(texto_final(r.respuesta))
print(f"\nSustentada: {'sí' if r.respuesta.sustentada else 'no'}")

if r.citas:
    print("Fuentes:")
for c in r.citas:
    if c.verificada:
        x = c.resultado  # los datos de la cita salen de la base, no del modelo
        hasta = x.vigente_hasta.isoformat() if x.vigente_hasta else "sin fecha de fin"
        print(
            f"  - {x.entidad}, {x.codigo}, {x.articulo} | vigente desde {x.vigente_desde.isoformat()} hasta {hasta}"
        )
        print(f"    {x.fuente_url}")
    else:
        f = c.fuente
        print(
            f"  - ADVERTENCIA, cita no verificada: {f.entidad}, {f.norma}, {f.articulo} no estaba en el contexto"
        )

if r.meta:
    print(
        f"\nModelo: {r.meta.modelo} | tokens {r.meta.tokens_entrada} + {r.meta.tokens_salida} | "
        f"stop_reason {r.meta.stop_reason} | {r.meta.latencia_ms} ms"
    )
else:
    print("\nNo se llamó al modelo.")
costo = f"USD {r.costo_usd:.8f}" if r.costo_usd is not None else "sin precio conocido"
print(f"Costo estimado de la consulta (embedding + modelo): {costo}")
