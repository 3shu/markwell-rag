"""Juez de fidelidad sobre una corrida de evaluar_fidelidad.py, comparado con las etiquetas manuales.

Lee las respuestas guardadas (no vuelve a llamar al asistente), busca en la base
el texto de los fragmentos que recibió el modelo y le pide al juez un veredicto
por pregunta. Si la pregunta tiene etiqueta manual para esa corrida, los compara.

Uso:
    uv run python juzgar_fidelidad.py eval/resultados/fidelidad-20261024-134942.jsonl --ids P01 P08 P23 P25 P26
    uv run python juzgar_fidelidad.py eval/resultados/<archivo> --modelo <id de otro modelo o perfil>
"""

import argparse
import json
import logging
import sys
from datetime import date, datetime
from pathlib import Path

from markwell_rag import db
from markwell_rag.bedrock import BedrockError
from markwell_rag.juez import PROMPT_JUEZ, FragmentoJuzgado, juzgar, resumir_concordancia
from markwell_rag.llm import ClienteBedrock, SalidaInvalidaError

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")

parser = argparse.ArgumentParser()
parser.add_argument("resultados", type=Path, help="JSON Lines de evaluar_fidelidad.py")
parser.add_argument("--ids", nargs="+", help="juzga solo estas preguntas")
parser.add_argument("--etiquetas", type=Path, default=Path("eval/etiquetas_fidelidad.jsonl"))
parser.add_argument("--modelo", help="modelo del juez; por defecto BEDROCK_MODEL_ID")
parser.add_argument("--salida", type=Path, help="por defecto eval/resultados/juez-<fecha y hora>.jsonl")
args = parser.parse_args()

if not args.resultados.exists():
    sys.exit(f"No existe {args.resultados}")
respuestas = [json.loads(linea) for linea in args.resultados.open(encoding="utf-8") if linea.strip()]
respuestas = [r for r in respuestas if "error" not in r]
if args.ids:
    faltan = set(args.ids) - {r["id"] for r in respuestas}
    if faltan:
        sys.exit(f"No están en {args.resultados.name} (o fueron salidas inválidas): {sorted(faltan)}")
    respuestas = [r for r in respuestas if r["id"] in args.ids]

# Una etiqueta manual vale para la respuesta que se leyó: la clave es (corrida, id).
manuales = {}
if args.etiquetas.exists():
    for linea in args.etiquetas.open(encoding="utf-8"):
        if linea.strip():
            e = json.loads(linea)
            manuales[(e["corrida"], e["id"])] = e["etiqueta"]

salida = args.salida or Path(f"eval/resultados/juez-{datetime.now():%Y%m%d-%H%M%S}.jsonl")
salida.parent.mkdir(parents=True, exist_ok=True)

SQL_FRAGMENTO = """
    SELECT f.texto, v.vigente_desde
    FROM fragmento f
    JOIN version_documento v ON v.id = f.version_id
    JOIN documento d ON d.id = v.documento_id
    WHERE d.codigo = %s AND f.articulo = %s
      AND v.vigente_desde <= %s AND (v.vigente_hasta IS NULL OR %s < v.vigente_hasta)
"""


def fragmentos_de(conn, recuperados: list[str], fecha: date) -> list[FragmentoJuzgado]:
    fragmentos = []
    for etiqueta in recuperados:
        # "SUGESE 03-10 Art. 3" -> ("SUGESE 03-10", "Art. 3"). Supone códigos de dos palabras, como los del catálogo.
        partes = etiqueta.split(" ", 2)
        codigo, articulo = " ".join(partes[:2]), partes[2]
        filas = conn.execute(SQL_FRAGMENTO, (codigo, articulo, fecha, fecha)).fetchall()
        if len(filas) != 1:
            sys.exit(f"{etiqueta} al {fecha}: {len(filas)} fragmentos en la base (se esperaba 1). ¿Cambió la base desde la corrida?")
        texto, vigente_desde = filas[0]
        fragmentos.append(FragmentoJuzgado(codigo, articulo, vigente_desde, texto))
    return fragmentos


evaluador = ClienteBedrock(model_id=args.modelo, system_prompt=PROMPT_JUEZ, max_tokens=2000, temperature=0.0)
print(f"Juez: {evaluador.model_id} | corrida: {args.resultados.name} | preguntas: {len(respuestas)}\n")

pares = []
invalidas = []
desacuerdos = []
costo = 0.0
with db.conectar() as conn, salida.open("w", encoding="utf-8") as archivo:
    for r in respuestas:
        fragmentos = fragmentos_de(conn, r["recuperados"], date.fromisoformat(r["fecha"]))
        try:
            evaluacion, meta = juzgar(evaluador, r["pregunta"], r["respuesta"], fragmentos)
        except SalidaInvalidaError as e:
            invalidas.append(r["id"])
            print(f"{r['id']}  ERROR salida inválida: {e}")
            continue
        except BedrockError as e:
            sys.exit(f"{r['id']}: {type(e).__name__}: {e}")
        costo += meta.costo_usd or 0.0
        manual = manuales.get((args.resultados.name, r["id"]))
        if manual:
            pares.append((manual, evaluacion.veredicto))
            if manual != evaluacion.veredicto:
                desacuerdos.append((r["id"], manual, evaluacion))
        comparacion = "" if manual is None else f" | manual: {manual:<8} | {'coincide' if manual == evaluacion.veredicto else 'NO COINCIDE'}"
        print(f"{r['id']}  juez: {evaluacion.veredicto:<8}{comparacion}")
        print(f"      {evaluacion.justificacion}")
        archivo.write(json.dumps({
            "id": r["id"],
            "corrida": args.resultados.name,
            "modelo_juez": evaluador.model_id,
            "veredicto": evaluacion.veredicto,
            "manual": manual,
            "justificacion": evaluacion.justificacion,
            "afirmaciones": [a.model_dump() for a in evaluacion.afirmaciones],
            "costo_usd": meta.costo_usd,
        }, ensure_ascii=False) + "\n")

print(f"\nSalidas inválidas: {len(invalidas)} {invalidas if invalidas else ''}")
print(f"Con etiqueta manual: {len(pares)}")
for nombre, valor in resumir_concordancia(pares).items():
    print(f"{nombre:<26}{'sin casos' if valor is None else f'{valor:>6.2f}'}")
print(f"\nCosto estimado del juez: USD {costo:.4f}")
print(f"Detalle: {salida}")

for id_, manual, evaluacion in desacuerdos:
    print(f"\n{id_}: manual {manual}, juez {evaluacion.veredicto}")
    for a in evaluacion.afirmaciones:
        print(f"  [{a.respaldo}] {a.texto} ({a.fragmento or 'sin fragmento'})")
