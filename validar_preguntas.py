"""Revisa eval/preguntas.jsonl antes de medir: formato, ids únicos y que cada artículo esperado
exista como fragmento vigente en la fecha de la pregunta.

Uso:
    uv run python validar_preguntas.py
"""

import json
import sys
from datetime import date
from pathlib import Path

from markwell_rag import db

RUTA = Path("eval/preguntas.jsonl")
CAMPOS = {"id", "pregunta", "fecha", "esperados", "nota"}

errores: list[str] = []
preguntas = []
for n, linea in enumerate(RUTA.read_text(encoding="utf-8").splitlines(), 1):
    if not linea.strip():
        continue
    try:
        p = json.loads(linea)
    except json.JSONDecodeError as e:
        errores.append(f"línea {n}: JSON inválido ({e})")
        continue
    faltan = CAMPOS - p.keys()
    if faltan:
        errores.append(f"línea {n}: faltan los campos {sorted(faltan)}")
        continue
    try:
        p["fecha"] = date.fromisoformat(p["fecha"])
    except ValueError:
        errores.append(f"{p['id']}: fecha inválida {p['fecha']!r} (formato AAAA-MM-DD)")
        continue
    preguntas.append(p)

ids = [p["id"] for p in preguntas]
for id_ in sorted({i for i in ids if ids.count(i) > 1}):
    errores.append(f"{id_}: id repetido")

with db.conectar() as conn:
    for p in preguntas:
        for e in p["esperados"]:
            (n,) = conn.execute(
                """
                SELECT count(*)
                FROM fragmento f
                JOIN version_documento v ON v.id = f.version_id
                JOIN documento d ON d.id = v.documento_id
                WHERE d.codigo = %s AND f.articulo = %s
                  AND v.vigente_desde <= %s AND (v.vigente_hasta IS NULL OR v.vigente_hasta > %s)
                """,
                (e["norma"], e["articulo"], p["fecha"], p["fecha"]),
            ).fetchone()
            if n != 1:
                errores.append(f"{p['id']}: {e['norma']} {e['articulo']} no es un fragmento vigente al {p['fecha']} ({n} filas)")

respondibles = sum(1 for p in preguntas if p["esperados"])
print(f"Preguntas: {len(preguntas)} | respondibles: {respondibles} | controles: {len(preguntas) - respondibles}")
for error in errores:
    print(f"  - {error}")
if errores:
    sys.exit(f"{len(errores)} problema(s). Corrige eval/preguntas.jsonl antes de medir.")
print("Sin problemas.")
