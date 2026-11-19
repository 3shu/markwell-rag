"""Crea negativos sintéticos para calibrar el juez de fidelidad.

Cada negativo parte de una respuesta con etiqueta manual "fiel", cambia un solo texto y deja
documentados el cambio y la etiqueta que le corresponde. El contexto es el mismo que vio el
asistente, así que la etiqueta se conoce por construcción: no depende de lo que diga el juez.

Uso:
    uv run python crear_negativos.py
"""

import json
import sys
from pathlib import Path

ORIGEN = Path("eval/resultados/fidelidad-20261111-153110.jsonl")
SALIDA = Path("eval/negativos-s17.jsonl")
ETIQUETAS = Path("eval/etiquetas_fidelidad.jsonl")

# (id, pregunta de origen, texto original, texto nuevo, etiqueta, por qué)
CAMBIOS = [
    (
        "N01", "P01",
        "dentro de treinta días naturales",
        "dentro de quince días hábiles",
        "no fiel",
        "cambia el plazo, que es la afirmación central; el 06-13 Art. 3 dice treinta días naturales",
    ),
    (
        "N02", "P01",
        "en torno al avance en las investigaciones de siniestros reportados.",
        "en torno al avance en las investigaciones de siniestros reportados. Si la aseguradora no responde "
        "en ese plazo, el reclamo se considera aceptado a favor del consumidor.",
        "parcial",
        "agrega una consecuencia (silencio positivo) que ningún fragmento establece; el plazo sigue respaldado",
    ),
    (
        "N03", "P17",
        "como mínimo una vez al año",
        "como mínimo cuatro veces al año",
        "parcial",
        "cambia la frecuencia de la formación continua, uno de seis puntos; el resto sigue respaldado",
    ),
    (
        "N04", "P17",
        "deben ser aprobados por el Órgano de Dirección y formar parte del Manual de Políticas y Procedimientos.",
        "deben ser aprobados por el Órgano de Dirección y formar parte del Manual de Políticas y Procedimientos. "
        "Según el Reglamento sobre Inclusión y Acceso al Seguro, estos programas deben durar al menos 40 horas.",
        "parcial",
        "atribuye una duración mínima a un reglamento que no está entre los fragmentos (regla 5 del juez)",
    ),
    (
        "N05", "P23",
        "no tienen un plazo de conservación explícitamente establecido en los documentos proporcionados",
        "deben conservarse como mínimo diez años después de cerrado el reclamo",
        "no fiel",
        "inventa el plazo que la pregunta pide; el 06-13 Art. 6 obliga a conservar el expediente sin fijar plazo",
    ),
    (
        "N06", "P25",
        "Los documentos incluidos no contienen una definición explícita de qué es un operador de seguros autoexpedibles.",
        "Un operador de seguros autoexpedibles es un empleado de la entidad aseguradora que vende seguros "
        "autoexpedibles en las oficinas de esta.",
        "no fiel",
        "inventa la definición central; en ese contexto la definición del 03-10 está derogada y no hay otra",
    ),
]

if SALIDA.exists():
    sys.exit(f"{SALIDA} ya existe. No se modificó nada.")
etiquetadas = {
    (e["corrida"], e["id"]): e["etiqueta"]
    for e in map(json.loads, ETIQUETAS.read_text(encoding="utf-8").splitlines())
}
if any(corrida == SALIDA.name for corrida, _ in etiquetadas):
    sys.exit(f"{ETIQUETAS} ya tiene etiquetas de {SALIDA.name}. No se modificó nada.")

respuestas = {r["id"]: r for r in map(json.loads, ORIGEN.read_text(encoding="utf-8").splitlines()) if "error" not in r}
negativos, etiquetas = [], []
for id_, origen, viejo, nuevo, etiqueta, motivo in CAMBIOS:
    if etiquetadas.get((ORIGEN.name, origen)) != "fiel":
        sys.exit(f"{id_}: {origen} no tiene etiqueta manual 'fiel' en {ORIGEN.name}. No se modificó nada.")
    r = respuestas[origen]
    if r["respuesta"].count(viejo) != 1:
        sys.exit(f"{id_}: el texto a cambiar no aparece exactamente una vez en {origen}. No se modificó nada.")
    negativos.append({
        "id": id_,
        "pregunta": r["pregunta"],
        "fecha": r["fecha"],
        "recuperados": r["recuperados"],
        "contexto": r["contexto"],
        "respuesta": r["respuesta"].replace(viejo, nuevo, 1),
        "origen": {"corrida": ORIGEN.name, "id": origen},
        "cambio": {"antes": viejo, "despues": nuevo, "motivo": motivo},
    })
    etiquetas.append({"id": id_, "corrida": SALIDA.name, "etiqueta": etiqueta, "nota": f"sesión 17, sintético desde {origen}: {motivo}"})

SALIDA.write_text("".join(json.dumps(n, ensure_ascii=False) + "\n" for n in negativos), encoding="utf-8")
with ETIQUETAS.open("a", encoding="utf-8") as archivo:
    for e in etiquetas:
        archivo.write(json.dumps(e, ensure_ascii=False) + "\n")
for e in etiquetas:
    print(f"{e['id']}  {e['etiqueta']:<8} {e['nota']}")
print(f"\n{len(negativos)} negativos en {SALIDA}; {len(etiquetas)} etiquetas agregadas a {ETIQUETAS}")
