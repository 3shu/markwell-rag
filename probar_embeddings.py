import logging
import math
import sys

from markwell_rag.bedrock import BedrockError
from markwell_rag.embeddings import ClienteEmbeddings

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

PREGUNTA = "¿En cuánto tiempo debe la aseguradora pagar un reclamo?"

# Frases de ejemplo escritas para el experimento. No son normativa real.
FRASES = {
    "A. Mismo tema, otras palabras": "La compañía tiene un período máximo para entregar la indemnización una vez que el asegurado completa la documentación.",
    "B. Mismas palabras, otro tema": "El plazo para presentar un reclamo en la oficina de atención al cliente es de cinco días.",
    "C. Seguros, otro asunto": "Las primas del seguro de automóviles se calculan según el historial del conductor.",
    "D. Sin relación": "El restaurante del centro abre todos los días a las ocho de la mañana.",
}


def similitud_coseno(a: list[float], b: list[float]) -> float:
    producto = sum(x * y for x, y in zip(a, b))
    return producto / (
        math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    )


dimensiones = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
cliente = ClienteEmbeddings(dimensiones=dimensiones)

try:
    pregunta = cliente.vectorizar(PREGUNTA)
    frases = cliente.vectorizar_lote(list(FRASES.values()))
except BedrockError as e:
    print(f"{type(e).__name__}: {e}")
    sys.exit(1)

norma = math.sqrt(sum(x * x for x in pregunta.vector))
print(f"\nDimensiones: {len(pregunta.vector)} | norma del vector: {norma:.4f}")
print(f"Primeros 5 valores: {[round(x, 4) for x in pregunta.vector[:5]]}")

print(f"\nPregunta: {PREGUNTA}\n")
ranking = sorted(
    zip(FRASES, frases),
    key=lambda par: similitud_coseno(pregunta.vector, par[1].vector),
    reverse=True,
)
for etiqueta, e in ranking:
    print(f"{similitud_coseno(pregunta.vector, e.vector):.4f}  {etiqueta}")

tokens = pregunta.tokens + sum(e.tokens for e in frases)
print(f"\nTokens totales: {tokens}")
