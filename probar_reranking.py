"""Primera llamada al reranker con las frases del experimento de la sesión 5 (no usa la base de datos)."""

import logging
import sys
from datetime import date

from markwell_rag.bedrock import BedrockError
from markwell_rag.repositorio import Resultado
from markwell_rag.reranking import ClienteReranking

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

PREGUNTA = "¿En cuánto tiempo debe la aseguradora pagar un reclamo?"

# Las mismas frases de probar_embeddings.py. No son normativa real.
FRASES = {
    "A. Mismo tema, otras palabras": "La compañía tiene un período máximo para entregar la indemnización una vez que el asegurado completa la documentación.",
    "B. Mismas palabras, otro tema": "El plazo para presentar un reclamo en la oficina de atención al cliente es de cinco días.",
    "C. Seguros, otro asunto": "Las primas del seguro de automóviles se calculan según el historial del conductor.",
    "D. Sin relación": "El restaurante del centro abre todos los días a las ocho de la mañana.",
}

# El reranker recibe Resultado, como los que devuelve buscar(); aquí los armamos a mano.
candidatos = [
    Resultado(
        fragmento_id=i,
        entidad="PRUEBA",
        codigo="PRUEBA",
        titulo="Frase de ejemplo",
        articulo=etiqueta,
        texto=texto,
        vigente_desde=date(2026, 1, 1),
        vigente_hasta=None,
        fuente_url="-",
        distancia=0.0,
    )
    for i, (etiqueta, texto) in enumerate(FRASES.items(), start=1)
]

try:
    reordenados = ClienteReranking().reordenar(
        PREGUNTA, candidatos, top_n=len(candidatos)
    )
except BedrockError as e:
    print(f"{type(e).__name__}: {e}")
    sys.exit(1)

print(f"\nPregunta: {PREGUNTA}")
for r in reordenados:
    print(f"{r.relevancia:.4f}  {r.resultado.articulo}")
