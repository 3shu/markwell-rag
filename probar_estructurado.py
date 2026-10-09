import logging

from markwell_rag.esquemas import RespuestaRegulatoria
from markwell_rag.llm import BedrockError, ClienteBedrock

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

# Documento inventado solo para probar el flujo de citas. No es normativa real.
DOCUMENTO_FICTICIO = """[Documento de prueba ficticio; no es normativa real]
Entidad: SUGESE
Norma: Circular de prueba PRUEBA-001-2026
Vigente desde: 2026-01-15
Artículo 4. La aseguradora deberá resolver todo reclamo en un plazo máximo de 30 días naturales."""

PREGUNTA = "¿Cuál es el plazo para resolver un reclamo de seguro?"

CASOS = {
    "1. Sin documentos": PREGUNTA,
    "2. Con documento ficticio": f"Documentos:\n{DOCUMENTO_FICTICIO}\n\nPregunta: {PREGUNTA}",
}

cliente = ClienteBedrock()

for titulo, consulta in CASOS.items():
    print(f"\n=== {titulo} ===")
    try:
        r, _ = cliente.consultar_estructurado(consulta, RespuestaRegulatoria)
        print(r.model_dump_json(indent=2))
    except BedrockError as e:
        print(f"{type(e).__name__}: {e}")
