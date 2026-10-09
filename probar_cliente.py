import logging

from markwell_rag.llm import BedrockError, ClienteBedrock

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

PREGUNTA = "¿Qué dice la normativa vigente de SUGESE sobre el plazo para pagar un reclamo de seguro?"

cliente = ClienteBedrock()

print("=== 1. Sin streaming ===")
r = cliente.consultar(PREGUNTA)
print(r.texto)

print("\n=== 2. Con streaming ===")
r = cliente.consultar_stream(PREGUNTA)
print()

print("\n=== 3. Errores controlados ===")
for modelo in ["us.anthropic.modelo-inexistente", "us.anthropic.claude-haiku-5-5"]:
    try:
        ClienteBedrock(model_id=modelo).consultar("Hola")
        print(f"{modelo}: respondió correctamente")
    except BedrockError as e:
        print(f"{modelo}: {type(e).__name__}: {e}")
