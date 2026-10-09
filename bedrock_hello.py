import os

import boto3
from dotenv import load_dotenv

load_dotenv()

session = boto3.Session(
    profile_name=os.environ["AWS_PROFILE"],
    region_name=os.environ["AWS_REGION"],
)
client = session.client("bedrock-runtime")

response = client.converse(
    modelId=os.environ["BEDROCK_MODEL_ID"],
    messages=[
        {
            "role": "user",
            "content": [
                {"text": "¿Qué dice la normativa vigente de SUGESE sobre el plazo para pagar un reclamo de seguro? Cita el artículo exacto."}
            ],
        }
    ],
    inferenceConfig={"maxTokens": 400, "temperature": 0.2},
)

print(response["output"]["message"]["content"][0]["text"])

usage = response["usage"]
print(f"\nTokens: entrada={usage['inputTokens']}, salida={usage['outputTokens']}")
