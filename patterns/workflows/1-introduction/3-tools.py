import json
import os
import requests
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

# 🔐 Cargar variables de entorno (como OPENAI_API_KEY)
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

print("✅ Cliente OpenAI inicializado")

# --------------------------------------------------------------
# 🧰 Definición de la función externa que usaremos
# --------------------------------------------------------------

def get_weather(latitude, longitude):
    """Consulta clima en una API pública usando latitud y longitud."""
    print(f"🌍 Llamando a get_weather con lat: {latitude}, lon: {longitude}")
    response = requests.get(
        f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
    )
    data = response.json()
    print("📦 Respuesta cruda de la API del tiempo:", json.dumps(data, indent=2))
    return data["current"]

# --------------------------------------------------------------
# 🧠 Paso 1: El modelo recibe la instrucción y las herramientas
# --------------------------------------------------------------

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current temperature for provided coordinates in celsius.",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number"},
                    "longitude": {"type": "number"},
                    "date": {"type": "number"},
                },
                "required": ["latitude", "longitude"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
]

system_prompt = "You are a helpful weather assistant."

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "What's the weather like in Paris today?"},
]

print("\n💬 Enviando mensaje inicial al modelo con herramientas...")

completion = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools,
)

# --------------------------------------------------------------
# 🕵️‍♂️ Paso 2: El modelo decide si usar una herramienta
# --------------------------------------------------------------

print("\n🤖 El modelo respondió:")
print(json.dumps(completion.model_dump(), indent=2))

tool_calls = completion.choices[0].message.tool_calls
if not tool_calls:
    print("⚠️ El modelo no pidió usar ninguna herramienta.")
else:
    print(f"🛠️ El modelo pidió usar {len(tool_calls)} herramienta(s):")

# --------------------------------------------------------------
# ⚙️ Paso 3: Ejecutamos la función solicitada
# --------------------------------------------------------------

def call_function(name, args):
    print(f"\n📞 Llamando a función '{name}' con argumentos:", args)
    if name == "get_weather":
        return get_weather(**args)

for tool_call in tool_calls:
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)

    # Añadir mensaje de tool_call a la conversación
    messages.append(completion.choices[0].message)

    result = call_function(name, args)
    print(f"✅ Resultado de {name}:", result)

    # Añadir resultado como mensaje de tool
    messages.append(
        {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result)}
    )

# --------------------------------------------------------------
# 🧠 Paso 4: Volvemos a llamar al modelo con el resultado
# --------------------------------------------------------------

class WeatherResponse(BaseModel):
    temperature: float = Field(
        description="The current temperature in celsius for the given location."
    )
    response: str = Field(
        description="A natural language response to the user's question."
    )

print("\n🔁 Enviando resultado de la herramienta de vuelta al modelo...")

completion_2 = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=messages,
    tools=tools,
    response_format=WeatherResponse,
)

# --------------------------------------------------------------
# ✅ Paso 5: Mostramos la respuesta final
# --------------------------------------------------------------

final_response = completion_2.choices[0].message.parsed

print("\n🎉 Respuesta estructurada del modelo:")
print("🌡️  Temperatura:", final_response.temperature)
print("🗨️  Respuesta natural:", final_response.response)
