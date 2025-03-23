import json
import os
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

# --------------------------------------------------------------
# 🔐 Paso 0: Cargar API Key desde .env
# --------------------------------------------------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
print("✅ Cliente OpenAI inicializado")

# --------------------------------------------------------------
# 🧠 Paso 1: Definir función de búsqueda en knowledge base
# --------------------------------------------------------------

def search_kb(question: str):
    print(f"\n🔍 Buscando información para: '{question}'")

    # Construir ruta absoluta a kb.json
    script_dir = os.path.dirname(__file__)
    kb_path = os.path.join(script_dir, "kb.json")
    print(f"📂 Intentando abrir archivo: {kb_path}")

    # Cargar JSON
    with open(kb_path, "r") as f:
        kb_data = json.load(f)
    print("📦 Datos cargados del archivo:")
    print(json.dumps(kb_data, indent=2))
    return kb_data

# --------------------------------------------------------------
# 🧠 Paso 2: Preparar herramientas y mensaje inicial
# --------------------------------------------------------------

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_kb",
            "description": "Get the answer to the user's question from the knowledge base.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                },
                "required": ["question"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
]

system_prompt = "You are a helpful assistant that answers questions from the knowledge base about our e-commerce store."

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "What is the return policy?"},
]

print("\n📨 Enviando pregunta al modelo con herramientas: ", messages)

completion = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools,
)

# --------------------------------------------------------------
# 🔎 Paso 3: El modelo decide si usar una herramienta
# --------------------------------------------------------------
print("\n🤖 Respuesta del modelo:")
print(json.dumps(completion.model_dump(), indent=2))

# Verificar tool_calls
tool_calls = completion.choices[0].message.tool_calls
if not tool_calls:
    print("⚠️ El modelo no usó ninguna herramienta.")
else:
    print(f"🛠️ Herramientas a ejecutar: {[t.function.name for t in tool_calls]}")

# --------------------------------------------------------------
# ⚙️ Paso 4: Ejecutar función solicitada por el modelo
# --------------------------------------------------------------

def call_function(name, args):
    if name == "search_kb":
        return search_kb(**args)

for tool_call in tool_calls:
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    print("messages before append", messages)
    # Guardar mensaje original
    messages.append(completion.choices[0].message)
    print("messages after append", messages)
    result = call_function(name, args)
    print(f"✅ Resultado de la función '{name}':", result)

    # Agregar respuesta a la conversación
    messages.append(
        {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result)}
    )
    print("messages tools: ", messages)

# --------------------------------------------------------------
# 🧠 Paso 5: Enviar resultado al modelo para que genere la respuesta final
# --------------------------------------------------------------

class KBResponse(BaseModel):
    answer: str = Field(description="The answer to the user's question.")
    source: int = Field(description="The record id of the answer.")

completion_2 = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=messages,
    tools=tools,
    response_format=KBResponse,
)

final_response = completion_2.choices[0].message.parsed

print("\n🎯 Respuesta final del modelo: ", completion_2)
print("🗨️  Respuesta:", final_response.answer)
print("📎 Fuente (ID):", final_response.source)

# --------------------------------------------------------------
# 🌤️ Extra: Pregunta fuera del dominio
# --------------------------------------------------------------

print("\n🌤️ Pregunta fuera del scope de la KB...")

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "What is the weather in Tokyo?"},
]

print("messages last: ", messages)

completion_3 = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=messages,
    tools=tools,
)

print("🧠 Respuesta directa del modelo:")
print(completion_3.choices[0].message.content)
