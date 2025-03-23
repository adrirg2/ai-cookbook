import os
from dotenv import load_dotenv  # 👈 importante

load_dotenv()  # 👈 carga el archivo .env

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
print("API Key cargada:", os.getenv("OPENAI_API_KEY"))

completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You're a helpful assistant."},
        {"role": "user", "content": "Write a limerick about the Python programming language."},
    ],
)

response = completion.choices[0].message.content
print(response)
