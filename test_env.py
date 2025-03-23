import os
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
print("Usando dotenv en:", dotenv_path)

load_dotenv(dotenv_path)

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("❌ La API Key NO se ha cargado")
else:
    print("✅ API Key cargada correctamente:", api_key[:10] + "...")
