import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client_gemini = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

print("\nAVAILABLE MODELS:\n")

for model in client.models.list():
    print(model.name)