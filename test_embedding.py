from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2"
)

vector = embeddings.embed_query(
    "What is the World Trade Organization?"
)

print(len(vector))
print(vector[:10])