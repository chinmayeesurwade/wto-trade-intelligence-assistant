import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_collection(
    name="wto_knowledge_base"
)

query = "Trade for Peace"

query_embedding = model.encode(query).tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=10
)

for i, doc in enumerate(results["documents"][0], start=1):
    print(f"\n===== RESULT {i} =====\n")
    print(doc[:1500])