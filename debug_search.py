import chromadb
from sentence_transformers import SentenceTransformer

print("Loading model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_collection(
    "wto_knowledge_base"
)

query = "Trade for Peace WTO accession"

embedding = model.encode(
    query
).tolist()

results = collection.query(
    query_embeddings=[embedding],
    n_results=50
)

found = False

for i, doc in enumerate(results["documents"][0]):

    if "trade for peace" in doc.lower():

        print("\n================================")
        print("FOUND AT POSITION:", i + 1)
        print("================================\n")

        print(doc)

        found = True

if not found:
    print("\nTRADE FOR PEACE NOT FOUND IN TOP 50 RESULTS")