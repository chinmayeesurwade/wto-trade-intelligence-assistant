import chromadb

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_collection(
    name="wto_knowledge_base"
)

print("Total documents:", collection.count())