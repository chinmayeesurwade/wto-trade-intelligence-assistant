import chromadb

client = chromadb.PersistentClient(path="./chroma_db")

collections = client.list_collections()

print("\nCollections found:\n")

for c in collections:
    print(c.name)