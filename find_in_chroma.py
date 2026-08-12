import chromadb

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_collection(
    name="wto_knowledge_base"
)

results = collection.get()

found = False

for i, doc in enumerate(results["documents"]):

    if "trade for peace" in doc.lower():

        found = True

        print("\nFOUND!")
        print("ID:", results["ids"][i])
        print("\nDOCUMENT:\n")
        print(doc[:3000])

if not found:
    print("Trade for Peace not found in ChromaDB")