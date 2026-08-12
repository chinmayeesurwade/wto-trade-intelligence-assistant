import os
import json
import shutil

import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# =====================================
# DELETE OLD DATABASE
# =====================================

if os.path.exists("chroma_db"):
    shutil.rmtree("chroma_db")
    print("Old ChromaDB deleted.")

# =====================================
# LOAD EMBEDDING MODEL
# =====================================

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

# =====================================
# CREATE CHROMADB
# =====================================

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_or_create_collection(
    name="wto_knowledge_base"
)

# =====================================
# TEXT SPLITTER
# =====================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

doc_count = 0

# =====================================
# PROCESS FILES
# =====================================

for file in os.listdir("data/cleaned"):

    if not file.endswith(".json"):
        continue

    filepath = os.path.join("data/cleaned", file)

    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    chunks = splitter.split_text(data["content"])

    for i, chunk in enumerate(chunks):

        embedding = model.encode(chunk).tolist()

        collection.add(
            ids=[f"{file}_{i}"],
            documents=[chunk],
            embeddings=[embedding],
            metadatas=[{
                "title": data.get("title", ""),
                "url": data.get("url", "")
            }]
        )

        doc_count += 1

        if doc_count % 50 == 0:
            print(f"Processed {doc_count} chunks...")

print("\n=====================================")
print(f"Stored {doc_count} chunks in ChromaDB")
print("=====================================")