import json
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

total_chunks = 0

for file in os.listdir("data/cleaned"):

    if not file.endswith(".json"):
        continue

    path = os.path.join("data/cleaned", file)

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    chunks = splitter.split_text(data["content"])

    total_chunks += len(chunks)

print(f"Total chunks created: {total_chunks}")