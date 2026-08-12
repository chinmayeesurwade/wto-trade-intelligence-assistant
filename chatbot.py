import os

import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai

# =====================================
# LOAD ENV VARIABLES
# =====================================

load_dotenv()

# =====================================
# GEMINI CLIENT
# =====================================

client_gemini = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# =====================================
# EMBEDDING MODEL
# =====================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# =====================================
# CHROMADB
# =====================================

client_db = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client_db.get_collection(
    name="wto_knowledge_base"
)

# =====================================
# CHAT LOOP
# =====================================

print("\nWTO RAG Assistant Ready")
print("Type 'exit' to quit\n")

while True:

    question = input("You: ")

    if question.lower() == "exit":
        break

    # =====================================
    # QUERY EXPANSION
    # =====================================

    search_query = question

    if "trade for peace" in question.lower():

        search_query = """
        Trade for Peace WTO accession initiative
        fragile conflict affected countries
        Trade for Peace
        WTO membership
        """

    # =====================================
    # EMBEDDING
    # =====================================

    query_embedding = embedding_model.encode(
        search_query
    ).tolist()

    # =====================================
    # RETRIEVE DOCUMENTS
    # =====================================

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=20
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    # =====================================
    # DEBUG OUTPUT
    # =====================================

    print("\n" + "=" * 80)
    print("TOP RETRIEVED DOCUMENTS")
    print("=" * 80)

    for i, doc in enumerate(documents[:3], start=1):

        print(f"\n----- DOCUMENT {i} -----\n")
        print(doc[:500])

    # =====================================
    # CONTEXT
    # =====================================

    context = "\n\n".join(documents)

    # =====================================
    # PROMPT
    # =====================================

    prompt = f"""
You are a WTO trade expert.

Answer ONLY from the WTO context provided.

If the answer exists in the context,
provide a detailed answer.

If the answer is not present,
say:

"I could not find that information in the WTO knowledge base."

WTO CONTEXT:

{context}

QUESTION:

{question}

ANSWER:
"""

    # =====================================
    # GEMINI
    # =====================================

    try:

        response = client_gemini.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        )

        print("\nAssistant:\n")
        print(response.text)

        print("\nSources:")

        seen = set()

        for meta in metadatas:

            url = meta.get("url", "")

            if url and url not in seen:

                print("-", url)
                seen.add(url)

    except Exception as e:

        print("\nERROR:")
        print(e)

    print("\n" + "=" * 60 + "\n")