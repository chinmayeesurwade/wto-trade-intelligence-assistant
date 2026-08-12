import os
import chromadb
import streamlit as st

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai

# ==========================
# LOAD ENV
# ==========================

load_dotenv()

# ==========================
# GEMINI CLIENT
# ==========================

client_gemini = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)
# ==========================
# EMBEDDING MODEL
# ==========================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

# ==========================
# CHROMADB
# ==========================

client_db = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client_db.get_collection(
    name="wto_knowledge_base"
)

# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="WTO Trade Intelligence Assistant",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 WTO Trade Intelligence Assistant")

st.caption(
    "Ask questions about WTO accession, trade policy, membership, dispute settlement and WTO agreements."
)

# ==========================
# CHAT HISTORY
# ==========================

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):
            with st.expander("📚 Sources"):
                for source in message["sources"]:
                    st.write(source)

# ==========================
# CHAT INPUT
# ==========================

question = st.chat_input(
    "Ask a WTO question..."
)

# ==========================
# QUERY
# ==========================

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        try:

            # --------------------------
            # EMBEDDING
            # --------------------------

            query_embedding = (
                embedding_model
                .encode(question)
                .tolist()
            )

            # --------------------------
            # SEARCH CHROMA
            # --------------------------

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=10
            )

            documents = results["documents"][0]

            context = "\n\n".join(
                documents[:5]
            )

            # --------------------------
            # PROMPT
            # --------------------------

            prompt = f"""
You are a WTO trade expert.

IMPORTANT RULES:

1. Answer ONLY using the provided WTO context.
2. If information exists in the context, answer it.
3. Do NOT say information is unavailable if it appears in the context.
4. Give a detailed answer.
5. Quote relevant information when needed.
6. Respond naturally like a chatbot.
7. Do not show the retrieved context.

CONTEXT:

{context}

QUESTION:

{question}

ANSWER:
"""

            # --------------------------
            # GEMINI CALL
            # --------------------------

            response = (
                client_gemini.models.generate_content(
                    model="models/gemini-3.5-flash",
                    contents=prompt
                )
            )

            answer = response.text

            st.markdown(answer)

            # --------------------------
            # SOURCES
            # --------------------------

            sources = []

            shown = set()

            for meta in results["metadatas"][0]:

                url = meta.get("url")

                if (
                    url
                    and url not in shown
                ):
                    shown.add(url)
                    sources.append(url)

            if sources:

                with st.expander(
                    "📚 Sources"
                ):
                    for source in sources:
                        st.write(source)

            # --------------------------
            # SAVE MESSAGE
            # --------------------------

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                }
            )

        except Exception as e:

            st.error(
                "Error while generating answer"
            )

            st.error(str(e))