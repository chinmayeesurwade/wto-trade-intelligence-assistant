import os
import time

import chromadb
import streamlit as st

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="WTO Trade Intelligence Assistant",
    page_icon="🌐",
    layout="wide"
)


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error(
        "Gemini API key is not configured. "
        "Please add GEMINI_API_KEY to Streamlit Secrets."
    )
    st.stop()


# ============================================================
# GEMINI CLIENT
# ============================================================

client_gemini = genai.Client(
    api_key=GEMINI_API_KEY
)

MODEL_NAME = "models/gemini-3.5-flash"


# ============================================================
# EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


embedding_model = load_embedding_model()


# ============================================================
# CHROMADB
# ============================================================

import os

DB_PATH = os.path.join(
    os.path.dirname(__file__),
    "chroma_db"
)

client_db = chromadb.PersistentClient(
    path=DB_PATH
)
collection = client_db.get_collection(
    name="wto_knowledge_base"
)


# ============================================================
# PAGE HEADER
# ============================================================

st.title("🌐 WTO Trade Intelligence Assistant")

st.caption(
    "Ask questions about WTO accession, trade policy, membership, "
    "dispute settlement and WTO agreements."
)


# ============================================================
# CHAT HISTORY
# ============================================================

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


# ============================================================
# GEMINI RESPONSE FUNCTION
# ============================================================

def generate_gemini_answer(prompt, max_retries=3):

    for attempt in range(max_retries):

        try:

            response = client_gemini.models.generate_content(
                model=MODEL_NAME,
                contents=prompt
            )

            return response.text

        except Exception as e:

            error_message = str(e)

            # ------------------------------------------------
            # QUOTA EXCEEDED
            # ------------------------------------------------

            if "429" in error_message or "RESOURCE_EXHAUSTED" in error_message:

                if attempt < max_retries - 1:

                    wait_time = 5 * (attempt + 1)

                    time.sleep(wait_time)

                    continue

                return None, "quota"

            # ------------------------------------------------
            # TEMPORARY SERVICE UNAVAILABLE
            # ------------------------------------------------

            if "503" in error_message or "UNAVAILABLE" in error_message:

                if attempt < max_retries - 1:

                    wait_time = 3 * (attempt + 1)

                    time.sleep(wait_time)

                    continue

                return None, "unavailable"

            # ------------------------------------------------
            # OTHER ERROR
            # ------------------------------------------------

            return None, "other"

    return None, "other"


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask a WTO question..."
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    # --------------------------------------------------------
    # SAVE USER MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)


    # --------------------------------------------------------
    # ASSISTANT RESPONSE
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        try:

            # =================================================
            # 1. CREATE QUESTION EMBEDDING
            # =================================================

            query_embedding = (
                embedding_model
                .encode(question)
                .tolist()
            )


            # =================================================
            # 2. SEARCH CHROMADB
            # =================================================

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=10
            )

            documents = results["documents"][0]


            # =================================================
            # 3. BUILD CONTEXT
            # =================================================

            context = "\n\n".join(
                documents[:5]
            )


            # =================================================
            # 4. BUILD PROMPT
            # =================================================

            prompt = f"""
You are a WTO trade expert.

IMPORTANT RULES:

1. Answer ONLY using the provided WTO context.
2. If the information exists in the context, answer the question.
3. Do NOT invent facts that are not supported by the context.
4. Give a clear and useful answer.
5. Use bullet points or numbered lists when appropriate.
6. Explain WTO terminology when necessary.
7. Do not reveal the retrieved context.
8. If the context genuinely does not contain enough information, clearly say that the available WTO knowledge base does not contain enough information to answer confidently.

WTO CONTEXT:

{context}

QUESTION:

{question}

ANSWER:
"""


            # =================================================
            # 5. GENERATE GEMINI RESPONSE
            # =================================================

            result = generate_gemini_answer(
                prompt,
                max_retries=3
            )


            # =================================================
            # 6. HANDLE GEMINI RESULT
            # =================================================

            if isinstance(result, tuple):

                answer, error_type = result

            else:

                answer = result
                error_type = None


            # ------------------------------------------------
            # QUOTA ERROR
            # ------------------------------------------------

            if error_type == "quota":

                answer = (
                    "⚠️ The AI generation service has temporarily "
                    "reached its usage limit. Your WTO knowledge base "
                    "is working correctly, but the Gemini API quota "
                    "has been exhausted. Please try again later."
                )

                st.warning(answer)


            # ------------------------------------------------
            # SERVICE UNAVAILABLE
            # ------------------------------------------------

            elif error_type == "unavailable":

                answer = (
                    "⚠️ The AI generation service is temporarily "
                    "unavailable. Please try again in a few moments."
                )

                st.warning(answer)


            # ------------------------------------------------
            # OTHER ERROR
            # ------------------------------------------------

            elif error_type == "other":

                answer = (
                    "⚠️ I couldn't generate the answer because "
                    "the AI service returned an unexpected error. "
                    "Please try again."
                )

                st.warning(answer)


            # ------------------------------------------------
            # SUCCESS
            # ------------------------------------------------

            else:

                st.markdown(answer)


            # =================================================
            # 7. COLLECT SOURCES
            # =================================================

            sources = []

            shown = set()

            for meta in results["metadatas"][0]:

                if meta:

                    url = meta.get("url")

                    if (
                        url
                        and url not in shown
                    ):

                        shown.add(url)

                        sources.append(url)


            # =================================================
            # 8. DISPLAY SOURCES
            # =================================================

            if sources:

                with st.expander("📚 Sources"):

                    for source in sources:
                        st.write(source)


            # =================================================
            # 9. SAVE ASSISTANT MESSAGE
            # =================================================

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                }
            )


        except Exception:

            answer = (
                "⚠️ Something went wrong while processing "
                "your question. Please try again."
            )

            st.error(answer)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": []
                }
            )