"""Main Streamlit page for the AI Research & Document Assistant.

This file mainly handles what the user sees and clicks. PDF processing,
embeddings, search, and answer generation live in separate Python modules.
"""

import requests
import streamlit as st

from ai_provider import check_connection
from config import USE_GEMINI
from pdf_processor import extract_pdf_chunks
from rag_pipeline import answer_question
from vector_store import build_vector_database


# Page settings must be configured before other Streamlit elements are shown.
st.set_page_config(page_title="AI Research & Document Assistant", page_icon="📚", layout="wide")

st.title("📚 AI Research & Document Assistant")
st.write(
    "Upload a research PDF, ask questions about it, and inspect the passages "
    "used to prepare each answer."
)
st.caption("A student project using Python, Streamlit, embeddings, ChromaDB, and RAG.")

if USE_GEMINI:
    st.info("AI provider: Google Gemini (cloud mode). Document text is sent to the configured API.")
else:
    st.info("AI provider: Ollama (local mode on your computer).")

with st.sidebar:
    st.header("How this app works")
    st.markdown(
        "1. Upload a PDF.\n"
        "2. Extract text and page numbers.\n"
        "3. Split text into overlapping chunks.\n"
        "4. Create embeddings and store them in ChromaDB.\n"
        "5. Retrieve passages related to your question.\n"
        "6. Generate an answer and show the source passages."
    )
    st.divider()
    st.subheader("Privacy and limitations")
    if USE_GEMINI:
        st.write("PDF text is sent to Google's Gemini API. Do not upload confidential documents.")
    else:
        st.write("Ollama processes requests locally, provided your Ollama service is running locally.")
    st.write("The vector index is temporary and may disappear when the app restarts.")
    st.write("Scanned PDFs need OCR, which is not included in this version.")

# Streamlit reruns the script after a click, so session_state keeps the index alive.
if "collection" not in st.session_state:
    st.session_state.collection = None
if "chroma_client" not in st.session_state:
    st.session_state.chroma_client = None
if "indexed_filename" not in st.session_state:
    st.session_state.indexed_filename = None

connection_ok, connection_message = check_connection()
if not connection_ok:
    st.warning(connection_message)
    if not USE_GEMINI:
        st.code("ollama serve\nollama pull llama3.2:1b\nollama pull nomic-embed-text", language="bash")

uploaded_file = st.file_uploader("Upload a research paper (PDF)", type=["pdf"])
if uploaded_file is not None:
    st.write(f"**Selected file:** {uploaded_file.name}")

    if st.button("Process PDF", type="primary", disabled=not connection_ok):
        try:
            with st.spinner("Reading the PDF and splitting its text..."):
                chunks = extract_pdf_chunks(uploaded_file.getvalue(), uploaded_file.name)

            if not chunks:
                st.error("No selectable text was found. This may be a scanned PDF; OCR is not included yet.")
            else:
                st.success(f"Extracted {len(chunks)} text passages from the PDF.")
                progress_bar = st.progress(0, text="Preparing embeddings...")
                with st.spinner("Creating embeddings and indexing passages..."):
                    client, collection = build_vector_database(chunks, progress_bar)

                st.session_state.chroma_client = client
                st.session_state.collection = collection
                st.session_state.indexed_filename = uploaded_file.name
                st.success("PDF processed. You can now ask questions about it.")

        except requests.HTTPError as error:
            # Show a short provider response to make common API errors easier to debug.
            details = error.response.text[:1000] if error.response is not None else ""
            st.error("The AI provider returned an HTTP error. Check model access, API key, quota, and model name.")
            if details:
                st.code(details)
        except Exception as error:
            st.error(f"Could not process the PDF: {error}")

st.divider()
st.subheader("Ask a question")
if st.session_state.collection is None:
    st.info("Upload a PDF and click **Process PDF** before asking questions.")
else:
    st.caption(f"Currently indexed: {st.session_state.indexed_filename}")
    question = st.text_area(
        "Your question",
        placeholder="Example: What research problem are the authors trying to solve?",
        height=100,
    )

    if st.button("Get answer", type="primary"):
        if not question.strip():
            st.warning("Please enter a question first.")
        else:
            try:
                with st.spinner("Finding useful passages and preparing an answer..."):
                    answer, sources = answer_question(st.session_state.collection, question.strip())

                st.subheader("Answer")
                st.markdown(answer)
                st.subheader("Retrieved source passages")
                st.caption("Check these passages against the original PDF; AI answers can still be wrong.")

                for index, source in enumerate(sources, start=1):
                    label = f"Source {index} — {source['source']}, page {source['page']}"
                    with st.expander(label):
                        st.write(source["text"])
            except requests.HTTPError as error:
                details = error.response.text[:1000] if error.response is not None else ""
                st.error("The AI request failed. Check the provider, model, API key, or quota.")
                if details:
                    st.code(details)
            except Exception as error:
                st.error(f"Could not answer the question: {error}")

st.divider()
st.caption("Portfolio project • RAG research assistant • Always verify answers using the original paper.")
