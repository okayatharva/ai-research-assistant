"""Settings used by the AI Research & Document Assistant.

I keep the model names and chunk settings in one place so they are easy to
find and change while learning or testing the project.
"""

import os
import streamlit as st

# Ollama runs on your own computer. Start it before using local mode.
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:1b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# Gemini is optional. If a key is available, the app uses Gemini instead.
GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash")
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")

# Chunk overlap helps keep a sentence or idea together across chunk boundaries.
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
TOP_K = 4


def get_gemini_api_key() -> str:
    """Read the Gemini key from Streamlit secrets or an environment variable."""
    try:
        # This works on Streamlit Community Cloud when a secret is configured.
        key = st.secrets.get("GEMINI_API_KEY", "")
        if key:
            return key
    except Exception:
        # Locally, secrets.toml may not exist. That is okay.
        pass

    return os.getenv("GEMINI_API_KEY", "")


GEMINI_API_KEY = get_gemini_api_key()
USE_GEMINI = bool(GEMINI_API_KEY)
