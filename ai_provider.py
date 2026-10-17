"""Small helpers for talking to Gemini or a local Ollama installation."""

import requests

from config import (
    GEMINI_API_KEY,
    GEMINI_CHAT_MODEL,
    GEMINI_EMBED_MODEL,
    OLLAMA_CHAT_MODEL,
    OLLAMA_EMBED_MODEL,
    OLLAMA_URL,
    USE_GEMINI,
)


def check_connection() -> tuple[bool, str]:
    """Check whether the selected provider is configured and ready to use."""
    if USE_GEMINI:
        return True, "Gemini API key found."

    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        response.raise_for_status()
        names = [model.get("name", "") for model in response.json().get("models", [])]

        # Ollama model names may include a tag, for example model:latest.
        has_chat = any(name == OLLAMA_CHAT_MODEL or name.startswith(OLLAMA_CHAT_MODEL + ":") for name in names)
        has_embedding = any(name == OLLAMA_EMBED_MODEL or name.startswith(OLLAMA_EMBED_MODEL + ":") for name in names)

        if not has_chat or not has_embedding:
            return False, (
                "Ollama is running, but a required model is missing. Run "
                f"'ollama pull {OLLAMA_CHAT_MODEL}' and "
                f"'ollama pull {OLLAMA_EMBED_MODEL}' in Terminal."
            )
        return True, "Ollama is running and both required models are available."
    except requests.RequestException:
        return False, "Cannot connect to Ollama. Start Ollama or configure GEMINI_API_KEY for Gemini mode."


def get_embedding(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
    """Turn text into a list of numbers that represents its meaning (embedding)."""
    if USE_GEMINI:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_EMBED_MODEL}:embedContent"
        payload = {
            "model": f"models/{GEMINI_EMBED_MODEL}",
            "content": {"parts": [{"text": text}]},
            "taskType": task_type,
        }
        response = requests.post(
            url,
            headers={"x-goog-api-key": GEMINI_API_KEY},
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["embedding"]["values"]

    response = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": OLLAMA_EMBED_MODEL, "input": text},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["embeddings"][0]


def generate_answer(prompt: str) -> str:
    """Send a prompt to the selected language model and return its answer."""
    if USE_GEMINI:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_CHAT_MODEL}:generateContent"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 1200},
        }
        response = requests.post(
            url,
            headers={"x-goog-api-key": GEMINI_API_KEY},
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        candidates = response.json().get("candidates", [])
        if not candidates:
            raise ValueError("Gemini did not return an answer. Try a shorter question.")
        parts = candidates[0].get("content", {}).get("parts", [])
        answer = "\n".join(part.get("text", "") for part in parts if part.get("text"))
    else:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": OLLAMA_CHAT_MODEL, "prompt": prompt, "stream": False},
            timeout=180,
        )
        response.raise_for_status()
        answer = response.json().get("response", "")

    answer = answer.strip()
    if not answer:
        raise ValueError("The AI provider returned an empty answer.")
    return answer
