"""Create a temporary ChromaDB collection and search it for relevant text."""

import uuid

import chromadb

from ai_provider import get_embedding
from config import TOP_K


def build_vector_database(chunks: list[dict], progress_bar=None):
    """Embed each chunk and save it in a temporary in-memory ChromaDB collection."""
    if not chunks:
        raise ValueError("There is no PDF text to index.")

    # An in-memory client keeps this simple for a student demo. Data is not
    # intended to persist after the app session or server restarts.
    client = chromadb.Client()
    collection = client.create_collection(name=f"research_{uuid.uuid4().hex[:12]}")

    for index, chunk in enumerate(chunks):
        embedding = get_embedding(chunk["text"])
        collection.add(
            ids=[f"chunk_{index}"],
            documents=[chunk["text"]],
            embeddings=[embedding],
            metadatas=[{"page": chunk["page"], "source": chunk["source"]}],
        )

        # A progress bar is optional, which makes this function easier to test.
        if progress_bar is not None:
            progress_bar.progress(
                (index + 1) / len(chunks),
                text=f"Indexing passage {index + 1} of {len(chunks)}...",
            )

    return client, collection


def retrieve_passages(collection, question: str) -> list[dict]:
    """Find the stored passages whose meaning is closest to the question."""
    count = collection.count()
    if count == 0:
        return []

    question_embedding = get_embedding(question, task_type="RETRIEVAL_QUERY")
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=min(TOP_K, count),
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    # Keep the original text and its source details together for the UI.
    return [
        {
            "text": document,
            "page": metadata.get("page", "Unknown"),
            "source": metadata.get("source", "Unknown"),
            "distance": distance,
        }
        for document, metadata, distance in zip(documents, metadatas, distances)
    ]
