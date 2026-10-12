
# AI Research & Document Assistant

A Retrieval-Augmented Generation (RAG) application that helps
users explore research papers, ask questions, and inspect
the document passages used to support answers.

## Live Demo

Coming soon: add the deployed Streamlit URL here.

## Problem

Research papers can be long and difficult to navigate.
This project explores how document retrieval and a language
model can help readers find relevant information while
keeping the source passages available for verification.

## Features

- Upload PDF research papers
- Extract text with PyMuPDF
- Split document text into searchable passages
- Generate embeddings with Ollama or Gemini
- Retrieve relevant passages using ChromaDB
- Generate answers using Ollama or Gemini
- Display source passages and PDF page numbers

## Architecture

1. PDF text extraction: PyMuPDF
2. Chunking: Python
3. Embeddings: Ollama or Gemini API
4. Vector storage and retrieval: ChromaDB
5. Answer generation: Ollama or Gemini API
6. User interface: Streamlit

## Run locally

Install Ollama and start it on your Mac.

Download the required models:

```bash
ollama pull llama3.2:1b
ollama pull nomic-embed-text
```

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Start the app:

```bash
python -m streamlit run app.py
```

Open the local URL printed in the terminal.

## Cloud deployment

The cloud version uses the Gemini API.
Configure `GEMINI_API_KEY` in the Streamlit app's
Secrets settings. Never commit the API key to GitHub.

Deploy the repository using Streamlit Community Cloud.

## Limitations

- Generated answers may be inaccurate or incomplete.
- Retrieved evidence should be checked against the original.
- Scanned PDFs may require OCR.
- The initial vector collection is held in memory.
- Cloud inference requires a valid API key and available model.
- API usage may be subject to quotas and pricing.

## Future improvements

- Evaluate retrieval using a set of questions with known answers.
- Compare chunk sizes and embedding models.
- Add OCR for scanned PDFs.
- Support multiple documents and persistent storage.
- Improve citation verification and error handling.

## Author


