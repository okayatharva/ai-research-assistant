# AI Research & Document Assistant

A Retrieval-Augmented Generation (RAG) application that helps users explore research papers and PDF documents by retrieving relevant passages and generating AI-powered answers.

## Features

- Upload and process PDF documents
- Extract text and page information using PyMuPDF
- Split documents into searchable text chunks
- Generate embeddings and perform semantic search
- Retrieve relevant passages using ChromaDB
- Ask questions about uploaded research papers
- Generate answers using an AI language model
- Inspect retrieved source passages
- Interactive web interface built with Streamlit
- Support for Google Gemini API and local Ollama inference

## How It Works

1. Upload a PDF document.
2. Extract its text and page information.
3. Split the text into smaller chunks.
4. Create embeddings for semantic search.
5. Retrieve relevant passages from ChromaDB.
6. Generate an answer based on the retrieved context.
7. Review the answer and supporting passages.

**Architecture:**

PDF Upload → Text Extraction → Chunking → Embeddings → ChromaDB Retrieval → AI Answer Generation → Source Passages

## Technology Stack

- Python
- Streamlit
- PyMuPDF
- ChromaDB
- Retrieval-Augmented Generation (RAG)
- Google Gemini API
- Ollama

## Limitations

- Scanned PDFs without extractable text may not work correctly.
- AI-generated answers can be incomplete or incorrect.
- Retrieved passages should be checked against the original document.
- Cloud inference may send document text to the configured AI provider.
- Local Ollama inference requires Ollama to be installed and running.

## Live Demo

The public Streamlit demo link will be added after deployment is verified.

## Run Locally

Clone the repository:

```bash
git clone https://github.com/okayatharva/ai-research-assistant.git
cd ai-research-assistant
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app.py
```

Configure any required API keys through environment variables or Streamlit secrets. Never commit API keys or credentials to GitHub.

## Author

Atharva Vichare

