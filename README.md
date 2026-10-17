# AI Research & Document Assistant

A student-built Retrieval-Augmented Generation (RAG) application for asking questions about research papers and checking AI-generated answers against the original PDF passages.

## 🚀 Live Demo

**[Try AI Research & Document Assistant](https://ai-research-assistant-ilfjgs4q4k4qludscdjoh9.streamlit.app/)**

**[View Source Code on GitHub](https://github.com/okayatharva/ai-research-assistant)**

## What It Can Do

- Upload PDF research papers and extract selectable text with page numbers.
- Split document text into overlapping chunks for better retrieval.
- Generate embeddings and index document chunks in ChromaDB.
- Retrieve passages relevant to a user's question using semantic search.
- Generate answers with references to the source filename and page number.
- Display retrieved passages so users can verify answers against the original document.
- Use local Ollama models by default or Google Gemini when `GEMINI_API_KEY` is configured.

## Project Structure

```text
ai-research-assistant/
├── app.py                  # Streamlit interface and user interactions
├── config.py               # Model names and shared settings
├── ai_provider.py          # AI connections, embeddings, and responses
├── pdf_processor.py        # PDF text extraction and chunking
├── vector_store.py         # ChromaDB indexing and semantic retrieval
├── rag_pipeline.py         # Connects retrieval to answer generation
├── requirements.txt        # Python dependencies
├── .env.example            # Example environment configuration
├── .gitignore              # Files excluded from Git
└── .streamlit/
    └── config.toml         # Streamlit configuration
```

## Run Locally on macOS

Clone the repository and enter the project directory:

```bash
git clone https://github.com/okayatharva/ai-research-assistant.git
cd ai-research-assistant

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Option A: Run with Ollama

Install Ollama from its [official website](https://ollama.com/) if it is not already installed.

Download the models used by the application:

```bash
ollama pull llama3.2:1b
ollama pull nomic-embed-text
```

Start the Ollama application, then run Streamlit:

```bash
streamlit run app.py
```

If the application cannot connect to Ollama, check that the Ollama service is running and that the configured model names match the downloaded models.

### Option B: Run with Google Gemini

Obtain an API key through [Google AI Studio](https://aistudio.google.com/).

Set the key in your terminal session:

```bash
export GEMINI_API_KEY="paste_your_key_here"
streamlit run app.py
```

For Streamlit Community Cloud, configure `GEMINI_API_KEY` in the application's **Secrets** settings rather than committing it to GitHub.

**Privacy note:** Gemini mode sends relevant document text to Google's API. Do not upload confidential or sensitive documents unless you are authorized to share them.

## How the RAG Workflow Works

1. **PDF extraction:** `pdf_processor.py` extracts document text while preserving page numbers.
2. **Text chunking:** The extracted text is divided into smaller, overlapping chunks.
3. **Embedding generation:** `ai_provider.py` converts document chunks into numerical representations.
4. **Indexing:** `vector_store.py` stores the chunks, embeddings, filenames, and page metadata in ChromaDB.
5. **Question retrieval:** The user's question is embedded and compared with the indexed document chunks.
6. **Answer generation:** `rag_pipeline.py` sends the question and relevant passages to the selected language model.
7. **Source verification:** `app.py` displays the generated answer and retrieved passages so users can check the supporting evidence.

## Technologies Used

- Python
- Streamlit
- Retrieval-Augmented Generation (RAG)
- ChromaDB
- Ollama
- Google Gemini API (optional)
- PyMuPDF
- Semantic search and text embeddings

## Current Limitations

- Scanned PDFs without selectable text are not supported because OCR is not included.
- The ChromaDB collection is stored in memory, so indexed documents may need to be uploaded again after an application restart.
- AI-generated answers can be incomplete or incorrect. Always verify important claims against the original document.
- Ollama mode requires a running local Ollama service. A cloud deployment needs an AI provider accessible from the cloud environment.
- Gemini mode sends relevant document text to Google's API.
- This is a learning and portfolio project, not a replacement for reading the original research paper.

## Learning Notes

I built this project to understand how a RAG pipeline connects PDF processing, text chunking, embeddings, vector search, and language-model-based answer generation.

The code is separated into smaller Python modules, each with a specific responsibility. The project is intended to support hands-on learning, experimentation, and further improvements to document-based question answering.
