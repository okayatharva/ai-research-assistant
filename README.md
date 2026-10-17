# AI Research & Document Assistant

A student-built Retrieval-Augmented Generation (RAG) app for asking questions about research papers and checking the PDF passages behind the answers.

## What it can do

- Upload a PDF and extract selectable text with page numbers.
- Split the text into overlapping chunks.
- Create embeddings and index the chunks in ChromaDB.
- Search for passages related to a question.
- Generate an answer with citations to the source filename and page.
- Show the retrieved passages so answers can be checked against the original document.
- Use local Ollama by default, or Google Gemini when `GEMINI_API_KEY` is configured.

## Project structure

```text
ai-research-assistant/
├── app.py                 # Streamlit page and user interactions
├── config.py              # Model names and shared settings
├── ai_provider.py         # Connection checks, embeddings, and AI responses
├── pdf_processor.py       # PDF text extraction and chunking
├── vector_store.py        # ChromaDB indexing and semantic retrieval
├── rag_pipeline.py        # Connects retrieval to answer generation
├── requirements.txt       # Python packages
├── .gitignore             # Files that should not be committed
└── .streamlit/config.toml # Basic Streamlit settings
```

## Run locally on macOS

Open Terminal and move into the folder where you cloned the repository:

```bash
git clone https://github.com/okayatharva/ai-research-assistant.git
cd ai-research-assistant
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Option A: Run with Ollama (local model)

Install Ollama from its official website if you do not already have it. Start the Ollama app, then run:

```bash
ollama pull llama3.2:1b
ollama pull nomic-embed-text
```

Start the app:

```bash
streamlit run app.py
```

If the app cannot connect to Ollama, make sure Ollama is running. Some installations also allow starting it with `ollama serve` in a separate Terminal window.

### Option B: Run with Gemini

Set your key in the Terminal session before starting Streamlit:

```bash
export GEMINI_API_KEY="paste_your_key_here"
streamlit run app.py
```

Do not paste your real API key into Python files, README, screenshots, or GitHub. For Streamlit Community Cloud, add `GEMINI_API_KEY` through the app's Secrets settings. When using Gemini, document text is sent to Google's API; do not upload confidential PDFs.

## How the RAG workflow works

1. `pdf_processor.py` extracts text and keeps the PDF page number.
2. The text is split into smaller, overlapping chunks.
3. `ai_provider.py` creates an embedding for each chunk.
4. `vector_store.py` saves the text, embedding, filename, and page in ChromaDB.
5. When a question is asked, it is embedded and compared with the stored chunks.
6. `rag_pipeline.py` sends the most relevant passages and question to the language model.
7. `app.py` shows the answer and the retrieved passages for checking.

## Current limitations

- Scanned PDFs without selectable text are not supported because OCR is not included.
- The ChromaDB collection is in memory, so it is temporary and may disappear after a restart.
- Answers may be incomplete or incorrect. Always check the original source passages.
- Gemini mode sends relevant document text to Google's API. Ollama mode depends on the local Ollama service and downloaded models.
- This is a learning/portfolio project, not a replacement for reading the original research paper.

## Learning notes

The code is split into smaller files so each part has one main job. Comments explain why the steps are there. Try changing one thing at a time, run the app, and check what changes.
