# Local Multi-PDF RAG System

A lightweight, fully local Retrieval-Augmented Generation (RAG) system built from scratch in Python. It allows you to chat with multiple PDF documents simultaneously using open-source models running locally via LM Studio.

## 🚀 Features

* **Multi-PDF Support** – Extracts and processes text from multiple PDF files seamlessly.
* **Persistent Vector Store** – Uses **ChromaDB** to cache document embeddings locally, preventing redundant computation on subsequent runs.
* **100% Local & Private** – Powered by LM Studio (e.g., Gemma / Llama for chat and Nomic embeddings). No data leaves your machine.
* **Source Tracking** – Injects metadata into prompts so the LLM can cite which PDF file each answer comes from.
* **Chat History Persistence** – Conversation history is saved to ChromaDB and restored automatically on the next session.
* **Streamlit UI** – Clean web interface with session history sidebar, export to JSON, and one-click history clear.

## 🛠️ Tech Stack

| Layer | Library |
|---|---|
| Language | Python 3.10+ |
| UI | Streamlit |
| LLM & Embeddings | LM Studio (OpenAI-compatible API) |
| Vector Database | ChromaDB |
| PDF Parsing | pypdf |

## 📁 Project Structure

```
RAG/
├── rag_backend.py      # Chunking, embeddings, ChromaDB, RAG logic
├── chatbotUI.py        # Streamlit front-end
├── chroma_db/          # Auto-generated — persistent vector store (git-ignored)
└── *.pdf               # Your PDF source files
```

## 📦 Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/matanmay/RAG-LLMStudio.git
cd RAG-LLMStudio
```

### 2. Install dependencies

```bash
pip install openai pypdf chromadb streamlit
```

### 3. Configure LM Studio

* Download and install **[LM Studio](https://lmstudio.ai)**
* Load an embedding model — recommended: `text-embedding-nomic-embed-text-v1.5`
* Load a chat model — recommended: `google/gemma-4-e2b` or any Llama 3.x variant
* Start the local server on port `1234`

### 4. Add your PDFs

Place your PDF files in the project root and update `PDF_FILES` in `rag_backend.py`:

```python
PDF_FILES: list[str] = [
    "your-document.pdf",
    "another-document.pdf",
]
```

### 5. Run the application

```bash
streamlit run chatbotUI.py
```

Open your browser at `http://localhost:8501`.

> **First run:** PDFs are chunked, embedded, and saved to `./chroma_db` — this may take a minute.  
> **Subsequent runs:** The existing database is loaded instantly; no reprocessing needed.

## 💡 How It Works

1. **Ingestion & Chunking**  
   On first run, PDFs are parsed and split into 600-character chunks with 200-character overlap.

2. **Embedding & Storage**  
   Each chunk is embedded via LM Studio and stored in a local ChromaDB collection (`pdf_knowledge_base`).

3. **Smart Loading**  
   On subsequent runs, the system detects the populated collection and skips reprocessing entirely.

4. **Retrieval & Generation**  
   For each query, the question is embedded and the top-3 most relevant chunks are retrieved from ChromaDB. They are injected into a structured prompt with source metadata and sent to the local LLM for a grounded, cited answer.

5. **Chat History**  
   Each Q&A turn is persisted to a separate ChromaDB collection (`chat_history`) and reloaded at the start of every session.

## ⚙️ Configuration

All tuneable parameters live at the top of `rag_backend.py`:

```python
EMBEDDING_MODEL  = "text-embedding-nomic-embed-text-v1.5"
CHAT_MODEL       = "google/gemma-4-e2b"
EMBEDDING_DIM    = 768   # must match embedding model output
CHUNK_SIZE       = 600
CHUNK_OVERLAP    = 200
TOP_K            = 3     # chunks retrieved per query
DB_PATH          = "./chroma_db"
```

## 🗂️ Streamlit UI — Sidebar Features

| Feature | Description |
|---|---|
| Session history | Numbered list of questions asked this session |
| Export (JSON) | Download the full conversation as a JSON file |
| Clear history | Wipes the UI and ChromaDB chat history |
| DB status | Shows how many chunks are loaded |

## 🔒 Privacy

All processing happens locally. No API calls are made to external services. Your documents and conversations never leave your machine.
