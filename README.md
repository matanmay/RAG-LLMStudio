# Local Multi-PDF RAG System

A lightweight, fully local Retrieval-Augmented Generation (RAG) system built from scratch in Python. This system allows you to chat with multiple PDF documents simultaneously using open-source models running locally via LM Studio.

## 🚀 Features
* **Multi-PDF Support:** Extracts and processes text from multiple PDF files seamlessly.
* **Persistent Vector Store:** Uses **ChromaDB** to cache document embeddings locally on disk, preventing redundant embedding generation on subsequent runs.
* **100% Local & Private:** Powered by LM Studio (Llama 3.2 for chat and Nomic-embed for vector embeddings). No data leaves your machine.
* **Source Tracking:** The system injects metadata into the prompt, allowing the LLM to cite which specific PDF file the answer came from.

## 🛠️ Tech Stack
* **Language:** Python 3.10+
* **LLM & Embeddings Orchestration:** LM Studio (OpenAI-compatible client)
* **Vector Database:** ChromaDB
* **PDF Parsing:** PyPDF
* **Math & Vector Operations:** NumPy

## 📦 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [[https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)](https://github.com/matanmay/RAG-LLMStudio.git)
   cd RAG-LLMStudio

```

2. **Install dependencies:**
```bash
pip install openai numpy pypdf chromadb

```

3. **Configure LM Studio:**
* Download and start **LM Studio**.
* Load your text embedding model (e.g., `text-embedding-nomic-embed-text-v1.5`).
* Load your LLM (e.g., `llama-3.2-1b-instruct`).
* Start the Local Server on port `1234`.


4. **Run the Application:**
* Place your PDF files in the root directory.
* Update the `pdf_files` list in the script with your filenames.
* Run the script:


```bash
python main.py

```

## 💡 How it Works

1. **Ingestion & Chunking:** On the first run, the script parses the PDFs, splits them into 1000-character chunks with a 200-character overlap.
2. **Embedding & Storage:** Chunks are converted into vector embeddings and saved into a local folder (`./chroma_db`).
3. **Smart Loading:** On future runs, the script detects the existing database and loads the data instantly, skipping the expensive PDF processing phase.
4. **Retrieval & QA:** When a query is made, ChromaDB fetches the top $K$ relevant chunks, compiles them into a structured prompt with source citations, and sends them to the local LLM for a grounded, hallucination-free answer.
