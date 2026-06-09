# Local Multi-PDF RAG System

A lightweight, fully local Retrieval-Augmented Generation (RAG) system built from scratch in Python. It allows you to chat with multiple PDF documents simultaneously using open-source models running locally via LM Studio.

## 🚀 Features

* **Multi-PDF Support** – Extracts and processes text from multiple PDF files seamlessly.
* **Persistent Vector Store** – Uses **ChromaDB** to cache document embeddings locally, preventing redundant computation on subsequent runs.
* **100% Local & Private** – Powered by LM Studio (e.g., Llama 3.x for chat and Nomic embeddings). No data leaves your machine.
* **Source Tracking** – Injects metadata into prompts so the LLM can cite which PDF file each answer comes from.

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **LLM & Embeddings:** LM Studio (OpenAI-compatible API)
* **Vector Database:** ChromaDB
* **PDF Parsing:** PyPDF
* **Numerical Operations:** NumPy

## 📦 Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/matanmay/RAG-LLMStudio.git
cd RAG-LLMStudio
```

### 2. Install dependencies

```bash
pip install openai numpy pypdf chromadb
```

### 3. Configure LM Studio

* Download and install **LM Studio**
* Load an embedding model (e.g., `text-embedding-nomic-embed-text-v1.5`)
* Load a chat model (e.g., `llama-3.2-1b-instruct`)
* Start the local server on port `1234`

### 4. Run the application

* Place your PDF files in the project root directory
* Update the `pdf_files` list in `main.py` with your filenames
* Run:

```bash
python main.py
```

## 💡 How It Works

1. **Ingestion & Chunking**
   On first run, PDFs are parsed and split into 1000-character chunks with 200-character overlap.

2. **Embedding & Storage**
   Each chunk is converted into embeddings and stored in a local ChromaDB database (`./chroma_db`).

3. **Smart Loading**
   On subsequent runs, the system detects the existing database and skips reprocessing.

4. **Retrieval & QA**
   For each query, ChromaDB retrieves the top-K relevant chunks, injects them into a structured prompt with source metadata, and sends it to the local LLM for grounded answers.
