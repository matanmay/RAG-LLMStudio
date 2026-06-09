from openai import OpenAI
import numpy as np
from pypdf import PdfReader
import os
import chromadb  # <--- New library for Vector Database

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
)

EMBEDDING_MODEL = "text-embedding-nomic-embed-text-v1.5"
# CHAT_MODEL = "llama-3.2-1b-instruct"
CHAT_MODEL = "google/gemma-4-e2b"

# ==========================================
# PDF Processing Functions
# ==========================================

def load_and_chunk_pdf(pdf_path, chunk_size=600, chunk_overlap=200):
    """
    Reads a single PDF file, extracts its text, and splits it into smaller chunks.
    """
    reader = PdfReader(pdf_path)
    full_text = ""
    
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"
            
    chunks = []
    start = 0
    while start < len(full_text):
        end = start + chunk_size
        chunk_text = full_text[start:end].strip()
        chunks.append(chunk_text)
        start += (chunk_size - chunk_overlap)
        
    return chunks

# ==========================================
# Custom Embedding Function for ChromaDB
# ==========================================

def get_embedding(text):
    """
    Generates embedding vector for a given text string using LM Studio.
    """
    text = text.replace("\n", " ")
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[text]
    )
    return response.data[0].embedding

# ==========================================
# Database Initialization (ChromaDB)
# ==========================================

# Define the local folder where the database files will be stored permanently
DB_PATH = "./chroma_db"
is_new_db = not os.path.exists(DB_PATH)

# Initialize the persistent Chroma client (saves data to disk)
chroma_client = chromadb.PersistentClient(path=DB_PATH)

# Create or get the collection (think of it like a table in a standard DB)
collection = chroma_client.get_or_create_collection(name="pdf_knowledge_base")

# List of PDF files to check/process
pdf_files = [
    "A.pdf",
    "B.pdf"
]

# If the database directory doesn't exist, we need to populate it for the first time
if is_new_db:
    print("Database not found. Creating a new one and processing PDFs...")
    
    all_chunks = []
    all_metadata = []
    all_ids = []
    all_embeddings = []
    
    chunk_counter = 0
    for pdf_file in pdf_files:
        if os.path.exists(pdf_file):
            print(f"Processing: {pdf_file}...")
            file_chunks = load_and_chunk_pdf(pdf_file)
            
            for chunk in file_chunks:
                all_chunks.append(chunk)
                # Metadata allows us to filter or display the source later
                all_metadata.append({"source": os.path.basename(pdf_file)})
                # Chroma requires a unique string ID for every record
                all_ids.append(f"id_{chunk_counter}")
                
                # Generate embedding for this specific chunk
                embedding = get_embedding(chunk)
                all_embeddings.append(embedding)
                
                chunk_counter += 1
        else:
            print(f"Warning: File not found -> {pdf_file}")
            
    # Save everything to the database in bulk
    if all_chunks:
        print(f"Saving {len(all_chunks)} chunks to ChromaDB...")
        collection.add(
            embeddings=all_embeddings,
            documents=all_chunks,
            metadatas=all_metadata,
            ids=all_ids
        )
        print("Database populated and saved successfully!")
else:
    print("Existing Database found! Loading data from disk instantly (Skipping PDF processing).")
    print(f"Total items in DB: {collection.count()}")

# ==========================================
# Retrieval & RAG QA Logic
# ==========================================

def retrieve_from_db(query, top_k=3):
    """
    Queries ChromaDB directly. The DB handles the vector comparison automatically.
    """
    # Step 1: Embed the user's question
    query_embedding = get_embedding(query)
    
    # Step 2: Query the database using the vector
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    # Step 3: Format the results into a clean list of dictionaries
    formatted_docs = []
    for i in range(len(results['documents'][0])):
        formatted_docs.append({
            "text": results['documents'][0][i],
            "source": results['metadatas'][0][i]['source']
        })
        
    return formatted_docs

def ask_rag(question):
    """
    Retrieves context from the Vector DB and answers the question using the LLM.
    """
    # Retrieve relevant data from DB instead of local python list
    context_docs = retrieve_from_db(question)
    
    context_blocks = []
    for doc in context_docs:
        block = f"[Source File: {doc['source']}]\n{doc['text']}"
        context_blocks.append(block)
        
    context = "\n---\n".join(context_blocks)

    prompt = f"""
Answer the question using only the provided context. If the answer cannot be found in the context, say "I don't know".
Always mention which source file the information comes from if available in the context.

Context:
{context}

Question:
{question}
"""
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content

# ==========================================
# Execution Example
# ==========================================

question = "Why the authors use RAG? and what is it?"
answer = ask_rag(question)

print("\nAnswer:")
print(answer)