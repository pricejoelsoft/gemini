import os
import chromadb

# --- LlamaIndex Imports ---
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding

# --- Configuration ---
# LlamaIndex will automatically use the GOOGLE_API_KEY environment variable.
# We explicitly configure the models for clarity.
Settings.llm = Gemini(model="gemini-2.5-flash") # Use Gemini for the final answer generation
Settings.embed_model = GeminiEmbedding(model_name="models/embedding-001") # Use Gemini for vector creation
PERSIST_DIR = "./chroma_db"
DATA_DIR = "./data"

print("Starting RAG Pipeline Setup...")

# 1. LOAD: Read documents from the local directory
documents = SimpleDirectoryReader(DATA_DIR).load_data()
print(f"Loaded {len(documents)} document(s) from {DATA_DIR}.")

# 2. INDEX: Create or Load the Vector Store Index
# This step chunks the documents, converts them into vectors using the embedding model, 
# and stores them in the Chroma vector database.
try:
    # Attempt to load the existing index
    db = chromadb.PersistentClient(path=PERSIST_DIR)
    chroma_collection = db.get_or_create_collection("company_policy")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=chroma_collection
    )
    print("Loaded existing vector index.")
except Exception:
    # If loading fails, create a new one
    db = chromadb.PersistentClient(path=PERSIST_DIR)
    chroma_collection = db.get_or_create_collection("company_policy")
    index = VectorStoreIndex.from_documents(documents)
    print("Created and persisted new vector index.")


# 3. QUERY: Ask a question that requires knowledge from the policy.txt
query_engine = index.as_query_engine()

question = "I worked remotely for 4 days last week. Is this allowed by the company policy, and what is the specific cost per mile for travel?"

print(f"\n3. User Query: {question}")

# This sends the query to LlamaIndex, which performs:
# A. Retrieval: Converts the question to a vector and finds relevant chunks.
# B. Generation: Sends the relevant chunks + the question to Gemini.
response = query_engine.query(question)

print("\n--- GEMINI RAG RESPONSE (Grounded in policy.txt) ---")
print(response.response)
print("\n--- SOURCE NODES (The proof) ---")

# Access the source nodes to see which policy text was used as context
for node in response.source_nodes:
    print(f"File: {node.metadata.get('file_name')}, Score: {node.score:.2f}")
    # Print the chunk text that was retrieved
    print(f"Context Snippet: {node.text.strip()[:100]}...\n") 
print("-----------------------------------------------------")