import os
import json
import chromadb
from google import genai
from google.genai import types
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.google_genai import GoogleGenAI # NEW LLM IMPORT
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding # NEW EMBEDDING IMPORT

# --- Configuration (using the upgraded imports) ---
Settings.llm = GoogleGenAI(model="gemini-2.5-flash") 
Settings.embed_model = GoogleGenAIEmbedding(model_name="models/embedding-001") 
PERSIST_DIR = "./chroma_db"
DATA_DIR = "./data"
client = genai.Client()

# --- 1. TOOL DEFINITION (Function Calling) ---
def get_current_weather(city: str) -> str:
    """Returns the current weather for a specific city."""
    city = city.lower()
    if "boston" in city:
        return json.dumps({"temperature": "12°C", "conditions": "Cloudy"})
    else:
        return json.dumps({"error": "City Not Found", "code": 404})

# --- 2. RAG SETUP (Indexing Policy Data) ---
try:
    documents = SimpleDirectoryReader(DATA_DIR).load_data()
    db = chromadb.PersistentClient(path=PERSIST_DIR)
    chroma_collection = db.get_or_create_collection("company_policy")
    index = VectorStoreIndex.from_documents(documents, collection=chroma_collection)
    query_engine_rag = index.as_query_engine()
    print("RAG Index successfully loaded/created.")
except Exception as e:
    print(f"RAG Indexing Failed (Quota Error Expected): {e}")
    # The agent will only be able to use the Function Calling tool if RAG fails

# --- 3. AGENT CONFIGURATION (Persona + Tools) ---
rag_persona = "You are a highly professional Corporate Information Assistant. You must answer questions using external tools if possible, otherwise rely on your general knowledge. Maintain a formal, concise tone."

config = types.GenerateContentConfig(
    system_instruction=rag_persona,
    tools=[get_current_weather], 
)

chat = client.chats.create(
    model="gemini-2.5-flash",
    config=config,
)

# --- 4. THE ULTIMATE AGENT EXECUTION ---
print("\n--- ULTIMATE AGENT TEST ---")

# Combine RAG answer and Tool answer (Note: This is a simplified call)
def run_ultimate_query(prompt: str):
    # 4a. Try RAG first (for policy questions)
    rag_response = query_engine_rag.query(prompt)
    rag_context = rag_response.response.strip()

    # 4b. Send a final prompt to the Gemini Chat with the RAG context and the tool
    # The agent decides whether to use the RAG context or the Function Calling tool.
    
    # NOTE: To simplify the final step and ensure the agent *uses* the tools/persona, 
    # we'll use the Chat object (which is tool-enabled) and manually prepend the RAG context.
    
    # Inject RAG Context into the Chat Message
    final_prompt = (
        f"CONTEXT (from company policy): {rag_context}\n\n"
        f"Based on the CONTEXT and your available tools, answer the user's question: {prompt}"
    )

    response = chat.send_message(final_prompt)
    return response.text.strip()


# TEST 1: RAG Question (Uses Index/policy.txt)
query_rag = "What is the policy regarding remote work and how much is the mileage reimbursement rate?"
response_rag = run_ultimate_query(query_rag)
print(f"\nQUERY 1 (RAG + Persona):\nUser: {query_rag}\nAgent: {response_rag}")


# TEST 2: Function Calling Question (Uses get_current_weather function)
query_tool = "What are the current weather conditions in Boston?"
response_tool = chat.send_message(query_tool)
print(f"\nQUERY 2 (Tool + Persona):\nUser: {query_tool}\nAgent: {response_tool.text}")