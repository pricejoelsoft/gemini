import os 
import json 
import chromadb
from google import genai
from google.genai import types
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding 

#Config 
Settings.llm = GoogleGenAI(model="gemini-2.5-flash")
Settings.embed_model = GoogleGenAIEmbedding(model_name="models/embedding-001")
PERSIST_DIR = "./chroma_db"
DATA_DIR = "./data"
client = genai.Client()


#1
def get_current_weather(city: str) -> str:
    """Returns the current weather for a specified city."""
    city = city.lower()
    if "boston" in city:
        return json.dumps({"temperature": "12C", "conditions": "Cloudy"})
    else:
        return json.dumps({"error": "City not found", "code": 404})

#2
query_engine_rag = None

try: 
    #2a
    documents = SimpleDirectoryReader(DATA_DIR).load_data()
    db = chromadb.PersistentClient(path=PERSIST_DIR)
    chroma_collection = db.get_or_create_collection("company_policy")
    index = VectorStoreIndex.from_documents(documents, collection=chroma_collection)

    query_engine_rag = index.as_query_engine()

    print("RAG Index successfully loaded")
except Exception as e:
    print("RAG Indexing Failed: {e.__class__.__name__}. RAG functionality diabled.")


#3
rag_persona = "You are a highly professional Corporate Information Assistant.  You must answer questions using external tools if possible, otherwise rely on your general knowledge.  Maintain a formal, concise tone."

# Register the tool using the Gemini API's expected format
weather_tool = types.Tool(
    function=get_current_weather,
    name="get_current_weather",
    description="Returns the current weather for a specified city."
)

config = types.GenerateContentConfig(
    system_instructions=rag_persona,
    tools=[weather_tool]
)

chat = client.chat.create(
    model="gemini-2.5-flash",
    config=config
)

#4
def run_ultimate_query(prompt: str):
    rag_context = None
    if query_engine_rag is not None:
        print("(Agent attempting RAG query.)")
        rag_response = query_engine_rag.query(prompt)
        rag_context = rag_response.response.strip()
    else:
        rag_context = "RAG CONTEXT UNAVAILABLE. Cannot access internal documents."
        print("Agent skipping RAG: Engine not initialized.")

    final_prompt = (
        f"RAG CONTEXT: {rag_context}\n\n"
        f"Based ONLY on the CONTEXT and your available tools, answer the user's question: {prompt}"
    )

    response = chat.send_message(final_prompt)
    return response.text.strip()


query_rag = "What is the policy regarding remote work and how much is the mileage reimbursement rate?"
response_rag = run_ultimate_query(query_rag)

query_tool =  "What are the current weather conditions in Boston?"
response_tool = chat.send_message(query_tool)
print(f"\nQuery 2 (Tool + Persona:\nUser: {query_tool}\nAgent: {response_tool.text})")