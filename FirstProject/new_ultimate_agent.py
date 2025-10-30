import os
import json
import chromadb
from google import genai
from google.genai import types
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core import SQLDatabase
from llama_index.core.query_engine import NLSQLTableQueryEngine
from sqlalchemy import create_engine, MetaData, Table, Column, String, Integer, text

# --- Global Configuration & Setup ---

# Initialize the RAG engine variable in the global scope 
# This prevents the NameError if the RAG indexing fails due to quota.
query_engine_sql = None

# LlamaIndex Global Settings
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


# --- 2. SQL AGENT SETUP (Text-to-SQL) ---

# CRITICAL FIX: The engine must be defined globally for the dispose() call at the end.
engine = create_engine("sqlite:///:memory:")

try:
    print("Starting SQL Agent Setup...")

    # Define the table schema and metadata
    metadata_obj = MetaData()
    employee_table = Table(
        'employee_info',
        metadata_obj,
        Column('employee_id', Integer, primary_key=True),
        Column('name', String(50)),
        Column('department', String(50)),
        Column('salary', Integer),
    )
    metadata_obj.create_all(engine)

    # Insert sample data into the table
    with engine.connect() as connection:
        connection.execute(
            employee_table.insert(),
            [
                {'name': 'Alice Johnson', 'department': 'Marketing', 'salary': 65000},
                {'name': 'Bob Smith', 'department': 'Sales', 'salary': 92000},
                {'name': 'Charlie Brown', 'department': 'Marketing', 'salary': 70000},
                {'name': 'David Lee', 'department': 'Sales', 'salary': 88000},
                {'name': 'Emily Davis', 'department': 'Finance', 'salary': 105000},
            ]
        )
        connection.commit()

    # Wrap the SQL Engine with LlamaIndex's SQLDatabase abstraction
    sql_database = SQLDatabase(engine, include_tables=['employee_info'])

    # Create the Natural Language to SQL Query Engine
    query_engine_sql = NLSQLTableQueryEngine(
        sql_database=sql_database,
        tables=['employee_info'],
        llm=Settings.llm,
        synthesize_response=True,
        verbose=True
    )
    print("✅ SQL Query Engine successfully initialized and ready.")
except Exception as e:
    # This branch is executed on the 429 Quota Error during RAG indexing
    print(f"❌ SQL Engine Setup Failed: {e.__class__.__name__}. Defaulting to Function Calling only.")

# --- 3. AGENT CONFIGURATION & CHAT CREATION ---
rag_persona = "You are a highly professional Corporate Information Assistant. You must answer questions using external tools if possible, otherwise rely on your general knowledge. Maintain a formal, concise tone."

config = types.GenerateContentConfig(
    system_instruction=rag_persona,
    tools=[get_current_weather],
)

chat = client.chats.create(
    model="gemini-2.5-flash",
    config=config,
)


# --- 4. THE ULTIMATE AGENT EXECUTION FUNCTION ---
def run_ultimate_query(prompt: str):
    # 1. ORCHESTRATION LOGIC: Check if the SQL engine was successfully created
    global query_engine_sql  # Not needed for reading, but good practice for clarity

    if query_engine_sql is not None:
        # A. SQL Retrieval is available: Query the database
        print("\n(Agent attempting SQL query via LlamaIndex.)")
        try:
            rag_response = query_engine_sql.query(prompt)
            # The RAG context is the final synthesized NL response from the SQL query
            rag_context = rag_response.response.strip()
        except Exception as e:
            # Handle potential SQL generation error inside LlamaIndex
            rag_context = f"SQL Query failed: Could not process request. Error: {e.__class__.__name__}"
            print(f"(SQL Query Failed inside LlamaIndex: {rag_context})")
    else:
        # B. SQL Retrieval is NOT available (due to initialization failure)
        rag_context = "SQL DATA UNAVAILABLE. Cannot access employee information."
        print(f"(Agent skipping SQL: Engine not initialized.)")

    # 2. GENERATION STEP: Send the prompt + context to the Gemini Chat
    final_prompt = (
        f"RAG CONTEXT: {rag_context}\n\n"
        f"Based ONLY on the CONTEXT and your available tools, answer the user's question: {prompt}"
    )

    response = chat.send_message(final_prompt)
    return response.text.strip()


# --- TEST QUERIES ---
print("\n--- ULTIMATE AGENT TEST (Full Orchestration) ---")

# TEST 1: SQL Query (Checks the in-memory database)
query_sql = "Which department has the highest total salary and what is the maximum salary in the Sales department?"
response_sql = run_ultimate_query(query_sql)
print(f"\nQUERY 1 (SQL Agent Test):\nUser: {query_sql}\nAgent: {response_sql}")

# TEST 2: Function Calling Question (Uses get_current_weather function)
query_tool = "What are the current weather conditions in Boston?"
response_tool = chat.send_message(query_tool)
print(f"\nQUERY 2 (Tool Test):\nUser: {query_tool}\nAgent: {response_tool.text}")

# --- FINAL CLEANUP (CRITICAL for PyCharm/IDE) ---
print("\nPerforming final database cleanup...")
engine.dispose()
print("\nPerforming final database cleanup...complete")