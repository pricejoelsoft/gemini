import os
import json
import chromadb
from google import genai
from google.genai import types
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
# --- SQL Imports ---
from llama_index.core import SQLDatabase
from llama_index.core.query_engine import NLSQLTableQueryEngine
from sqlalchemy import create_engine, MetaData, Table, Column, String, Integer, text
from sqlalchemy.engine.url import URL  # Useful for complex connection strings

# --- Global Configuration & Setup ---
Settings.llm = GoogleGenAI(model="gemini-2.5-flash")
Settings.embed_model = GoogleGenAIEmbedding(model_name="models/embedding-001")
PERSIST_DIR = "./chroma_db"
DATA_DIR = "./data"
client = genai.Client()

# Initialize the engine variable in the global scope for the final cleanup step
engine = None
query_engine_sql = None  # Initialize the query engine to None


# --- 1. TOOL DEFINITION (Function Calling) ---
def get_current_weather(city: str) -> str:
    """Returns the current weather for a specific city."""
    city = city.lower()
    if "boston" in city:
        return json.dumps({"temperature": "12°C", "conditions": "Cloudy"})
    else:
        return json.dumps({"error": "City Not Found", "code": 404})


# --- 2. SQL SERVER CONNECTION & AGENT SETUP ---

# >>>>> CRITICAL: REPLACE THESE PLACEHOLDERS <<<<<
DB_SERVER = "localhost"
DB_PORT = 1433
DB_NAME = "employees"
DB_USER = "sa"
DB_PASSWORD = "D00242861"
# -----------------------------------------------

# The SQLAlchemy URL format for SQL Server using the PyODBC driver
CONNECTION_URL = URL.create(
    "mssql+pyodbc",
    username=DB_USER,
    password=DB_PASSWORD,
    host=DB_SERVER,
    port=DB_PORT,
    database=DB_NAME,
    query={"driver": "ODBC Driver 17 for SQL Server"}  # Ensure this driver name is correct!
)

try:
    print("Starting SQL Server Agent Setup...")

    # Create the engine object (will attempt connection immediately)
    engine = create_engine(CONNECTION_URL)

    # 2a. Wrap the SQL Engine with LlamaIndex's SQLDatabase abstraction
    # NOTE: LlamaIndex automatically reflects (reads) the schema of tables in the database.
    # Update the include_tables list in your Python script:
    # List of tables already included in the database wrapper:
    TABLE_LIST = ['employee_info', 'Product_Catalog', 'Sales_Data'] # Define this list globally

    # Then, create the SQLDatabase object:
    sql_database = SQLDatabase(engine, include_tables=TABLE_LIST)

    # And use the list directly when creating the query engine:
    query_engine_sql = NLSQLTableQueryEngine(
        sql_database=sql_database,
        tables=TABLE_LIST, # <--- Pass the original list directly
        llm=Settings.llm,
        synthesize_response=True,
        verbose=True
    )
    
    print("✅ SQL Query Engine successfully connected to server and ready.")

except Exception as e:
    # This block catches connection errors, failed drivers, or LlamaIndex schema errors
    print(f"❌ SQL Engine Setup Failed: {e.__class__.__name__}. Check ODBC Driver and DB credentials.")
    query_engine_sql = None  # Ensure the engine is explicitly None if it fails

# --- 3. AGENT CONFIGURATION & CHAT CREATION ---
rag_persona = "You are a highly professional Corporate Information Assistant. You will translate user requests into SQL queries and provide only data-driven answers. Maintain a formal, concise tone."

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
    if query_engine_sql is not None:
        # A. SQL Retrieval is available: Query the database
        print("\n(Agent attempting SQL query via LlamaIndex.)")
        try:
            # LlamaIndex generates, executes, and returns the NL response from DB data
            rag_response = query_engine_sql.query(prompt)
            rag_context = rag_response.response.strip()
        except Exception as e:
            # Catch errors during query generation/execution
            rag_context = f"SQL Query failed: Could not process request. Error: {e.__class__.__name__}"
            print(f"(SQL Query Failed inside LlamaIndex: {rag_context})")
    else:
        # B. SQL Retrieval is NOT available (initialization failed)
        rag_context = "SQL DATA UNAVAILABLE. Cannot access the database."
        print(f"(Agent skipping SQL: Engine not initialized.)")

    # 2. GENERATION STEP: Send the prompt + context to the Gemini Chat
    final_prompt = (
        f"DATA CONTEXT: {rag_context}\n\n"
        f"Based ONLY on the DATA CONTEXT and your available tools, answer the user's question: {prompt}"
    )

    response = chat.send_message(final_prompt)
    return response.text.strip()


# --- TEST QUERIES ---
print("\n--- ULTIMATE AGENT TEST (Full Orchestration) ---")

# TEST 1: Database Query (Requires Text-to-SQL logic)
query_sql = "Show me the names of all employees in the Sales department and who has the highest salary."
response_sql = run_ultimate_query(query_sql)
print(f"\nQUERY 1 (SQL Agent Test):\nUser: {query_sql}\nAgent: {response_sql}")

# TEST 2: Function Calling Question (Uses get_current_weather tool)
query_tool = "What are the current weather conditions in Boston?"
response_tool = chat.send_message(query_tool)
print(f"\nQUERY 2 (Tool Test):\nUser: {query_tool}\nAgent: {response_tool.text}")

# --- FINAL CLEANUP (CRITICAL for PyCharm/IDE) ---
print("\nPerforming final database cleanup...")
if engine:
    engine.dispose()
print("Process finished and resources released.")