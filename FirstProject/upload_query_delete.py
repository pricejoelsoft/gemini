import os 
import json
from urllib import response
from google import genai


#--- SETUP ---
client = genai.Client()
FILE_PATH = "report.txt"
MODEL_NAME = "gemini-2.5-flash"

uploaded_file = None

try:
    print(f"1. Attempted to upload file: {FILE_PATH}...")
    

    #---2. UPLOAD FILE ---
    # client.files.upload handles the heavy lifting and returns a file object (reference).
    # This file object is now the reference token for the document in Gemini's memory.

    uploaded_file = client.files.upload(
        file = FILE_PATH,
        config = {
            "display_name": "Q3-Technical-Report",
            "mime_type": "text/plain"
        }
    )

    print(f"    Upload successful. File Name: {uploaded_file.name}")
    print(f"    The file is ready for analysis at URI: {uploaded_file.uri}")

    prompt = (
        "Based *only* on the provided report, what was the primary quantitative "
        "finding in Q3, and what is the specific recommendation to mitigate "
        "the second main risk? Respond in bullet points."
    )

    print("\n2. Sending complex mutimodal query to the model...")
    resonse = client.models.generate_content(
        model=MODEL_NAME,
        contents=[prompt, uploaded_file]
    )

    print("\n--- MODEL ANALYSIS (RAG without DB) ---")
    print(response.text)
    print("---------------------------------------")
    
except FileNotFoundError: 
    print(f"\n[ERROR] File not found. Please ensure '{FILE_PATH}' exists in your directory.")
except Exception as e:
    print(f"Error Occured {e}")

finally:
    if uploaded_file:
        print(f"\n3. Cleaning up: Deleting uploaded file '{uploaded_file.name}' from Gemini storage...")
        client.files.delete(name=uploaded_file.name)
        print("    File deleted successfully.")