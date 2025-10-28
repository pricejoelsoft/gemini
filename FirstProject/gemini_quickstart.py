import os 
from google import genai

try:
    Client = genai.Client()
except Exception as e:
    print(f"Error initializing GenAI Client: {e}")
    print("Please ensure your GEMINI_API_KEY is set correctly as an environment variable.")
        
    exit()
prompt = "Explain why learning the Gemini API is a smart career move in one paragraph."
model_name = "gemini-2.5-flash"

print(f"sending prompt to model: {model_name}...")

response = Client.models.generate_content(
    model=model_name,
    contents=prompt
)


print("\n---Gemini Response---")
print(response.text)
print("----------------------")