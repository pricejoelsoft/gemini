import os
import json
from google import genai
from google.genai import types

# --- 1. DEFINE YOUR TOOL (PYTHON FUNCTION) ---
def get_current_weather(city: str) -> str:
    """
    Returns the current weather for a specific city.
    Args:
        city: The city name, e.g., 'San Francisco' or 'Tokyo'.
    """
    city = city.lower()
    if "boston" in city:
        return json.dumps({"temperature": "12°C", "conditions": "Partly Cloudy", "wind": "15 kph"})
    elif "tokyo" in city:
        return json.dumps({"temperature": "25°C", "conditions": "Sunny", "wind": "8 kph"})
    else:
        return json.dumps({"error": "City Not Found", "code": 404})


# --- 2. INITIALIZE CLIENT AND CONFIGURATION ---
try:
    client = genai.Client()
except Exception as e:
    print(f"[ERROR] Failed to initialize client. Is GEMINI_API_KEY set? Error: {e}")
    exit()

# FIX: Remove the 'enable_automatic_function_calling' flag, as your SDK version rejects it.
# We only pass the tools list via the config.
config = types.GenerateContentConfig(
    tools=[get_current_weather],
    # enable_automatic_function_calling=True <--- REMOVED TO FIX ERROR
)


# --- 3. START THE CHAT SESSION ---
# We use the configured model and config to create the chat session.
try:
    chat = client.chats.create(
        model="gemini-2.5-flash",
        config=config # Pass the config object here
    )
    print("Agent Chat Session successfully created with the weather tool.")
except Exception as e:
    print(f"\n[ERROR] Failed to create chat session: {e}")
    exit()


# --- 4. TEST THE AGENT ---
user_prompt_1 = "What is the weather like in Boston today? And how about Tokyo?"
print(f"\nUser: {user_prompt_1}")

# Send the message. We rely on the model's default behavior to call the function.
response_1 = chat.send_message(user_prompt_1)

# The final response is the model's natural language summary of the function's result.
print(f"\nAgent: {response_1.text}")