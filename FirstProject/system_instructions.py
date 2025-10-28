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

client = genai.Client()


science_persona = """You are a highly professional, academic physics assistant. Your tone must be formal, serious, and concise. You MUST decline non-scientific questions politely."""


config = types.GenerateContentConfig(
    system_instruction=science_persona,
    tools=[get_current_weather]
)


try:
    chat = client.chats.create(
        model="gemini-2.5-flash",
        config=config
    )
    print("Agent Chat Session successfully created with the weather tool with a defined persona.")
except Exception as e:
    print(f"[ERROR] Failed to initialize client. Is GEMINI_API_KEY set? Error: {e}")
    exit()

prompt_1 = "explain the fundamental principles of quantum entanglement"
print(f"User 1: {prompt_1}")
response_1 = chat.send_message(prompt_1)
print(f"Agent 1: {response_1.text}")

prompt_2 = "what is the weather like in boston and tokyo today?"
print(f"User 2: {prompt_2}")
response_2 = chat.send_message(prompt_2)
print(f"Agent 2: {response_2.text}")



#### lesson part 2

print("\n---FULL CONVERSATION HISTORY---")

full_history = chat.get_history()

for turn_index, content in enumerate(full_history):
    role = content.role.upper()

    parts_summary = []

    for part in content.parts:
        if part.text:
            parts_summary.append(f"TEXT: '{part.text}'...")
        elif part.function_call:
            args = dict(part.function_call.args)
            parts_summary.append(f"FUNCTION CALL: {part.function_call.name}({args})")
        elif part.function_response:
            parts_summary.append(f"FUNCTION RESPONSE: {part.function_response.response}")
            parts_summary.append(f"THOUGHT: {part.thought_signature}")

    print(f"\v[Turn {turn_index} - {role}]")

    for summary in parts_summary:
        print(f"    > {summary}")


print("Get Token Count")

history_tokens = client.models.count_tokens(
    model="gemini-2.5-pro",
    contents=chat.get_history()
)


print(f"\nTotal tokens in History (Input Cost): {history_tokens.total_tokens}")


print(f"\n---USAGE METADATA---")
print(response_2.usage_metadata)


messages_to_keep = 2
if len(full_history) > messages_to_keep:
    print(f"Truncating history: Removing {len(full_history) - messages_to_keep} messages...")
    del full_history[messages_to_keep]
else :
    print(f"History is sort enough; no truncation needed.")

