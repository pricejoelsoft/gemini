import os 
import json 
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

client = genai.Client()
MODEL_NAME = "gemini-2.5-flash"

class ProductReview(BaseModel):
    """Structured data model for a product review summary"""
    product_name: str = Field(description="The formal, full name of the product.")
    sentiment_score: int = Field(description="The sentiment rating from 1 (bad) to 10 (excellent).")
    key_pros: list[str] = Field(description="A list of 2-3 main positive points about the product.")
    key_con: list[str] = Field(description="A list of 2-3 main negative points about the product.")

config = types.GenerateContentConfig(
    response_mime_type="application/json",
    response_schema=ProductReview
)


review_text = "I bought this laptop last month. The battery life is amazing, 10 hours easily! It does get hot when I game, which is annoying. The screen quality is perfect for video editing. I wish the keyboard was quieter."

print("1. Generating structured JSON output...")

response = client.models.generate_content(
    model=MODEL_NAME,
    contents=[
        f"Analyze the following user review and extract the structured data:",
        review_text
    ], 
    config=config
)



try:
    review_data: ProductReview = response.parsed
    print("\n--- JSON DATA EXTRACTED (Valid Pydantic Object) ---")
    print(f"Product: {review_data.product_name}")
    print(f"Score: {review_data.sentiment_score}/10")
    print(f"Pros: {review_data.key_pros}")
    print(f"Cons: {review_data.key_con}")
    print("-----------------------------------------------------")



    long_prompt= "Write a 5-paragraph analysis of the impact of Large Language Models on the future of professional coding jobs, maintaining a highly optimistic but realistic tone."

    print("\n2. Generating long-form content with the model...")
    print("Agent: ", end="")


    stream = client.models.generate_content_stream(
        model= MODEL_NAME,
        contents=[long_prompt]
    )


    for chunk in stream: 
        if chunk.text:
            print(chunk.text, end="", flush=True)

    print("\n--- END OF STREAMED CONTENT ---")


except Exception as e:
    print(f"\n[ERROR] Failed to parse JSON response: {e}")

'''###
    The formal, full name of the product.")
    sentiment_score: int = Field(description="The sentiment rating from 1 (bad) to 10 (excellent).", ge=1, le=10)
    key_pros: list[str] = Field(description="A list of 2-3 main positive points about the product.")
    key_cons: list[str] = Field(description="A list of 2-3 main negative points about the product.")###
    
    
    
    
    
    I bought this laptop last month. The battery life is amazing, 10 hours easily! It does get hot when I game, which is annoying. The screen quality is perfect for video editing. I wish the keyboard was quieter."

print("1. Generating structured JSON output...")

response = client.models.generate_content(
    model=MODEL_NAME,
    contents=[
        f"Analyze the following user review and extract the structured data:",
        review_text
    ],
    config=config
)

# --- 4. PARSE THE RESPONSE (EASY!) ---
# The SDK's '.parsed' attribute gives you a ready-to-use Pydantic object,
# eliminating the need for error-prone string parsing.
try:
    # Cast the parsed response to your Pydantic class for type safety
    review_data: ProductReview = response.parsed
    
    print("\n--- JSON DATA EXTRACTED (Valid Pydantic Object) ---")
    print(f"Product: {review_data.product_name}")
    print(f"Score: {review_data.sentiment_score}/10")
    print(f"Pros: {review_data.key_pros}")
    print("-----------------------------------------------------")

except Exception as e:
    print(f"\n[ERROR] Failed to parse JSON response: {e}")
    '''