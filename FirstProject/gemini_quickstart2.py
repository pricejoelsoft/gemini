import os
import io # <--- New import for handling binary data
from google import genai
from google.genai import types
from PIL import Image 

# Initialize the client (assumes GEMINI_API_KEY environment variable is set)
client = genai.Client()

# --- Multimodal API Call ---
image_path = 'image1.jpg'
mime_type = 'image/jpeg' # Ensure this matches your image file type

# 1. Open the image file using the Pillow library
try:
    img = Image.open(image_path)
    
    # 2. Convert the PIL Image object into raw bytes
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format=img.format or 'jpeg') # Save it to bytes
    img_bytes = img_byte_arr.getvalue()

except FileNotFoundError:
    print(f"\n[ERROR] Image file not found: {image_path}")
    print("Please ensure you have a JPEG image named 'image1.jpg' in the current directory.")
    exit()

# 3. Define the multimodal contents using Part.from_bytes
multimodal_contents = [
    # CORRECT WAY: Use from_bytes with the raw data and mime type
    types.Part.from_bytes(
        data=img_bytes, 
        mime_type=mime_type
    ), 
    
    # The text instruction for the model
    "Describe this image in detail and write a caption for it."
]

print("Sending multimodal prompt (Image + Text) to Gemini...")

# 4. Call the API
try:
    multimodal_response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=multimodal_contents
    )
    
    # 5. Print the model's response
    print("\n--- MULTIMODAL RESPONSE ---")
    print(multimodal_response.text)
    print("---------------------------")

except Exception as e:
    print(f"\n[ERROR] API Call Failed: {e}")