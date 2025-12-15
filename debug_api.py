import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("No API key found")
    exit(1)

client = genai.Client(api_key=api_key)
try:
    # Try listing models if the client supports it, or just try a simple generation
    # Newer google-genai might not have list_models directly on client
    # let's try to generate with a fallback model
    print("Attempting to generate with gemini-1.5-flash...")
    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents="Hello"
    )
    print("Success:", response.text)
except Exception as e:
    print("Error:", e)

try:
    print("\nAttempting to generate with gemini-2.0-flash-exp...")
    response = client.models.generate_content(
        model="gemini-2.0-flash-exp",
        contents="Hello"
    )
    print("Success:", response.text)
except Exception as e:
    print("Error:", e)
