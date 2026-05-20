from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

response = client.chat.completions.create(
    messages=[{"role": "user", "content": "Say hello"}],
    model="llama-3.1-8b-instant"
)

print(response.choices[0].message.content)