from dotenv import load_dotenv
from openai import OpenAI
import os
import base64

load_dotenv()

# Your Prompt
prompt = "Extract text from the image and don't provide any extra answer."

# Image file path
image_path = "C:/Users/acer/Pictures/Screenshots/Screenshot 2025-01-26 105603.png"

# Initialize Client
client = OpenAI(
    api_key=os.getenv("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")
)

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
# Encode the image
base64_image = encode_image(image_path)

# Create chat completion
try:
    chat_completion = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=300,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
    )

    print(chat_completion.choices[0].message.content)
except Exception as e:
    print(f"An error occured: {e}")