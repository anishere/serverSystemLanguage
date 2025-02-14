from pydantic import BaseModel

# models/speechToTextAI.py

from pydantic import BaseModel

class SpeechToTextResponse(BaseModel):
    detected_language: str
    transcript: str

