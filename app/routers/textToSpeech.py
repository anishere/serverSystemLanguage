import base64
import io
from fastapi import APIRouter, HTTPException
from app.models.textToSpeech import TextToSpeechRequest
import os
from gtts import gTTS
from app.security.security import get_api_key

# Tạo instance router
router = APIRouter(prefix="/textToSpeechAI", tags=["textToSpeechAI"])

def text_to_speech(text: str, lang: str) -> str:
    """
    Chuyển đổi văn bản thành âm thanh và trả về Base64 string.
    """
    try:
        # Tạo đối tượng gTTS từ văn bản
        tts = gTTS(text=text, lang=lang, slow=False)

        # Lưu vào bộ nhớ thay vì file
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)  # Reset vị trí đọc về đầu file

        # Mã hóa file thành base64
        base64_audio = base64.b64encode(audio_buffer.read()).decode("utf-8")

        return base64_audio
    except Exception as e:
        raise RuntimeError(f"Error during text-to-speech conversion: {e}")

@router.post("/textToSpeech")
async def text_to_speech_api(request: TextToSpeechRequest, api_key: str = get_api_key):
    """
    API chuyển văn bản thành âm thanh và trả về Base64 string.
    """
    if not request.text:
        raise HTTPException(status_code=400, detail="Text is required")

    try:
        base64_audio = text_to_speech(request.text, request.lang)
        return {"audio_base64": base64_audio}  # ✅ Trả về chuỗi base64
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
