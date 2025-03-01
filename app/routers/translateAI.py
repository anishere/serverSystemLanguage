# routers/translateAI.py
from fastapi import APIRouter, HTTPException
from app.models.translateAI import TranslateRequest, TranslateResponse
from app.config import settings
import openai  # Thêm thư viện OpenAI
from app.security.security import get_api_key

# Cấu hình client OpenAI
client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)

# Tạo instance router
router = APIRouter(prefix="/translateAI", tags=["translateAI"])

def translate_text_with_gpt(text: str, src_lang: str, tgt_lang: str) -> str:
    """
    Dịch văn bản từ ngôn ngữ nguồn sang ngôn ngữ đích sử dụng GPT-4o Mini.
    """
    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a professional translator. Your task is to translate the given text from the source language to the target language accurately and naturally, assuming it is for a general audience and from a standard context unless specified otherwise by the user."
                    "To ensure the highest quality, follow these steps:"
                    "1.Read and understand the entire text to grasp its context and meaning."
                    "2.Translate the text into the target language, aiming for a natural and fluent expression."
                    "3.Review your translation to check for any grammatical errors, misinterpretations, or awkward phrasing."
                    "4.Make necessary corrections to refine the translation."
                    "5.Ensure that the final translation maintains the same tone and style as the original text."
                    "Your final output should be only the translated text, without any additional explanations or notes."
                )
            },
            {
                "role": "user",
                "content": f"Translate this text from {src_lang} to {tgt_lang}: {text}"
            }
        ]
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.5
        )

        return response.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"Error during translation: {e}")

@router.post("/translate", response_model=TranslateResponse)
async def translate(request: TranslateRequest, api_key: str = get_api_key):
    """
    API dịch văn bản từ ngôn ngữ A sang ngôn ngữ B.
    """
    if not request.text or not request.src_lang or not request.tgt_lang:
        raise HTTPException(status_code=400, detail="Invalid input parameters")

    try:
        translated_text = translate_text_with_gpt(request.text, request.src_lang, request.tgt_lang)
        return TranslateResponse(translated_text=translated_text)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
