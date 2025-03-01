from fastapi import APIRouter, HTTPException, UploadFile, File
from app.models.imgToTextTranslate import ImgToTextTranslateResponse, TranslatedFragment
from app.config import settings
import openai
import base64
from app.security.security import get_api_key
import json

# Cấu hình OpenAI
client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)

# Khởi tạo Router
router = APIRouter(prefix="/imgToTextTranslate", tags=["imgToTextTranslate"])

def encode_image(file: UploadFile) -> str:
    """ Chuyển đổi file ảnh thành chuỗi base64. """
    try:
        image_bytes = file.file.read()
        return base64.b64encode(image_bytes).decode("utf-8")
    except Exception as e:
        raise RuntimeError(f"Error encoding image: {e}")

def extract_text_from_image(image_base64: str) -> str:
    """ Trích xuất văn bản từ hình ảnh bằng GPT-4o Mini. """
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract text from the image and don't provide any extra answer."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
        ]

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=1000,
            messages=messages
        )

        return response.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"Error during text extraction: {e}")

def analyze_multilingual_text(text: str) -> dict:
    """ Phân tích ngôn ngữ của văn bản bằng GPT-4o Mini. """
    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a multilingual text analyzer. Detect all languages in the given text, "
                    "extract the corresponding text segments, and return them in a structured JSON format."
                    "Use Google's language codes (e.g., 'en' for English, 'fr' for French, 'zh' for Chinese)."
                    "Example output:\n"
                    "{\n"
                    "  \"language\": [\n"
                    "    {\"name\": \"English\", \"code\": \"en\", \"text\": \"Hello my name is Linh.\"},\n"
                    "    {\"name\": \"Chinese\", \"code\": \"zh\", \"text\": \"你好，我的名字是Linh。\"}\n"
                    "  ]\n"
                    "}"
                )
            },
            {
                "role": "user",
                "content": f"Analyze this text: {text}"
            }
        ]

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.5,
            messages=messages
        )

        return json.loads(response.choices[0].message.content.strip())

    except Exception as e:
        raise RuntimeError(f"Error during text analysis: {e}")

def translate_text(text: str, target_lang: str) -> str:
    """ Dịch văn bản bằng GPT-4o Mini """
    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a professional translator. Translate the given text to the target language."
                    "Return only the translated text without additional explanation."
                )
            },
            {
                "role": "user",
                "content": f"Translate this text to {target_lang}: {text}"
            }
        ]

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.5,
            messages=messages
        )

        return response.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"Error during text translation: {e}")

@router.post("/extract-translate", response_model=ImgToTextTranslateResponse)
async def extract_and_translate_text(file: UploadFile = File(...), target_lang: str = "en", api_key: str = get_api_key):
    """
    API nhận file ảnh, trích xuất văn bản, nhận diện ngôn ngữ, và dịch sang ngôn ngữ đích.
    """
    if not file:
        raise HTTPException(status_code=400, detail="No image file uploaded")

    try:
        # Bước 1: Trích xuất văn bản từ ảnh
        image_base64 = encode_image(file)
        extracted_text = extract_text_from_image(image_base64)

        if not extracted_text:
            raise HTTPException(status_code=400, detail="No text found in the image")

        # Bước 2: Phân tích ngôn ngữ của văn bản trích xuất
        analysis_result = analyze_multilingual_text(extracted_text)

        translations = []

        for lang in analysis_result.get("language", []):
            original_text = lang["text"]
            detected_language = lang["name"]
            language_code = lang["code"]

            # Bước 3: Dịch từng đoạn văn bản sang ngôn ngữ đích
            translated_text = translate_text(original_text, target_lang)

            translations.append(
                TranslatedFragment(
                    original_text=original_text,
                    detected_language=detected_language,
                    language_code=language_code,
                    translated_text=translated_text
                )
            )

        return ImgToTextTranslateResponse(
            extracted_text=extracted_text,
            translations=translations
        )

    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

