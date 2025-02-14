# routers/imgToText.py
from fastapi import APIRouter, HTTPException, UploadFile, File
from app.models.imgToText import ImgToTextResponse
from app.config import settings
import openai
import base64
from app.security.security import get_api_key

# Cấu hình client OpenAI
client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)

# Tạo instance router
router = APIRouter(prefix="/imgToText", tags=["imgToText"])

def encode_image(file: UploadFile) -> str:
    """
    Chuyển đổi file ảnh thành chuỗi base64.
    """
    try:
        image_bytes = file.file.read()  # Đọc dữ liệu file ảnh
        return base64.b64encode(image_bytes).decode("utf-8")
    except Exception as e:
        raise RuntimeError(f"Error encoding image: {e}")

def extract_text_from_image(image_base64: str) -> str:
    """
    Trích xuất văn bản từ hình ảnh bằng OpenAI GPT-4o Mini.
    """
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
            max_tokens=300,
            messages=messages
        )

        return response.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"Error during text extraction: {e}")

@router.post("/extract", response_model=ImgToTextResponse)
async def extract_text(file: UploadFile = File(...), api_key: str = get_api_key):
    """
    API nhận file ảnh, chuyển đổi sang base64 và trích xuất văn bản.
    """
    if not file:
        raise HTTPException(status_code=400, detail="No image file uploaded")

    try:
        # Chuyển đổi file ảnh sang base64
        image_base64 = encode_image(file)

        # Trích xuất văn bản từ hình ảnh
        extracted_text = extract_text_from_image(image_base64)

        return ImgToTextResponse(extracted_text=extracted_text)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
