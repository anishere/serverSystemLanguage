from fastapi import APIRouter, HTTPException, UploadFile, File
from app.models.imgToText import ImgToTextResponse
from app.config import settings
import base64
from app.security.security import get_api_key
from chatbot.services.img_to_text_agent import ImgToTextAgent
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Tạo instance router
router = APIRouter(prefix="/imgToText", tags=["imgToText"])

# Khởi tạo ImgToTextAgent một lần
img_to_text_agent = ImgToTextAgent()

def encode_image(file: UploadFile) -> str:
    """
    Chuyển đổi file ảnh thành chuỗi base64.
    """
    try:
        image_bytes = file.file.read()  # Đọc dữ liệu file ảnh
        return base64.b64encode(image_bytes).decode("utf-8")
    except Exception as e:
        raise RuntimeError(f"Error encoding image: {e}")

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
        
        # Reset file cursor để có thể đọc lại nếu cần
        file.file.seek(0)
        
        # Đóng gói thông tin trích xuất vào câu hỏi theo format "EXTRACT|image_base64"
        formatted_question = f"EXTRACT|{image_base64}"
        logger.info(f"Processing image extraction. Image size: {len(image_base64)} bytes")
        
        # Gọi ImgToTextAgent với câu hỏi đã định dạng
        result = img_to_text_agent.get_workflow().compile().invoke(
            input={
                "question": formatted_question,
                "generation": "",
                "documents": []
            }
        )
        
        extracted_text = result["generation"]
        logger.info(f"Text extraction completed. Result length: {len(extracted_text)}")
        
        return ImgToTextResponse(extracted_text=extracted_text)
    except RuntimeError as e:
        logger.error(f"Runtime error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")