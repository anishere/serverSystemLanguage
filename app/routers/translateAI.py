# app/routers/translateAI.py
from fastapi import APIRouter, HTTPException
from app.models.translateAI import TranslateRequest, TranslateResponse
from app.config import settings
from app.security.security import get_api_key
from chatbot.services.translate_agent import TranslatorAgent
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Tạo instance router
router = APIRouter(prefix="/translateAI", tags=["translateAI"])

# Khởi tạo TranslatorAgent một lần
translator_agent = TranslatorAgent(model_name="openai")

@router.post("/translate", response_model=TranslateResponse)
async def translate(request: TranslateRequest, api_key: str = get_api_key):
    """
    API dịch văn bản từ ngôn ngữ A sang ngôn ngữ B.
    """
    if not request.text or not request.src_lang or not request.tgt_lang:
        raise HTTPException(status_code=400, detail="Invalid input parameters")

    try:
        # Đóng gói thông tin dịch thuật vào câu hỏi theo format
        # "TRANSLATE|src_lang|tgt_lang|text_to_translate"
        formatted_question = f"TRANSLATE|{request.src_lang}|{request.tgt_lang}|{request.text}"
        logger.info(f"Formatted question: {formatted_question}")
        
        # Gọi TranslatorAgent với câu hỏi đã định dạng
        result = translator_agent.get_workflow().compile().invoke(
            input={"question": formatted_question}
        )
        
        logger.info(f"Translation result: {result}")
        
        # Lấy kết quả từ generation
        return TranslateResponse(translated_text=result["generation"])
    except Exception as e:
        logger.error(f"Translation error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))