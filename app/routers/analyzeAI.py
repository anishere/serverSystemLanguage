from fastapi import APIRouter, HTTPException, BackgroundTasks
from app.models.analyzeAI import AnalyzeRequest, AnalyzeResponse, LanguageFragment
from app.config import settings
from app.security.security import get_api_key
from chatbot.services.multilingual_analyzer_agent import MultilingualAnalyzerAgent
import logging
import json
import time

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Tạo instance router
router = APIRouter(prefix="/analyzeAI", tags=["analyzeAI"])

# Cache đơn giản trong bộ nhớ
in_memory_cache = {}

# Thời gian cache (giây)
CACHE_TIMEOUT = 3600

def get_cache_key(text: str) -> str:
    """Tạo khóa cache từ văn bản."""
    import hashlib
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def set_cache(key: str, value, timeout: int = CACHE_TIMEOUT):
    """Lưu giá trị vào cache."""
    in_memory_cache[key] = (time.time() + timeout, value)

def get_cache(key: str):
    """Lấy giá trị từ cache."""
    if key in in_memory_cache:
        expiry, value = in_memory_cache[key]
        if time.time() < expiry:
            return value
        else:
            del in_memory_cache[key]
    return None

def clean_cache():
    """Xóa các bản ghi cache cũ."""
    current_time = time.time()
    keys_to_remove = []
    
    for key, (expiry, _) in in_memory_cache.items():
        if current_time > expiry:
            keys_to_remove.append(key)
    
    for key in keys_to_remove:
        del in_memory_cache[key]

# Khởi tạo MultilingualAnalyzerAgent một lần
analyzer_agent = MultilingualAnalyzerAgent(model_name="openai")

@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest, background_tasks: BackgroundTasks, api_key: str = get_api_key):
    """
    API phân tích văn bản đa ngôn ngữ.
    
    Args:
        request (AnalyzeRequest): Yêu cầu phân tích văn bản.
        background_tasks (BackgroundTasks): Tác vụ nền FastAPI.
        api_key (str): API key xác thực.
        
    Returns:
        AnalyzeResponse: Kết quả phân tích ngôn ngữ.
        
    Raises:
        HTTPException: Nếu có lỗi xảy ra trong quá trình phân tích.
    """
    if not request.text:
        raise HTTPException(status_code=400, detail="Text is required")

    try:
        # Kiểm tra cache
        cache_key = get_cache_key(request.text)
        cached_result = get_cache(cache_key)
        
        if cached_result:
            logger.info("Cache hit for analyze request")
            return AnalyzeResponse(language=cached_result)
            
        # Đóng gói thông tin phân tích vào câu hỏi theo format "ANALYZE|text_to_analyze"
        formatted_question = f"ANALYZE|{request.text}"
        logger.info(f"Formatted question: {formatted_question[:100]}...")
        
        # Gọi MultilingualAnalyzerAgent với câu hỏi đã định dạng
        result = analyzer_agent.get_workflow().compile().invoke(
            input={
                "question": formatted_question,
                "generation": "",
                "documents": []
            }
        )
        
        logger.info(f"Analysis result type: {type(result['generation'])}")
        
        # Phân tích kết quả JSON để chuyển thành response API
        try:
            # Nếu là chuỗi JSON, thử parse
            if isinstance(result['generation'], str):
                result_json = json.loads(result['generation'])
            else:
                result_json = result['generation']
                
            # Lấy danh sách ngôn ngữ từ kết quả phân tích
            language_fragments = []
            for lang in result_json.get("language", []):
                language_fragments.append(
                    LanguageFragment(
                        name=lang.get("name", "Unknown"),
                        code=lang.get("code", "und"),
                        text=lang.get("text", "")
                    )
                )
            
            # Lưu vào cache
            set_cache(cache_key, language_fragments)
            
            # Lên lịch làm sạch cache
            background_tasks.add_task(clean_cache)
            
            # Trả kết quả phân tích
            return AnalyzeResponse(language=language_fragments)
        except (json.JSONDecodeError, AttributeError) as e:
            logger.error(f"Error parsing analysis result: {str(e)}")
            return AnalyzeResponse(
                language=[LanguageFragment(
                    name="Error", 
                    code="err", 
                    text=f"Failed to parse analysis result: {str(e)}"
                )]
            )
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))