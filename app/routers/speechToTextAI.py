from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Header, BackgroundTasks
import os
import json
from app.models.speechToTextAI import SpeechToTextResponse
from app.security.security import get_api_key
from chatbot.services.speech_to_text_agent import SpeechToTextAgent
import logging
from typing import Optional

# Thiết lập logging chi tiết hơn
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Khởi tạo router
router = APIRouter(prefix="/api", tags=["speechToTextAI"])

# Khởi tạo SpeechToTextAgent
stt_agent = SpeechToTextAgent()

@router.post("/speech-to-text/", response_model=SpeechToTextResponse)
async def speech_to_text_endpoint(
    file: UploadFile = File(...), 
    language: Optional[str] = Query(None, description="Mã ngôn ngữ ISO-639 (ví dụ: vi, en, ja) để hỗ trợ nhận dạng"), 
    background_tasks: BackgroundTasks = None,
    x_audio_language_hint: Optional[str] = Header(None, description="Gợi ý ngôn ngữ qua header"),
    x_audio_quality: Optional[str] = Header("high", description="Chất lượng audio (low/medium/high)"),
    api_key: str = get_api_key
):
    """
    API tự động nhận diện ngôn ngữ và chuyển giọng nói thành văn bản bằng OpenAI Whisper API.
    
    Args:
        file (UploadFile): File âm thanh cần chuyển đổi.
        language (str, optional): Mã ngôn ngữ để hướng dẫn nhận diện (ví dụ: vi, en, ja).
        background_tasks (BackgroundTasks): Tác vụ nền FastAPI.
        x_audio_language_hint (str, optional): Gợi ý ngôn ngữ qua header (ưu tiên cao hơn tham số language).
        x_audio_quality (str, optional): Thông tin chất lượng audio giúp xử lý tốt hơn.
        api_key (str): API key xác thực.
        
    Returns:
        SpeechToTextResponse: Kết quả chuyển đổi gồm văn bản và ngôn ngữ nhận diện.
    """
    start_time = __import__("time").time()  # Đo thời gian xử lý
    
    try:
        # Ưu tiên ngôn ngữ hint từ header
        detect_language = x_audio_language_hint or language
        
        # Log thông tin request
        logger.info(f"Processing audio file: {file.filename} | Size: {file.size if hasattr(file, 'size') else 'unknown'} | Language hint: {detect_language}")
        
        # Đọc dữ liệu file
        file_content = await file.read()
        file_size = len(file_content)
        
        # Kiểm tra nhanh kích thước file
        if file_size == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        # Kiểm tra giới hạn kích thước (25MB cho Whisper API)
        max_size = 25 * 1024 * 1024
        if file_size > max_size:
            raise HTTPException(status_code=400, detail=f"File too large: {file_size} bytes. Maximum allowed: {max_size} bytes")
        
        # Lưu file tạm thời
        temp_file_path = stt_agent.save_temporary_file(file_content, file.filename)
        
        # Thêm thông tin hữu ích để debug
        logger.info(f"Temporary file saved at: {temp_file_path} | Size: {file_size} bytes")
        
        # Chuẩn bị thông tin cho agent - truyền thêm thông tin về chất lượng audio nếu cần
        formatted_question = f"TRANSCRIBE|{temp_file_path}|{detect_language if detect_language else ''}"
        if x_audio_quality and x_audio_quality.lower() != "high":
            # Thêm thông tin chất lượng vào log để tương lai có thể xử lý cụ thể hơn
            logger.info(f"Audio quality specified: {x_audio_quality}")
        
        # Gọi agent để xử lý
        result = stt_agent.get_workflow().compile().invoke(
            input={
                "question": formatted_question,
                "generation": "",
                "documents": []
            }
        )
        
        # Parse kết quả JSON từ generation
        try:
            transcription_result = json.loads(result["generation"])
            
            # Kiểm tra nếu có lỗi
            if "error" in transcription_result and transcription_result["error"]:
                raise HTTPException(status_code=500, detail=transcription_result["error"])
            
            # Kiểm tra kết quả trống
            if not transcription_result.get("transcript"):
                logger.warning(f"Empty transcript returned for file: {file.filename}")
            
            # Ghi log kết quả thành công
            processing_time = __import__("time").time() - start_time
            logger.info(f"Transcription successful | Language: {transcription_result['detected_language']} | " +
                       f"Length: {len(transcription_result['transcript'])} chars | Time: {processing_time:.2f}s")
            
            # Lên lịch xóa file tạm sau khi xử lý - sử dụng background_tasks nếu có
            if background_tasks and os.path.exists(temp_file_path):
                background_tasks.add_task(lambda p: os.remove(p) if os.path.exists(p) else None, temp_file_path)
                logger.info(f"Scheduled cleanup for temporary file: {temp_file_path}")
            else:
                # Xóa file ngay lập tức nếu không có background_tasks
                try:
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
                        logger.info(f"Removed temporary file: {temp_file_path}")
                except Exception as e:
                    logger.warning(f"Failed to remove temporary file: {e}")
            
            # Trả về kết quả
            return SpeechToTextResponse(
                detected_language=transcription_result["detected_language"],
                transcript=transcription_result["transcript"]
            )
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing result JSON: {e}")
            # Log raw output to aid debugging
            logger.error(f"Raw output: {result['generation'][:200]}...")
            raise HTTPException(status_code=500, detail=f"Error processing transcription result: {e}")

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Xử lý lỗi chung
        logger.error(f"Error in speech-to-text endpoint: {str(e)}", exc_info=True)
        
        # Cố gắng xóa file tạm nếu có lỗi
        try:
            if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                logger.info(f"Cleaned up temporary file after error: {temp_file_path}")
        except Exception as cleanup_error:
            logger.warning(f"Error during cleanup: {cleanup_error}")
        
        raise HTTPException(status_code=500, detail=f"Transcription error: {str(e)}")