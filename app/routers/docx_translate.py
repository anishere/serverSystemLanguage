import os
import time
import json
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Request, Depends, Security
from fastapi.responses import FileResponse, JSONResponse
from app.security.security import get_api_key
from app.models.docx_translate import DocxTranslationRequest, DocxTranslationResponse, SupportedLanguage
from app.config import settings
import logging
import uuid

# Import DocxTranslatorAgent
from chatbot.services.docx_translate_agent import DocxTranslatorAgent

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Tạo instance router
router = APIRouter(prefix="/document", tags=["document"])

# Khởi tạo DocxTranslatorAgent một lần
docx_translator_agent = DocxTranslatorAgent(model_name="openai")

# Create uploads, downloads and status directories if they don't exist
SERVER_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
UPLOAD_DIR = SERVER_DIR / "uploads"
DOWNLOAD_DIR = SERVER_DIR / "downloads"
STATUS_DIR = SERVER_DIR / "status"
UPLOAD_DIR.mkdir(exist_ok=True)
DOWNLOAD_DIR.mkdir(exist_ok=True)
STATUS_DIR.mkdir(exist_ok=True)

def update_status_file(status_file_path, updates, preserve_started_at=True):
    """
    Helper function to update the status file properly
    
    Args:
        status_file_path: Path to the status file
        updates: Dict containing fields to update
        preserve_started_at: Whether to preserve the original started_at value
    """
    try:
        # Read existing data if file exists
        if os.path.exists(status_file_path):
            with open(status_file_path, 'r', encoding='utf-8') as f:
                status_data = json.load(f)
            
            # Keep the original started_at if preserve_started_at is True
            original_started_at = status_data.get("started_at")
        else:
            status_data = {}
            original_started_at = None
        
        # Update fields
        status_data.update(updates)
        
        # Preserve started_at if needed
        if preserve_started_at and "started_at" in updates and original_started_at:
            status_data["started_at"] = original_started_at
        
        # Always update updated_at
        status_data["updated_at"] = time.time()
        
        # Write back to file
        with open(status_file_path, 'w', encoding='utf-8') as f:
            json.dump(status_data, f, ensure_ascii=False, indent=2)
        
        return status_data
    except Exception as e:
        logger.error(f"Error updating status file: {e}", exc_info=True)
        return None

@router.post("/translate-docx", response_model=DocxTranslationResponse)
async def translate_docx(
    background_tasks: BackgroundTasks,
    request: Request,
    file: UploadFile = File(...),
    target_language: SupportedLanguage = Form(...),
    model: Optional[str] = Form("gpt-4o-mini"),
    temperature: Optional[float] = Form(0.3),
    workers: Optional[int] = Form(4),
    api_key: str = get_api_key
):
    """
    API dịch tài liệu DOCX từ ngôn ngữ nguồn sang ngôn ngữ đích.
    """
    # Kiểm tra định dạng tệp
    if not file.filename.lower().endswith('.docx'):
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ tệp DOCX")
    
    # Tạo một ID duy nhất cho công việc dịch - sử dụng timestamp
    timestamp = int(time.time())
    translation_id = str(timestamp)
    
    # Tạo tên tệp duy nhất để tránh trùng lặp
    original_filename = file.filename
    base_name = Path(original_filename).stem
    input_file_path = UPLOAD_DIR / f"{base_name}_{timestamp}.docx"
    output_file_path = DOWNLOAD_DIR / f"{base_name}_{target_language}_{timestamp}.docx"
    status_file_path = STATUS_DIR / f"{translation_id}.json"
    
    # Lưu tệp tải lên
    try:
        with open(input_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi lưu tệp: {str(e)}")
    
    # Hàm xử lý dịch trong background
    def process_translation():
        try:
            # Khởi tạo trạng thái ban đầu
            initial_status = {
                "id": translation_id,
                "status": "started",
                "progress": 0,
                "started_at": time.time(),
                "updated_at": time.time(),
                "filename": output_file_path.name,
                "target_language": target_language,
                "message": "Đang chuẩn bị dịch tài liệu...",
                "estimated_time_remaining": None
            }
            
            # Lưu trạng thái ban đầu
            update_status_file(status_file_path, initial_status, preserve_started_at=False)
                
            # Đóng gói thông tin dịch thuật vào câu hỏi theo format
            # "TRANSLATE_DOCX|input_path|output_path|target_lang|model|temperature|workers"
            formatted_question = f"TRANSLATE_DOCX|{input_file_path}|{output_file_path}|{target_language}|{model}|{temperature}|{workers}"
            logger.info(f"Formatted question: {formatted_question}")
            
            # Gọi DocxTranslatorAgent với câu hỏi đã định dạng
            result = docx_translator_agent.get_workflow().compile().invoke(
                input={"question": formatted_question}
            )
            
            logger.info(f"Translation result: {result}")
            
            # Cập nhật trạng thái hoàn thành
            completion_status = {
                "status": "completed",
                "progress": 100,
                "message": "Dịch tài liệu hoàn tất",
                "estimated_time_remaining": 0
            }
            
            # Lưu trạng thái hoàn thành (giữ nguyên started_at)
            update_status_file(status_file_path, completion_status)
            
            # Dọn dẹp tệp tải lên sau khi xử lý
            if input_file_path.exists():
                os.remove(input_file_path)
                
        except Exception as e:
            logger.error(f"Lỗi trong quá trình dịch: {e}", exc_info=True)
            
            # Cập nhật trạng thái lỗi
            error_status = {
                "status": "error",
                "message": f"Lỗi: {str(e)}",
                "error": str(e)
            }
            
            # Lưu trạng thái lỗi (giữ nguyên started_at)
            update_status_file(status_file_path, error_status)
    
    # Thêm tác vụ dịch vào hàng đợi background
    background_tasks.add_task(process_translation)
    
    # Tạo URL tải xuống và kiểm tra trạng thái
    base_url = str(request.base_url).rstrip('/')
    download_url = f"{base_url}/document/download/{output_file_path.name}"
    status_url = f"{base_url}/document/status/{translation_id}"
    
    return DocxTranslationResponse(
        filename=output_file_path.name,
        message="Quá trình dịch đã bắt đầu. Tài liệu sẽ có sẵn tại URL đã cung cấp khi hoàn thành.",
        download_url=download_url,
        status_url=status_url,
        translation_id=translation_id
    )

@router.get("/status/{translation_id}")
async def check_translation_status(translation_id: str):
    """
    Kiểm tra trạng thái tiến trình dịch
    """
    status_file_path = STATUS_DIR / f"{translation_id}.json"
    
    if not status_file_path.exists():
        raise HTTPException(status_code=404, detail="Không tìm thấy công việc dịch với ID cung cấp")
    
    try:
        with open(status_file_path, 'r', encoding='utf-8') as f:
            status_data = json.load(f)
        
        return JSONResponse(content=status_data)
    except Exception as e:
        logger.error(f"Lỗi khi đọc trạng thái: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Lỗi khi đọc trạng thái: {str(e)}")

@router.get("/download/{filename}")
async def download_translated_document(filename: str):
    """
    Tải xuống tài liệu đã dịch
    """
    file_path = DOWNLOAD_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Không tìm thấy tệp hoặc dịch thuật đang xử lý")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )

@router.get("/languages")
async def get_supported_languages():
    """
    Lấy danh sách các ngôn ngữ được hỗ trợ
    """
    return {lang.name: lang.value for lang in SupportedLanguage} 