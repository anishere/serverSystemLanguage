import os
import time
import json
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Request, Depends, Security
from fastapi.responses import FileResponse, JSONResponse
from app.security.security import get_api_key
from app.models.docx_translate import DocxTranslationRequest, DocxTranslationResponse
from app.config import settings
import logging
import uuid
import re
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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

# Các mẫu regex để phát hiện URL, email, và tên miền - tương tự maintest.py
URL_PATTERN = re.compile(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+(?:/[-\w%!$&\'()*+,;=:@/~]*)?(?:\?[-\w%!$&\'()*+,;=:@/~]*)?(?:#[-\w%!$&\'()*+,;=:@/~]*)?')
EMAIL_PATTERN = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
DOMAIN_PATTERN = re.compile(r'(?<!\w)(?:www\.)?[a-zA-Z0-9][-a-zA-Z0-9]*(?:\.[a-zA-Z0-9][-a-zA-Z0-9]*)+(?!\w)')

def setup_openai_api(api_key, target_lang="en", model="gpt-4o-mini", temperature=0.3, request_timeout=60):
    """
    Thiết lập hàm gọi API OpenAI để dịch văn bản - tích hợp từ maintest.py
    
    Returns:
        callable: Hàm dịch văn bản sử dụng OpenAI API
    """
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST"]
    )
    session.mount('https://', HTTPAdapter(max_retries=retries))
    
    api_url = "https://api.openai.com/v1/chat/completions"
    
    def preprocess_text(text):
        """
        Tiền xử lý văn bản để bảo vệ URLs, emails, và tên miền
        Thay thế chúng bằng placeholders để tránh bị dịch
        """
        if not text or not text.strip():
            return text, {}
        
        # Dictionary để lưu trữ ánh xạ giữa placeholders và giá trị thực
        placeholders = {}
        
        # Hàm thay thế và lưu trữ trong dict
        def replace_and_store(pattern, prefix):
            nonlocal text, placeholders
            
            def replacer(match):
                original = match.group(0)
                placeholder = f"___{prefix}_{len(placeholders)}___"
                placeholders[placeholder] = original
                return placeholder
            
            text = pattern.sub(replacer, text)
        
        # Xử lý URLs
        replace_and_store(URL_PATTERN, "URL")
        # Xử lý emails
        replace_and_store(EMAIL_PATTERN, "EMAIL")
        # Xử lý tên miền
        replace_and_store(DOMAIN_PATTERN, "DOMAIN")
        
        return text, placeholders
    
    def postprocess_text(text, placeholders):
        """
        Hậu xử lý văn bản để khôi phục URLs, emails, và tên miền từ placeholders
        """
        if not text or not placeholders:
            return text
        
        # Thay thế các placeholders bằng giá trị thực
        for placeholder, original in placeholders.items():
            text = text.replace(placeholder, original)
        
        return text
    
    def translate_text(text):
        """
        Dịch văn bản sử dụng OpenAI API
        """
        if not text or not text.strip():
            return text
        
        try:
            # Tiền xử lý để bảo vệ URLs, emails, và domains
            processed_text, placeholders = preprocess_text(text)
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            # Sử dụng prompt chuyên nghiệp
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a professional translator. Translate the given text "
                        f"to {target_lang} accurately and naturally. "
                        "Maintain the original style, formatting, and tone. "
                        "DO NOT translate placeholders that look like ___URL_X___, ___EMAIL_X___, or ___DOMAIN_X___. "
                        "For ambiguous words, consider the context to choose the appropriate meaning (e.g., 'course' could be translated differently in educational contexts versus geographical contexts). "
                        "Return ONLY the translated text without explanations or notes."
                    )
                },
                {
                    "role": "user",
                    "content": f"Translate this text to {target_lang}:\n\n{processed_text}"
                }
            ]
            
            data = {
                "model": model,
                "messages": messages,
                "temperature": temperature
            }
            
            response = session.post(
                api_url,
                headers=headers,
                data=json.dumps(data),
                timeout=request_timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                translated_text = result["choices"][0]["message"]["content"].strip()
                
                # Kiểm tra nếu văn bản trả về là thông báo lỗi
                error_messages = [
                    "It seems that there is no text provided",
                    "Please provide the text you would like",
                    "There is no text to translate",
                    "No content to translate"
                ]
                
                if any(error_msg in translated_text for error_msg in error_messages):
                    return text
                
                # Hậu xử lý để khôi phục URLs, emails và domains
                final_text = postprocess_text(translated_text, placeholders)
                
                return final_text
            else:
                logger.error(f"Lỗi API: {response.status_code}")
                if response.status_code == 429:
                    logger.warning("Rate limit exceeded. Đợi và thử lại...")
                    time.sleep(20)  # Đợi 20 giây trước khi thử lại
                    return translate_text(text)  # Thử lại
                
                return text
                
        except requests.exceptions.Timeout:
            logger.error(f"Timeout khi gọi API OpenAI sau {request_timeout}s")
            return text
        except requests.exceptions.RequestException as e:
            logger.error(f"Lỗi kết nối khi gọi API OpenAI: {e}")
            return text
        except Exception as e:
            logger.error(f"Lỗi không xác định khi gọi API OpenAI: {e}")
            return text
    
    return translate_text

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

def generate_progress_bar(progress, width=100):
    """
    Tạo thanh tiến trình dạng unicode cho client
    
    Args:
        progress: Giá trị phần trăm tiến trình (0-100)
        width: Độ rộng của thanh tiến trình
        
    Returns:
        Dict chứa thông tin về thanh tiến trình
    """
    # Đảm bảo giá trị progress hợp lệ
    progress = max(0, min(100, progress))
    
    # Ký tự lấp đầy
    fill_char = '█'
    # Số lượng ký tự cần lấp đầy dựa trên tỷ lệ phần trăm
    filled_length = int(width * progress // 100)
    
    # Tạo thanh tiến trình
    bar = fill_char * filled_length + ' ' * (width - filled_length)
    
    # Tạo chuỗi hoàn chỉnh như terminal: 45%|████████████████████████████████                  | 45/100 [00:15<00:18,  3.03it/s]
    return {
        "progress_percent": progress,
        "progress_bar": bar,
        "progress_text": f"{progress}%|{bar}| {progress}/{100}"
    }

@router.post("/translate-docx", response_model=DocxTranslationResponse)
async def translate_docx(
    background_tasks: BackgroundTasks,
    request: Request,
    file: UploadFile = File(...),
    target_language: str = Form(...),
    style: Optional[str] = Form("General"),
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
    
    # Tạo một ID duy nhất cho công việc dịch - sử dụng UUID
    translation_id = str(uuid.uuid4())
    timestamp = int(time.time())
    
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
            # Khởi tạo trạng thái ban đầu với thông tin chi tiết hơn
            initial_status = {
                "id": translation_id,
                "status": "started",
                "progress": 0,
                "started_at": time.time(),
                "updated_at": time.time(),
                "filename": output_file_path.name,
                "target_language": target_language,
                "style": style,
                "model": model,
                "temperature": temperature,
                "workers": workers,
                "message": "Đang chuẩn bị dịch tài liệu...",
                "estimated_time_remaining": None,
                "document_analysis": None,
                "processing_details": {
                    "current_step": "init",
                    "total_paragraphs": 0,
                    "processed_paragraphs": 0,
                    "total_files": 0,
                    "processed_files": 0,
                    "cache_stats": None
                },
                # Thông tin hiển thị tiến trình
                "progress_display": {
                    "progress_bar": generate_progress_bar(0),
                    "stage_info": "Khởi tạo",
                    "current_file": "",
                    "elapsed_time_str": "00:00",
                    "eta_str": "--:--",
                    "speed": ""
                }
            }
            
            # Lưu trạng thái ban đầu
            update_status_file(status_file_path, initial_status, preserve_started_at=False)
            
            # Cấu hình hàm callback để cập nhật trạng thái tiến trình
            def progress_callback(current, total):
                nonlocal initial_status
                
                # Tính toán phần trăm hoàn thành
                progress_percent = min(99, int(current / total * 100)) if total > 0 else 0
                
                # Tính toán thời gian còn lại
                elapsed_time = time.time() - initial_status["started_at"]
                if progress_percent > 0:
                    time_per_percent = elapsed_time / progress_percent
                    estimated_time_remaining = time_per_percent * (100 - progress_percent)
                    
                    # Tính tốc độ xử lý
                    items_per_second = current / elapsed_time if elapsed_time > 0 else 0
                    
                    # Định dạng thời gian
                    elapsed_mins, elapsed_secs = divmod(int(elapsed_time), 60)
                    elapsed_hrs, elapsed_mins = divmod(elapsed_mins, 60)
                    
                    if estimated_time_remaining:
                        eta_mins, eta_secs = divmod(int(estimated_time_remaining), 60)
                        eta_hrs, eta_mins = divmod(eta_mins, 60)
                        eta_str = f"{eta_hrs:02d}:{eta_mins:02d}:{eta_secs:02d}"
                    else:
                        eta_str = "--:--"
                else:
                    estimated_time_remaining = None
                    items_per_second = 0
                    elapsed_mins, elapsed_secs = divmod(int(elapsed_time), 60)
                    elapsed_hrs, elapsed_mins = divmod(elapsed_mins, 60)
                    eta_str = "--:--"
                
                # Định dạng thời gian đã trôi qua
                elapsed_time_str = f"{elapsed_hrs:02d}:{elapsed_mins:02d}:{elapsed_secs:02d}"
                
                # Thanh tiến trình kiểu terminal
                # Tạo thông tin hiển thị tiến trình kiểu terminal
                progress_display = {
                    "progress_bar": generate_progress_bar(progress_percent),
                    "stage_info": "Xử lý các đoạn văn bản",
                    "current_item": f"{current}/{total} đoạn",
                    "elapsed_time_str": elapsed_time_str,
                    "eta_str": eta_str,
                    "speed": f"{items_per_second:.2f} đoạn/giây",
                    "terminal_style": f"{progress_percent}%|{'█' * (progress_percent // 2)}{' ' * (50 - progress_percent // 2)}| {current}/{total} [{elapsed_time_str}<{eta_str}, {items_per_second:.2f}it/s]"
                }
                
                # Cập nhật thông tin tiến trình
                status_updates = {
                    "status": "processing",
                    "progress": progress_percent,
                    "updated_at": time.time(),
                    "message": f"Đang dịch tài liệu... {progress_percent}%",
                    "estimated_time_remaining": estimated_time_remaining,
                    "processing_details": {
                        "current_step": "translating",
                        "processed_paragraphs": current,
                        "total_paragraphs": total
                    },
                    "progress_display": progress_display
                }
                
                # Lưu trạng thái cập nhật
                update_status_file(status_file_path, status_updates)
            
            # Cách 1: Sử dụng docx_translator_agent
            # Đóng gói thông tin dịch thuật vào câu hỏi theo format
            # "TRANSLATE_DOCX|input_path|output_path|target_lang|model|temperature|workers|status_file|style"
            formatted_question = f"TRANSLATE_DOCX|{input_file_path}|{output_file_path}|{target_language}|{model}|{temperature}|{workers}|{status_file_path}|{style}"
            logger.info(f"Formatted question: {formatted_question}")
            
            # Gọi DocxTranslatorAgent với câu hỏi đã định dạng
            result = docx_translator_agent.get_workflow().compile().invoke(
                input={"question": formatted_question}
            )
            
            logger.info(f"Translation result: {result}")
            
            # Cập nhật trạng thái hoàn thành với thông tin chi tiết
            # Tạo thanh tiến trình hoàn thành 100%
            progress_bar = generate_progress_bar(100)
            
            # Tính thời gian đã trôi qua
            elapsed_time = time.time() - initial_status["started_at"]
            elapsed_mins, elapsed_secs = divmod(int(elapsed_time), 60)
            elapsed_hrs, elapsed_mins = divmod(elapsed_mins, 60)
            elapsed_time_str = f"{elapsed_hrs:02d}:{elapsed_mins:02d}:{elapsed_secs:02d}"
            
            completion_status = {
                "status": "completed",
                "progress": 100,
                "message": "Dịch tài liệu hoàn tất",
                "estimated_time_remaining": 0,
                "progress_display": {
                    "progress_bar": progress_bar,
                    "stage_info": "Hoàn thành",
                    "elapsed_time_str": elapsed_time_str,
                    "eta_str": "00:00",
                    "terminal_style": f"100%|{'█' * 50}| Hoàn thành [{elapsed_time_str}, Xong!]"
                }
            }
            
            # Lưu trạng thái hoàn thành (giữ nguyên started_at)
            update_status_file(status_file_path, completion_status)
            
            # Dọn dẹp tệp tải lên sau khi xử lý
            if input_file_path.exists():
                os.remove(input_file_path)
                
        except Exception as e:
            logger.error(f"Lỗi trong quá trình dịch: {e}", exc_info=True)
            
            # Tính thời gian đã trôi qua cho đến khi lỗi
            elapsed_time = time.time() - initial_status["started_at"]
            elapsed_mins, elapsed_secs = divmod(int(elapsed_time), 60)
            elapsed_hrs, elapsed_mins = divmod(elapsed_mins, 60)
            elapsed_time_str = f"{elapsed_hrs:02d}:{elapsed_mins:02d}:{elapsed_secs:02d}"
            
            # Cập nhật trạng thái lỗi với thông tin chi tiết
            error_status = {
                "status": "error",
                "message": f"Lỗi: {str(e)}",
                "error": str(e),
                "error_details": {
                    "type": type(e).__name__,
                    "traceback": logger.formatException(e) if hasattr(logger, 'formatException') else str(e)
                },
                "progress_display": {
                    "progress_bar": generate_progress_bar(0),
                    "stage_info": "Lỗi",
                    "elapsed_time_str": elapsed_time_str,
                    "eta_str": "--:--",
                    "terminal_style": f"Lỗi sau [{elapsed_time_str}]: {str(e)}"
                }
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
        
        # Tính toán thêm các thông tin chi tiết để trả về
        if "started_at" in status_data and status_data["progress"] < 100 and status_data["status"] != "error":
            elapsed_time = time.time() - status_data["started_at"]
            status_data["elapsed_time"] = elapsed_time
            
            # Ước tính thời gian còn lại
            if status_data["progress"] > 0:
                time_per_percent = elapsed_time / status_data["progress"]
                estimated_time_remaining = time_per_percent * (100 - status_data["progress"])
                status_data["estimated_time_remaining"] = estimated_time_remaining
                
            # Thêm hoặc cập nhật thông tin hiển thị tiến trình nếu chưa có
            if "progress_display" not in status_data:
                # Định dạng thời gian đã trôi qua
                elapsed_mins, elapsed_secs = divmod(int(elapsed_time), 60)
                elapsed_hrs, elapsed_mins = divmod(elapsed_mins, 60)
                elapsed_time_str = f"{elapsed_hrs:02d}:{elapsed_mins:02d}:{elapsed_secs:02d}"
                
                # Định dạng thời gian còn lại
                if "estimated_time_remaining" in status_data and status_data["estimated_time_remaining"]:
                    eta_mins, eta_secs = divmod(int(status_data["estimated_time_remaining"]), 60)
                    eta_hrs, eta_mins = divmod(eta_mins, 60)
                    eta_str = f"{eta_hrs:02d}:{eta_mins:02d}:{eta_secs:02d}"
                else:
                    eta_str = "--:--"
                
                # Xác định thông tin giai đoạn hiện tại
                current_step = status_data.get("processing_details", {}).get("current_step", "processing")
                stage_info = "Đang xử lý"
                if current_step == "analyzing":
                    stage_info = "Phân tích tài liệu"
                elif current_step == "translating":
                    stage_info = "Dịch văn bản"
                
                # Tạo thông tin hiển thị progress bar
                processed = status_data.get("processing_details", {}).get("processed_paragraphs", 0)
                total = status_data.get("processing_details", {}).get("total_paragraphs", 100)
                
                # Tính tốc độ xử lý
                speed = processed / elapsed_time if elapsed_time > 0 and processed > 0 else 0
                
                # Tạo thanh tiến trình kiểu terminal
                terminal_style = f"{status_data['progress']}%|{'█' * (status_data['progress'] // 2)}{' ' * (50 - status_data['progress'] // 2)}| {processed}/{total} [{elapsed_time_str}<{eta_str}, {speed:.2f}it/s]"
                
                status_data["progress_display"] = {
                    "progress_bar": generate_progress_bar(status_data["progress"]),
                    "stage_info": stage_info,
                    "current_item": f"{processed}/{total} đoạn",
                    "elapsed_time_str": elapsed_time_str,
                    "eta_str": eta_str,
                    "speed": f"{speed:.2f} đoạn/giây",
                    "terminal_style": terminal_style
                }
        
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