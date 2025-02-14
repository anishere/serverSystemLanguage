# routers/speechToTextAI.py

from fastapi import APIRouter, UploadFile, File, HTTPException
import os
from uuid import uuid4
import whisper
import torch
from app.models.speechToTextAI import SpeechToTextResponse
from app.security.security import get_api_key

# Đảm bảo rằng model sử dụng CPU hoặc GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SIZE = "medium"  # Bạn có thể thay đổi model size tùy thuộc vào cấu hình máy
model = whisper.load_model(MODEL_SIZE, device=DEVICE)

# Đặt thư mục tạm thời để lưu file âm thanh
UPLOAD_FOLDER = "uploads"

# Khởi tạo router
router = APIRouter(prefix="/api", tags=["speechToTextAI"])

def speech_to_text_with_whisper(audio_path: str) -> dict:
    """
    Nhận diện ngôn ngữ và chuyển giọng nói thành văn bản bằng OpenAI Whisper (GPU nếu có).
    """
    try:
        # Chạy mô hình Whisper để nhận diện ngôn ngữ và chuyển thành văn bản
        result = model.transcribe(audio_path, language=None)

        return {
            "detected_language": result["language"],  # Ngôn ngữ nhận diện được
            "transcript": result["text"]  # Văn bản được chuyển đổi từ giọng nói
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during STT processing: {str(e)}")

@router.post("/speech-to-text/", response_model=SpeechToTextResponse)
async def speech_to_text_endpoint(file: UploadFile = File(...), api_key: str = get_api_key):
    """
    API tự động nhận diện ngôn ngữ và chuyển giọng nói thành văn bản bằng OpenAI Whisper (hỗ trợ GPU).
    
    Trả về:
    {
        "detected_language": "vi",
        "transcript": "Xin chào, tôi đang thử nghiệm nhận diện giọng nói."
    }
    """
    try:
        # Lưu tệp âm thanh vào thư mục tạm thời
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        file_path = os.path.join(UPLOAD_FOLDER, f"{uuid4()}_{file.filename}")
        
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Gọi hàm xử lý STT
        result = speech_to_text_with_whisper(file_path)

        # Xóa file sau khi xử lý
        os.remove(file_path)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
