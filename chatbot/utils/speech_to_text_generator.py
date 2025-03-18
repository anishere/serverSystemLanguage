import openai
from app.config import settings
import logging
import os
from typing import Dict, Any, Optional

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpeechToTextGenerator:
    """
    Lớp SpeechToTextGenerator chịu trách nhiệm chuyển đổi âm thanh thành văn bản
    sử dụng API Whisper của OpenAI, cấu hình tương tự mô hình medium cục bộ.
    """

    def __init__(self) -> None:
        """
        Khởi tạo SpeechToTextGenerator với OpenAI client.
        """
        self.client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        # Các tham số tương tự mô hình Whisper medium cục bộ
        self.temperature = 0  # Giảm xuống 0 để giống với tham số mặc định của local model
        self.initial_prompt = None  # Không sử dụng prompt, giống như mô hình cục bộ

    def transcribe(self, audio_file_path: str, language: Optional[str] = None) -> Dict[str, Any]:
        """
        Chuyển đổi file âm thanh thành văn bản sử dụng Whisper API với cấu hình tối ưu.
        
        Args:
            audio_file_path (str): Đường dẫn đến file âm thanh.
            language (Optional[str]): Mã ngôn ngữ của âm thanh (tùy chọn).
            
        Returns:
            Dict[str, Any]: Kết quả chuyển đổi bao gồm văn bản và ngôn ngữ nhận diện.
        """
        try:
            logger.info(f"Transcribing audio file: {audio_file_path}")
            
            # Mở file âm thanh để đọc dưới dạng binary
            with open(audio_file_path, "rb") as audio_file:
                # Cấu hình tương tự mô hình cục bộ medium
                response = self.client.audio.transcriptions.create(
                    model="whisper-1",  # Sử dụng whisper-1 mới nhất
                    file=audio_file,
                    language=language,  # Nếu None, sẽ tự động nhận diện giống local
                    response_format="verbose_json",
                    temperature=self.temperature,  # Temperature=0 giống cục bộ
                    prompt=self.initial_prompt   # Không sử dụng prompt, giống cục bộ
                )
            
            # Lấy kết quả và ngôn ngữ nhận diện
            transcript = response.text
            detected_language = response.language
            
            logger.info(f"Transcription complete. Detected language: {detected_language}")
            
            # Trả về cùng định dạng như model cục bộ
            return {
                "detected_language": detected_language,
                "transcript": transcript
            }
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            return {
                "error": str(e),
                "detected_language": "unknown",
                "transcript": ""
            }

    def validate_audio_file(self, file_path: str) -> bool:
        """
        Kiểm tra xem file có tồn tại và có phải là file âm thanh hợp lệ không.
        
        Args:
            file_path (str): Đường dẫn đến file âm thanh.
            
        Returns:
            bool: True nếu file hợp lệ, False nếu không.
        """
        # Kiểm tra file có tồn tại không
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return False
        
        # Kiểm tra định dạng file - danh sách định dạng mà mô hình cục bộ hỗ trợ tốt
        valid_extensions = ['.mp3', '.mp4', '.mpeg', '.mpga', '.m4a', '.wav', '.webm']
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext not in valid_extensions:
            logger.error(f"Invalid file format: {file_ext}. Supported formats: {valid_extensions}")
            return False
        
        # Kiểm tra kích thước file (giới hạn của OpenAI là 25MB)
        max_size = 25 * 1024 * 1024  # 25MB in bytes
        file_size = os.path.getsize(file_path)
        
        if file_size > max_size:
            logger.error(f"File too large: {file_size} bytes. Maximum allowed: {max_size} bytes")
            return False
        
        return True