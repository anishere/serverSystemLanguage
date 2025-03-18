from langgraph.graph import END, StateGraph, START
from typing import Dict, Any, List
from app.config import settings
from chatbot.utils.speech_to_text_generator import SpeechToTextGenerator
import os
import logging
import uuid

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Đặt thư mục tạm thời để lưu file âm thanh
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class SpeechToTextAgent:
    """
    Lớp SpeechToTextAgent chịu trách nhiệm điều phối quá trình chuyển đổi âm thanh thành văn bản.
    """

    def __init__(self) -> None:
        """
        Khởi tạo SpeechToTextAgent với STT Generator.
        """
        self.stt_generator = SpeechToTextGenerator()

    def transcribe_audio(self, state) -> Dict[str, Any]:
        """
        Chuyển đổi âm thanh thành văn bản.
        
        Args:
            state: Trạng thái đồ thị hiện tại.
            
        Returns:
            Dict[str, Any]: Trạng thái đồ thị mới bao gồm kết quả chuyển đổi.
        """
        # Format câu hỏi: "TRANSCRIBE|temp_file_path|language_code"
        parts = state["question"].split("|")
        
        if len(parts) >= 2 and parts[0] == "TRANSCRIBE":
            audio_file_path = parts[1]
            language = parts[2] if len(parts) > 2 and parts[2] else None
            
            # Kiểm tra file
            if not self.stt_generator.validate_audio_file(audio_file_path):
                error_message = f"Invalid audio file: {audio_file_path}"
                logger.error(error_message)
                return {
                    "generation": '{"error": "' + error_message + '", "detected_language": "unknown", "transcript": ""}',
                    "documents": state.get("documents", [])
                }
            
            # Chuyển đổi âm thanh thành văn bản
            try:
                result = self.stt_generator.transcribe(audio_file_path, language)
                
                # Định dạng kết quả thành JSON string để dễ dàng parse bên router
                import json
                result_json = json.dumps(result)
                
                # Cố gắng xóa file tạm sau khi xử lý
                try:
                    if os.path.exists(audio_file_path):
                        os.remove(audio_file_path)
                        logger.info(f"Temporary file removed: {audio_file_path}")
                except Exception as e:
                    logger.warning(f"Failed to remove temporary file: {e}")
                
                return {
                    "generation": result_json,
                    "documents": state.get("documents", [])
                }
            except Exception as e:
                error_message = f"Error during audio transcription: {str(e)}"
                logger.error(error_message)
                return {
                    "generation": '{"error": "' + error_message + '", "detected_language": "unknown", "transcript": ""}',
                    "documents": state.get("documents", [])
                }
        else:
            error_message = "Invalid request format. Use: TRANSCRIBE|file_path|language_code"
            logger.error(error_message)
            return {
                "generation": '{"error": "' + error_message + '", "detected_language": "unknown", "transcript": ""}',
                "documents": state.get("documents", [])
            }

    def get_workflow(self):
        """
        Thiết lập luồng xử lý của quá trình chuyển đổi âm thanh thành văn bản.
        
        Returns:
            StateGraph: Đồ thị trạng thái định nghĩa luồng xử lý.
        """
        from typing_extensions import TypedDict
        
        class GraphState(TypedDict):
            question: str
            generation: str
            documents: List[str]
            
        workflow = StateGraph(GraphState)
        workflow.add_node("transcribe", self.transcribe_audio)
        workflow.add_edge(START, "transcribe")
        workflow.add_edge("transcribe", END)
        
        return workflow

    @staticmethod
    def save_temporary_file(file_data: bytes, filename: str = None) -> str:
        """
        Lưu dữ liệu file tạm thời và trả về đường dẫn.
        
        Args:
            file_data (bytes): Dữ liệu file âm thanh.
            filename (str, optional): Tên file gốc.
            
        Returns:
            str: Đường dẫn đến file tạm thời.
        """
        # Tạo tên file duy nhất
        unique_filename = f"{uuid.uuid4()}_{filename if filename else 'audio'}"
        file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        
        # Lưu file
        with open(file_path, "wb") as f:
            f.write(file_data)
        
        logger.info(f"Temporary file saved: {file_path}")
        return file_path