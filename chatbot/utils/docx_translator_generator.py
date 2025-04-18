# chatbot/utils/docx_translator_generator.py
import os
import sys
import re
from typing import Optional, Dict, Any, List, Tuple, Callable
from pathlib import Path
import importlib.util
import time
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from chatbot.utils.custom_prompt import CustomPrompt

# Import the existing modules correctly
def import_module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Import the docx_translator module from the server directory
server_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
docx_translator = import_module_from_file("docx_translator", os.path.join(server_dir, "docx_translator.py"))

# Get the classes/functions we need
DocxTranslator = docx_translator.DocxTranslator


class DocxTranslatorGenerator:
    """
    Lớp DocxTranslatorGenerator chịu trách nhiệm dịch tài liệu DOCX
    sử dụng mô hình ngôn ngữ và DocxTranslator.
    """

    def __init__(self, llm) -> None:
        """
        Khởi tạo DocxTranslatorGenerator với mô hình ngôn ngữ (LLM).

        Args:
            llm: Mô hình ngôn ngữ được sử dụng để dịch văn bản.
        """
        self.llm = llm
        
        # Xây dựng prompt từ CustomPrompt
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", CustomPrompt.DOCX_TRANSLATE_PROMPT),
                ("human", "Here is the text to translate to {target_lang} with {style} style:\n\n{text}")
            ]
        )

        # Xây dựng pipeline xử lý
        self.chain = prompt | llm | StrOutputParser()
    
    def get_chain(self) -> RunnableSequence:
        """
        Trả về chuỗi pipeline xử lý để dịch văn bản.
        """
        return self.chain
    
    def generate_progress_bar(self, progress, width=50):
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
    
    def preprocess_text(self, text: str) -> str:
        """
        Tiền xử lý văn bản trước khi dịch để xử lý các vấn đề phổ biến.
        Bảo vệ URL, email và các định dạng đặc biệt.
        
        Args:
            text: Văn bản cần xử lý
            
        Returns:
            str: Văn bản đã xử lý
        """
        if not text or not text.strip():
            return text
        
        # 1. Bảo vệ URL, email và các định dạng đặc biệt
        protected_segments, text = self._protect_special_formats(text)
        
        # 2. Xử lý từ bị dính nhau (thiếu khoảng trắng)
        # Regex tìm những chỗ nên có khoảng trắng giữa từ thường và từ viết hoa
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        
        # 3. Xử lý từ bị dính với số (ngoại trừ các định dạng phổ biến như h1, h2, p3)
        text = re.sub(r'([a-zA-Z][a-zA-Z])(\d)', r'\1 \2', text)
        text = re.sub(r'(\d)([a-zA-Z][a-zA-Z])', r'\1 \2', text)
        
        # 4. Xử lý từ dính với ký tự đặc biệt
        text = re.sub(r'([a-zA-Z])([^\w\s])', r'\1 \2', text)
        text = re.sub(r'([^\w\s])([a-zA-Z])', r'\1 \2', text)
        
        # 5. Sửa các trường hợp như "formationof" -> "formation of"
        common_prefixes = ['pre', 'post', 're', 'un', 'in', 'dis', 'over', 'under', 'co', 'sub', 'inter', 'trans']
        for prefix in common_prefixes:
            if len(prefix) < 3 or prefix not in text.lower():
                continue
            
            # Tìm và sửa các từ không có khoảng trắng giữa tiền tố và phần còn lại
            pattern = fr'({prefix})([a-z]{{2,}})'
            text = re.sub(pattern, fr'\1 \2', text, flags=re.IGNORECASE)
        
        # 6. Sửa trường hợp có nhiều khoảng trắng thành một khoảng trắng
        text = re.sub(r'\s+', ' ', text)
        
        # 7. Khôi phục các định dạng đặc biệt đã bảo vệ
        text = self._restore_protected_segments(text, protected_segments)
        
        return text.strip()
    
    def _protect_special_formats(self, text: str) -> Tuple[List[str], str]:
        """
        Bảo vệ URL, email và các định dạng đặc biệt bằng cách thay thế chúng 
        bằng các placeholder và lưu trữ chúng để khôi phục sau.
        
        Args:
            text: Văn bản đầu vào
            
        Returns:
            Tuple[List[str], str]: Danh sách các định dạng đặc biệt và văn bản đã thay thế
        """
        # Các pattern cần bảo vệ
        patterns = [
            # URL
            r'https?://[^\s]+',
            r'www\.[^\s]+\.[a-zA-Z]{2,}[^\s]*',
            # Email
            r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            # File paths
            r'[a-zA-Z]:\\[^\s]+',
            r'/[a-zA-Z0-9_/.-]+',
            # Mã ID và mã định danh
            r'\b\d+[A-Za-z]+-[A-Za-z0-9]+\b',
            # Placeholders đặc biệt
            r'___[A-Z]+_\d+___',
            # Mã định danh kỹ thuật
            r'\b[A-Z]+\d+[A-Z]+-[A-Z]+\b'
        ]
        
        # Danh sách lưu trữ các segment cần bảo vệ
        protected_segments = []
        
        # Tìm và thay thế từng pattern
        for i, pattern in enumerate(patterns):
            matches = re.finditer(pattern, text)
            offset = 0
            for match in matches:
                # Lấy nội dung matched và vị trí
                start, end = match.span()
                start += offset
                end += offset
                matched_text = text[start:end]
                
                # Tạo placeholder
                placeholder = f"__PROTECTED_{len(protected_segments)}__"
                
                # Lưu nội dung matched
                protected_segments.append(matched_text)
                
                # Thay thế trong text
                text = text[:start] + placeholder + text[end:]
                
                # Cập nhật offset cho lần tiếp theo
                offset += len(placeholder) - len(matched_text)
        
        return protected_segments, text
    
    def _restore_protected_segments(self, text: str, protected_segments: List[str]) -> str:
        """
        Khôi phục các định dạng đặc biệt đã được bảo vệ
        
        Args:
            text: Văn bản với các placeholder
            protected_segments: Danh sách các định dạng đặc biệt đã bảo vệ
            
        Returns:
            str: Văn bản đã khôi phục
        """
        for i, segment in enumerate(protected_segments):
            placeholder = f"__PROTECTED_{i}__"
            text = text.replace(placeholder, segment)
        
        return text
    
    def translate_text(self, text: str, target_lang: str = "en", style: str = "General") -> str:
        """
        Dịch một đoạn văn bản sử dụng chain đã được thiết lập.
        Bảo vệ URLs, email, website khỏi bị dịch.
        """
        if not text or not text.strip():
            return text
        
        # 1. Bảo vệ URLs, email, và các định dạng đặc biệt trước khi dịch
        protected_segments, text_to_translate = self._protect_special_formats(text)
        
        # 2. Tiền xử lý văn bản (xử lý các vấn đề về khoảng trắng)
        processed_text = text_to_translate
        processed_text = re.sub(r'([a-z])([A-Z])', r'\1 \2', processed_text)
        processed_text = re.sub(r'([a-zA-Z][a-zA-Z])(\d)', r'\1 \2', processed_text)
        processed_text = re.sub(r'(\d)([a-zA-Z][a-zA-Z])', r'\1 \2', processed_text)
        processed_text = re.sub(r'([a-zA-Z])([^\w\s])', r'\1 \2', processed_text)
        processed_text = re.sub(r'([^\w\s])([a-zA-Z])', r'\1 \2', processed_text)
        processed_text = re.sub(r'\s+', ' ', processed_text).strip()
        
        # 3. Dịch văn bản đã xử lý, giữ nguyên placeholder
        result = self.chain.invoke({
            "text": processed_text,
            "target_lang": target_lang,
            "style": style
        })
        
        # 4. Kiểm tra và loại bỏ các cụm không mong muốn
        unwanted_phrases = [
            "I'm sorry", 
            "Sorry", 
            "There is no text provided", 
            "Translate this text to",
            "Here is the translation",
            "Translation:"
        ]
        
        for phrase in unwanted_phrases:
            if phrase in result:
                result = result.replace(phrase, "").strip()
        
        # 5. Khôi phục các định dạng đặc biệt (URLs, email, v.v.) trong kết quả dịch
        final_result = self._restore_protected_segments(result, protected_segments)
        
        return final_result

    def _update_status(self, status_file: Optional[str], status_data: Dict[str, Any]) -> None:
        """
        Cập nhật tệp trạng thái với thông tin tiến trình mới
        
        Args:
            status_file: Đường dẫn đến tệp trạng thái
            status_data: Thông tin trạng thái cần cập nhật
        """
        if not status_file:
            return
            
        try:
            # Đọc dữ liệu hiện tại nếu tệp tồn tại
            if os.path.exists(status_file):
                with open(status_file, 'r', encoding='utf-8') as f:
                    current_data = json.load(f)
                
                # Giữ nguyên thời gian bắt đầu
                if "started_at" in current_data and "started_at" not in status_data:
                    status_data["started_at"] = current_data["started_at"]
                
                # Giữ các thông tin khác không thay đổi
                if "id" in current_data and "id" not in status_data:
                    status_data["id"] = current_data["id"]
                
                if "filename" in current_data and "filename" not in status_data:
                    status_data["filename"] = current_data["filename"]
                
                if "target_language" in current_data and "target_language" not in status_data:
                    status_data["target_language"] = current_data["target_language"]
                
                # Cập nhật dữ liệu hiện tại
                current_data.update(status_data)
                status_data = current_data
            
            # Luôn cập nhật thời gian updated_at
            status_data["updated_at"] = time.time()
            
            # Ghi lại vào tệp
            with open(status_file, 'w', encoding='utf-8') as f:
                json.dump(status_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Lỗi khi cập nhật trạng thái: {e}")

    def translate_docx(
        self, 
        input_path: str, 
        output_path: str, 
        target_lang: str, 
        model: str = "gpt-4o-mini", 
        temperature: float = 0.3, 
        workers: int = 4,
        api_key: Optional[str] = None,
        status_file: Optional[str] = None,  # Thêm tham số status_file để theo dõi tiến trình
        progress_callback: Optional[Callable[[int, int], None]] = None,  # Thêm callback cho tiến trình
        style: str = "General"  # Thêm tham số style
    ) -> str:
        """
        Dịch tài liệu DOCX từ ngôn ngữ nguồn sang ngôn ngữ đích.
        
        Args:
            input_path: Đường dẫn đến tập tin DOCX cần dịch
            output_path: Đường dẫn lưu tập tin DOCX đã dịch
            target_lang: Mã ngôn ngữ đích (en, vi, zh, etc.)
            model: Tên mô hình OpenAI để sử dụng
            temperature: Thông số nhiệt độ cho mô hình dịch
            workers: Số lượng worker cho xử lý đa luồng
            api_key: OpenAI API key (sử dụng để khởi tạo hàm dịch)
            status_file: Tệp JSON để lưu trạng thái tiến trình
            progress_callback: Hàm callback để theo dõi tiến trình
            style: Kiểu dịch (General, Professional, Technology, Medical, v.v.)
            
        Returns:
            str: Đường dẫn đến tập tin đã dịch
        """
        # Chuẩn bị paths
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        # Tạo thư mục đích nếu cần
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Khởi tạo trạng thái nếu status_file được cung cấp
        start_time = time.time()
        if status_file:
            status_data = {
                "status": "started",
                "progress": 0,
                "started_at": start_time,
                "updated_at": start_time,
                "filename": output_path.name,
                "target_language": target_lang,
                "style": style,
                "model": model,
                "temperature": temperature,
                "message": "Đang chuẩn bị dịch tài liệu...",
                "estimated_time_remaining": None,
                "progress_display": {
                    "progress_bar": self.generate_progress_bar(0),
                    "stage_info": "Khởi tạo",
                    "current_item": "Đang chuẩn bị",
                    "elapsed_time_str": "00:00:00",
                    "eta_str": "--:--:--",
                    "terminal_style": "0%|                                                  | 0/? [00:00<?, ?it/s]"
                }
            }
            self._update_status(status_file, status_data)
        
        # Tạo hàm callback kết hợp để cập nhật trạng thái và gọi progress_callback nếu có
        def combined_progress_callback(current, total):
            # Cập nhật trạng thái
            if status_file:
                progress_percent = min(99, int(current / total * 100)) if total > 0 else 0
                elapsed_time = time.time() - start_time
                
                # Định dạng thời gian đã trôi qua
                elapsed_mins, elapsed_secs = divmod(int(elapsed_time), 60)
                elapsed_hrs, elapsed_mins = divmod(elapsed_mins, 60)
                elapsed_time_str = f"{elapsed_hrs:02d}:{elapsed_mins:02d}:{elapsed_secs:02d}"
                
                # Ước tính thời gian còn lại
                if progress_percent > 0:
                    time_per_percent = elapsed_time / progress_percent
                    estimated_time_remaining = time_per_percent * (100 - progress_percent)
                    
                    # Định dạng thời gian còn lại
                    eta_mins, eta_secs = divmod(int(estimated_time_remaining), 60)
                    eta_hrs, eta_mins = divmod(eta_mins, 60)
                    eta_str = f"{eta_hrs:02d}:{eta_mins:02d}:{eta_secs:02d}"
                    
                    # Tính tốc độ xử lý
                    items_per_second = current / elapsed_time if elapsed_time > 0 else 0
                else:
                    estimated_time_remaining = None
                    eta_str = "--:--:--"
                    items_per_second = 0
                
                # Tạo thanh tiến trình kiểu terminal
                terminal_style = f"{progress_percent}%|{'█' * (progress_percent // 2)}{' ' * (50 - progress_percent // 2)}| {current}/{total} [{elapsed_time_str}<{eta_str}, {items_per_second:.2f}it/s]"
                
                update_data = {
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
                    "progress_display": {
                        "progress_bar": self.generate_progress_bar(progress_percent),
                        "stage_info": "Dịch văn bản",
                        "current_item": f"{current}/{total} đoạn",
                        "elapsed_time_str": elapsed_time_str,
                        "eta_str": eta_str,
                        "speed": f"{items_per_second:.2f} đoạn/giây",
                        "terminal_style": terminal_style
                    }
                }
                self._update_status(status_file, update_data)
            
            # Gọi callback ban đầu nếu có
            if progress_callback:
                progress_callback(current, total)
        
        try:
            # Cập nhật trạng thái phân tích tài liệu
            if status_file:
                self._update_status(status_file, {
                    "status": "analyzing",
                    "message": "Đang phân tích cấu trúc tài liệu...",
                    "processing_details": {
                        "current_step": "analyzing"
                    },
                    "progress_display": {
                        "progress_bar": self.generate_progress_bar(0),
                        "stage_info": "Phân tích cấu trúc",
                        "current_item": "Đang quét tài liệu",
                        "elapsed_time_str": "00:00:00",
                        "eta_str": "--:--:--",
                        "terminal_style": "Phân tích cấu trúc tài liệu..."
                    }
                })
            
            # Tạo hàm dịch văn bản sử dụng chain đã thiết lập
            def translate_func(text):
                if not text or not text.strip():
                    return text
                return self.translate_text(text, target_lang, style)
            
            # Khởi tạo DocxTranslator với cấu hình đa luồng
            translator = DocxTranslator(
                translate_func=translate_func,
                max_workers=workers
            )
            
            # Dịch file và trả về đường dẫn
            result_path = translator.translate_docx_complete(
                str(input_path),
                str(output_path),
                progress_callback=combined_progress_callback  # Truyền hàm callback kết hợp
            )
            
            # Lấy thông tin phân tích tài liệu từ translator
            document_analysis = {}
            if hasattr(translator, 'document_analysis') and translator.document_analysis:
                document_analysis = translator.document_analysis
                
            # Lấy thông tin về cache
            cache_stats = {}
            if hasattr(translator, 'cache') and hasattr(translator.cache, 'get_stats'):
                cache_stats = translator.cache.get_stats()
            
            # Tính thời gian đã trôi qua
            elapsed_time = time.time() - start_time
            elapsed_mins, elapsed_secs = divmod(int(elapsed_time), 60)
            elapsed_hrs, elapsed_mins = divmod(elapsed_mins, 60)
            elapsed_time_str = f"{elapsed_hrs:02d}:{elapsed_mins:02d}:{elapsed_secs:02d}"
            
            # Tạo thanh tiến trình hoàn thành 100%
            terminal_style = f"100%|{'█' * 50}| Hoàn thành [{elapsed_time_str}, Xong!]"
            
            # Cập nhật trạng thái hoàn thành với thông tin phân tích
            if status_file:
                final_status = {
                    "status": "completed",
                    "progress": 100,
                    "updated_at": time.time(),
                    "message": "Dịch tài liệu hoàn tất",
                    "estimated_time_remaining": 0,
                    "document_analysis": document_analysis,
                    "processing_details": {
                        "current_step": "completed",
                        "cache_stats": cache_stats
                    },
                    "progress_display": {
                        "progress_bar": self.generate_progress_bar(100),
                        "stage_info": "Hoàn thành",
                        "current_item": "Xong",
                        "elapsed_time_str": elapsed_time_str,
                        "eta_str": "00:00:00",
                        "speed": "",
                        "terminal_style": terminal_style
                    }
                }
                self._update_status(status_file, final_status)
            
            return result_path
            
        except Exception as e:
            # Tính thời gian đã trôi qua khi xảy ra lỗi
            elapsed_time = time.time() - start_time
            elapsed_mins, elapsed_secs = divmod(int(elapsed_time), 60)
            elapsed_hrs, elapsed_mins = divmod(elapsed_mins, 60)
            elapsed_time_str = f"{elapsed_hrs:02d}:{elapsed_mins:02d}:{elapsed_secs:02d}"
            
            # Tạo thông tin hiển thị lỗi kiểu terminal
            terminal_style = f"Lỗi sau [{elapsed_time_str}]: {str(e)}"
            
            # Cập nhật trạng thái lỗi
            if status_file:
                error_status = {
                    "status": "error",
                    "updated_at": time.time(),
                    "message": f"Lỗi: {str(e)}",
                    "error": str(e),
                    "error_details": {
                        "type": type(e).__name__,
                        "traceback": str(e)
                    },
                    "progress_display": {
                        "progress_bar": self.generate_progress_bar(0),
                        "stage_info": "Lỗi",
                        "current_item": "",
                        "elapsed_time_str": elapsed_time_str,
                        "eta_str": "--:--:--",
                        "terminal_style": terminal_style
                    }
                }
                self._update_status(status_file, error_status)
            
            raise e 