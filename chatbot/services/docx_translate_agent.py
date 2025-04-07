# chatbot/services/docx_translate_agent.py
import os
import json
import time
from pathlib import Path
from typing import Dict, Any
from langgraph.graph import END, StateGraph, START
from chatbot.utils.graph_state import GraphState
from chatbot.utils.docx_translator_generator import DocxTranslatorGenerator
from chatbot.utils.llm import LLM
from app.config import settings


class DocxTranslatorAgent:
    """
    Lớp DocxTranslatorAgent chịu trách nhiệm điều phối quá trình dịch thuật tài liệu DOCX.
    """

    def __init__(self, model_name=None) -> None:
        """
        Khởi tạo DocxTranslatorAgent với các thành phần chính.
        """
        model = model_name or settings.LLM_NAME
        self.llm = LLM().get_llm(model)
        self.docx_translator_generator = DocxTranslatorGenerator(self.llm)
        
        # Thiết lập thư mục uploads và downloads
        SERVER_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        self.UPLOAD_DIR = SERVER_DIR / "uploads"
        self.DOWNLOAD_DIR = SERVER_DIR / "downloads"
        self.STATUS_DIR = SERVER_DIR / "status"
        self.UPLOAD_DIR.mkdir(exist_ok=True)
        self.DOWNLOAD_DIR.mkdir(exist_ok=True)
        self.STATUS_DIR.mkdir(exist_ok=True)

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

    def update_status_file(self, status_file_path, updates, preserve_started_at=True):
        """
        Cập nhật tệp trạng thái với thông tin tiến trình mới
        
        Args:
            status_file_path: Đường dẫn đến tệp trạng thái
            updates: Dict chứa các trường cần cập nhật
            preserve_started_at: Có giữ nguyên giá trị started_at ban đầu hay không
        """
        try:
            # Đọc dữ liệu hiện tại nếu tệp tồn tại
            if os.path.exists(status_file_path):
                with open(status_file_path, 'r', encoding='utf-8') as f:
                    status_data = json.load(f)
                
                # Giữ nguyên thời gian bắt đầu nếu cần
                original_started_at = status_data.get("started_at")
            else:
                status_data = {}
                original_started_at = None
            
            # Cập nhật trường
            status_data.update(updates)
            
            # Giữ nguyên started_at nếu cần
            if preserve_started_at and "started_at" in updates and original_started_at:
                status_data["started_at"] = original_started_at
            
            # Luôn cập nhật updated_at
            status_data["updated_at"] = time.time()
            
            # Ghi lại vào tệp
            with open(status_file_path, 'w', encoding='utf-8') as f:
                json.dump(status_data, f, ensure_ascii=False, indent=2)
            
            return status_data
        except Exception as e:
            print(f"Lỗi khi cập nhật trạng thái: {e}")
            return None

    def translate_docx(self, state: GraphState) -> Dict[str, Any]:
        """
        Dịch tài liệu DOCX từ ngôn ngữ nguồn sang ngôn ngữ đích.
        """
        # Phân tích thông tin từ state
        # Format: "TRANSLATE_DOCX|input_path|output_path|target_lang|model|temperature|workers|status_file"
        parts = state["question"].split("|")
        if len(parts) >= 4 and parts[0] == "TRANSLATE_DOCX":
            input_path = parts[1]
            output_path = parts[2]
            target_lang = parts[3]
            
            # Thông số tùy chọn
            model = parts[4] if len(parts) > 4 else "gpt-4o-mini"
            temperature = float(parts[5]) if len(parts) > 5 else 0.3
            workers = int(parts[6]) if len(parts) > 6 else 4
            status_file = parts[7] if len(parts) > 7 else None
            
            # Kiểm tra tệp tồn tại
            if not os.path.exists(input_path):
                error_message = f"ERROR|Không tìm thấy tệp: {input_path}"
                if status_file:
                    self.update_status_file(status_file, {
                        "status": "error",
                        "message": error_message,
                        "error": error_message,
                        "progress_display": {
                            "progress_bar": self.generate_progress_bar(0),
                            "stage_info": "Lỗi",
                            "terminal_style": f"Lỗi: {error_message}"
                        }
                    })
                return {"generation": error_message}
            
            # Tạo tệp status từ tên tệp đầu ra nếu không được cung cấp
            if not status_file:
                filename = os.path.basename(output_path)
                translation_id = os.path.splitext(filename)[0].split('_')[-1]  # Lấy timestamp từ tên file
                status_file = os.path.join(self.STATUS_DIR, f"{translation_id}.json")
            
            # Cập nhật trạng thái đầu tiên - đang chuẩn bị
            start_time = time.time()
            initial_status = {
                "status": "preparing",
                "progress": 0,
                "started_at": start_time,
                "message": "Đang phân tích cấu trúc tài liệu...",
                "processing_details": {
                    "current_step": "analyzing",
                    "total_paragraphs": 0,
                    "processed_paragraphs": 0
                },
                "progress_display": {
                    "progress_bar": self.generate_progress_bar(0),
                    "stage_info": "Phân tích tài liệu",
                    "current_item": "Đang chuẩn bị",
                    "elapsed_time_str": "00:00:00",
                    "eta_str": "--:--:--",
                    "terminal_style": "0%|                                                  | 0/? [00:00<?, ?it/s]"
                }
            }
            self.update_status_file(status_file, initial_status)
            
            # Thực hiện dịch thuật tài liệu
            try:
                # Cấu hình callback để theo dõi tiến trình
                def progress_callback(current, total):
                    if not status_file:
                        return
                    
                    # Tính toán phần trăm hoàn thành
                    progress_percent = min(99, int(current / total * 100)) if total > 0 else 0
                    
                    # Đọc trạng thái hiện tại
                    try:
                        with open(status_file, 'r', encoding='utf-8') as f:
                            status_data = json.load(f)
                        
                        # Tính toán thời gian đã trôi qua
                        elapsed_time = time.time() - status_data.get("started_at", time.time())
                        
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
                        
                        # Tạo thông tin cập nhật
                        progress_status = {
                            "status": "processing",
                            "progress": progress_percent,
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
                        
                        # Cập nhật tệp trạng thái
                        self.update_status_file(status_file, progress_status)
                    except Exception as e:
                        print(f"Lỗi cập nhật tiến trình: {e}")
                
                # Gọi hàm dịch với callback
                result_path = self.docx_translator_generator.translate_docx(
                    input_path=input_path,
                    output_path=output_path,
                    target_lang=target_lang,
                    model=model,
                    temperature=temperature,
                    workers=workers,
                    status_file=status_file,  # Truyền đường dẫn tới file status
                    progress_callback=progress_callback  # Truyền hàm callback
                )
                
                # Tính toán thời gian đã trôi qua
                elapsed_time = time.time() - start_time
                elapsed_mins, elapsed_secs = divmod(int(elapsed_time), 60)
                elapsed_hrs, elapsed_mins = divmod(elapsed_mins, 60)
                elapsed_time_str = f"{elapsed_hrs:02d}:{elapsed_mins:02d}:{elapsed_secs:02d}"
                
                # Tạo thanh tiến trình hoàn thành
                terminal_style = f"100%|{'█' * 50}| Hoàn thành [{elapsed_time_str}, Xong!]"
                
                # Cập nhật trạng thái hoàn thành
                completion_status = {
                    "status": "completed",
                    "progress": 100,
                    "message": "Dịch tài liệu hoàn tất",
                    "estimated_time_remaining": 0,
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
                self.update_status_file(status_file, completion_status)
                
                return {"generation": f"SUCCESS|{result_path}"}
            except Exception as e:
                error_message = str(e)
                
                # Tính toán thời gian đã trôi qua khi xảy ra lỗi
                elapsed_time = time.time() - start_time
                elapsed_mins, elapsed_secs = divmod(int(elapsed_time), 60)
                elapsed_hrs, elapsed_mins = divmod(elapsed_mins, 60)
                elapsed_time_str = f"{elapsed_hrs:02d}:{elapsed_mins:02d}:{elapsed_secs:02d}"
                
                # Tạo thông tin lỗi với định dạng terminal
                terminal_style = f"Lỗi sau [{elapsed_time_str}]: {error_message}"
                
                # Cập nhật trạng thái lỗi
                error_status = {
                    "status": "error",
                    "message": f"Lỗi: {error_message}",
                    "error": error_message,
                    "error_details": {
                        "type": type(e).__name__
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
                self.update_status_file(status_file, error_status)
                
                return {"generation": f"ERROR|{error_message}"}
        else:
            error_message = "Định dạng không hợp lệ. Sử dụng: TRANSLATE_DOCX|input_path|output_path|target_lang|model|temperature|workers|status_file"
            return {"generation": f"ERROR|{error_message}"}

    def get_workflow(self):
        """
        Thiết lập luồng xử lý của quá trình dịch thuật tài liệu.
        """
        workflow = StateGraph(GraphState)
        workflow.add_node("translate_docx", self.translate_docx)
        workflow.add_edge(START, "translate_docx")
        workflow.add_edge("translate_docx", END)
        
        return workflow