# chatbot/services/docx_translate_agent.py
import os
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

    def translate_docx(self, state: GraphState) -> Dict[str, Any]:
        """
        Dịch tài liệu DOCX từ ngôn ngữ nguồn sang ngôn ngữ đích.
        """
        # Phân tích thông tin từ state
        # Format: "TRANSLATE_DOCX|input_path|output_path|target_lang|model|temperature|workers"
        parts = state["question"].split("|")
        if len(parts) >= 4 and parts[0] == "TRANSLATE_DOCX":
            input_path = parts[1]
            output_path = parts[2]
            target_lang = parts[3]
            
            # Thông số tùy chọn
            model = parts[4] if len(parts) > 4 else "gpt-4o-mini"
            temperature = float(parts[5]) if len(parts) > 5 else 0.3
            workers = int(parts[6]) if len(parts) > 6 else 4
            
            # Kiểm tra tệp tồn tại
            if not os.path.exists(input_path):
                return {"generation": f"ERROR|Không tìm thấy tệp: {input_path}"}
            
            # Tạo tệp status từ tên tệp đầu ra
            filename = os.path.basename(output_path)
            translation_id = os.path.splitext(filename)[0].split('_')[-1]  # Lấy timestamp từ tên file
            status_file = os.path.join(self.STATUS_DIR, f"{translation_id}.json")
            
            # Thực hiện dịch thuật tài liệu
            try:
                result_path = self.docx_translator_generator.translate_docx(
                    input_path=input_path,
                    output_path=output_path,
                    target_lang=target_lang,
                    model=model,
                    temperature=temperature,
                    workers=workers,
                    status_file=status_file  # Truyền đường dẫn tới file status
                )
                
                return {"generation": f"SUCCESS|{result_path}"}
            except Exception as e:
                return {"generation": f"ERROR|{str(e)}"}
        else:
            return {"generation": "ERROR|Định dạng không hợp lệ. Sử dụng: TRANSLATE_DOCX|input_path|output_path|target_lang|model|temperature|workers"}

    def get_workflow(self):
        """
        Thiết lập luồng xử lý của quá trình dịch thuật tài liệu.
        """
        workflow = StateGraph(GraphState)
        workflow.add_node("translate_docx", self.translate_docx)
        workflow.add_edge(START, "translate_docx")
        workflow.add_edge("translate_docx", END)
        
        return workflow