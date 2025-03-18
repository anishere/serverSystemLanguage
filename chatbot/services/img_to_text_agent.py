from langgraph.graph import END, StateGraph, START
from typing import Dict, Any, List
from app.config import settings
from chatbot.utils.img_to_text_generator import ImgToTextGenerator

class ImgToTextAgent:
    """
    Lớp ImgToTextAgent chịu trách nhiệm điều phối quá trình trích xuất văn bản từ ảnh.
    """

    def __init__(self, model_name=None) -> None:
        """
        Khởi tạo ImgToTextAgent với các thành phần chính.
        
        Args:
            model_name (str, optional): Tên mô hình LLM ('openai' hoặc 'gemini'). 
                                       Mặc định sử dụng 'openai' vì cần GPT-Vision.
        """
        self.img_to_text_generator = ImgToTextGenerator()

    def extract_text(self, state) -> Dict[str, Any]:
        """
        Trích xuất văn bản từ ảnh.
        
        Args:
            state: Trạng thái đồ thị hiện tại chứa thông tin ảnh.
            
        Returns:
            Dict[str, Any]: Trạng thái đồ thị mới bao gồm kết quả trích xuất.
        """
        # Format câu hỏi: "EXTRACT|image_base64"
        parts = state["question"].split("|", 1)
        if len(parts) == 2 and parts[0] == "EXTRACT":
            image_base64 = parts[1]
            
            # Thực hiện trích xuất văn bản
            extracted_text = self.img_to_text_generator.extract_text(image_base64)
            
            # Trả về theo cấu trúc của GraphState hiện tại
            return {
                "generation": extracted_text,
                "documents": state.get("documents", [])  # Giữ nguyên documents hiện có
            }
        else:
            return {
                "generation": "Invalid format. Use: EXTRACT|image_base64",
                "documents": state.get("documents", [])
            }

    def get_workflow(self):
        """
        Thiết lập luồng xử lý của quá trình trích xuất văn bản từ ảnh.
        
        Returns:
            StateGraph: Đồ thị trạng thái định nghĩa luồng xử lý.
        """
        from typing_extensions import TypedDict
        
        class GraphState(TypedDict):
            question: str
            generation: str
            documents: List[str]
            
        workflow = StateGraph(GraphState)
        workflow.add_node("extract_text", self.extract_text)
        workflow.add_edge(START, "extract_text")
        workflow.add_edge("extract_text", END)
        
        return workflow