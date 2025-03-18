from langgraph.graph import END, StateGraph, START
from typing import Dict, Any, List
from app.config import settings
from chatbot.utils.multilingual_analyzer_generator import MultilingualAnalyzerGenerator

class MultilingualAnalyzerAgent:
    """
    Lớp MultilingualAnalyzerAgent chịu trách nhiệm điều phối quá trình phân tích đa ngôn ngữ.
    """

    def __init__(self, model_name=None) -> None:
        """
        Khởi tạo MultilingualAnalyzerAgent với các thành phần chính.
        
        Args:
            model_name (str, optional): Tên mô hình LLM ('openai' hoặc 'gemini'). 
                                       Nếu không cung cấp, sẽ sử dụng mô hình từ cấu hình.
        """
        from chatbot.utils.llm import LLM
        model = model_name or settings.DEFAULT_LLM
        llm_instance = LLM()
        self.llm = llm_instance.get_llm(model)
        self.analyzer_generator = MultilingualAnalyzerGenerator(self.llm)

    def analyze(self, state) -> Dict[str, Any]:
        """
        Phân tích văn bản đa ngôn ngữ.
        
        Args:
            state: Trạng thái đồ thị hiện tại chứa câu hỏi người dùng.
            
        Returns:
            Dict[str, Any]: Trạng thái đồ thị mới bao gồm kết quả phân tích.
        """
        # Phân tích câu hỏi để lấy text
        # Format câu hỏi: "ANALYZE|text_to_analyze"
        parts = state["question"].split("|", 1)
        if len(parts) == 2 and parts[0] == "ANALYZE":
            text = parts[1]
            
            # Thực hiện phân tích
            analysis_result = self.analyzer_generator.analyze(text)
            
            # Trả về theo cấu trúc của GraphState hiện tại
            return {
                "generation": analysis_result,
                "documents": state.get("documents", [])  # Giữ nguyên documents hiện có
            }
        else:
            return {
                "generation": "Invalid analysis format. Use: ANALYZE|text",
                "documents": state.get("documents", [])
            }

    def get_workflow(self):
        """
        Thiết lập luồng xử lý của quá trình phân tích.
        
        Returns:
            StateGraph: Đồ thị trạng thái định nghĩa luồng xử lý phân tích đa ngôn ngữ.
        """
        from typing_extensions import TypedDict
        
        class GraphState(TypedDict):
            question: str
            generation: str
            documents: List[str]
            
        workflow = StateGraph(GraphState)
        workflow.add_node("analyze", self.analyze)
        workflow.add_edge(START, "analyze")
        workflow.add_edge("analyze", END)
        
        return workflow