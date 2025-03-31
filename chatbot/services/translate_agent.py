# chatbot/services/translate_agent.py
from langgraph.graph import END, StateGraph, START
from chatbot.utils.graph_state import GraphState
from chatbot.utils.translator_generator import TranslatorGenerator
from chatbot.utils.llm import LLM
from typing import Dict, Any
from app.config import settings

class TranslatorAgent:
    """
    Lớp TranslatorAgent chịu trách nhiệm điều phối quá trình dịch thuật.
    """

    def __init__(self, model_name=None) -> None:
        """
        Khởi tạo TranslatorAgent với các thành phần chính.
        """
        model = model_name or settings.LLM_NAME
        self.llm = LLM().get_llm(model)
        self.translator_generator = TranslatorGenerator(self.llm)

    def translate(self, state: GraphState) -> Dict[str, Any]:
        """
        Dịch văn bản từ ngôn ngữ nguồn sang ngôn ngữ đích.
        """
        # Phân tích câu hỏi để lấy text, src_lang, tgt_lang, style
        # Format câu hỏi: "TRANSLATE|src_lang|tgt_lang|style|text_to_translate"
        parts = state["question"].split("|", 4)
        if len(parts) == 5 and parts[0] == "TRANSLATE":
            src_lang = parts[1]
            tgt_lang = parts[2]
            style = parts[3]
            text = parts[4]
            
            # Thực hiện dịch thuật với style
            translated_text = self.translator_generator.translate(text, src_lang, tgt_lang, style)
            
            # Trả về theo cấu trúc của GraphState
            return {"generation": translated_text}
        else:
            return {"generation": "Invalid translation format. Use: TRANSLATE|source_language|target_language|style|text"}

    def get_workflow(self):
        """
        Thiết lập luồng xử lý của quá trình dịch thuật.
        """
        workflow = StateGraph(GraphState)
        workflow.add_node("translate", self.translate)
        workflow.add_edge(START, "translate")
        workflow.add_edge("translate", END)
        
        return workflow