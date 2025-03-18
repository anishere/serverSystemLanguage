from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from chatbot.utils.custom_prompt import CustomPrompt
import json
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultilingualAnalyzerGenerator:
    """
    Lớp MultilingualAnalyzerGenerator chịu trách nhiệm phân tích văn bản đa ngôn ngữ
    sử dụng mô hình ngôn ngữ.
    """

    def __init__(self, llm) -> None:
        """
        Khởi tạo MultilingualAnalyzerGenerator với mô hình ngôn ngữ (LLM).

        Args:
            llm: Mô hình ngôn ngữ được sử dụng để phân tích văn bản.
        """
        # Xây dựng prompt cho phân tích đa ngôn ngữ
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", CustomPrompt.MULTILINGUAL_ANALYSIS_PROMPT),
                ("human", "Analyze this text for language changes: {text}")
            ]
        )

        # Xây dựng pipeline xử lý
        self.chain = prompt | llm | StrOutputParser()

    def get_chain(self) -> RunnableSequence:
        """
        Trả về chuỗi pipeline xử lý để phân tích văn bản.
        
        Returns:
            RunnableSequence: Pipeline LangChain để phân tích văn bản.
        """
        return self.chain
        
    def analyze(self, text: str) -> str:
        """
        Phân tích văn bản đa ngôn ngữ để xác định các đoạn của từng ngôn ngữ.
        
        Args:
            text (str): Văn bản cần phân tích.
            
        Returns:
            str: Kết quả phân tích định dạng JSON.
        """
        result = self.chain.invoke({"text": text})
        
        # Cố gắng phân tích kết quả trả về dưới dạng JSON
        try:
            # Nếu kết quả là chuỗi JSON, thử chuyển đổi thành đối tượng Python
            if result.strip().startswith('{') and result.strip().endswith('}'):
                result_json = json.loads(result)
                # Định dạng lại kết quả để hiển thị đẹp hơn
                return json.dumps(result_json, ensure_ascii=False, indent=2)
            
            # Nếu không phải JSON, trả về kết quả nguyên bản
            return result
        except json.JSONDecodeError:
            logger.warning(f"Result is not valid JSON: {result[:100]}...")
            # Nếu không phải JSON, trả về kết quả nguyên bản
            return result