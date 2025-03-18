# chatbot/utils/translator_generator.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from chatbot.utils.custom_prompt import CustomPrompt

class TranslatorGenerator:
    """
    Lớp TranslatorGenerator chịu trách nhiệm dịch văn bản từ ngôn ngữ nguồn sang ngôn ngữ đích
    sử dụng mô hình ngôn ngữ.
    """

    def __init__(self, llm) -> None:
        """
        Khởi tạo TranslatorGenerator với mô hình ngôn ngữ (LLM).

        Args:
            llm: Mô hình ngôn ngữ được sử dụng để dịch văn bản.
        """
        # Xây dựng prompt cho dịch thuật
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", CustomPrompt.TRANSLATE_PROMPT),
                ("human", "Translate this text from {src_lang} to {tgt_lang}: {text}")
            ]
        )

        # Xây dựng pipeline xử lý
        self.chain = prompt | llm | StrOutputParser()

    def get_chain(self) -> RunnableSequence:
        """
        Trả về chuỗi pipeline xử lý để dịch văn bản.
        """
        return self.chain
        
    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """
        Dịch văn bản từ ngôn ngữ nguồn sang ngôn ngữ đích.
        """
        return self.chain.invoke({
            "text": text,
            "src_lang": src_lang,
            "tgt_lang": tgt_lang
        })