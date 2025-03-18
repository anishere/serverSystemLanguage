import openai
from app.config import settings
import logging
from chatbot.utils.custom_prompt import CustomPrompt

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImgToTextGenerator:
    """
    Lớp ImgToTextGenerator chịu trách nhiệm trích xuất văn bản từ ảnh
    sử dụng API OpenAI Vision.
    """

    def __init__(self) -> None:
        """
        Khởi tạo ImgToTextGenerator với OpenAI client.
        """
        self.client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = "gpt-4o-mini"
        self.max_tokens = 1000

    def extract_text(self, image_base64: str) -> str:
        """
        Trích xuất văn bản từ ảnh dưới dạng base64.
        
        Args:
            image_base64 (str): Ảnh dưới dạng chuỗi base64.
            
        Returns:
            str: Văn bản được trích xuất từ ảnh.
        """
        try:
            messages = [
                {
                    "role": "system",
                    "content": CustomPrompt.IMG_TO_TEXT_PROMPT
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                    ]
                }
            ]

            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=messages
            )

            extracted_text = response.choices[0].message.content
            logger.info(f"Text extracted successfully. Length: {len(extracted_text)}")
            return extracted_text
        except Exception as e:
            logger.error(f"Error during text extraction: {e}")
            return f"Error during text extraction: {e}"