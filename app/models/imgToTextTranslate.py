from pydantic import BaseModel
from typing import List

# Mô hình dữ liệu cho từng đoạn văn bản sau khi phân tích
class TranslatedFragment(BaseModel):
    original_text: str
    detected_language: str
    language_code: str
    translated_text: str

# Mô hình dữ liệu đầu ra API trích xuất & dịch văn bản từ ảnh
class ImgToTextTranslateResponse(BaseModel):
    extracted_text: str
    translations: List[TranslatedFragment]
