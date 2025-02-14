# models/translateAI.py
from pydantic import BaseModel

# Mô hình dữ liệu đầu vào yêu cầu dịch văn bản
class TranslateRequest(BaseModel):
    text: str
    src_lang: str
    tgt_lang: str

# Mô hình dữ liệu đầu ra cho kết quả dịch
class TranslateResponse(BaseModel):
    translated_text: str
