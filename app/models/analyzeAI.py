from pydantic import BaseModel
from typing import List

# Mô hình dữ liệu đầu vào yêu cầu phân tích văn bản đa ngôn ngữ
class AnalyzeRequest(BaseModel):
    text: str

# Mô hình dữ liệu đầu ra cho kết quả phân tích văn bản đa ngôn ngữ
class LanguageFragment(BaseModel):
    name: str
    code: str
    text: str

class AnalyzeResponse(BaseModel):
    language: List[LanguageFragment]
