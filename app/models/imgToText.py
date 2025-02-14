# models/imgToText.py
from pydantic import BaseModel

# Mô hình dữ liệu đầu ra cho kết quả trích xuất văn bản
class ImgToTextResponse(BaseModel):
    extracted_text: str
