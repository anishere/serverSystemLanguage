from pydantic import BaseModel

# Mô hình yêu cầu đầu vào
class TextToSpeechRequest(BaseModel):
    text: str
    lang: str = "en"  # Ngôn ngữ mặc định là tiếng Anh

# Mô hình trả về kết quả
class TextToSpeechResponse(BaseModel):
    audio_base64: str
