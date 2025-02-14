from fastapi import FastAPI
from app.routers import base
from app.routers import translateAI
from app.routers import analyzeAI
from app.routers import speechToTextAI
from app.routers import textToSpeech
from app.routers import imgToText
from fastapi.middleware.cors import CORSMiddleware

# Tạo instance của FastAPI
app = FastAPI()

# Cấu hình CORS

app.add_middleware(
CORSMiddleware,
    allow_origins=["*"], # Cho phép mọi nguồn (có thể thay "_" bằng một danh sách các domain cụ thể)
    allow_credentials=True,
    allow_methods=["*"], # Cho phép tất cả các phương thức HTTP (GET, POST, ...)
    allow_headers=["*"], # Cho phép tất cả các header
)

# Include các router vào ứng dụng chính
app.include_router(base.router)
app.include_router(translateAI.router)
app.include_router(analyzeAI.router)
app.include_router(speechToTextAI.router)
app.include_router(textToSpeech.router)
app.include_router(imgToText.router)

# @app.route("/favicon.ico") lỗi khi khởi chạy
# def favicon():
#     return "", 204


@app.get("/")
def read_root():
    return {"message": "Welcome to my FastAPI application"}
