from fastapi import APIRouter
from fastapi import FastAPI, File, UploadFile, Header, HTTPException, Request, Form  # noqa: E402, F401

from app.security.security import get_api_key
from app.models.base import Base

from chatbot.services.files_chat_agent import FilesChatAgent  # noqa: E402
from app.config import settings

# Tạo router cho người dùng
router = APIRouter(prefix="/base", tags=["base"])

# response_model=Base
@router.post("/base-url/")
async def base_url(
    api_key: str = get_api_key,  # Khóa API để xác thực
    base_data: str = Form(""),
):
    #
    # return Base(base_data=base_data)

    settings.LLM_NAME = "openai"

    _question = "Nhu cầu tuyển dụng trong nhóm kinh doanh/bán hàng có đang giữ vững vị trí dẫn đầu và gia tăng không?"
    chat = FilesChatAgent("demo\data_vector").get_workflow().compile().invoke(
        input={
            "question": _question,
        }
    )

    print(chat)

    print("generation", chat["generation"])

    return chat["generation"]