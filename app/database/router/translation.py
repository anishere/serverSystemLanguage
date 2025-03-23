from fastapi import APIRouter, Depends, HTTPException, Query, Path
from app.database.connection import get_db
from app.database.api.translation_api import (
    save_translation_history, 
    get_translation_history,
    delete_translation_history,
    delete_all_translation_history  
)
from sqlalchemy.orm import Session
from pydantic import BaseModel
from app.security.security import get_api_key
from typing import Optional, List
from datetime import datetime

router = APIRouter(
    prefix="/history/translation",
    tags=["translation_history"]
)

class TranslationHistoryCreate(BaseModel):
    user_id: int
    input_text: str
    output_text: str
    source_language: str
    target_language: str

@router.post("/", summary="Lưu lịch sử dịch thuật")
def create_translation_history(
    history_data: TranslationHistoryCreate,
    db: Session = Depends(get_db),
    api_key: str = get_api_key
):
    """
    Lưu lịch sử dịch thuật
    
    - **user_id**: ID của người dùng
    - **input_text**: Văn bản gốc cần dịch
    - **output_text**: Văn bản đã được dịch
    - **source_language**: Ngôn ngữ nguồn
    - **target_language**: Ngôn ngữ đích
    """
    return save_translation_history(
        db,
        history_data.user_id,
        history_data.input_text,
        history_data.output_text,
        history_data.source_language,
        history_data.target_language
    )

@router.get("/", summary="Lấy danh sách lịch sử dịch thuật")
def get_translation_history_endpoint(
    user_id: int,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None, 
    source_language: Optional[str] = None,
    target_language: Optional[str] = None,
    sort_by_date: str = Query("desc", regex="^(asc|desc)$"),
    db: Session = Depends(get_db),
    api_key: str = get_api_key
):
    """
    Lấy danh sách lịch sử dịch thuật với các tùy chọn lọc
    
    - **user_id**: ID của người dùng (bắt buộc)
    - **skip**: Số bản ghi bỏ qua (để phân trang)
    - **limit**: Số bản ghi tối đa trả về
    - **start_date**: Lọc từ ngày (định dạng ISO)
    - **end_date**: Lọc đến ngày (định dạng ISO)
    - **source_language**: Lọc theo ngôn ngữ nguồn
    - **target_language**: Lọc theo ngôn ngữ đích
    - **sort_by_date**: Sắp xếp theo thời gian ("asc" hoặc "desc")
    """
    return get_translation_history(
        db=db,
        user_id=user_id,
        skip=skip,
        limit=limit,
        start_date=start_date,
        end_date=end_date,
        source_language=source_language,
        target_language=target_language,
        sort_by_date=sort_by_date
    )

@router.delete("/delete-all", summary="Xóa tất cả lịch sử dịch thuật")
def delete_all_translation_history_endpoint(
    user_id: int = Query(..., description="ID của người dùng"),
    db: Session = Depends(get_db),
    api_key: str = get_api_key
):
    """
    Xóa tất cả lịch sử dịch thuật của một người dùng
    
    - **user_id**: ID của người dùng
    """
    return delete_all_translation_history(db, user_id)

@router.delete("/{history_id}", summary="Xóa một bản ghi lịch sử dịch")
def delete_translation_history_endpoint(
    history_id: int = Path(..., description="ID của bản ghi lịch sử cần xóa"),
    user_id: int = Query(..., description="ID của người dùng (để xác minh quyền sở hữu)"),
    db: Session = Depends(get_db),
    api_key: str = get_api_key
):
    """
    Xóa một bản ghi lịch sử dịch thuật cụ thể
    
    - **history_id**: ID của bản ghi lịch sử cần xóa (path parameter)
    - **user_id**: ID của người dùng (query parameter, để xác minh quyền sở hữu)
    
    Lưu ý: Người dùng chỉ có thể xóa bản ghi lịch sử của chính họ
    """
    return delete_translation_history(db, history_id, user_id)

