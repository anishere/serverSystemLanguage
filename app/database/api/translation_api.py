from typing import List
from sqlalchemy.orm import Session
from app.database.models import TranslationHistory, CreditTransaction, TransactionType, User
from fastapi import HTTPException
import math
from datetime import datetime, timedelta, timezone

def save_translation_history(
    db: Session, 
    user_id: int, 
    input_text: str, 
    output_text: str, 
    source_language: str, 
    target_language: str
):
    """
    Lưu lịch sử dịch thuật
    """
    try:
        # Tính số ký tự
        character_count = len(input_text)
        
        # Ước tính credits sử dụng (1 credit/100 ký tự)
        credits_used = max(1, math.ceil(character_count / 1))
        
        # Tạo timezone UTC+7 (Vietnam)
        vietnam_tz = timezone(timedelta(hours=7))
        vietnam_time = datetime.now(vietnam_tz)
        
        # Tạo bản ghi lịch sử dịch - TẠO ĐỐI TƯỢNG TranslationHistory thay vì gọi lại hàm
        translation_history = TranslationHistory(
            user_id=user_id,
            input_text=input_text,
            output_text=output_text,
            source_language=source_language,
            target_language=target_language,
            credits_used=credits_used,
            created_at=vietnam_time
        )
        
        db.add(translation_history)
        db.commit()
        db.refresh(translation_history)
        
        return {
            "message": "Đã lưu lịch sử dịch thành công",
            "translation_id": translation_history.id,
            "credits_used": credits_used,
            "character_count": character_count
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Lỗi khi lưu lịch sử dịch: {str(e)}")
    
def get_translation_history(
    db: Session,
    user_id: int,
    skip: int = 0,
    limit: int = 100,
    start_date: datetime = None,
    end_date: datetime = None,
    source_language: str = None,
    target_language: str = None,
    sort_by_date: str = "desc"
):
    """
    Lấy danh sách lịch sử dịch thuật theo các tiêu chí lọc
    """
    query = db.query(TranslationHistory).filter(TranslationHistory.user_id == user_id)
    
    # Lọc theo ngày nếu có
    if start_date:
        query = query.filter(TranslationHistory.created_at >= start_date)
    if end_date:
        query = query.filter(TranslationHistory.created_at <= end_date)
    
    # Lọc theo ngôn ngữ nếu có
    if source_language:
        query = query.filter(TranslationHistory.source_language == source_language)
    if target_language:
        query = query.filter(TranslationHistory.target_language == target_language)
    
    # Sắp xếp theo thời gian
    if sort_by_date.lower() == "asc":
        query = query.order_by(TranslationHistory.created_at.asc())
    else:
        query = query.order_by(TranslationHistory.created_at.desc())
    
    # Đếm tổng số bản ghi
    total_count = query.count()
    
    # Áp dụng phân trang
    results = query.offset(skip).limit(limit).all()
    
    # Chuẩn bị dữ liệu trả về
    history_list = []
    for item in results:
        history_list.append({
            "id": item.id,
            "user_id": item.user_id,
            "input_text": item.input_text,
            "output_text": item.output_text,
            "source_language": item.source_language,
            "target_language": item.target_language,
            "created_at": item.created_at.isoformat(),
            "character_count": item.character_count,
            "credits_used": float(item.credits_used)
        })
    
    return {
        "total": total_count,
        "limit": limit,
        "offset": skip,
        "items": history_list
    }

def delete_translation_history(
    db: Session,
    history_id: int,
    user_id: int
):
    """
    Xóa một bản ghi lịch sử dịch thuật
    
    Args:
        db: Phiên database
        history_id: ID của bản ghi lịch sử cần xóa
        user_id: ID của người dùng (để xác thực quyền sở hữu)
        
    Returns:
        dict: Thông báo kết quả
    """
    # Tìm bản ghi
    history_item = db.query(TranslationHistory).filter(
        TranslationHistory.id == history_id
    ).first()
    
    # Kiểm tra tồn tại
    if not history_item:
        raise HTTPException(status_code=404, detail="Không tìm thấy bản ghi lịch sử")
    
    # Kiểm tra quyền sở hữu
    if history_item.user_id != user_id:
        raise HTTPException(status_code=403, detail="Không có quyền xóa bản ghi này")
    
    try:
        # Xóa bản ghi
        db.delete(history_item)
        db.commit()
        
        return {
            "message": "Đã xóa bản ghi lịch sử thành công",
            "id": history_id
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Lỗi khi xóa lịch sử dịch: {str(e)}")

def delete_all_translation_history(
    db: Session,
    user_id: int
):
    """
    Xóa tất cả lịch sử dịch thuật của một người dùng
    
    Args:
        db: Phiên database
        user_id: ID của người dùng
        
    Returns:
        dict: Thông báo kết quả
    """
    try:
        # Xóa tất cả bản ghi của người dùng
        result = db.query(TranslationHistory).filter(
            TranslationHistory.user_id == user_id
        ).delete(synchronize_session=False)
        
        db.commit()
        
        return {
            "message": f"Đã xóa {result} bản ghi lịch sử dịch thuật",
            "deleted_count": result
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Lỗi khi xóa lịch sử dịch: {str(e)}")