from sqlalchemy.orm import Session
from app.database.models import CreditTransaction, TransactionType, User
from fastapi import HTTPException
from datetime import datetime

def save_credit_transaction(
    db: Session, 
    user_id: int, 
    amount: int, 
    transaction_type: str = "purchase",
    payment_method: str = None
):
    """
    Lưu lịch sử giao dịch credits
    """
    try:
        # Xác định loại giao dịch
        tx_type = TransactionType.purchase if transaction_type.lower() == "purchase" else TransactionType.usage
        
        # Tạo bản ghi giao dịch
        credit_transaction = CreditTransaction(
            user_id=user_id,
            amount=amount,
            transaction_type=tx_type,
            created_at=datetime.utcnow(),
            payment_method=payment_method
        )
        
        db.add(credit_transaction)
        db.commit()
        db.refresh(credit_transaction)
        
        return {
            "message": "Đã lưu lịch sử giao dịch credits thành công",
            "transaction_id": credit_transaction.id,
            "amount": amount,
            "transaction_type": transaction_type,
            "payment_method": payment_method
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Lỗi khi lưu lịch sử giao dịch: {str(e)}")
    
def get_credit_transactions(
    db: Session,
    user_id: int,
    skip: int = 0,
    limit: int = 100,
    start_date: datetime = None,
    end_date: datetime = None,
    transaction_type: str = None,
    sort_by_date: str = "desc"
):
    """
    Lấy danh sách lịch sử giao dịch credits theo các tiêu chí lọc
    """
    query = db.query(CreditTransaction).filter(CreditTransaction.user_id == user_id)
    
    # Lọc theo ngày nếu có
    if start_date:
        query = query.filter(CreditTransaction.created_at >= start_date)
    if end_date:
        query = query.filter(CreditTransaction.created_at <= end_date)
    
    # Lọc theo loại giao dịch nếu có
    if transaction_type:
        if transaction_type.lower() == "purchase":
            query = query.filter(CreditTransaction.transaction_type == TransactionType.purchase)
        elif transaction_type.lower() == "usage":
            query = query.filter(CreditTransaction.transaction_type == TransactionType.usage)
    
    # Sắp xếp theo thời gian
    if sort_by_date.lower() == "asc":
        query = query.order_by(CreditTransaction.created_at.asc())
    else:
        query = query.order_by(CreditTransaction.created_at.desc())
    
    # Đếm tổng số bản ghi
    total_count = query.count()
    
    # Áp dụng phân trang
    results = query.offset(skip).limit(limit).all()
    
    # Chuẩn bị dữ liệu trả về
    transaction_list = []
    for item in results:
        transaction_list.append({
            "id": item.id,
            "user_id": item.user_id,
            "amount": float(item.amount),
            "transaction_type": item.transaction_type.value,
            "created_at": item.created_at.isoformat(),
            "payment_method": item.payment_method
        })
    
    return {
        "total": total_count,
        "limit": limit,
        "offset": skip,
        "items": transaction_list
    }