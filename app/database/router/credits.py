from fastapi import APIRouter, Depends, HTTPException, Query
from app.database.connection import get_db
from app.database.api.credit_api import save_credit_transaction, get_credit_transactions, get_revenue_credit_transactions, get_all_credit_transactions
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from app.security.security import get_api_key
from typing import Optional, List
from datetime import datetime

router = APIRouter(
    prefix="/history/credit",
    tags=["credit_history"]
)

class CreditTransactionCreate(BaseModel):
    user_id: int
    amount: int = Field(..., description="Số lượng credits (dương: nạp, âm: tiêu)")
    transaction_type: str = Field("purchase", description="Loại giao dịch: 'purchase' hoặc 'usage'")
    payment_method: Optional[str] = None

@router.post("/", summary="Lưu lịch sử giao dịch credits")
def create_credit_transaction(
    transaction_data: CreditTransactionCreate,
    db: Session = Depends(get_db),
    api_key: str = get_api_key
):
    """
    Lưu lịch sử giao dịch credits
    
    - **user_id**: ID của người dùng
    - **amount**: Số lượng credits (dương: nạp, âm: tiêu)
    - **transaction_type**: Loại giao dịch: 'purchase' hoặc 'usage'
    - **payment_method**: Phương thức thanh toán (tùy chọn)
    """
    return save_credit_transaction(
        db,
        transaction_data.user_id,
        transaction_data.amount,
        transaction_data.transaction_type,
        transaction_data.payment_method
    )

@router.get("/", summary="Lấy danh sách lịch sử giao dịch credits")
def get_credit_transactions_endpoint(
    user_id: int,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None, 
    transaction_type: Optional[str] = Query(None, regex="^(purchase|usage|all)$"),
    sort_by_date: str = Query("desc", regex="^(asc|desc)$"),
    db: Session = Depends(get_db),
    api_key: str = get_api_key
):
    """
    Lấy danh sách lịch sử giao dịch credits với các tùy chọn lọc
    
    - **user_id**: ID của người dùng (bắt buộc)
    - **skip**: Số bản ghi bỏ qua (để phân trang)
    - **limit**: Số bản ghi tối đa trả về
    - **start_date**: Lọc từ ngày (định dạng ISO)
    - **end_date**: Lọc đến ngày (định dạng ISO)
    - **transaction_type**: Lọc theo loại giao dịch ("purchase", "usage" hoặc "all")
    - **sort_by_date**: Sắp xếp theo thời gian ("asc" hoặc "desc")
    """
    # Xử lý trường hợp transaction_type = 'all'
    if transaction_type == 'all':
        transaction_type = None
        
    return get_credit_transactions(
        db=db,
        user_id=user_id,
        skip=skip,
        limit=limit,
        start_date=start_date,
        end_date=end_date,
        transaction_type=transaction_type,
        sort_by_date=sort_by_date
    )

@router.get("/revenue", summary="Lấy danh sách lịch sử giao dịch nạp tiền (doanh thu)")
def get_revenue_transactions_endpoint(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    sort_by_date: str = Query("desc", regex="^(asc|desc)$"),
    db: Session = Depends(get_db),
    api_key: str = get_api_key
):
    """
    Lấy danh sách tất cả lịch sử giao dịch nạp tiền (doanh thu) trong hệ thống
    
    - **skip**: Số bản ghi bỏ qua (để phân trang)
    - **limit**: Số bản ghi tối đa trả về
    - **start_date**: Lọc từ ngày (định dạng ISO)
    - **end_date**: Lọc đến ngày (định dạng ISO)
    - **sort_by_date**: Sắp xếp theo thời gian ("asc" hoặc "desc")
    """
    return get_revenue_credit_transactions(
        db=db,
        skip=skip,
        limit=limit,
        start_date=start_date,
        end_date=end_date,
        sort_by_date=sort_by_date
    )

@router.get("/all", summary="Lấy tất cả giao dịch credits trong hệ thống")
def get_all_transactions_endpoint(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=1000),
    sort_order: str = Query("desc", regex="^(asc|desc)$"),
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    transaction_type: Optional[str] = Query(None, regex="^(purchase|usage|all)$"),
    db: Session = Depends(get_db),
    api_key: str = get_api_key
):
    """
    Lấy tất cả giao dịch credits trong hệ thống.
    
    - **skip**: Số bản ghi bỏ qua (phân trang)
    - **limit**: Số bản ghi trả về tối đa (phân trang)
    - **sort_order**: Thứ tự sắp xếp ("asc" hoặc "desc")
    - **start_date**: Ngày bắt đầu (định dạng ISO)
    - **end_date**: Ngày kết thúc (định dạng ISO)
    - **transaction_type**: Loại giao dịch (purchase, usage, all)
    """
    
    # Xử lý trường hợp transaction_type = 'all'
    if transaction_type == 'all':
        transaction_type = None
    
    # Lấy tất cả giao dịch
    return get_all_credit_transactions(
        db=db,
        skip=skip,
        limit=limit,
        sort_order=sort_order,
        start_date=start_date,
        end_date=end_date,
        transaction_type=transaction_type
    )