from fastapi import APIRouter, Depends, HTTPException, Query
from app.database.connection import get_db
from app.database.auth_api import (
    register_user, login_user, update_profile, change_password, 
    change_account_type, update_active_status, delete_user,
    reset_password, get_all_users, add_user_credits, subtract_user_credits
)
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from app.security.security import get_api_key
from typing import Optional

router = APIRouter(
    prefix="/auth",
    tags=["auth"]
)

# Định nghĩa model cho dữ liệu đầu vào
class UserRegister(BaseModel):
    username: str
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class UserUpdate(BaseModel):
    username: Optional[str] = None
    email: Optional[str] = None

class PasswordChange(BaseModel):
    current_password: str
    new_password: str

class PasswordReset(BaseModel):
    new_password: str = Field(..., min_length=6, description="Mật khẩu mới phải có ít nhất 6 ký tự")

class AccountTypeChange(BaseModel):
    account_type: str

class ActiveStatusUpdate(BaseModel):
    is_active: int

class CreditUpdate(BaseModel):
    amount: int = Field(..., gt=0, description="Số credits (phải lớn hơn 0)")

# Endpoint đăng ký
@router.post("/register")
def register(user: UserRegister, db: Session = Depends(get_db), api_key: str = get_api_key):
    return register_user(db, user.username, user.email, user.password)

# Endpoint đăng nhập
@router.post("/login")
def login(user: UserLogin, db: Session = Depends(get_db), api_key: str = get_api_key):
    return login_user(db, user.email, user.password)

# Endpoint cập nhật thông tin tài khoản
@router.put("/profile/{user_id}")
def update(user_id: int, user: UserUpdate, db: Session = Depends(get_db), api_key: str = get_api_key):
    return update_profile(db, user_id, user.username, user.email)

# Endpoint đổi mật khẩu
@router.put("/change-password/{user_id}")
def change_password_endpoint(user_id: int, password: PasswordChange, db: Session = Depends(get_db), api_key: str = get_api_key):
    return change_password(db, user_id, password.current_password, password.new_password)

# Endpoint thay đổi quyền account_type 
@router.put("/change-account-type/{user_id}")
def change_account_type_endpoint(user_id: int, account_type: AccountTypeChange, db: Session = Depends(get_db), api_key: str = get_api_key):
    return change_account_type(db, user_id, account_type.account_type)

# Endpoint cập nhật trạng thái is_active 
@router.put("/update-active-status/{user_id}")
def update_active_status_endpoint(user_id: int, status: ActiveStatusUpdate, db: Session = Depends(get_db), api_key: str = get_api_key):
    return update_active_status(db, user_id, status.is_active)

# Endpoint xóa tài khoản 
@router.delete("/delete/{user_id}")
def delete_user_endpoint(user_id: int, db: Session = Depends(get_db), api_key: str = get_api_key):
    return delete_user(db, user_id)

# Endpoint đặt lại mật khẩu (không cần mật khẩu cũ)
@router.put("/reset-password/{user_id}", summary="Đặt lại mật khẩu mới cho người dùng")
def reset_password_endpoint(
    user_id: int, 
    password: PasswordReset, 
    db: Session = Depends(get_db), 
    api_key: str = get_api_key
):
    """
    Đặt lại mật khẩu mới cho người dùng mà không cần xác minh mật khẩu cũ.
    
    - **user_id**: ID của người dùng cần đặt lại mật khẩu
    - **new_password**: Mật khẩu mới (tối thiểu 6 ký tự)
    
    Chỉ admin mới nên có quyền sử dụng API này.
    """
    return reset_password(db, user_id, password.new_password)

# Endpoint lấy danh sách tất cả người dùng
@router.get("/users", summary="Lấy danh sách tất cả người dùng")
def get_users(
    skip: int = Query(0, ge=0, description="Số lượng bản ghi bỏ qua"),
    limit: int = Query(100, ge=1, le=100, description="Số lượng bản ghi tối đa trả về"),
    db: Session = Depends(get_db), 
    api_key: str = get_api_key
):
    """
    Lấy danh sách tất cả người dùng trong hệ thống.
    
    - **skip**: Số lượng bản ghi bỏ qua (pagination)
    - **limit**: Số lượng bản ghi tối đa trả về (tối đa 100)
    
    Trả về danh sách người dùng, bao gồm thông tin cơ bản.
    Không bao gồm mật khẩu người dùng.
    """
    return get_all_users(db, skip, limit)

# Endpoint cộng credits
@router.put("/add-credits/{user_id}", summary="Cộng credits vào tài khoản người dùng")
def add_credits_endpoint(
    user_id: int, 
    credit_data: CreditUpdate, 
    db: Session = Depends(get_db), 
    api_key: str = get_api_key
):
    """
    Cộng một số lượng credits vào tài khoản của người dùng.
    
    - **user_id**: ID của người dùng
    - **amount**: Số lượng credits cần cộng (phải lớn hơn 0)
    
    Trả về thông tin số dư hiện tại của người dùng sau khi cộng.
    """
    return add_user_credits(db, user_id, credit_data.amount)

# Endpoint trừ credits
@router.put("/subtract-credits/{user_id}", summary="Trừ credits từ tài khoản người dùng")
def subtract_credits_endpoint(
    user_id: int, 
    credit_data: CreditUpdate, 
    db: Session = Depends(get_db), 
    api_key: str = get_api_key
):
    """
    Trừ một số lượng credits từ tài khoản của người dùng.
    
    - **user_id**: ID của người dùng
    - **amount**: Số lượng credits cần trừ (phải lớn hơn 0)
    
    Trả về thông tin số dư hiện tại của người dùng sau khi trừ.
    """
    return subtract_user_credits(db, user_id, credit_data.amount)