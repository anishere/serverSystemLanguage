from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from app.database.connection import get_db
from app.database.models import CreditTransaction, TransactionType, User
from fastapi import Depends, HTTPException, status
from passlib.context import CryptContext
from datetime import datetime

# Khởi tạo context để mã hóa mật khẩu
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_password_hash(password: str):
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str):
    return pwd_context.verify(plain_password, hashed_password)

# Logic xử lý đăng ký
def register_user(db: Session, username: str, email: str, password: str):
    try:
        hashed_password = get_password_hash(password)
        new_user = User(username=username, email=email, password=hashed_password, credits=5000)
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        return {"message": "Đăng ký thành công", "user_id": new_user.user_id, "credits": new_user.credits}
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=400, detail="Email đã tồn tại")

# Logic xử lý đăng nhập
def login_user(db: Session, email: str, password: str):
    user = db.query(User).filter(User.email == email).first()
    if not user or not verify_password(password, user.password):
        raise HTTPException(status_code=401, detail="Email hoặc mật khẩu không đúng")
    user.last_login = datetime.utcnow()
    db.commit()
    # Trả về thông tin người dùng (trừ mật khẩu)
    return {
        "message": "Đăng nhập thành công",
        "user": {
            "user_id": user.user_id,
            "username": user.username,
            "email": user.email,
            "created_at": user.created_at,
            "updated_at": user.updated_at,
            "last_login": user.last_login,
            "is_active": user.is_active,
            "credits": user.credits,
            "account_type": user.account_type
        }
    }

# Logic cập nhật thông tin tài khoản
def update_profile(db: Session, user_id: int, username: str = None, email: str = None):
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Người dùng không tồn tại")
    if username:
        user.username = username
    if email:
        user.email = email
    user.updated_at = datetime.utcnow()
    db.commit()
    return {"message": "Cập nhật thông tin thành công"}

# Logic đổi mật khẩu
def change_password(db: Session, user_id: int, current_password: str, new_password: str):
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user or not verify_password(current_password, user.password):
        raise HTTPException(status_code=401, detail="Mật khẩu hiện tại không đúng")
    user.password = get_password_hash(new_password)
    user.updated_at = datetime.utcnow()
    db.commit()
    return {"message": "Đổi mật khẩu thành công"}

# Logic thay đổi quyền account_type
def change_account_type(db: Session, user_id: int, new_account_type: str):
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Người dùng không tồn tại")
    # Có thể thêm logic kiểm tra giá trị hợp lệ cho new_account_type nếu cần
    valid_account_types = ["0", "1", "2"]  # Ví dụ: "0" = user, "1" = admin, "2" = moderator
    if new_account_type not in valid_account_types:
        raise HTTPException(status_code=400, detail="Loại tài khoản không hợp lệ")
    user.account_type = new_account_type
    user.updated_at = datetime.utcnow()
    db.commit()
    return {"message": "Thay đổi quyền tài khoản thành công", "account_type": new_account_type}

# Logic cập nhật trạng thái is_active 
def update_active_status(db: Session, user_id: int, is_active: int):
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Người dùng không tồn tại")
    if is_active not in [0, 1]:  # Chỉ cho phép giá trị 0 hoặc 1
        raise HTTPException(status_code=400, detail="Trạng thái không hợp lệ, chỉ chấp nhận 0 hoặc 1")
    user.is_active = is_active
    user.updated_at = datetime.utcnow()
    db.commit()
    return {"message": "Cập nhật trạng thái thành công", "is_active": user.is_active}

# Logic xóa tài khoản 
def delete_user(db: Session, user_id: int):
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Người dùng không tồn tại")
    db.delete(user)  # Xóa cứng
    db.commit()
    return {"message": "Xóa tài khoản thành công"}

# Logic đặt lại mật khẩu (không cần mật khẩu cũ)
def reset_password(db: Session, user_id: int, new_password: str):
    """
    Hàm đặt lại mật khẩu mới cho người dùng mà không cần biết mật khẩu cũ.
    Thường dùng cho admin hoặc chức năng quên mật khẩu.
    """
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Người dùng không tồn tại")
    
    # Kiểm tra độ dài mật khẩu tối thiểu
    if len(new_password) < 6:
        raise HTTPException(status_code=400, detail="Mật khẩu phải có ít nhất 6 ký tự")
    
    user.password = get_password_hash(new_password)
    user.updated_at = datetime.utcnow()
    db.commit()
    return {"message": "Đặt lại mật khẩu thành công"}

# Logic lấy danh sách tất cả người dùng
def get_all_users(db: Session, skip: int = 0, limit: int = 100):
    """
    Lấy danh sách tất cả người dùng với phân trang.
    Mặc định trả về tối đa 100 người dùng mỗi lần.
    """
    users = db.query(User).offset(skip).limit(limit).all()
    return [
        {
            "id": user.user_id,
            "username": user.username,
            "email": user.email,
            "created_at": user.created_at,
            "updated_at": user.updated_at,
            "last_login": user.last_login,
            "is_active": user.is_active,
            "credits": user.credits,
            "account_type": user.account_type
        } for user in users
    ]

# Logic cộng credits cho người dùng
def add_user_credits(db: Session, user_id: int, amount: int):
    """
    Cộng một số lượng credits vào tài khoản của người dùng
    """
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Người dùng không tồn tại")
    
    if amount <= 0:
        raise HTTPException(status_code=400, detail="Số tiền phải lớn hơn 0")
    
    user.credits += amount
    user.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(user)
    
    return {
        "message": f"Đã cộng {amount} credits cho người dùng",
        "current_credits": user.credits,
        "user_id": user.user_id,
        "username": user.username
    }

# Logic trừ credits từ người dùng
def subtract_user_credits(db: Session, user_id: int, amount: int):
    """
    Trừ một số lượng credits từ tài khoản của người dùng
    """
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Người dùng không tồn tại")
    
    if amount <= 0:
        raise HTTPException(status_code=400, detail="Số tiền phải lớn hơn 0")
    
    if user.credits < amount:
        raise HTTPException(status_code=400, detail="Số dư không đủ để trừ")
    
    user.credits -= amount
    user.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(user)
    
    return {
        "message": f"Đã trừ {amount} credits từ người dùng",
        "current_credits": user.credits,
        "user_id": user.user_id,
        "username": user.username
    }