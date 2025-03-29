# router/config.py
from fastapi import APIRouter, Depends, HTTPException
from app.database.connection import get_db
from app.database.api import config as config_api
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional

router = APIRouter(
    prefix="/config",
    tags=["config"]
)

# Định nghĩa model cho dữ liệu đầu vào khi cập nhật
class ConfigUpdate(BaseModel):
    name_web: Optional[str] = None
    address_web: Optional[str] = None
    logo_link: Optional[str] = None
    api_key: Optional[str] = None
    name_owner: Optional[str] = None
    phone_1: Optional[str] = None
    phone_2: Optional[str] = None
    google_map_link: Optional[str] = None
    price: Optional[int] = None
    address: Optional[str] = None
    email: Optional[str] = None

@router.get("/")
def get_config(db: Session = Depends(get_db)):
    """
    Lấy thông tin cấu hình
    """
    return config_api.get_config(db)

@router.put("/")
def update_config(config_data: ConfigUpdate, db: Session = Depends(get_db)):
    """
    Cập nhật thông tin cấu hình
    """
    # Chuyển đổi dữ liệu thành dict và loại bỏ các giá trị None
    update_data = {k: v for k, v in config_data.dict().items() if v is not None}
    if not update_data:
        raise HTTPException(status_code=400, detail="Không có dữ liệu để cập nhật")
    return config_api.update_config(db, update_data)