# api/config.py
from sqlalchemy.orm import Session
from app.database.models import Config
from fastapi import HTTPException

def get_config(db: Session):
    """
    Lấy thông tin cấu hình từ bảng config
    """
    config = db.query(Config).first()
    if not config:
        raise HTTPException(status_code=404, detail="Không tìm thấy thông tin cấu hình")

    # Chuẩn bị dữ liệu trả về
    config_data = {
        "id": config.id,
        "name_web": config.name_web,
        "address_web": config.address_web,
        "logo_link": config.logo_link,  # logo_link giờ là chuỗi base64
        "api_key": config.api_key,
        "name_owner": config.name_owner,
        "phone_1": config.phone_1,
        "phone_2": config.phone_2,
        "google_map_link": config.google_map_link,
        "price": config.price
    }

    return config_data

def update_config(db: Session, config_data: dict):
    """
    Cập nhật thông tin cấu hình
    """
    try:
        config = db.query(Config).first()
        if not config:
            # Nếu chưa có bản ghi, tạo mới
            config = Config(**config_data)
            db.add(config)
        else:
            # Cập nhật các trường
            for key, value in config_data.items():
                setattr(config, key, value)
        
        db.commit()
        db.refresh(config)
        return {
            "message": "Cập nhật cấu hình thành công",
            "config": config
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Lỗi khi cập nhật cấu hình: {str(e)}")