from app.database.connection import Base, engine
# Import tất cả models
from app.database.models import *

def create_tables():
    # Tạo tất cả bảng cùng lúc
    Base.metadata.create_all(bind=engine)