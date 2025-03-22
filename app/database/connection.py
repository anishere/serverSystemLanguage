from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

# Cấu hình thông tin kết nối đến MySQL (XAMPP)
SQLALCHEMY_DATABASE_URL = "mysql+mysqlconnector://root:@localhost:3306/systemlanguaguedb"

# Tạo engine kết nối
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_size=5,  # Số kết nối tối đa trong pool
    max_overflow=10,  # Số kết nối bổ sung khi vượt quá pool_size
    pool_timeout=30,  # Thời gian chờ khi không có kết nối trống
    pool_pre_ping=True  # Kiểm tra kết nối trước khi sử dụng
)

# Tạo SessionLocal để quản lý session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Tạo Base cho các model
Base = declarative_base()

# Dependency để lấy session database
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
