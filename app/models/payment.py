from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from app.database.connection import Base

class PaymentTransaction(Base):
    """Lưu trữ thông tin giao dịch thanh toán PayOS"""
    __tablename__ = "payment_transactions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    order_code = Column(String(100), nullable=False, unique=True, index=True)
    amount = Column(Integer, nullable=False)
    credits_to_add = Column(Integer, nullable=False)
    status = Column(String(50), default="PENDING")
    checkout_url = Column(Text, nullable=True)
    qr_code = Column(Text, nullable=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())

# Thêm vào model User hiện có
"""
# Trong file User model, thêm dòng này:
from app.database.models.payment import PaymentTransaction
payments = relationship("PaymentTransaction", backref="user")
"""