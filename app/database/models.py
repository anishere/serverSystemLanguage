from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, DECIMAL, Boolean, BigInteger, Enum
from sqlalchemy.orm import relationship
from app.database.connection import Base
import enum
from datetime import datetime

class User(Base):
    __tablename__ = "users"
    
    user_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    username = Column(String(100), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    password = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)
    credits = Column(BigInteger, default=5000)
    account_type = Column(String(50), default="standard")
    
    # Relationships
    translations = relationship("TranslationHistory", back_populates="user")
    transactions = relationship("CreditTransaction", back_populates="user")

class TransactionType(enum.Enum):
    purchase = "purchase"
    usage = "usage"

class CreditTransaction(Base):
    __tablename__ = "credit_transactions"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.user_id", ondelete="RESTRICT"), nullable=False)
    amount = Column(DECIMAL(10, 2), nullable=False)
    transaction_type = Column(Enum(TransactionType), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    payment_method = Column(String(50), nullable=True)
    
    # Relationship
    user = relationship("User", back_populates="transactions")

class TranslationHistory(Base):
    __tablename__ = "translation_history"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)
    input_text = Column(Text, nullable=False)
    output_text = Column(Text, nullable=False)
    source_language = Column(String(20), nullable=False)
    target_language = Column(String(20), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    character_count = Column(Integer, nullable=False)
    credits_used = Column(DECIMAL(10, 2), default=0, nullable=False)
    
    # Relationship
    user = relationship("User", back_populates="translations")