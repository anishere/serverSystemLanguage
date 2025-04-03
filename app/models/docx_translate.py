from pydantic import BaseModel
from enum import Enum
from typing import Optional


class SupportedLanguage(str, Enum):
    """Supported languages for document translation"""
    ENGLISH = "en"
    VIETNAMESE = "vi"
    CHINESE = "zh"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    KOREAN = "ko"
    RUSSIAN = "ru"


class DocxTranslationRequest(BaseModel):
    """Request model for DOCX file translation"""
    target_language: SupportedLanguage
    model: Optional[str] = "gpt-4o-mini"
    temperature: Optional[float] = 0.3
    workers: Optional[int] = 4


class DocxTranslationResponse(BaseModel):
    """Response model for DOCX file translation"""
    filename: str
    message: str
    download_url: str
    status_url: Optional[str] = None
    translation_id: Optional[str] = None 