from pydantic import BaseModel
from enum import Enum
from typing import Optional, Dict, Any, List


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


class DocumentAnalysis(BaseModel):
    """Model for document analysis information"""
    total_paragraphs: int
    average_length: float
    xml_files: Optional[int] = None
    structure_details: Optional[Dict[str, Any]] = None


class ProgressBar(BaseModel):
    """Model for progress bar information"""
    progress_percent: int
    progress_bar: str
    progress_text: str


class ProgressDisplay(BaseModel):
    """Model for detailed progress display information, including terminal-style output"""
    progress_bar: Optional[ProgressBar] = None
    stage_info: str
    current_item: Optional[str] = None
    elapsed_time_str: str
    eta_str: str
    speed: Optional[str] = None
    terminal_style: str


class ProcessingDetails(BaseModel):
    """Model for detailed translation processing information"""
    current_step: str
    total_paragraphs: int = 0
    processed_paragraphs: int = 0
    total_files: int = 0
    processed_files: int = 0
    cache_stats: Optional[Dict[str, Any]] = None


class TranslationStatus(BaseModel):
    """Model for translation status information"""
    status: str
    progress: int
    message: str
    started_at: float
    updated_at: float
    elapsed_time: Optional[float] = None
    estimated_time_remaining: Optional[float] = None
    document_analysis: Optional[DocumentAnalysis] = None
    processing_details: Optional[ProcessingDetails] = None
    progress_display: Optional[ProgressDisplay] = None
    error: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None


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