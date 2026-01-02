"""Configuration settings for OCR Pipeline."""
import os
from dataclasses import dataclass

@dataclass
class Config:
    # API Settings
    BACKEND_HOST: str = "0.0.0.0"
    BACKEND_PORT: int = 8000
    
    # LLM Settings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")  # "openai" or "anthropic"
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    
    # DeepSeek OCR Settings
    OCR_PROVIDER: str = os.getenv("OCR_PROVIDER", "deepseek")  # "deepseek" or "tesseract"
    DEEPSEEK_OCR_MODE: str = os.getenv("DEEPSEEK_OCR_MODE", "local")  # "local" or "api"
    
    # For local DeepSeek OCR (requires GPU)
    DEEPSEEK_OCR_MODEL: str = "deepseek-ai/DeepSeek-OCR"
    DEEPSEEK_OCR_DEVICE: str = os.getenv("DEEPSEEK_OCR_DEVICE", "cuda")  # "cuda" or "cpu"
    
    # For DeepSeek OCR via Clarifai API
    CLARIFAI_PAT: str = os.getenv("CLARIFAI_PAT", "")
    CLARIFAI_BASE_URL: str = "https://api.clarifai.com/v2/users/deepseek-ai/apps/deepseek-ocr/models/DeepSeek-OCR/versions/1/outputs"
    
    # For self-hosted vLLM server
    DEEPSEEK_VLLM_URL: str = os.getenv("DEEPSEEK_VLLM_URL", "http://localhost:8001/v1")
    
    # Legacy Tesseract settings (fallback)
    TESSERACT_LANG: str = "vie+eng"  # Vietnamese + English
    
    # File Settings
    UPLOAD_DIR: str = "/tmp/ocr_uploads"
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB

config = Config()