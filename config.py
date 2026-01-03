"""Configuration settings for OCR Pipeline - API Only Mode."""
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
    
    # DeepSeek OCR API Settings
    # Option 1: Replicate API (Recommended - ~$0.011 per run, very cheap)
    REPLICATE_API_TOKEN: str = os.getenv("REPLICATE_API_TOKEN", "")
    
    # Option 2: Clarifai API (Has free tier)
    CLARIFAI_PAT: str = os.getenv("CLARIFAI_PAT", "")
    CLARIFAI_BASE_URL: str = "https://api.clarifai.com/v2/users/deepseek-ai/apps/deepseek-ocr/models/DeepSeek-OCR/versions/1/outputs"
    
    # Default OCR Provider: "replicate" (recommended) or "clarifai"
    OCR_PROVIDER: str = os.getenv("OCR_PROVIDER", "replicate")
    
    # File Settings
    UPLOAD_DIR: str = "/tmp/ocr_uploads"
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB

config = Config()