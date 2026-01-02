"""FastAPI Backend for OCR Pipeline with DeepSeek-OCR."""
import os
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from config import config
from ocr_module import extract_text_from_image, text_to_xml
from llm_module import paraphrase_xml, mock_paraphrase

# Create upload directory
os.makedirs(config.UPLOAD_DIR, exist_ok=True)

app = FastAPI(
    title="OCR Pipeline API (DeepSeek-OCR)",
    description="Pipeline: Image → DeepSeek-OCR → XML → LLM Paraphrase → Clean XML",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class OCRResponse(BaseModel):
    raw_text: str
    raw_xml: str
    paraphrased_xml: str
    ocr_provider: str
    success: bool
    message: str


class XMLParaphraseRequest(BaseModel):
    xml_content: str
    provider: Optional[str] = None
    use_mock: bool = False


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "healthy", "service": "OCR Pipeline API (DeepSeek-OCR)"}


@app.get("/config")
async def get_config():
    """Get current configuration (without sensitive data)."""
    return {
        "ocr_provider": config.OCR_PROVIDER,
        "ocr_mode": config.DEEPSEEK_OCR_MODE,
        "llm_provider": config.LLM_PROVIDER,
        "llm_model": config.LLM_MODEL,
        "openai_configured": bool(config.OPENAI_API_KEY),
        "anthropic_configured": bool(config.ANTHROPIC_API_KEY),
        "clarifai_configured": bool(config.CLARIFAI_PAT),
        "vllm_url": config.DEEPSEEK_VLLM_URL,
    }


@app.post("/ocr", response_model=OCRResponse)
async def process_ocr(
    file: UploadFile = File(...),
    ocr_provider: Optional[str] = Form(default=None),
    ocr_mode: Optional[str] = Form(default=None),
    ocr_prompt: Optional[str] = Form(default=None),
    use_mock_llm: bool = Form(default=True),
    llm_provider: Optional[str] = Form(default=None)
):
    """
    Complete OCR pipeline with DeepSeek-OCR:
    1. Extract text from image using DeepSeek-OCR
    2. Convert to structured XML
    3. Paraphrase Vietnamese text using LLM
    4. Return clean XML
    
    OCR Providers:
    - deepseek: DeepSeek-OCR (default)
    - tesseract: Tesseract OCR (fallback)
    - mock: Mock data for testing
    
    DeepSeek Modes (when ocr_provider=deepseek):
    - local: Run model locally (requires GPU)
    - vllm: Use vLLM server
    - api: Use Clarifai API
    """
    # Validate file type
    allowed_types = ["image/jpeg", "image/png", "image/gif", "image/bmp", "image/tiff"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type. Allowed: {', '.join(allowed_types)}"
        )
    
    # Read file
    image_data = await file.read()
    
    if len(image_data) > config.MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large (max 10MB)")
    
    try:
        # Step 1: OCR - Extract text from image using DeepSeek-OCR
        raw_text, provider_used = extract_text_from_image(
            image_data,
            provider=ocr_provider,
            mode=ocr_mode,
            prompt=ocr_prompt
        )
        
        if not raw_text:
            return OCRResponse(
                raw_text="",
                raw_xml="",
                paraphrased_xml="",
                ocr_provider=provider_used,
                success=False,
                message="No text detected in image"
            )
        
        # Step 2: Convert to XML
        raw_xml = text_to_xml(raw_text)
        
        # Step 3: Paraphrase with LLM
        if use_mock_llm:
            paraphrased_xml = mock_paraphrase(raw_xml)
        else:
            try:
                paraphrased_xml = await paraphrase_xml(raw_xml, llm_provider)
            except Exception as e:
                # Fallback to mock if LLM fails
                paraphrased_xml = mock_paraphrase(raw_xml)
        
        return OCRResponse(
            raw_text=raw_text,
            raw_xml=raw_xml,
            paraphrased_xml=paraphrased_xml,
            ocr_provider=provider_used,
            success=True,
            message="OCR pipeline completed successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.post("/paraphrase")
async def paraphrase_only(request: XMLParaphraseRequest):
    """Paraphrase Vietnamese text in existing XML."""
    try:
        if request.use_mock:
            result = mock_paraphrase(request.xml_content)
        else:
            result = await paraphrase_xml(request.xml_content, request.provider)
        
        return {
            "original_xml": request.xml_content,
            "paraphrased_xml": result,
            "success": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/text-to-xml")
async def convert_text_to_xml(text: str = Form(...)):
    """Convert plain text to XML structure."""
    try:
        xml_result = text_to_xml(text)
        return {"xml": xml_result, "success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.BACKEND_HOST, port=config.BACKEND_PORT)