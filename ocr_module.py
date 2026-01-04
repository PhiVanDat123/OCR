"""OCR module using DeepSeek-OCR via API (Replicate or Clarifai)."""
import re
import base64
import time
from io import BytesIO
from typing import Optional, Tuple
from PIL import Image

from config import config


def ocr_with_replicate(image: Image.Image, prompt: str = None) -> str:
    """
    Run OCR using DeepSeek-OCR via Replicate API.
    
    Replicate pricing: ~$0.011 per run (~90 runs per $1)
    Get API token at: https://replicate.com/account/api-tokens
    
    Args:
        image: PIL Image object
        prompt: Custom prompt for OCR (optional)
    
    Returns:
        Extracted text from image
    """
    import replicate
    
    if not config.REPLICATE_API_TOKEN:
        raise ValueError(
            "REPLICATE_API_TOKEN not configured.\n"
            "Get your token at: https://replicate.com/account/api-tokens\n"
            "Then set: export REPLICATE_API_TOKEN=your-token"
        )
    
    # Set API token
    import os
    os.environ["REPLICATE_API_TOKEN"] = config.REPLICATE_API_TOKEN
    
    # Default prompt
    if prompt is None:
        prompt = "Convert the document to markdown."
    
    # Ensure RGB and convert to base64
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    image_uri = f"data:image/png;base64,{img_base64}"
    
    # Run prediction with latest version hash (updated Jan 2026)
    output = replicate.run(
        "lucataco/deepseek-ocr:cb3b474fbfc56b1664c8c7841550bccecbe7b74c30e45ce938ffca1180b4dff5",
        input={
            "image": image_uri,
            "prompt": prompt
        }
    )
    
    return output


def ocr_with_clarifai(image: Image.Image, prompt: str = None) -> str:
    """
    Run OCR using DeepSeek-OCR via Clarifai API.
    
    Clarifai has a free tier with limited usage.
    Get PAT at: https://clarifai.com/settings/security
    
    Args:
        image: PIL Image object
        prompt: Custom prompt for OCR (optional)
    
    Returns:
        Extracted text from image
    """
    import requests
    
    if not config.CLARIFAI_PAT:
        raise ValueError(
            "CLARIFAI_PAT not configured.\n"
            "Get your token at: https://clarifai.com/settings/security\n"
            "Then set: export CLARIFAI_PAT=your-token"
        )
    
    # Default prompt
    if prompt is None:
        prompt = "Free OCR."
    
    # Ensure RGB and convert to base64
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    
    # Make request to Clarifai
    headers = {
        "Authorization": f"Key {config.CLARIFAI_PAT}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "inputs": [{
            "data": {
                "image": {"base64": img_base64},
                "text": {"raw": prompt}
            }
        }]
    }
    
    response = requests.post(
        config.CLARIFAI_BASE_URL,
        headers=headers,
        json=payload,
        timeout=120
    )
    
    if response.status_code != 200:
        raise RuntimeError(f"Clarifai API error: {response.text}")
    
    result = response.json()
    return result["outputs"][0]["data"]["text"]["raw"]


def mock_ocr(image: Image.Image) -> str:
    """Mock OCR function for demonstration/testing."""
    return """Hợp đồng mua bán
    
Bên A: Công ty TNHH ABC
Địa chỉ: 123 Đường Nguyễn Huệ, Quận 1, TP.HCM

Bên B: Ông Nguyễn Văn A
Địa chỉ: 456 Đường Lê Lợi, Quận 3, TP.HCM

Điều 1: Đối tượng hợp đồng
Bên A đồng ý bán cho Bên B sản phẩm theo danh sách đính kèm.

Điều 2: Giá trị hợp đồng
Tổng giá trị: 100.000.000 VNĐ (Một trăm triệu đồng)

Điều 3: Phương thức thanh toán
Thanh toán 50% khi ký hợp đồng, 50% còn lại khi giao hàng."""


def extract_text_from_image(
    image_data: bytes, 
    provider: str = None,
    prompt: str = None
) -> Tuple[str, str]:
    """
    Extract text from image using specified OCR API provider.
    
    Args:
        image_data: Raw image bytes
        provider: OCR provider ("replicate", "clarifai", or "mock")
        prompt: Custom prompt for DeepSeek OCR
    
    Returns:
        Tuple of (extracted_text, provider_used)
    """
    provider = provider or config.OCR_PROVIDER
    
    image = Image.open(BytesIO(image_data))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    try:
        if provider == "replicate":
            text = ocr_with_replicate(image, prompt)
            return text, "replicate"
        
        elif provider == "clarifai":
            text = ocr_with_clarifai(image, prompt)
            return text, "clarifai"
        
        elif provider == "mock":
            text = mock_ocr(image)
            return text, "mock"
        
        else:
            raise ValueError(f"Unknown OCR provider: {provider}. Use 'replicate', 'clarifai', or 'mock'")
            
    except Exception as e:
        # Fallback to mock if any error
        print(f"OCR error ({provider}): {e}")
        print("Falling back to mock OCR...")
        return mock_ocr(image), "mock (fallback)"


def text_to_xml(text: str) -> str:
    """
    Convert extracted text to structured XML format.
    
    Args:
        text: Raw text from OCR
    
    Returns:
        XML string with structured content
    """
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    xml_parts = ['<?xml version="1.0" encoding="UTF-8"?>', '<document>']
    
    current_section = None
    section_content = []
    
    for line in lines:
        # Check if line is a section header (starts with "Điều" or similar patterns)
        if re.match(r'^(Điều|Mục|Phần|Chương|##?\s)', line, re.IGNORECASE):
            # Save previous section
            if current_section:
                xml_parts.append(f'  <section title="{escape_xml(current_section)}">')
                for content in section_content:
                    xml_parts.append(f'    <paragraph>{escape_xml(content)}</paragraph>')
                xml_parts.append('  </section>')
            
            # Clean markdown headers if present
            clean_title = re.sub(r'^#+\s*', '', line)
            current_section = clean_title
            section_content = []
        elif re.match(r'^(Bên\s+[A-Z]|Địa chỉ|Tổng giá trị|Thanh toán|[A-Za-z]+):', line, re.IGNORECASE):
            # Key-value pairs
            if ':' in line:
                key, value = line.split(':', 1)
                xml_parts.append(f'  <field name="{escape_xml(key.strip())}">{escape_xml(value.strip())}</field>')
            else:
                section_content.append(line)
        elif current_section:
            section_content.append(line)
        else:
            # Header or standalone text
            if not any(char.isalnum() for char in line):
                continue
            xml_parts.append(f'  <header>{escape_xml(line)}</header>')
    
    # Save last section
    if current_section:
        xml_parts.append(f'  <section title="{escape_xml(current_section)}">')
        for content in section_content:
            xml_parts.append(f'    <paragraph>{escape_xml(content)}</paragraph>')
        xml_parts.append('  </section>')
    
    xml_parts.append('</document>')
    
    return '\n'.join(xml_parts)


def escape_xml(text: str) -> str:
    """Escape special XML characters."""
    return (text
            .replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
            .replace('"', '&quot;')
            .replace("'", '&apos;'))