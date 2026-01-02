"""OCR module using DeepSeek-OCR for extracting text from images."""
import re
import base64
from io import BytesIO
from typing import Optional, Tuple
from PIL import Image

from config import config

# Global model cache for local inference
_model_cache = {}


def get_deepseek_model():
    """Load and cache DeepSeek-OCR model for local inference."""
    if "model" not in _model_cache:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
            
            print("Loading DeepSeek-OCR model... (this may take a few minutes)")
            
            model_name = config.DEEPSEEK_OCR_MODEL
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            
            # Try flash attention first, fallback to standard
            try:
                model = AutoModel.from_pretrained(
                    model_name,
                    _attn_implementation='flash_attention_2',
                    trust_remote_code=True,
                    use_safetensors=True
                )
            except Exception:
                model = AutoModel.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    use_safetensors=True
                )
            
            # Move to device
            if config.DEEPSEEK_OCR_DEVICE == "cuda":
                import torch
                model = model.eval().cuda().to(torch.bfloat16)
            else:
                model = model.eval()
            
            _model_cache["model"] = model
            _model_cache["tokenizer"] = tokenizer
            print("DeepSeek-OCR model loaded successfully!")
            
        except ImportError as e:
            raise RuntimeError(
                f"Missing dependencies for local DeepSeek-OCR: {e}\n"
                "Install with: pip install torch transformers einops addict easydict"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load DeepSeek-OCR model: {e}")
    
    return _model_cache["model"], _model_cache["tokenizer"]


def ocr_with_deepseek_local(image: Image.Image, prompt: str = None) -> str:
    """
    Run OCR using local DeepSeek-OCR model.
    Requires NVIDIA GPU with CUDA.
    """
    model, tokenizer = get_deepseek_model()
    
    # Default prompt for Vietnamese documents
    if prompt is None:
        prompt = "<image>\n<|grounding|>Convert the document to markdown."
    
    # Ensure RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Run inference
    result = model.chat(
        image=image,
        msgs=[{"role": "user", "content": prompt}],
        tokenizer=tokenizer
    )
    
    return result


def ocr_with_deepseek_vllm(image: Image.Image, prompt: str = None) -> str:
    """
    Run OCR using DeepSeek-OCR via vLLM server (OpenAI-compatible API).
    Requires running vLLM server with DeepSeek-OCR model.
    """
    from openai import OpenAI
    
    # Default prompt
    if prompt is None:
        prompt = "Free OCR."
    
    # Ensure RGB and convert to base64
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    
    # Create client
    client = OpenAI(
        api_key="EMPTY",
        base_url=config.DEEPSEEK_VLLM_URL,
        timeout=300
    )
    
    # Make request
    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-OCR",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                },
                {"type": "text", "text": prompt}
            ]
        }],
        max_tokens=4096,
        temperature=0.0,
        extra_body={
            "skip_special_tokens": False,
            "vllm_xargs": {
                "ngram_size": 30,
                "window_size": 90,
                "whitelist_token_ids": [128821, 128822],  # <td>, </td>
            }
        }
    )
    
    return response.choices[0].message.content


def ocr_with_clarifai(image: Image.Image, prompt: str = None) -> str:
    """
    Run OCR using DeepSeek-OCR via Clarifai API.
    Requires CLARIFAI_PAT environment variable.
    """
    import requests
    
    if not config.CLARIFAI_PAT:
        raise ValueError("CLARIFAI_PAT not configured")
    
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


def ocr_with_tesseract(image: Image.Image, lang: str = None) -> str:
    """Fallback OCR using Tesseract."""
    try:
        import pytesseract
        lang = lang or config.TESSERACT_LANG
        return pytesseract.image_to_string(image, lang=lang)
    except ImportError:
        return mock_ocr(image)


def mock_ocr(image: Image.Image) -> str:
    """Mock OCR function for demonstration."""
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
    mode: str = None,
    prompt: str = None
) -> Tuple[str, str]:
    """
    Extract text from image using specified OCR provider.
    
    Args:
        image_data: Raw image bytes
        provider: OCR provider ("deepseek", "tesseract", or "mock")
        mode: For DeepSeek - "local", "vllm", or "api" (Clarifai)
        prompt: Custom prompt for DeepSeek OCR
    
    Returns:
        Tuple of (extracted_text, provider_used)
    """
    provider = provider or config.OCR_PROVIDER
    mode = mode or config.DEEPSEEK_OCR_MODE
    
    image = Image.open(BytesIO(image_data))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    try:
        if provider == "deepseek":
            if mode == "local":
                text = ocr_with_deepseek_local(image, prompt)
                return text, "deepseek-local"
            elif mode == "vllm":
                text = ocr_with_deepseek_vllm(image, prompt)
                return text, "deepseek-vllm"
            elif mode == "api":
                text = ocr_with_clarifai(image, prompt)
                return text, "deepseek-clarifai"
            else:
                raise ValueError(f"Unknown DeepSeek mode: {mode}")
        
        elif provider == "tesseract":
            text = ocr_with_tesseract(image)
            return text, "tesseract"
        
        elif provider == "mock":
            text = mock_ocr(image)
            return text, "mock"
        
        else:
            raise ValueError(f"Unknown OCR provider: {provider}")
            
    except Exception as e:
        # Fallback to mock if any error
        print(f"OCR error ({provider}/{mode}): {e}")
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