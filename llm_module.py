"""LLM module for paraphrasing Vietnamese text in XML."""
import re
from typing import Optional
from config import config


async def paraphrase_xml(xml_content: str, provider: str = None) -> str:
    """
    Paraphrase Vietnamese text in XML using LLM.
    
    Args:
        xml_content: XML string with Vietnamese text
        provider: LLM provider ("openai" or "anthropic")
    
    Returns:
        XML with paraphrased Vietnamese text
    """
    provider = provider or config.LLM_PROVIDER
    
    prompt = f"""Bạn là một chuyên gia về ngôn ngữ tiếng Việt. 
Hãy paraphrase (viết lại) các đoạn văn bản tiếng Việt trong XML sau đây, 
giữ nguyên cấu trúc XML và các thẻ, chỉ thay đổi nội dung văn bản bên trong các thẻ.
Đảm bảo giữ nguyên ý nghĩa nhưng sử dụng cách diễn đạt khác.

XML gốc:
{xml_content}

Trả về XML đã được paraphrase:"""

    if provider == "openai":
        from openai import AsyncOpenAI
        
        if not config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not configured")
        
        client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
        response = await client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content
    
    elif provider == "anthropic":
        from anthropic import AsyncAnthropic
        
        if not config.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY not configured")
        
        client = AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)
        response = await client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


def mock_paraphrase(xml_content: str) -> str:
    """
    Mock paraphrase function for demonstration/testing.
    Simply adds "[Paraphrased]" prefix to content in XML tags.
    """
    def replace_content(match):
        tag = match.group(1)
        content = match.group(2)
        closing_tag = match.group(3)
        
        # Skip XML declaration and empty content
        if tag.startswith('?') or not content.strip():
            return match.group(0)
        
        # Add paraphrase marker
        paraphrased = f"[Đã viết lại] {content}"
        return f"<{tag}>{paraphrased}</{closing_tag}>"
    
    # Match XML tags with content
    pattern = r'<([^/>]+)>([^<]+)</([^>]+)>'
    result = re.sub(pattern, replace_content, xml_content)
    
    return result