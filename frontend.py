"""Gradio Frontend for OCR Pipeline - API Only Mode."""
import gradio as gr
import requests
import xml.dom.minidom
from typing import Tuple, Optional

# Configuration
BACKEND_URL = "http://localhost:8000"


def format_xml(xml_string: str) -> str:
    """Format XML with proper indentation."""
    try:
        dom = xml.dom.minidom.parseString(xml_string)
        return dom.toprettyxml(indent="  ")
    except:
        return xml_string


def check_backend_status() -> str:
    """Check if backend is running and return status."""
    try:
        response = requests.get(f"{BACKEND_URL}/config", timeout=2)
        if response.status_code == 200:
            cfg = response.json()
            status = "✅ Backend đang hoạt động\n\n"
            status += f"📝 OCR Provider: {cfg.get('ocr_provider', 'N/A')}\n"
            status += f"{'✅' if cfg.get('replicate_configured') else '⚠️'} Replicate API\n"
            status += f"{'✅' if cfg.get('clarifai_configured') else '⚠️'} Clarifai API\n"
            status += f"{'✅' if cfg.get('openai_configured') else '⚠️'} OpenAI API\n"
            status += f"{'✅' if cfg.get('anthropic_configured') else '⚠️'} Anthropic API"
            return status
        else:
            return "❌ Backend không phản hồi"
    except:
        return "❌ Không kết nối được Backend\n\nChạy: python backend.py"


def process_ocr(
    image,
    ocr_provider: str,
    ocr_prompt: str,
    use_mock_llm: bool,
    llm_provider: str
) -> Tuple[str, str, str, str]:
    """
    Process OCR pipeline.
    
    Returns:
        Tuple of (raw_text, raw_xml, paraphrased_xml, status_message)
    """
    if image is None:
        return "", "", "", "❌ Vui lòng upload ảnh trước"
    
    try:
        # Save image to bytes
        import io
        from PIL import Image
        
        # Convert numpy array to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        # Save to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        # Prepare request
        files = {"file": ("image.png", img_byte_arr, "image/png")}
        data = {
            "ocr_provider": ocr_provider,
            "use_mock_llm": str(use_mock_llm).lower(),
            "llm_provider": llm_provider
        }
        
        # Add custom prompt if provided
        if ocr_prompt and ocr_prompt.strip():
            data["ocr_prompt"] = ocr_prompt.strip()
        
        # Send request
        response = requests.post(
            f"{BACKEND_URL}/ocr",
            files=files,
            data=data,
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            raw_text = result.get("raw_text", "")
            raw_xml = format_xml(result.get("raw_xml", ""))
            paraphrased_xml = format_xml(result.get("paraphrased_xml", ""))
            provider_used = result.get("ocr_provider", "N/A")
            status = f"✅ Xử lý thành công! (Provider: {provider_used})"
            return raw_text, raw_xml, paraphrased_xml, status
        else:
            error = response.json().get("detail", "Unknown error")
            return "", "", "", f"❌ Lỗi: {error}"
            
    except requests.exceptions.ConnectionError:
        return "", "", "", "❌ Không kết nối được với backend. Hãy chạy `python backend.py`"
    except Exception as e:
        return "", "", "", f"❌ Lỗi: {str(e)}"


def paraphrase_xml_manual(
    xml_content: str,
    use_mock_llm: bool,
    llm_provider: str
) -> Tuple[str, str]:
    """
    Paraphrase XML content manually.
    
    Returns:
        Tuple of (paraphrased_xml, status_message)
    """
    if not xml_content or not xml_content.strip():
        return "", "❌ Vui lòng nhập XML"
    
    try:
        response = requests.post(
            f"{BACKEND_URL}/paraphrase",
            json={
                "xml_content": xml_content,
                "provider": llm_provider,
                "use_mock": use_mock_llm
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            paraphrased = format_xml(result.get("paraphrased_xml", ""))
            return paraphrased, "✅ Paraphrase thành công!"
        else:
            error = response.json().get("detail", "Unknown error")
            return "", f"❌ Lỗi: {error}"
            
    except Exception as e:
        return "", f"❌ Lỗi: {str(e)}"


def text_to_xml_convert(text: str) -> Tuple[str, str]:
    """
    Convert text to XML.
    
    Returns:
        Tuple of (xml_result, status_message)
    """
    if not text or not text.strip():
        return "", "❌ Vui lòng nhập text"
    
    try:
        response = requests.post(
            f"{BACKEND_URL}/text-to-xml",
            data={"text": text},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            xml_result = format_xml(result.get("xml", ""))
            return xml_result, "✅ Chuyển đổi thành công!"
        else:
            error = response.json().get("detail", "Unknown error")
            return "", f"❌ Lỗi: {error}"
            
    except Exception as e:
        return "", f"❌ Lỗi: {str(e)}"


# Custom CSS
custom_css = """
.main-title {
    text-align: center;
    color: #1E88E5;
    margin-bottom: 0.5rem;
}
.sub-title {
    text-align: center;
    color: #666;
    font-size: 1rem;
    margin-bottom: 1.5rem;
}
.status-box {
    padding: 10px;
    border-radius: 5px;
    background-color: #f5f5f5;
}
"""

# Build Gradio Interface
with gr.Blocks(css=custom_css, title="DeepSeek OCR Pipeline") as demo:
    
    # Header
    gr.Markdown(
        """
        # 📝 DeepSeek OCR Pipeline
        ### Ảnh → DeepSeek-OCR (API) → XML → LLM Paraphrase → XML sạch
        """,
        elem_classes=["main-title"]
    )
    
    with gr.Row():
        # Left Column - Settings & Upload
        with gr.Column(scale=1):
            gr.Markdown("## ⚙️ Cài đặt")
            
            # OCR Settings
            with gr.Group():
                gr.Markdown("### 📝 OCR Settings")
                ocr_provider = gr.Dropdown(
                    choices=["replicate", "clarifai", "mock"],
                    value="replicate",
                    label="OCR Provider",
                    info="replicate (~$0.011/run) | clarifai (free tier) | mock (test)"
                )
                
                ocr_prompt = gr.Textbox(
                    label="Custom Prompt (tùy chọn)",
                    placeholder="Convert the document to markdown.",
                    lines=2
                )
            
            # LLM Settings
            with gr.Group():
                gr.Markdown("### 🤖 LLM Settings")
                use_mock_llm = gr.Checkbox(
                    label="Sử dụng Mock LLM (Demo)",
                    value=True,
                    info="Bật để test mà không cần API key"
                )
                
                llm_provider = gr.Dropdown(
                    choices=["openai", "anthropic"],
                    value="openai",
                    label="LLM Provider",
                    interactive=True
                )
            
            # Backend Status
            with gr.Group():
                gr.Markdown("### 📊 Backend Status")
                status_display = gr.Textbox(
                    label="",
                    value=check_backend_status(),
                    lines=6,
                    interactive=False
                )
                refresh_btn = gr.Button("🔄 Refresh Status", size="sm")
                refresh_btn.click(
                    fn=check_backend_status,
                    outputs=status_display
                )
        
        # Right Column - Main Content
        with gr.Column(scale=2):
            with gr.Tabs():
                # Tab 1: OCR Processing
                with gr.TabItem("🚀 Xử lý OCR"):
                    with gr.Row():
                        with gr.Column():
                            image_input = gr.Image(
                                label="📤 Upload ảnh",
                                type="pil",
                                height=300
                            )
                            process_btn = gr.Button(
                                "🚀 Xử lý OCR",
                                variant="primary",
                                size="lg"
                            )
                            ocr_status = gr.Textbox(
                                label="Trạng thái",
                                interactive=False,
                                lines=1
                            )
                    
                    gr.Markdown("### 📋 Kết quả")
                    
                    with gr.Tabs():
                        with gr.TabItem("📝 Raw Text"):
                            raw_text_output = gr.Textbox(
                                label="Text trích xuất từ OCR",
                                lines=10,
                                interactive=False
                            )
                        
                        with gr.TabItem("📄 Raw XML"):
                            raw_xml_output = gr.Code(
                                label="XML gốc",
                                language="html",
                                lines=15
                            )
                            download_raw_xml = gr.Button("📥 Tải XML gốc", size="sm")
                        
                        with gr.TabItem("✨ Paraphrased XML"):
                            paraphrased_xml_output = gr.Code(
                                label="XML sau khi paraphrase",
                                language="html",
                                lines=15
                            )
                            download_paraphrased_xml = gr.Button("📥 Tải XML đã xử lý", size="sm")
                    
                    # Process button click
                    process_btn.click(
                        fn=process_ocr,
                        inputs=[image_input, ocr_provider, ocr_prompt, use_mock_llm, llm_provider],
                        outputs=[raw_text_output, raw_xml_output, paraphrased_xml_output, ocr_status]
                    )
                
                # Tab 2: Manual Paraphrase
                with gr.TabItem("📝 Paraphrase XML"):
                    gr.Markdown("### Nhập XML để paraphrase")
                    
                    manual_xml_input = gr.Code(
                        label="XML Input",
                        language="html",
                        lines=10,
                        value='<?xml version="1.0" encoding="UTF-8"?>\n<document>\n  <paragraph>Nội dung tiếng Việt...</paragraph>\n</document>'
                    )
                    
                    paraphrase_btn = gr.Button("🤖 Paraphrase", variant="primary")
                    paraphrase_status = gr.Textbox(label="Trạng thái", interactive=False, lines=1)
                    
                    manual_paraphrase_output = gr.Code(
                        label="Kết quả",
                        language="html",
                        lines=10
                    )
                    
                    paraphrase_btn.click(
                        fn=paraphrase_xml_manual,
                        inputs=[manual_xml_input, use_mock_llm, llm_provider],
                        outputs=[manual_paraphrase_output, paraphrase_status]
                    )
                
                # Tab 3: Text to XML
                with gr.TabItem("🔄 Text → XML"):
                    gr.Markdown("### Chuyển text thành XML")
                    
                    text_input = gr.Textbox(
                        label="Text Input",
                        placeholder="Nhập văn bản tiếng Việt để chuyển thành XML...",
                        lines=8
                    )
                    
                    convert_btn = gr.Button("🔄 Chuyển đổi", variant="primary")
                    convert_status = gr.Textbox(label="Trạng thái", interactive=False, lines=1)
                    
                    xml_output = gr.Code(
                        label="Kết quả XML",
                        language="html",
                        lines=10
                    )
                    
                    convert_btn.click(
                        fn=text_to_xml_convert,
                        inputs=[text_input],
                        outputs=[xml_output, convert_status]
                    )
    
    # Footer
    gr.Markdown(
        """
        ---
        <center>
        📝 DeepSeek OCR Pipeline v2.1 (API Only) | Backend: FastAPI | Frontend: Gradio
        
        Pipeline: Image → DeepSeek-OCR API → XML Structure → LLM Paraphrase → Clean XML
        
        Supports: Replicate API (~$0.011/run) | Clarifai API (free tier)
        </center>
        """,
        elem_classes=["sub-title"]
    )


# Launch
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True
    )