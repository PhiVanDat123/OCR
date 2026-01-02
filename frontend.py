"""Streamlit Frontend for OCR Pipeline with DeepSeek-OCR."""
import streamlit as st
import requests
from io import BytesIO
import xml.dom.minidom

# Configuration
BACKEND_URL = "http://localhost:8000"

# Page config
st.set_page_config(
    page_title="DeepSeek OCR Pipeline - Tiếng Việt",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .provider-badge {
        background-color: #e3f2fd;
        border-radius: 5px;
        padding: 0.3rem 0.6rem;
        font-size: 0.8rem;
        color: #1565c0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🔍 DeepSeek OCR Pipeline</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Ảnh → DeepSeek-OCR → XML → LLM Paraphrase → XML sạch</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Cài đặt OCR")
    
    ocr_provider = st.selectbox(
        "OCR Provider",
        options=["deepseek", "tesseract", "mock"],
        index=0,
        help="DeepSeek-OCR cho kết quả tốt nhất với văn bản phức tạp"
    )
    
    ocr_mode = None
    if ocr_provider == "deepseek":
        ocr_mode = st.selectbox(
            "DeepSeek Mode",
            options=["local", "vllm", "api"],
            index=0,
            help="""
            - local: Chạy model trên máy (cần GPU NVIDIA)
            - vllm: Dùng vLLM server
            - api: Dùng Clarifai API
            """
        )
        
        ocr_prompt = st.text_area(
            "Custom Prompt (tùy chọn)",
            placeholder="<image>\\n<|grounding|>Convert the document to markdown.",
            height=80,
            help="Để trống để dùng prompt mặc định"
        )
    else:
        ocr_prompt = None
    
    st.divider()
    
    st.header("🤖 Cài đặt LLM")
    
    use_mock_llm = st.checkbox("Sử dụng Mock LLM (Demo)", value=True, 
                               help="Bật để test mà không cần API key")
    
    llm_provider = st.selectbox(
        "LLM Provider",
        options=["openai", "anthropic"],
        disabled=use_mock_llm
    )
    
    st.divider()
    
    st.header("📊 Pipeline Flow")
    st.markdown("""
    1. **Upload ảnh** 📤
    2. **DeepSeek-OCR** 🔍
    3. **Chuyển thành XML** 📝
    4. **LLM paraphrase** 🤖
    5. **Hiển thị kết quả** ✅
    """)
    
    st.divider()
    
    # Check backend status
    try:
        response = requests.get(f"{BACKEND_URL}/config", timeout=2)
        if response.status_code == 200:
            st.success("✅ Backend đang hoạt động")
            cfg = response.json()
            st.caption(f"OCR: {cfg.get('ocr_provider', 'N/A')}")
            st.caption(f"Mode: {cfg.get('ocr_mode', 'N/A')}")
        else:
            st.error("❌ Backend không phản hồi")
    except:
        st.error("❌ Không kết nối được Backend")
        st.info("Chạy: `python backend.py`")


def format_xml(xml_string: str) -> str:
    """Format XML with proper indentation."""
    try:
        dom = xml.dom.minidom.parseString(xml_string)
        return dom.toprettyxml(indent="  ")
    except:
        return xml_string


def render_xml_as_html(xml_string: str) -> str:
    """Convert XML to readable HTML display."""
    import re
    # Simple XML to display conversion
    formatted = xml_string.replace("<", "&lt;").replace(">", "&gt;")
    formatted = re.sub(r'&lt;(\/?[\w-]+)', r'<span style="color:#1E88E5">&lt;\1</span>', formatted)
    formatted = re.sub(r'(\w+)=', r'<span style="color:#E91E63">\1</span>=', formatted)
    return f'<pre style="background:#f5f5f5; padding:1rem; border-radius:5px; overflow-x:auto;">{formatted}</pre>'


# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Upload ảnh")
    
    uploaded_file = st.file_uploader(
        "Chọn ảnh chứa văn bản tiếng Việt",
        type=["png", "jpg", "jpeg", "gif", "bmp", "tiff"],
        help="Hỗ trợ: PNG, JPG, JPEG, GIF, BMP, TIFF"
    )
    
    if uploaded_file:
        st.image(uploaded_file, caption="Ảnh đã upload", use_container_width=True)
        
        if st.button("🚀 Xử lý OCR", type="primary", use_container_width=True):
            with st.spinner("Đang xử lý với DeepSeek-OCR..."):
                try:
                    # Prepare request
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    data = {
                        "ocr_provider": ocr_provider,
                        "use_mock_llm": str(use_mock_llm).lower(),
                        "llm_provider": llm_provider
                    }
                    
                    # Add DeepSeek-specific options
                    if ocr_mode:
                        data["ocr_mode"] = ocr_mode
                    if ocr_prompt:
                        data["ocr_prompt"] = ocr_prompt
                    
                    # Send request
                    response = requests.post(
                        f"{BACKEND_URL}/ocr",
                        files=files,
                        data=data,
                        timeout=120  # Longer timeout for DeepSeek
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.session_state["ocr_result"] = result
                        st.success(f"✅ Xử lý thành công! (Provider: {result.get('ocr_provider', 'N/A')})")
                    else:
                        st.error(f"❌ Lỗi: {response.json().get('detail', 'Unknown error')}")
                        
                except requests.exceptions.ConnectionError:
                    st.error("❌ Không kết nối được với backend. Hãy chạy `python backend.py`")
                except Exception as e:
                    st.error(f"❌ Lỗi: {str(e)}")

with col2:
    st.header("📋 Kết quả")
    
    if "ocr_result" in st.session_state:
        result = st.session_state["ocr_result"]
        
        # Show OCR provider used
        st.info(f"🔍 OCR Provider: **{result.get('ocr_provider', 'N/A')}**")
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📝 Raw Text", "📄 Raw XML", "✨ Paraphrased XML", "🔄 So sánh"])
        
        with tab1:
            st.subheader("Text trích xuất từ OCR")
            st.text_area("", result["raw_text"], height=300, disabled=True)
        
        with tab2:
            st.subheader("XML gốc")
            st.code(format_xml(result["raw_xml"]), language="xml")
            st.download_button(
                "📥 Tải XML gốc",
                result["raw_xml"],
                file_name="raw_output.xml",
                mime="application/xml"
            )
        
        with tab3:
            st.subheader("XML sau khi paraphrase")
            st.code(format_xml(result["paraphrased_xml"]), language="xml")
            st.download_button(
                "📥 Tải XML đã xử lý",
                result["paraphrased_xml"],
                file_name="paraphrased_output.xml",
                mime="application/xml"
            )
        
        with tab4:
            st.subheader("So sánh trước/sau")
            compare_col1, compare_col2 = st.columns(2)
            
            with compare_col1:
                st.markdown("**XML gốc:**")
                st.code(format_xml(result["raw_xml"]), language="xml")
            
            with compare_col2:
                st.markdown("**XML sau paraphrase:**")
                st.code(format_xml(result["paraphrased_xml"]), language="xml")
    else:
        st.info("👆 Upload ảnh và nhấn 'Xử lý OCR' để xem kết quả")


# Additional section: Manual XML input
st.divider()
st.header("🔧 Công cụ bổ sung")

tool_tab1, tool_tab2 = st.tabs(["📝 Paraphrase XML thủ công", "📄 Text → XML"])

with tool_tab1:
    st.subheader("Nhập XML để paraphrase")
    manual_xml = st.text_area(
        "XML Input",
        placeholder='<?xml version="1.0" encoding="UTF-8"?>\n<document>\n  <paragraph>Nội dung tiếng Việt...</paragraph>\n</document>',
        height=200
    )
    
    if st.button("🤖 Paraphrase", disabled=not manual_xml):
        with st.spinner("Đang xử lý..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/paraphrase",
                    json={
                        "xml_content": manual_xml,
                        "provider": llm_provider,
                        "use_mock": use_mock_llm
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    paraphrased = response.json()["paraphrased_xml"]
                    st.subheader("Kết quả:")
                    st.code(format_xml(paraphrased), language="xml")
                else:
                    st.error(f"Lỗi: {response.json().get('detail', 'Unknown')}")
            except Exception as e:
                st.error(f"Lỗi: {str(e)}")

with tool_tab2:
    st.subheader("Chuyển text thành XML")
    manual_text = st.text_area(
        "Text Input",
        placeholder="Nhập văn bản tiếng Việt để chuyển thành XML...",
        height=200
    )
    
    if st.button("📄 Chuyển đổi", disabled=not manual_text):
        with st.spinner("Đang chuyển đổi..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/text-to-xml",
                    data={"text": manual_text},
                    timeout=10
                )
                
                if response.status_code == 200:
                    xml_result = response.json()["xml"]
                    st.subheader("Kết quả XML:")
                    st.code(format_xml(xml_result), language="xml")
                else:
                    st.error(f"Lỗi: {response.json().get('detail', 'Unknown')}")
            except Exception as e:
                st.error(f"Lỗi: {str(e)}")


# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.9rem;">
    <p>🔍 DeepSeek OCR Pipeline v2.0 | Backend: FastAPI | Frontend: Streamlit</p>
    <p>Pipeline: Image → DeepSeek-OCR → XML Structure → LLM Paraphrase → Clean XML</p>
    <p>Supports: Local GPU, vLLM Server, Clarifai API</p>
</div>
""", unsafe_allow_html=True)