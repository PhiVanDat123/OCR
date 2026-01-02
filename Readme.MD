# DeepSeek OCR Pipeline - Xử lý văn bản tiếng Việt

Pipeline xử lý OCR hoàn chỉnh sử dụng **DeepSeek-OCR**: **Ảnh → DeepSeek-OCR → XML → LLM Paraphrase → XML sạch**

## 🚀 Tính năng mới với DeepSeek-OCR

- **Layout-aware OCR**: Nhận diện cấu trúc văn bản, bảng, tiêu đề
- **Markdown output**: Xuất trực tiếp sang Markdown có cấu trúc
- **Đa ngôn ngữ**: Hỗ trợ tốt tiếng Việt và các ngôn ngữ khác
- **3 chế độ chạy**: Local GPU, vLLM Server, Clarifai API

## 🏗️ Kiến trúc

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Upload    │────▶│  DeepSeek   │────▶│   Text to   │────▶│     LLM     │
│   Image     │     │    OCR      │     │     XML     │     │  Paraphrase │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
         [Local GPU]  [vLLM Server] [Clarifai API]
```

## 📁 Cấu trúc dự án

```
ocr_pipeline/
├── backend.py          # FastAPI backend
├── frontend.py         # Streamlit frontend
├── ocr_module.py       # DeepSeek OCR module
├── llm_module.py       # LLM paraphrase module
├── config.py           # Cấu hình
├── requirements.txt    # Dependencies
├── run.sh              # Script khởi chạy
└── README.md           # Hướng dẫn
```

## 🚀 Cài đặt & Chạy

### 1. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 2. Cài đặt DeepSeek-OCR

**Cách 1: Chạy Local (cần GPU NVIDIA với CUDA 11.8+)**

```bash
# Cài đặt PyTorch với CUDA
pip install torch==2.6.0 torchvision --index-url https://download.pytorch.org/whl/cu118

# Cài flash-attention (tùy chọn, tăng tốc)
pip install flash-attn==2.7.3 --no-build-isolation

# Model sẽ tự động tải khi chạy lần đầu (~10GB)
```

**Cách 2: Dùng vLLM Server**

```bash
# Cài vLLM
pip install vllm

# Chạy server
vllm serve deepseek-ai/DeepSeek-OCR --port 8001

# Set environment
export DEEPSEEK_OCR_MODE=vllm
export DEEPSEEK_VLLM_URL=http://localhost:8001/v1
```

**Cách 3: Dùng Clarifai API (không cần GPU)**

```bash
# Đăng ký tại https://clarifai.com và lấy PAT
export CLARIFAI_PAT=your-personal-access-token
export DEEPSEEK_OCR_MODE=api
```

### 3. Cấu hình

```bash
# OCR Settings
export OCR_PROVIDER=deepseek        # deepseek, tesseract, mock
export DEEPSEEK_OCR_MODE=local      # local, vllm, api

# LLM Settings (cho paraphrase)
export OPENAI_API_KEY=your-key      # hoặc
export ANTHROPIC_API_KEY=your-key
export LLM_PROVIDER=openai          # openai, anthropic
```

### 4. Khởi chạy

```bash
# Chạy script tự động
chmod +x run.sh
./run.sh

# Hoặc chạy thủ công
# Terminal 1:
python backend.py

# Terminal 2:
streamlit run frontend.py
```

### 5. Truy cập

- **Frontend UI:** http://localhost:8501
- **Backend API:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs

## 📖 API Endpoints

### `POST /ocr`

Xử lý OCR với DeepSeek-OCR.

**Request:**
- `file`: Ảnh (multipart/form-data)
- `ocr_provider`: "deepseek" | "tesseract" | "mock" (default: deepseek)
- `ocr_mode`: "local" | "vllm" | "api" (for deepseek)
- `ocr_prompt`: Custom prompt (optional)
- `use_mock_llm`: bool (default: true)
- `llm_provider`: "openai" | "anthropic"

**Response:**
```json
{
  "raw_text": "Text trích xuất từ ảnh",
  "raw_xml": "<?xml ...>",
  "paraphrased_xml": "<?xml ...>",
  "ocr_provider": "deepseek-local",
  "success": true,
  "message": "..."
}
```

## 🔧 DeepSeek OCR Prompts

```python
# Chuyển document sang Markdown (mặc định)
"<image>\n<|grounding|>Convert the document to markdown."

# OCR đơn giản
"<image>\nFree OCR."

# Parse bảng
"<image>\n<|grounding|>Extract tables as markdown."

# Parse hình ảnh/biểu đồ
"<image>\nParse the figure."
```

## 💻 Yêu cầu hệ thống

| Mode | GPU | RAM | VRAM |
|------|-----|-----|------|
| Local | NVIDIA CUDA 11.8+ | 32GB+ | 12GB+ |
| vLLM | NVIDIA CUDA 11.8+ | 32GB+ | 24GB+ |
| Clarifai API | Không cần | 8GB | - |

## 🐛 Troubleshooting

**Lỗi "CUDA out of memory":**
- Giảm kích thước ảnh trước khi upload
- Dùng mode `api` (Clarifai) thay vì `local`

**Lỗi "Model loading failed":**
- Kiểm tra kết nối internet (model ~10GB)
- Đảm bảo đủ dung lượng ổ cứng

**Lỗi Clarifai API:**
- Kiểm tra CLARIFAI_PAT đã set đúng
- Kiểm tra quota API còn không

## 📄 License

MIT License