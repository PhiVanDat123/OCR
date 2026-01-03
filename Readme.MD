# DeepSeek OCR Pipeline - API Only Mode

Pipeline xử lý OCR hoàn chỉnh sử dụng **DeepSeek-OCR qua API**: **Ảnh → DeepSeek-OCR (API) → XML → LLM Paraphrase → XML sạch**

## 🚀 Tính năng

- **Không cần GPU**: Sử dụng API thay vì chạy model local
- **Chi phí thấp**: Replicate ~$0.011/lần chạy (~90 lần/$1)
- **Layout-aware OCR**: Nhận diện cấu trúc văn bản, bảng, tiêu đề
- **Markdown output**: Xuất trực tiếp sang Markdown có cấu trúc
- **Đa ngôn ngữ**: Hỗ trợ tốt tiếng Việt và ~100 ngôn ngữ khác

## 🏗️ Kiến trúc

```
┌─────────────┐     ┌─────────────────┐     ┌─────────────┐     ┌─────────────┐
│   Upload    │────▶│  DeepSeek-OCR   │────▶│   Text to   │────▶│     LLM     │
│   Image     │     │   (API)         │     │     XML     │     │  Paraphrase │
└─────────────┘     └─────────────────┘     └─────────────┘     └─────────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
         [Replicate]              [Clarifai]
         ~$0.011/run              Free tier
```

## 📁 Cấu trúc dự án

```
ocr_pipeline/
├── backend.py          # FastAPI backend
├── frontend.py         # Streamlit frontend
├── ocr_module.py       # DeepSeek OCR module (API only)
├── llm_module.py       # LLM paraphrase module
├── config.py           # Cấu hình
├── requirements.txt    # Dependencies
└── README.md           # Hướng dẫn
```

## 🔑 Lấy API Keys

### Option 1: Replicate API (Khuyên dùng - ~$0.011/lần chạy)

1. Đăng ký tại [replicate.com](https://replicate.com)
2. Vào [Account API Tokens](https://replicate.com/account/api-tokens)
3. Tạo token mới
4. Set environment variable:
```bash
export REPLICATE_API_TOKEN=r8_xxxxxxxxxxxxxxxxxxxx
```

**Pricing**: ~$0.011/lần chạy = ~90 lần OCR với $1

### Option 2: Clarifai API (Có Free Tier)

1. Đăng ký tại [clarifai.com](https://clarifai.com)
2. Vào [Security Settings](https://clarifai.com/settings/security)
3. Tạo Personal Access Token (PAT)
4. Set environment variable:
```bash
export CLARIFAI_PAT=your-personal-access-token
```

**Pricing**: Free tier có giới hạn, sau đó tính phí

## 🚀 Cài đặt & Chạy

### 1. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 2. Cấu hình API Keys

```bash
# OCR API (chọn 1 trong 2)
export REPLICATE_API_TOKEN=your-replicate-token     # Khuyên dùng
# hoặc
export CLARIFAI_PAT=your-clarifai-pat

# OCR Provider mặc định
export OCR_PROVIDER=replicate    # replicate, clarifai, hoặc mock

# LLM Settings (cho paraphrase - tùy chọn)
export OPENAI_API_KEY=your-openai-key
# hoặc
export ANTHROPIC_API_KEY=your-anthropic-key
export LLM_PROVIDER=openai      # openai hoặc anthropic
```

### 3. Khởi chạy

```bash
# Terminal 1: Backend
python backend.py

# Terminal 2: Frontend
streamlit run frontend.py
```

### 4. Truy cập

- **Frontend UI:** http://localhost:8501
- **Backend API:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs

## 📖 API Endpoints

### `POST /ocr`

Xử lý OCR với DeepSeek-OCR qua API.

**Request:**
- `file`: Ảnh (multipart/form-data)
- `ocr_provider`: "replicate" | "clarifai" | "mock" (default: replicate)
- `ocr_prompt`: Custom prompt (optional)
- `use_mock_llm`: bool (default: true)
- `llm_provider`: "openai" | "anthropic"

**Response:**
```json
{
  "raw_text": "Text trích xuất từ ảnh",
  "raw_xml": "<?xml ...>",
  "paraphrased_xml": "<?xml ...>",
  "ocr_provider": "replicate",
  "success": true,
  "message": "..."
}
```

### `GET /config`

Kiểm tra cấu hình hiện tại.

**Response:**
```json
{
  "ocr_provider": "replicate",
  "replicate_configured": true,
  "clarifai_configured": false,
  "openai_configured": true,
  "anthropic_configured": false
}
```

## 🔧 OCR Prompts

DeepSeek-OCR hỗ trợ nhiều loại prompt khác nhau:

```python
# Chuyển document sang Markdown (mặc định)
"Convert the document to markdown."

# OCR đơn giản
"Free OCR."

# Parse bảng
"Extract tables as markdown."

# Parse hình ảnh/biểu đồ
"Parse the figure."

# Trích xuất text
"Extract the text in the image."
```

## 💰 So sánh chi phí

| Provider | Giá | Ghi chú |
|----------|-----|---------|
| Replicate | ~$0.011/run | Khuyên dùng, ổn định |
| Clarifai | Free tier | Giới hạn requests |
| Local GPU | $0 (có GPU) | Cần NVIDIA GPU 12GB+ VRAM |

## 💻 Yêu cầu hệ thống

- Python 3.8+
- Internet connection (để gọi API)
- **Không cần GPU**

## 🐛 Troubleshooting

**Lỗi "REPLICATE_API_TOKEN not configured":**
```bash
export REPLICATE_API_TOKEN=your-token
```

**Lỗi "CLARIFAI_PAT not configured":**
```bash
export CLARIFAI_PAT=your-token
```

**Lỗi timeout:**
- Kiểm tra kết nối internet
- Thử lại với ảnh nhỏ hơn

**Test với mock mode:**
```bash
export OCR_PROVIDER=mock
```

## 📄 License

MIT License