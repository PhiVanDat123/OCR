import replicate
import os

token = os.getenv("REPLICATE_API_TOKEN")
print("Token (first 10 chars):", token[:10] if token else "NOT SET")

# Thử lấy thông tin model
try:
    model = replicate.models.get("lucataco/deepseek-ocr")
    print("\n✅ Model found!")
    print("Name:", model.name)
    print("Owner:", model.owner)
    
    # Lấy versions
    versions = list(model.versions.list())
    if versions:
        latest = versions[0]
        print("\nLatest version ID:", latest.id)
        
        # Thử chạy với version cụ thể
        print("\nTrying to run with latest version...")
        output = replicate.run(
            f"lucataco/deepseek-ocr:{latest.id}",
            input={
                "image": "https://static.simonwillison.net/static/2025/ft.jpeg",
                "prompt": "Free OCR."
            }
        )
        print("SUCCESS:", output)
    else:
        print("No versions available")
        
except Exception as e:
    print("\n❌ ERROR:", type(e).__name__)
    print("Message:", e)