#!/usr/bin/env python3
import os
import sys
import base64
import json
import cv2
import openai

# Cấu hình client OpenAI từ biến môi trường
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if not client.api_key:
    print("Vui lòng thiết lập biến môi trường OPENAI_API_KEY.")
    sys.exit(1)

def adjust_image_size(image, min_width=100, min_height=100):
    """
    Điều chỉnh kích thước ảnh sao cho ảnh có ít nhất min_width x min_height.
    Nếu ảnh nhỏ hơn, sẽ phóng to theo tỉ lệ.
    """
    h, w = image.shape[:2]
    new_w = max(w, min_width)
    new_h = max(h, min_height)
    if new_w != w or new_h != h:
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return resized
    return image

def encode_cv2_image(image) -> str:
    """
    Mã hóa ảnh (numpy array) sang chuỗi base64 sau khi chuyển đổi thành JPEG.
    """
    try:
        success, buffer = cv2.imencode(".jpg", image)
        if not success:
            raise RuntimeError("Không thể mã hóa ảnh sang định dạng JPEG.")
        return base64.b64encode(buffer).decode("utf-8")
    except Exception as e:
        raise RuntimeError(f"Error encoding cv2 image: {e}")

def extract_text_from_image(image_base64: str) -> str:
    """
    Trích xuất văn bản và vị trí (bounding box) của từng khối văn bản từ ảnh bằng GPT‑4o‑mini.
    
    Prompt yêu cầu:
      - Trích xuất tất cả văn bản từ ảnh kèm theo vị trí chính xác của từng khối văn bản.
      - Trả về kết quả dưới dạng JSON array, mỗi phần tử chứa 'text' và 'bounding_box'
        (định dạng: [x, y, width, height]).
      - Không cung cấp thêm bất kỳ bình luận hay giải thích nào.
      - Ảnh được cung cấp dưới dạng data URL, đã được điều chỉnh sao cho rõ ràng (ít nhất 100x100 pixels).
    """
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Extract all the text from the image along with the most precise and accurate location of each text block. "
                            "For each text block, return the text and its bounding box coordinates (x, y, width, height) "
                            "in a JSON array. Do not provide any extra commentary or explanation. "
                            "The image provided is a clear, high-resolution image (at least 100x100 pixels)."
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                    }
                ]
            }
        ]
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"Error during text extraction: {e}")

def translate_text(text: str) -> str:
    """
    Dịch văn bản sử dụng GPT‑4o‑mini.
    
    Prompt yêu cầu: Dịch văn bản được cung cấp sang tiếng Anh, trả về kết quả chỉ chứa văn bản dịch.
    """
    try:
        messages = [
            {
                "role": "user",
                "content": (
                    f"Translate the following text into English: \"{text}\". "
                    "Return only the translated text without any extra commentary."
                )
            }
        ]
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=100
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"Error during text translation: {e}")

def overlay_translations(image, text_blocks):
    """
    Với từng khối văn bản (có 'text' và 'bounding_box'), dịch văn bản đó và overlay
    văn bản dịch lên vùng ảnh tương ứng.
    
    - Che đi vùng văn bản gốc bằng hình chữ nhật màu trắng.
    - Overlay văn bản dịch lên chính vị trí đó.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    text_color = (0, 0, 0)  # màu đen

    for block in text_blocks:
        orig_text = block.get("text", "")
        bbox = block.get("bounding_box", {})
        if not isinstance(bbox, dict) or not all(k in bbox for k in ["x", "y", "width", "height"]):
            continue
        x = bbox["x"]
        y = bbox["y"]
        w = bbox["width"]
        h = bbox["height"]
        
        # Dịch văn bản gốc
        translated = translate_text(orig_text)
        
        # Che khu vực văn bản gốc (vẽ hình chữ nhật màu trắng)
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), -1)
        
        # Tính toán vị trí để overlay văn bản (căn giữa vùng bounding box)
        text_size, _ = cv2.getTextSize(translated, font, font_scale, thickness)
        text_width, text_height = text_size
        text_x = x + (w - text_width) // 2
        text_y = y + (h + text_height) // 2
        cv2.putText(image, translated, (text_x, text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)
    return image

def clean_extraction_result(result: str) -> str:
    """
    Loại bỏ các ký tự markdown (```json và ``` ở đầu, cuối) nếu có.
    """
    cleaned = result.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json"):].strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()
    return cleaned

def main():
    if len(sys.argv) < 2:
        print("Usage: python img_to_text.py <image_file>")
        sys.exit(1)
    image_file = sys.argv[1]
    try:
        # Đọc ảnh gốc
        image = cv2.imread(image_file)
        if image is None:
            raise RuntimeError("Cannot read the image from the provided path.")
        
        # Điều chỉnh kích thước ảnh nếu cần (ít nhất 100x100 pixels)
        image = adjust_image_size(image, min_width=100, min_height=100)
        
        # Mã hóa ảnh đã điều chỉnh sang chuỗi base64
        image_base64 = encode_cv2_image(image)
        
        # Trích xuất văn bản và bounding box từ ảnh (kết quả là chuỗi JSON)
        extraction_result = extract_text_from_image(image_base64)
        print("Extraction result:")
        print(extraction_result)
        
        cleaned_result = clean_extraction_result(extraction_result)
        
        try:
            text_blocks = json.loads(cleaned_result)
            if not isinstance(text_blocks, list):
                raise ValueError("The JSON result is not an array.")
        except Exception as e:
            raise RuntimeError(f"Error parsing extraction result: {e}")
        
        # Overlay văn bản dịch lên ảnh dựa trên bounding box
        output_image = overlay_translations(image, text_blocks)
        output_path = "translated_output.jpg"
        cv2.imwrite(output_path, output_image)
        print(f"Output image saved as: {output_path}")
    except RuntimeError as e:
        print(e)
        sys.exit(1)

if __name__ == "__main__":
    main()
