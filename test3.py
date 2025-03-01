import cv2
import pytesseract
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import openai

# 🔹 Cấu hình OpenAI API (>1.0)
client = openai.OpenAI(api_key="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")

# 🔹 Cấu hình đường dẫn Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 🔹 Tiền xử lý ảnh để tăng độ chính xác OCR
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    
    # Chuyển ảnh sang grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Adaptive Thresholding (loại bỏ nền, làm rõ chữ)
    processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 31, 10)
    
    # Morphological Operations để làm sạch nhiễu
    kernel = np.ones((1,1), np.uint8)
    processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
    
    return processed

# 🔹 Nhận diện văn bản & vị trí bounding box từ ảnh
def extract_text_with_boxes(image_path):
    image = preprocess_image(image_path)
    
    # Lấy dữ liệu văn bản & bounding box từ pytesseract
    data = pytesseract.image_to_data(image, lang="vie+eng", output_type=pytesseract.Output.DICT)

    text_boxes = []
    for i in range(len(data["text"])):
        if int(data["conf"][i]) > 50:  # Chỉ lấy văn bản có độ tin cậy cao
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            text_boxes.append({"text": data["text"][i], "bbox": (x, y, x + w, y + h)})

    return text_boxes

# 🔹 Dịch văn bản bằng OpenAI GPT-4o Mini
def translate_text(text, target_language="Vietnamese"):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that translates text. "
                    "Always return only the translated text without any additional explanations."
                )
            },
            {
                "role": "user",
                "content": f"Translate this text from Auto to {target_language}: {text}"
            }
        ],
        max_tokens=100
    )
    return response.choices[0].message.content.strip()

# 🔹 Tính toán font size động để chữ dịch vừa khít bounding box
def calculate_fontsize(text, bbox_width, bbox_height, font_path="arial.ttf"):
    fontsize = bbox_height  # Bắt đầu với fontsize bằng chiều cao bbox
    font = ImageFont.truetype(font_path, fontsize)

    # Giảm fontsize nếu chữ quá rộng
    while font.getbbox(text)[2] > bbox_width - 4 and fontsize > 1:
        fontsize -= 1
        font = ImageFont.truetype(font_path, fontsize)

    # Tăng fontsize nếu chữ còn nhỏ hơn đáng kể so với bounding box
    while font.getbbox(text)[2] < bbox_width * 0.9 and fontsize < bbox_height:
        fontsize += 1
        font = ImageFont.truetype(font_path, fontsize)

    return font

# 🔹 Đè văn bản dịch lên ảnh gốc
def overlay_translated_text(image_path, output_path, target_language="Vietnamese"):
    text_boxes = extract_text_with_boxes(image_path)
    
    # Đọc ảnh bằng OpenCV và chuyển sang Pillow để xử lý text overlay
    img = cv2.imread(image_path)
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font_path = "arial.ttf"

    for box in text_boxes:
        original_text = box["text"]
        x_min, y_min, x_max, y_max = box["bbox"]

        translated_text = translate_text(original_text, target_language)
        print(f"Original: {original_text} -> Translated: {translated_text}")

        bbox_width = x_max - x_min
        bbox_height = y_max - y_min

        # Lấy font size phù hợp
        font = calculate_fontsize(translated_text, bbox_width, bbox_height, font_path)

        # Tính toán vị trí căn giữa chữ
        text_width, text_height = font.getbbox(translated_text)[2:4]
        x_text = x_min + (bbox_width - text_width) // 2
        y_text = y_min + (bbox_height - text_height) // 2 - 1

        # Vẽ nền trắng trước khi vẽ chữ mới
        draw.rectangle([x_min, y_min, x_max, y_max], fill="white")

        # Vẽ văn bản dịch
        draw.text((x_text, y_text), translated_text, font=font, fill="black")

    # Lưu ảnh cuối cùng
    final_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, final_img)
    print(f"Ảnh đã được lưu tại: {output_path}")

# 🔹 Chạy thử nghiệm
if __name__ == "__main__":
    input_image = "C:/Users/acer/Pictures/Screenshots/Screenshot 2025-01-26 105248.png"
    output_image = "translated_overlay.png"

    overlay_translated_text(input_image, output_image, "Vietnamese")
