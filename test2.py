import cv2
import pytesseract
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import openai
import torch
import concurrent.futures
from segment_anything import SamPredictor, sam_model_registry

# 🔹 Cấu hình OpenAI API (>1.0)
client = openai.OpenAI(api_key="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")

# 🔹 Cấu hình đường dẫn Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 🔹 Load mô hình SAM để tách nền văn bản
sam_checkpoint = "sam_vit_h_4b8939.pth"
sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
predictor = SamPredictor(sam)
sam.to("cuda" if torch.cuda.is_available() else "cpu")

# 🔹 Tiền xử lý ảnh để tăng độ chính xác OCR
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 31, 10)
    kernel = np.ones((1,1), np.uint8)
    processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
    return processed

# 🔹 Tách chữ bằng mô hình SAM
def segment_text(image_path):
    image = cv2.imread(image_path)
    predictor.set_image(image)
    masks, _, _ = predictor.predict()
    return masks

# 🔹 Nhận diện văn bản & vị trí bounding box từ ảnh
def extract_text_with_boxes(image_path):
    image = preprocess_image(image_path)
    data = pytesseract.image_to_data(image, lang="vie+eng", output_type=pytesseract.Output.DICT)

    text_boxes = []
    for i in range(len(data["text"])):
        if int(data["conf"][i]) > 60:  # Chỉ lấy văn bản có độ tin cậy cao
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            text_boxes.append({"text": data["text"][i], "bbox": (x, y, x + w, y + h)})

    return text_boxes

# 🔹 Dịch văn bản bằng OpenAI GPT-4o Mini (Chạy đa luồng)
def translate_text(text, target_language="Vietnamese"):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
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

# 🔹 Dịch toàn bộ văn bản bằng Đa Luồng (Tăng số luồng)
def translate_text_bulk(text_list, target_language="Vietnamese"):
    translated_texts = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:  # Tăng số luồng để dịch nhanh hơn
        future_to_text = {executor.submit(translate_text, text, target_language): text for text in text_list}
        for future in concurrent.futures.as_completed(future_to_text):
            try:
                translated_texts.append(future.result())
            except Exception as e:
                translated_texts.append("")  # Nếu lỗi, trả về chuỗi rỗng

    return translated_texts

# 🔹 Mở rộng kích thước ảnh nếu văn bản dịch dài hơn nhiều
def expand_image_if_needed(img, text_boxes, translated_texts, font_path="arial.ttf"):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(font_path, 30)  # Giữ nguyên kích thước chữ

    max_width = img_pil.width
    expand_needed = False

    for box, translated_text in zip(text_boxes, translated_texts):
        x_min, y_min, x_max, y_max = box["bbox"]
        text_width, _ = draw.textbbox((0, 0), translated_text, font=font)[2:4]

        if text_width > (x_max - x_min):
            expand_needed = True
            max_width = max(max_width, x_min + text_width + 10)

    if expand_needed:
        new_img = Image.new("RGB", (max_width, img_pil.height), "white")
        new_img.paste(img_pil, (0, 0))
        img = cv2.cvtColor(np.array(new_img), cv2.COLOR_RGB2BGR)
        return img
    return img

# 🔹 Đè văn bản dịch lên ảnh gốc
# 🔹 Đè văn bản dịch lên ảnh gốc
# 🔹 Tính toán font size tự động để tránh chữ đè lên nhau
def get_optimal_fontsize(text, bbox_width, bbox_height, font_path="arial.ttf"):
    """ Điều chỉnh kích thước font để vừa với bounding box """
    font_size = bbox_height  # Bắt đầu bằng kích thước bbox
    font = ImageFont.truetype(font_path, font_size)

    while font.getbbox(text)[2] > bbox_width and font_size > 10:
        font_size -= 1
        font = ImageFont.truetype(font_path, font_size)

    return font


# 🔹 Đè văn bản dịch lên ảnh gốc
def overlay_translated_text(image_path, output_path, target_language="Vietnamese"):
    text_boxes = extract_text_with_boxes(image_path)
    
    # Lấy toàn bộ văn bản cần dịch
    original_texts = [box["text"] for box in text_boxes]
    
    # Sử dụng đa luồng để dịch nhanh hơn
    translated_texts = translate_text_bulk(original_texts, target_language)
    
    # Đọc ảnh bằng OpenCV và chuyển sang Pillow để xử lý text overlay
    img = cv2.imread(image_path)

    # Kiểm tra xem có cần mở rộng ảnh không
    img = expand_image_if_needed(img, text_boxes, translated_texts)

    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font_path = "arial.ttf"

    # Sử dụng mô hình SAM để tách nền chữ cũ
    masks = segment_text(image_path)
    for mask in masks:
        resized_mask = cv2.resize(mask.astype(np.uint8) * 255, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        img[resized_mask > 0] = 255  # Làm trắng nền tại vị trí có chữ

    # Đè văn bản dịch lên ảnh với font size tự động điều chỉnh
    for box, translated_text in zip(text_boxes, translated_texts):
        x_min, y_min, x_max, y_max = box["bbox"]
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min

        print(f"Original: {box['text']} -> Translated: {translated_text}")

        # Lấy font size phù hợp
        font = get_optimal_fontsize(translated_text, bbox_width, bbox_height, font_path)

        # Vẽ nền trắng trước khi vẽ chữ mới
        draw.rectangle([x_min, y_min, x_max, y_max], fill="white")

        # Vẽ văn bản dịch
        draw.text((x_min, y_min), translated_text, font=font, fill="black")

    # Lưu ảnh cuối cùng
    final_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, final_img)
    print(f"✅ Ảnh đã được lưu tại: {output_path}")

# 🔹 Chạy thử nghiệm
if __name__ == "__main__":
    input_image = "C:/Users/acer/Pictures/Screenshots/Screenshot 2025-01-26 105248.png"
    output_image = "translated_overlay_sam.png"

    overlay_translated_text(input_image, output_image, "Vietnamese")
