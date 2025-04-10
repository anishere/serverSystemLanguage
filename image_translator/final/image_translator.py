#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Chương trình dịch văn bản từ ảnh sử dụng OCR và OpenAI GPT-4o-mini

Quy trình:
1. Đọc ảnh và tự động nhận dạng ngôn ngữ nguồn
2. Phát hiện vùng bounding box chứa văn bản (chỉ sử dụng EasyOCR)
3. Crop vùng bounding box
4. Tiền xử lý ảnh đã cắt (lọc nhiễu, tăng độ tương phản, làm sắc nét)
5. Sử dụng GPT để trích xuất văn bản từ ảnh đã xử lý
6. Dịch văn bản trích xuất
7. Vẽ văn bản đã dịch vào vùng crop (căn giữa)
8. Chèn vùng đã xử lý lại vào ảnh gốc

Yêu cầu:
- opencv-python
- numpy
- requests
- Pillow
- python-dotenv
- matplotlib
- easyocr
"""

import os
import argparse
import time
import shutil
import cv2
import sys
import numpy as np
import base64
import requests
import json
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv

# Tải biến môi trường
load_dotenv()

# Bảng chuyển đổi từ tên ngôn ngữ đầy đủ sang mã OCR
LANGUAGE_TO_OCR_CODE = {
    'Vietnamese': 'vi',
    'English': 'en',
    'Chinese': 'ch',
    'Chinese Traditional': 'ch_tra',
    'Japanese': 'ja',
    'Korean': 'ko',
    'French': 'fr',
    'German': 'de',
    'Italian': 'it',
    'Spanish': 'es',
    'Portuguese': 'pt',
    'Russian': 'ru',
    'Arabic': 'ar',
    'Hindi': 'hi',
    'Thai': 'th'
}

# Phần 1: Đọc ảnh và phát hiện ngôn ngữ

def read_image(image_path):
    """
    Đọc ảnh từ đường dẫn và trả về thông tin cơ bản.
    
    Args:
        image_path: Đường dẫn đến ảnh
        
    Returns:
        image: Ảnh đã đọc
    """
    print(f"Đọc ảnh từ: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"Lỗi: Không tìm thấy ảnh tại đường dẫn {image_path}")
        return None
    
    # Đọc ảnh
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Lỗi: Không thể đọc ảnh từ {image_path}")
        return None
    
    # Lấy thông tin ảnh
    height, width, channels = image.shape
    
    print(f"Thông tin ảnh:")
    print(f"- Kích thước: {width} x {height} pixels")
    print(f"- Số kênh màu: {channels}")
    print(f"- Loại dữ liệu: {image.dtype}")
    
    return image

def detect_language_from_image(image_path, api_key):
    """
    Nhận dạng ngôn ngữ từ ảnh sử dụng GPT-4o-mini.
    
    Args:
        image_path: Đường dẫn đến ảnh
        api_key: OpenAI API key
        
    Returns:
        tuple: (language_code, language_name) - mã ngôn ngữ ISO 639-1 và tên đầy đủ
    """
    print(f"Đang nhận dạng ngôn ngữ từ ảnh: {image_path}")
    
    try:
        # Mã hóa ảnh thành Base64
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Chuẩn bị payload
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        prompt = """
        Identify the primary language used in this image.
        
        Requirements:
        1. Look at all text elements in the image
        2. Determine the main language used
        3. Return ONLY the language code (2-letter ISO 639-1 code) and full language name in JSON format
        4. Use the format: {"code": "xx", "name": "Language Name"}
        5. For example: {"code": "en", "name": "English"} or {"code": "vi", "name": "Vietnamese"}
        
        Common language codes:
        - English: en
        - Vietnamese: vi
        - Chinese: zh
        - Japanese: ja
        - Korean: ko
        - French: fr
        - German: de
        - Spanish: es
        - Russian: ru
        
        Do not include any explanation or additional text, ONLY return the JSON object.
        """
        
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
            "max_tokens": 150
        }
        
        # Gọi API
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        response.raise_for_status()  # Kiểm tra lỗi HTTP
        result = response.json()
        
        # Lấy phản hồi JSON
        response_content = result['choices'][0]['message']['content']
        
        # Parse JSON
        language_info = json.loads(response_content)
        
        # Lấy mã ngôn ngữ và tên đầy đủ
        language_code = language_info.get('code', 'en')
        language_name = language_info.get('name', 'English')
        
        print(f"Ngôn ngữ được nhận dạng: {language_name} ({language_code})")
        
        return language_code, language_name
    
    except Exception as e:
        print(f"Lỗi khi nhận dạng ngôn ngữ: {e}")
        # Mặc định là tiếng Anh nếu có lỗi
        return "en", "English"

# Phần 2: Phát hiện vùng văn bản sử dụng EasyOCR

def find_font():
    """Tìm font phù hợp để hiển thị kết quả."""
    possible_font_paths = [
        './fonts/simfang.ttf',
        './simfang.ttf',
        'C:/Windows/Fonts/Arial.ttf',
        'C:/Windows/Fonts/calibri.ttf',
        'C:/Windows/Fonts/tahoma.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/Library/Fonts/Arial.ttf',
    ]
    
    for path in possible_font_paths:
        if os.path.exists(path):
            return path
    
    return None

def detect_text_with_easyocr(image_path, lang='en'):
    """
    Phát hiện vùng văn bản trong ảnh sử dụng EasyOCR.
    
    Args:
        image_path: Đường dẫn đến ảnh
        lang: Ngôn ngữ OCR
        
    Returns:
        List các tuples (bounding_box, text, confidence)
    """
    print(f"Phát hiện vùng văn bản trong ảnh bằng EasyOCR: {image_path}")
    
    start_time = time.time()
    
    try:
        import easyocr
        
        # Xác định ngôn ngữ
        languages = [lang]
        if lang != 'en':
            languages = ['en', lang]  # Thêm tiếng Anh để tăng độ chính xác
        
        # Khởi tạo EasyOCR reader
        print("Đang khởi tạo EasyOCR...")
        reader = easyocr.Reader(languages, gpu=False)
        print("Đã khởi tạo EasyOCR, đang thực hiện OCR...")
        
        # Thực hiện OCR
        result = reader.readtext(image_path)
        
        # Chuyển đổi kết quả về định dạng thống nhất
        text_areas = []
        
        for detection in result:
            bbox = detection[0]  # Bounding box dạng [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
            text = detection[1]  # Văn bản
            confidence = detection[2]  # Độ tin cậy
            
            # Chuyển đổi bbox từ 4 điểm thành danh sách tọa độ
            box = [
                [int(bbox[0][0]), int(bbox[0][1])],  # top-left
                [int(bbox[1][0]), int(bbox[1][1])],  # top-right
                [int(bbox[2][0]), int(bbox[2][1])],  # bottom-right
                [int(bbox[3][0]), int(bbox[3][1])]   # bottom-left
            ]
            
            text_areas.append((box, text, confidence))
        
        end_time = time.time()
        print(f"Thời gian phát hiện: {end_time - start_time:.2f} giây")
        print(f"Đã phát hiện {len(text_areas)} vùng văn bản")
        
        return text_areas
    
    except ImportError as e:
        print(f"Lỗi: Thiếu thư viện EasyOCR. Chi tiết: {e}")
        print("Cài đặt: pip install easyocr")
        return []
    except Exception as e:
        print(f"Lỗi khi phát hiện văn bản: {e}")
        return []

def merge_nearby_boxes(text_areas, horizontal_threshold=30, vertical_threshold=10):
    """
    Hợp nhất các bounding box gần nhau thành một bounding box lớn hơn.
    
    Args:
        text_areas: Danh sách các bounding box (box, text, confidence)
        horizontal_threshold: Ngưỡng khoảng cách ngang để hợp nhất (pixel)
        vertical_threshold: Ngưỡng khoảng cách dọc để hợp nhất (pixel)
        
    Returns:
        Danh sách bounding box đã hợp nhất
    """
    if not text_areas:
        return []
    
    # Sắp xếp các vùng theo tọa độ y (từ trên xuống dưới)
    sorted_areas = sorted(text_areas, key=lambda item: item[0][0][1])
    
    # Tính toán các phạm vi và trung tâm cho mỗi box
    box_info = []
    for box, text, confidence in sorted_areas:
        # Tính toạ độ min và max
        x_min = min(p[0] for p in box)
        y_min = min(p[1] for p in box)
        x_max = max(p[0] for p in box)
        y_max = max(p[1] for p in box)
        
        # Tính toạ độ trung tâm
        center_y = (y_min + y_max) / 2
        
        box_info.append({
            'box': box,
            'text': text,
            'confidence': confidence,
            'x_min': x_min,
            'y_min': y_min,
            'x_max': x_max,
            'y_max': y_max,
            'center_y': center_y
        })
    
    # Nhóm các box theo dòng (dựa theo tọa độ y trung bình)
    lines = []
    current_line = [box_info[0]]
    
    for i in range(1, len(box_info)):
        current_box = box_info[i]
        prev_box = current_line[-1]
        
        # Kiểm tra nếu box hiện tại thuộc cùng dòng với box trước đó
        if abs(current_box['center_y'] - prev_box['center_y']) <= vertical_threshold:
            current_line.append(current_box)
        else:
            # Bắt đầu dòng mới
            lines.append(current_line)
            current_line = [current_box]
    
    # Thêm dòng cuối cùng
    if current_line:
        lines.append(current_line)
    
    # Sắp xếp các box trong mỗi dòng theo tọa độ x (từ trái sang phải)
    for line in lines:
        line.sort(key=lambda box: box['x_min'])
    
    # Hợp nhất các box gần nhau trong cùng một dòng
    merged_boxes = []
    
    for line in lines:
        merged_line = []
        current_group = [line[0]]
        
        for i in range(1, len(line)):
            current_box = line[i]
            prev_box = current_group[-1]
            
            # Kiểm tra khoảng cách ngang
            if current_box['x_min'] - prev_box['x_max'] <= horizontal_threshold:
                # Box đủ gần để hợp nhất
                current_group.append(current_box)
            else:
                # Tạo box hợp nhất từ nhóm hiện tại
                merged_line.append(merge_group(current_group))
                # Bắt đầu nhóm mới
                current_group = [current_box]
        
        # Xử lý nhóm cuối cùng
        if current_group:
            merged_line.append(merge_group(current_group))
        
        merged_boxes.extend(merged_line)
    
    return merged_boxes

def merge_group(group):
    """
    Hợp nhất một nhóm box thành một box duy nhất.
    
    Args:
        group: Danh sách các box cần hợp nhất
        
    Returns:
        Tuple (box, text, confidence) đã hợp nhất
    """
    # Tìm tọa độ bao quanh tất cả các box
    x_min = min(box['x_min'] for box in group)
    y_min = min(box['y_min'] for box in group)
    x_max = max(box['x_max'] for box in group)
    y_max = max(box['y_max'] for box in group)
    
    # Tạo bounding box mới cho vùng đã hợp nhất
    merged_box = [
        [x_min, y_min],  # top-left
        [x_max, y_min],  # top-right
        [x_max, y_max],  # bottom-right
        [x_min, y_max]   # bottom-left
    ]
    
    # Nối các đoạn text với khoảng trắng
    merged_text = " ".join(box['text'] for box in group)
    
    # Lấy confidence trung bình
    avg_confidence = sum(box['confidence'] for box in group) / len(group)
    
    return (merged_box, merged_text, avg_confidence)

def generate_fallback_text_areas(image_path):
    """
    Tạo các vùng văn bản giả khi OCR không hoạt động.
    Phương pháp dự phòng này chia ảnh thành lưới các vùng.
    
    Args:
        image_path: Đường dẫn đến ảnh
        
    Returns:
        List các tuples (bounding_box, text, confidence)
    """
    # Đọc ảnh
    image = cv2.imread(image_path)
    if image is None:
        return []
    
    height, width = image.shape[:2]
    
    # Tạo lưới chia ảnh (3x3, 4x2 hoặc 2x2 tùy thuộc vào kích thước ảnh)
    if width > height * 2:  # Ảnh rộng hơn nhiều so với chiều cao
        rows, cols = 2, 4
    elif width > height:  # Ảnh rộng hơn một chút so với chiều cao
        rows, cols = 2, 3
    elif height > width * 2:  # Ảnh cao hơn nhiều so với chiều rộng
        rows, cols = 4, 2
    else:  # Ảnh gần vuông
        rows, cols = 3, 3
    
    text_areas = []
    cell_height = height // rows
    cell_width = width // cols
    
    # Tạo các vùng bounding box dựa trên lưới
    for row in range(rows):
        for col in range(cols):
            # Tính tọa độ của ô
            x1 = col * cell_width
            y1 = row * cell_height
            x2 = (col + 1) * cell_width
            y2 = (row + 1) * cell_height
            
            # Tạo bounding box
            box = [
                [x1, y1],  # top-left
                [x2, y1],  # top-right
                [x2, y2],  # bottom-right
                [x1, y2]   # bottom-left
            ]
            
            # Tạo khu vực vùng văn bản
            region = image[y1:y2, x1:x2]
            
            # Kiểm tra nếu vùng quá trống (nghĩa là chỉ có một màu)
            if region.size > 0:
                std_dev = np.std(region)
                if std_dev < 10:  # Vùng quá đồng nhất, có thể không có văn bản
                    continue
            
            # Thêm vào danh sách với text rỗng và độ tin cậy thấp
            text_areas.append((box, f"Region_{row}_{col}", 0.5))
    
    print(f"Đã tạo {len(text_areas)} vùng với phương pháp dự phòng")
    return text_areas

def detect_text_areas(image_path, lang='en', use_fallback=False):
    """
    Phát hiện các vùng văn bản trong ảnh.
    
    Args:
        image_path: Đường dẫn đến ảnh
        lang: Mã ngôn ngữ
        use_fallback: Sử dụng phương pháp dự phòng nếu OCR không hoạt động
        
    Returns:
        Danh sách các vùng văn bản (bounding_box, text, confidence)
    """
    print(f"Phát hiện văn bản trong ảnh {image_path} (ngôn ngữ: {lang})...")
    
    # Phát hiện văn bản sử dụng EasyOCR
    text_areas = detect_text_with_easyocr(image_path, lang)
    
    # Hợp nhất các bounding box gần nhau
    if text_areas:
        text_areas = merge_nearby_boxes(text_areas)
    
    # Nếu không phát hiện được vùng văn bản và cho phép sử dụng phương pháp dự phòng
    if not text_areas and use_fallback:
        print("Không phát hiện được vùng văn bản. Sử dụng phương pháp dự phòng...")
        text_areas = generate_fallback_text_areas(image_path)
    
    return text_areas

def draw_text_areas(image, text_areas, output_path):
    """
    Vẽ các vùng văn bản lên ảnh.
    
    Args:
        image: Ảnh gốc
        text_areas: Danh sách các vùng văn bản
        output_path: Đường dẫn lưu ảnh kết quả
    """
    # Tạo bản sao của ảnh để vẽ lên
    result_image = image.copy()
    
    # Vẽ bounding box cho từng vùng văn bản
    for idx, (box, text, confidence) in enumerate(text_areas):
        # Chuyển box thành dạng points cho hàm vẽ
        points = np.array(box).astype(np.int32).reshape((-1, 1, 2))
        
        # Vẽ bounding box
        cv2.polylines(result_image, [points], True, (0, 255, 0), 2)
        
        # Vẽ số thứ tự
        x, y = box[0]  # Góc trên bên trái
        cv2.putText(result_image, f"#{idx+1}", (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        # In thông tin
        print(f"Vùng #{idx+1}:")
        print(f"  - Tọa độ: {box}")
        print(f"  - Văn bản: {text}")
        print(f"  - Độ tin cậy: {confidence:.4f}")
    
    # Lưu ảnh kết quả
    cv2.imwrite(output_path, result_image)
    print(f"Đã lưu ảnh kết quả vào: {output_path}")
    
    return result_image

def save_detected_boxes_image(image_path, text_areas, output_path):
    """
    Lưu ảnh với các vùng đã phát hiện.
    
    Args:
        image_path: Đường dẫn đến ảnh gốc
        text_areas: Danh sách các vùng văn bản
        output_path: Đường dẫn lưu ảnh kết quả
    """
    # Đọc ảnh
    image = cv2.imread(image_path)
    if image is None:
        print(f"Không thể đọc ảnh từ {image_path}")
        return
    
    # Tạo bản sao của ảnh để vẽ lên
    result_image = image.copy()
    
    # Vẽ bounding box cho từng vùng văn bản
    for idx, (box, text, confidence) in enumerate(text_areas):
        # Chuyển box thành dạng points cho hàm vẽ
        points = np.array(box).astype(np.int32).reshape((-1, 1, 2))
        
        # Vẽ bounding box
        cv2.polylines(result_image, [points], True, (0, 255, 0), 2)
        
        # Vẽ số thứ tự
        x, y = points[0][0]
        cv2.putText(result_image, f"#{idx+1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
    # Lưu ảnh kết quả
    cv2.imwrite(output_path, result_image)
    print(f"Đã lưu ảnh kết quả vào: {output_path}")

# Phần 3: Cắt vùng bounding box và tô trắng

def crop_text_areas(image, text_areas, output_dir):
    """
    Cắt vùng văn bản từ ảnh gốc.
    
    Args:
        image: Ảnh gốc
        text_areas: Danh sách vùng văn bản
        output_dir: Thư mục lưu ảnh đã cắt
        
    Returns:
        Danh sách đường dẫn đến các ảnh đã cắt
    """
    # Tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)
    
    # Danh sách lưu đường dẫn các ảnh đã cắt
    cropped_images_paths = []
    
    # Thông tin ảnh
    height, width = image.shape[:2]
    
    # Cắt và lưu từng vùng văn bản
    for idx, (box, text, confidence) in enumerate(text_areas):
        # Chuyển đổi box thành dạng số nguyên
        box_points = np.array(box, dtype=np.int32)
        
        # Tính toán bounding rectangle
        rect = cv2.boundingRect(box_points)
        x, y, w, h = rect
        
        # Đảm bảo kích thước hợp lệ
        x = max(0, x)
        y = max(0, y)
        w = min(width - x, w)
        h = min(height - y, h)
        
        # Cắt vùng ảnh
        cropped = image[y:y+h, x:x+w]
        
        # Tạo đường dẫn lưu ảnh
        output_path = os.path.join(output_dir, f"crop_{idx+1}.jpg")
        
        # Lưu ảnh đã cắt
        cv2.imwrite(output_path, cropped)
        
        # Lưu thông tin
        cropped_images_paths.append({
            'path': output_path,
            'index': idx+1,
            'box': box,
            'text': text,
            'confidence': confidence,
            'rect': rect  # Lưu thông tin rectangle để dễ dàng chèn lại sau này
        })
        
        print(f"Đã cắt và lưu vùng #{idx+1} vào: {output_path}")
    
    return cropped_images_paths

def whiten_text_areas(image, text_areas):
    """
    Tô trắng vùng chữ trong ảnh gốc.
    
    Args:
        image: Ảnh gốc
        text_areas: Danh sách vùng văn bản
        
    Returns:
        Ảnh sau khi tô trắng các vùng văn bản
    """
    # Tạo bản sao của ảnh
    whitened_image = image.copy()
    
    # Tô trắng từng vùng văn bản
    for box, _, _ in text_areas:
        # Chuyển đổi box thành dạng số nguyên
        points = np.array(box, dtype=np.int32).reshape((-1, 1, 2))
        
        # Tô trắng vùng đa giác
        cv2.fillPoly(whitened_image, [points], (255, 255, 255))
    
    return whitened_image

# Phần 4: Tiền xử lý ảnh

def denoise_image(image):
    """
    Loại bỏ nhiễu từ ảnh.
    
    Args:
        image: Ảnh đầu vào
        
    Returns:
        Ảnh đã loại bỏ nhiễu
    """
    # Loại bỏ nhiễu không phá hủy cạnh
    if len(image.shape) == 3:
        denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    else:
        denoised = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
    
    return denoised

def enhance_contrast(image):
    """
    Cải thiện độ tương phản của ảnh.
    
    Args:
        image: Ảnh đầu vào
        
    Returns:
        Ảnh đã cải thiện độ tương phản
    """
    # Chuyển đổi không gian màu
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
    else:
        l = image.copy()
    
    # Áp dụng CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # Hợp nhất các kênh và chuyển lại không gian màu
    if len(image.shape) == 3:
        limg = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    else:
        enhanced = cl
    
    return enhanced

def sharpen_image(image):
    """
    Làm sắc nét ảnh.
    
    Args:
        image: Ảnh đầu vào
        
    Returns:
        Ảnh đã làm sắc nét
    """
    # Tạo kernel
    kernel = np.array([[-1, -1, -1],
                      [-1,  9, -1],
                      [-1, -1, -1]])
    
    # Áp dụng kernel
    sharpened = cv2.filter2D(image, -1, kernel)
    
    return sharpened

def preprocess_image(image_path, output_path):
    """
    Tiền xử lý ảnh để cải thiện độ chính xác khi trích xuất văn bản.
    
    Args:
        image_path: Đường dẫn đến ảnh đầu vào
        output_path: Đường dẫn lưu ảnh đã xử lý
        
    Returns:
        Ảnh đã xử lý và ảnh gốc
    """
    print(f"Đang xử lý ảnh: {image_path}")
    
    # Đọc ảnh
    image = cv2.imread(image_path)
    if image is None:
        print(f"Lỗi: Không thể đọc ảnh từ {image_path}")
        return None
    
    # Lưu ảnh gốc
    original = image.copy()
    
    # Loại bỏ nhiễu
    denoised = denoise_image(image)
    
    # Cải thiện độ tương phản
    enhanced = enhance_contrast(denoised)
    
    # Làm sắc nét
    sharpened = sharpen_image(enhanced)
    
    # Lưu ảnh đã xử lý
    cv2.imwrite(output_path, sharpened)
    print(f"Đã lưu ảnh đã xử lý vào: {output_path}")
    
    return sharpened, original

def process_images(cropped_images, output_dir, show=False):
    """
    Xử lý tất cả các ảnh đã cắt.
    
    Args:
        cropped_images: Danh sách thông tin các ảnh đã cắt
        output_dir: Thư mục lưu các ảnh đã xử lý
        show: Hiển thị kết quả
    
    Returns:
        Danh sách thông tin các ảnh đã xử lý
    """
    # Tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)
    
    # Danh sách lưu thông tin các ảnh đã xử lý
    processed_images = []
    
    # Xử lý từng ảnh
    for crop_info in cropped_images:
        # Tạo đường dẫn đầu ra
        output_path = os.path.join(output_dir, f"processed_{crop_info['index']}.jpg")
        
        # Tiền xử lý ảnh
        result, original = preprocess_image(crop_info['path'], output_path)
        
        if result is not None:
            # Lưu thông tin
            processed_info = crop_info.copy()
            processed_info['processed_path'] = output_path
            processed_images.append(processed_info)
            
            # Hiển thị kết quả nếu yêu cầu
            if show:
                # Đặt kích thước phù hợp cho cửa sổ hiển thị
                scale = 1
                if original.shape[1] > 800:
                    scale = 800 / original.shape[1]
                
                resized_original = cv2.resize(original, (0, 0), fx=scale, fy=scale)
                resized_result = cv2.resize(result, (0, 0), fx=scale, fy=scale)
                
                cv2.imshow(f"Original #{crop_info['index']}", resized_original)
                cv2.imshow(f"Processed #{crop_info['index']}", resized_result)
    
    # Lưu thông tin các ảnh đã xử lý
    processed_info_path = os.path.join(output_dir, "processed_info.txt")
    with open(processed_info_path, 'w', encoding='utf-8') as f:
        for info in processed_images:
            rect = info['rect']
            box_str = ",".join([f"{p[0]},{p[1]}" for p in info['box']])
            f.write(f"{info['index']}|{info['path']}|{info['processed_path']}|{box_str}|{rect[0]},{rect[1]},{rect[2]},{rect[3]}|{info['text']}|{info['confidence']}\n")
    
    print(f"Đã lưu thông tin các ảnh đã xử lý vào: {processed_info_path}")
    
    return processed_images

# Phần 5: Trích xuất văn bản

def extract_text_with_gpt(image_path, api_key):
    """
    Trích xuất văn bản từ ảnh sử dụng GPT-4o-mini.
    
    Args:
        image_path: Đường dẫn đến ảnh
        api_key: OpenAI API key
        
    Returns:
        str: Văn bản được trích xuất
    """
    print(f"Trích xuất văn bản từ ảnh {image_path}...")
    
    try:
        # Mã hóa ảnh thành Base64
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Chuẩn bị payload
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Sử dụng prompt được đề xuất để trích xuất chính xác hơn
        prompt = """
        Extract all text visible in the image accurately. Focus only on the text content.
        
        Guidelines:
        - Extract ALL text elements visible in the image
        - Maintain the original formatting where possible (paragraphs, bullet points)
        - Preserve text hierarchy (headings, subheadings, body text)
        - Include text from diagrams, charts, and tables
        - Keep numbers, dates, and special characters exactly as they appear
        - Do not add any explanations, interpretations or commentary
        - Do not describe the image itself
        
        Your output should ONLY contain the extracted text, presented as clearly as possible.
        """
        
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}
            ],
            "temperature": 0.1,
            "max_tokens": 500
        }
        
        # Gọi API
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        response.raise_for_status()  # Kiểm tra lỗi HTTP
        result = response.json()
        
        # Lấy văn bản
        extracted_text = result['choices'][0]['message']['content'].strip()
        
        print(f"Văn bản đã trích xuất: {extracted_text}")
        
        return extracted_text
    
    except Exception as e:
        print(f"Lỗi khi trích xuất văn bản từ ảnh {image_path}: {e}")
        return ""

def process_crops(processed_images, api_key, output_file):
    """
    Xử lý các ảnh đã xử lý để trích xuất văn bản.
    
    Args:
        processed_images: Danh sách thông tin các ảnh đã xử lý
        api_key: OpenAI API key
        output_file: Đường dẫn file đầu ra
    
    Returns:
        List thông tin các ảnh đã được trích xuất văn bản
    """
    processed_crops = []
    
    # Mở file để lưu kết quả
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("INDEX|PATH|PROCESSED_PATH|ORIGINAL_TEXT|EXTRACTED_TEXT\n")
        
        # Xử lý từng ảnh
        for crop_info in processed_images:
            print(f"\nXử lý ảnh #{crop_info['index']}: {crop_info.get('processed_path', crop_info.get('path'))}")
            
            # Sử dụng ảnh đã tiền xử lý nếu có, nếu không thì sử dụng ảnh gốc
            image_path = crop_info.get('processed_path', crop_info.get('path'))
            
            # Trích xuất văn bản
            extracted_text = extract_text_with_gpt(image_path, api_key)
            
            # Thêm văn bản đã trích xuất vào thông tin
            crop_info['extracted_text'] = extracted_text
            
            # Ghi kết quả vào file
            crop_path = crop_info.get('crop_path', crop_info.get('path'))
            processed_path = crop_info.get('processed_path', '')
            original_text = crop_info.get('text', '')
            
            f.write(f"{crop_info['index']}|{crop_path}|{processed_path}|{original_text}|{extracted_text}\n")
            
            # Thêm vào danh sách đã xử lý
            processed_crops.append(crop_info)
            
            # Chờ một chút để tránh giới hạn API
            time.sleep(0.5)
    
    print(f"Đã lưu văn bản trích xuất vào: {output_file}")
    
    return processed_crops

# Phần 6: Dịch văn bản

def translate_text(text, target_lang, api_key):
    """
    Dịch văn bản sang ngôn ngữ đích sử dụng GPT-4o-mini.
    
    Args:
        text: Văn bản cần dịch
        target_lang: Ngôn ngữ đích
        api_key: OpenAI API key
        
    Returns:
        str: Văn bản đã dịch
    """
    if not text.strip():
        return ""
    
    print(f"Dịch văn bản: {text}")
    
    try:
        # Xác định ngôn ngữ đích với tên đầy đủ
        language_map = {
            'vi': 'Vietnamese',
            'en': 'English',
            'fr': 'French',
            'de': 'German',
            'ja': 'Japanese',
            'ko': 'Korean',
            'zh': 'Chinese',
            'es': 'Spanish',
            'ru': 'Russian',
            'it': 'Italian',
            'pt': 'Portuguese',
            'nl': 'Dutch',
            'ar': 'Arabic',
            'hi': 'Hindi',
            'th': 'Thai',
            'id': 'Indonesian',
            'ms': 'Malay'
        }
        
        # Nếu đã truyền tên đầy đủ, sử dụng trực tiếp, nếu không thì tra trong bảng
        if target_lang in language_map.values():
            language_name = target_lang
        else:
            language_name = language_map.get(target_lang.lower(), 'English')
            # Nếu không tìm thấy trong map, giả định target_lang là tên đầy đủ
            if target_lang not in language_map:
                language_name = target_lang
        
        # Chuẩn bị payload
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Sử dụng prompt rõ ràng hơn với ngôn ngữ đầy đủ và hướng dẫn giữ lại phần trong ngoặc
        prompt = f"""
        Translate the following text to {language_name}.
        
        Guidelines:
        - Translate ALL text accurately
        - Maintain original formatting and structure
        - Preserve numbers, dates, and special characters
        - IMPORTANT: Keep all content inside parentheses exactly as is, like "(image translation)" or "(some text)"
        - For single words or short phrases, provide direct translation only
        - Do not add explanations or comments
        - Return ONLY the translated text
        
        Example:
        Original: "dịch ảnh có chữ (image translation)"
        Translation: "image text translation (image translation)"
        """
        
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": text}
            ],
            "temperature": 0.1,
            "max_tokens": 500
        }
        
        # Gọi API
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        response.raise_for_status()  # Kiểm tra lỗi HTTP
        result = response.json()
        
        # Lấy văn bản đã dịch
        translated_text = result['choices'][0]['message']['content'].strip()
        
        print(f"Văn bản đã dịch: {translated_text}")
        
        return translated_text
    
    except Exception as e:
        print(f"Lỗi khi dịch văn bản: {e}")
        return text  # Trả về văn bản gốc nếu có lỗi

def process_translations(extracted_texts, target_lang, api_key, output_file):
    """
    Xử lý dịch văn bản cho tất cả các mục.
    
    Args:
        extracted_texts: Danh sách thông tin các văn bản đã trích xuất
        target_lang: Ngôn ngữ đích
        api_key: OpenAI API key
        output_file: Đường dẫn file đầu ra
    
    Returns:
        List thông tin các văn bản đã dịch
    """
    translated_texts = []
    
    # Mở file để lưu kết quả
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("INDEX|PATH|EXTRACTED_TEXT|TRANSLATED_TEXT\n")
        
        # Xử lý từng mục
        for item in extracted_texts:
            print(f"\nXử lý văn bản #{item['index']}: {item['path']}")
            
            # Dịch văn bản
            translated_text = translate_text(item['extracted_text'], target_lang, api_key)
            
            # Thêm văn bản đã dịch vào thông tin
            item['translated_text'] = translated_text
            
            # Ghi kết quả vào file
            f.write(f"{item['index']}|{item['path']}|{item['extracted_text']}|{translated_text}\n")
            
            # Thêm vào danh sách đã xử lý
            translated_texts.append(item)
            
            # Chờ một chút để tránh giới hạn API
            time.sleep(0.5)
    
    print(f"Đã lưu văn bản đã dịch vào: {output_file}")
    
    return translated_texts

# Phần 7: Vẽ văn bản đã dịch

def find_font_for_drawing():
    """Tìm font phù hợp để vẽ văn bản dịch."""
    possible_font_paths = [
        './fonts/arial.ttf',
        './arial.ttf',
        'C:/Windows/Fonts/Arial.ttf',
        'C:/Windows/Fonts/calibri.ttf',
        'C:/Windows/Fonts/tahoma.ttf',
        'C:/Windows/Fonts/times.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/Library/Fonts/Arial.ttf',
    ]
    
    for path in possible_font_paths:
        if os.path.exists(path):
            return path
    
    return None

def get_text_dimensions(text, font_path, font_size):
    """
    Tính toán kích thước của văn bản với font và kích thước cho trước.
    
    Args:
        text: Văn bản cần đo
        font_path: Đường dẫn đến font
        font_size: Kích thước font
        
    Returns:
        (width, height): Kích thước của văn bản
    """
    try:
        font = ImageFont.truetype(font_path, font_size)
        # Tạo ảnh tạm để đo văn bản
        temp_img = Image.new('RGB', (1, 1))
        draw = ImageDraw.Draw(temp_img)
        
        # Tính kích thước với font đã chọn
        lines = text.split('\n')
        max_width = 0
        total_height = 0
        
        for line in lines:
            if not line.strip():  # Dòng trống
                total_height += int(font_size * 0.5)  # Khoảng trống nhỏ hơn
                continue
                
            bbox = draw.textbbox((0, 0), line, font=font)
            width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
            max_width = max(max_width, width)
            total_height += height
            
        # Thêm khoảng trống giữa các dòng
        if len(lines) > 1:
            line_spacing = int(font_size * 0.2)  # 20% kích thước font
            total_height += line_spacing * (len(lines) - 1)
        
        return (max_width, total_height)
    
    except Exception as e:
        print(f"Lỗi khi tính kích thước văn bản: {e}")
        # Ước tính kích thước dựa trên số ký tự
        return (len(text) * font_size // 2, font_size)

def find_optimal_font_size(text, max_width, max_height, font_path, min_size=8, max_size=100):
    """
    Tìm kích thước font tối ưu để văn bản vừa vặn với kích thước cho trước.
    
    Args:
        text: Văn bản cần vẽ
        max_width: Chiều rộng tối đa
        max_height: Chiều cao tối đa
        font_path: Đường dẫn đến font
        min_size: Kích thước font tối thiểu
        max_size: Kích thước font tối đa
        
    Returns:
        Kích thước font tối ưu
    """
    if not text.strip():
        return min_size
    
    # Thêm biên an toàn để tránh cắt chữ
    safety_margin = 0.98  # Giảm kích thước tối đa xuống % để tạo biên an toàn
    max_width_with_margin = max_width * safety_margin
    max_height_with_margin = max_height * safety_margin
    
    # Tìm kiếm nhị phân để xác định kích thước font tối ưu
    low, high = min_size, max_size
    optimal_size = min_size
    
    while low <= high:
        mid = (low + high) // 2
        width, height = get_text_dimensions(text, font_path, mid)
        
        # Kiểm tra xem text có vừa với kích thước không
        if width <= max_width_with_margin and height <= max_height_with_margin:
            optimal_size = mid
            low = mid + 1  # Tìm kích thước lớn hơn
        else:
            high = mid - 1  # Giảm kích thước
    
    # Điều chỉnh lại để đảm bảo hoàn toàn vừa vặn
    width, height = get_text_dimensions(text, font_path, optimal_size)
    
    # Nếu vẫn chưa vừa vặn, giảm dần kích thước đến khi vừa
    while (width > max_width_with_margin or height > max_height_with_margin) and optimal_size > min_size:
        optimal_size -= 1
        width, height = get_text_dimensions(text, font_path, optimal_size)
    
    # Giảm thêm 1 kích thước để đảm bảo an toàn
    if optimal_size > min_size:
        optimal_size -= 1
    
    return optimal_size

def hex_to_rgb(hex_color):
    """
    Chuyển đổi màu từ dạng HEX sang RGB.
    
    Args:
        hex_color: Mã màu HEX (ví dụ: "#FF0000")
        
    Returns:
        Tuple (R, G, B)
    """
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 3:
        hex_color = ''.join([c*2 for c in hex_color])
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def get_color(color_str):
    """
    Chuyển đổi chuỗi màu thành tuple RGB hoặc RGBA.
    
    Args:
        color_str: Tên màu hoặc mã HEX
        
    Returns:
        Tuple màu (R, G, B) hoặc (R, G, B, A)
    """
    # Bảng màu cơ bản
    color_map = {
        'white': (255, 255, 255),
        'black': (0, 0, 0),
        'red': (255, 0, 0),
        'green': (0, 255, 0),
        'blue': (0, 0, 255),
        'yellow': (255, 255, 0),
        'cyan': (0, 255, 255),
        'magenta': (255, 0, 255),
        'gray': (128, 128, 128),
        'grey': (128, 128, 128),
        'purple': (128, 0, 128),
        'orange': (255, 165, 0),
        'pink': (255, 192, 203),
        'brown': (165, 42, 42),
    }
    
    # Kiểm tra nếu là tên màu
    if color_str.lower() in color_map:
        return color_map[color_str.lower()]
    
    # Kiểm tra nếu là mã HEX
    if color_str.startswith('#'):
        return hex_to_rgb(color_str)
    
    # Mặc định là màu đen
    return (0, 0, 0)

def draw_text_on_image(image, text, font_path, font_size=20, bg_color='white', text_color='black'):
    """
    Vẽ văn bản lên ảnh với nền có màu, đảm bảo văn bản không bị cắt.
    
    Args:
        image: Ảnh OpenCV (numpy array)
        text: Văn bản cần vẽ
        font_path: Đường dẫn đến font
        font_size: Kích thước font (nếu = 0, tự động xác định)
        bg_color: Màu nền
        text_color: Màu chữ
        
    Returns:
        Ảnh đã vẽ văn bản
    """
    # Chuyển đổi ảnh OpenCV sang PIL
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    
    # Kích thước ảnh
    width, height = pil_image.size
    
    # Nếu font_size = 0, tự động xác định kích thước font tối ưu
    if font_size <= 0:
        if font_path:
            font_size = find_optimal_font_size(text, width, height, font_path, min_size=8)
        else:
            font_size = 12  # Kích thước mặc định nhỏ
    
    # Tạo ảnh mới với màu nền
    bg_color_rgb = get_color(bg_color)
    new_image = Image.new('RGB', (width, height), bg_color_rgb)
    
    # Tạo đối tượng vẽ
    draw = ImageDraw.Draw(new_image)
    
    # Chuẩn bị font
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception as e:
        print(f"Lỗi font: {e}, sử dụng font mặc định")
        font = ImageFont.load_default()
    
    # Tính toán kích thước văn bản
    text_color_rgb = get_color(text_color)
    
    # Kiểm tra kích thước văn bản để đảm bảo nó vừa vặn
    text_width, text_height = get_text_dimensions(text, font_path, font_size)
    
    # Chỉ giảm kích thước font khi văn bản thực sự lớn hơn bounding box
    if text_width > width*0.95 or text_height > height*0.95:
        # Giảm kích thước font để đảm bảo vừa vặn
        scaling_factor = min(width*0.95/text_width, height*0.95/text_height)
        new_font_size = max(8, int(font_size * scaling_factor))
        
        # Thông báo về việc giảm kích thước font
        print(f"Văn bản quá dài cho bounding box: giảm font size từ {font_size} xuống {new_font_size}")
        
        # Cập nhật font với kích thước mới
        try:
            font = ImageFont.truetype(font_path, new_font_size)
            font_size = new_font_size
            # Tính lại kích thước văn bản
            text_width, text_height = get_text_dimensions(text, font_path, font_size)
        except Exception:
            pass
    else:
        print(f"Văn bản vừa vặn với bounding box, giữ nguyên font size {font_size}")
    
    # Căn giữa văn bản
    lines = text.split('\n')
    y_offset = (height - text_height) // 2
    
    for line in lines:
        if not line.strip():  # Dòng trống
            y_offset += int(font_size * 0.5)
            continue
            
        line_width, line_height = get_text_dimensions(line, font_path, font_size)
        x_offset = (width - line_width) // 2
        
        # Đảm bảo x_offset không âm
        x_offset = max(0, x_offset)
        
        # Vẽ văn bản
        draw.text((x_offset, y_offset), line, font=font, fill=text_color_rgb)
        y_offset += line_height + int(font_size * 0.2)  # Thêm khoảng cách giữa các dòng
    
    # Chuyển đổi trở lại ảnh OpenCV
    result_image = np.array(new_image)
    result_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
    
    return result_image

def process_translated_images(translated_texts, crop_dir, output_dir, bg_color, text_color, font_size):
    """
    Xử lý các ảnh để vẽ văn bản đã dịch.
    
    Args:
        translated_texts: Danh sách thông tin các văn bản đã dịch
        crop_dir: Thư mục chứa ảnh gốc đã cắt
        output_dir: Thư mục lưu ảnh đã vẽ văn bản
        bg_color: Màu nền
        text_color: Màu chữ
        font_size: Kích thước font (0 = tự động)
        
    Returns:
        Danh sách thông tin các ảnh đã xử lý
    """
    # Tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)
    
    # Tìm font cho việc vẽ chữ
    font_path = find_font_for_drawing()
    if not font_path:
        print("Cảnh báo: Không tìm thấy font cho việc vẽ chữ, sử dụng font mặc định")
    
    # Danh sách lưu thông tin các ảnh đã xử lý
    processed_images = []
    
    # Xử lý từng ảnh
    for item in translated_texts:
        crop_path = item.get('path', item.get('crop_path', ''))
        if not crop_path or not os.path.exists(crop_path):
            print(f"Bỏ qua xử lý ảnh #{item['index']}: Không tìm thấy ảnh gốc")
            continue
        
        print(f"Xử lý ảnh #{item['index']}: {crop_path}")
        
        # Đọc ảnh gốc để lấy kích thước
        image = cv2.imread(crop_path)
        if image is None:
            print(f"Bỏ qua xử lý ảnh #{item['index']}: Không thể đọc ảnh")
            continue
        
        # Lấy văn bản đã dịch
        translated_text = item.get('translated_text', '')
        
        # Nếu không có văn bản đã dịch, sử dụng văn bản gốc
        if not translated_text:
            translated_text = item.get('extracted_text', item.get('text', ''))
        
        # Xác định kích thước font
        if font_size <= 0 and font_path:
            # Tự động xác định kích thước font
            height, width = image.shape[:2]
            font_size = find_optimal_font_size(translated_text, width, height, font_path)
        elif font_size <= 0:
            # Nếu không có font, sử dụng kích thước mặc định
            font_size = 20
        
        # Vẽ văn bản lên ảnh
        result_image = draw_text_on_image(image, translated_text, font_path, font_size, bg_color, text_color)
        
        # Tạo đường dẫn đầu ra
        output_path = os.path.join(output_dir, f"translated_{item['index']}.jpg")
        
        # Lưu ảnh kết quả
        cv2.imwrite(output_path, result_image)
        print(f"Đã lưu ảnh đã vẽ văn bản vào: {output_path}")
        
        # Lưu thông tin
        processed_info = item.copy()
        processed_info['output_path'] = output_path
        processed_images.append(processed_info)
    
    # Lưu thông tin các ảnh đã xử lý
    translated_info_path = os.path.join(output_dir, "translated_info.txt")
    with open(translated_info_path, 'w', encoding='utf-8') as f:
        f.write("INDEX|PATH|OUTPUT_PATH|EXTRACTED_TEXT|TRANSLATED_TEXT\n")
        for info in processed_images:
            crop_path = info.get('path', info.get('crop_path', ''))
            output_path = info['output_path']
            extracted_text = info.get('extracted_text', '')
            translated_text = info.get('translated_text', '')
            f.write(f"{info['index']}|{crop_path}|{output_path}|{extracted_text}|{translated_text}\n")
    
    print(f"Đã lưu thông tin các ảnh đã xử lý vào: {translated_info_path}")
    
    return processed_images

# Phần 8: Chèn vùng đã xử lý vào ảnh gốc

def load_translated_info(translated_info_path):
    """
    Đọc thông tin các ảnh đã dịch từ file.
    
    Args:
        translated_info_path: Đường dẫn đến file thông tin
        
    Returns:
        List thông tin các ảnh đã dịch
    """
    translated_images = []
    
    try:
        with open(translated_info_path, 'r', encoding='utf-8') as f:
            # Bỏ qua dòng header
            header = f.readline()
            
            for line in f:
                parts = line.strip().split('|')
                if len(parts) >= 5:
                    translated_images.append({
                        'index': int(parts[0]),
                        'crop_path': parts[1],
                        'output_path': parts[2],
                        'extracted_text': parts[3],
                        'translated_text': parts[4]
                    })
        
        print(f"Đã đọc thông tin {len(translated_images)} ảnh đã dịch từ {translated_info_path}")
    
    except Exception as e:
        print(f"Lỗi khi đọc file thông tin các ảnh đã dịch: {e}")
    
    return translated_images

def merge_translated_regions(original_image, translated_info, cropped_info=None):
    """
    Chèn các vùng đã dịch vào ảnh gốc.
    
    Args:
        original_image: Ảnh gốc
        translated_info: Thông tin các ảnh đã dịch
        cropped_info: Thông tin các ảnh đã cắt (nếu có)
        
    Returns:
        Ảnh kết quả
    """
    # Tạo bản sao của ảnh gốc
    result_image = original_image.copy()
    
    # Nếu có thông tin các ảnh đã cắt, sử dụng thông tin đó để lấy vị trí chèn
    if cropped_info:
        # Tạo ánh xạ giữa chỉ số và thông tin crop
        crop_map = {item['index']: item for item in cropped_info}
        
        # Chèn từng vùng đã dịch
        for item in translated_info:
            idx = item['index']
            
            # Lấy thông tin về vùng đã cắt
            if idx in crop_map:
                crop_item = crop_map[idx]
                x, y, w, h = crop_item['rect']
                
                # Đọc ảnh đã dịch
                translated_img = cv2.imread(item['output_path'])
                
                if translated_img is not None:
                    # Đảm bảo ảnh đã dịch có đúng kích thước với vùng cắt
                    if translated_img.shape[:2] != (h, w):
                        # Resize chính xác với kích thước gốc
                        translated_img = cv2.resize(translated_img, (w, h), interpolation=cv2.INTER_AREA)
                    
                    # Kiểm tra ranh giới để tránh lỗi
                    if y >= 0 and x >= 0 and y + h <= result_image.shape[0] and x + w <= result_image.shape[1]:
                        # Chèn vào ảnh gốc
                        result_image[y:y+h, x:x+w] = translated_img
                        print(f"Đã chèn vùng {idx} vào vị trí ({x}, {y}) với kích thước {w}x{h}")
                    else:
                        print(f"Cảnh báo: Vùng {idx} có tọa độ nằm ngoài ảnh gốc, bỏ qua")
    else:
        # Không có thông tin crop, dựa vào tên file để xác định vị trí chèn
        for item in translated_info:
            # Trích xuất thông tin từ tên file
            basename = os.path.basename(item['crop_path'])
            if basename.startswith('crop_') and '_' in basename:
                try:
                    # Phân tích tên file để lấy chỉ số
                    idx = int(basename.split('_')[1].split('.')[0])
                    
                    # Tìm vùng cắt trong ảnh gốc (đơn giản hóa, cần cải thiện)
                    # Giả định rằng các vùng được cắt từ trái sang phải, trên xuống dưới
                    height, width = original_image.shape[:2]
                    rows, cols = 3, 3  # Giả định có 3x3 vùng
                    cell_height, cell_width = height // rows, width // cols
                    
                    row = (idx - 1) // cols
                    col = (idx - 1) % cols
                    
                    x = col * cell_width
                    y = row * cell_height
                    w, h = cell_width, cell_height
                    
                    # Đọc ảnh đã dịch
                    translated_img = cv2.imread(item['output_path'])
                    
                    if translated_img is not None:
                        # Đảm bảo kích thước ảnh đã dịch khớp với vùng cắt
                        if translated_img.shape[:2] != (h, w):
                            translated_img = cv2.resize(translated_img, (w, h), interpolation=cv2.INTER_AREA)
                        
                        # Kiểm tra ranh giới
                        if y >= 0 and x >= 0 and y + h <= result_image.shape[0] and x + w <= result_image.shape[1]:
                            # Chèn vào ảnh gốc
                            result_image[y:y+h, x:x+w] = translated_img
                            print(f"Đã chèn vùng {idx} vào vị trí ({x}, {y}) với kích thước {w}x{h}")
                        else:
                            print(f"Cảnh báo: Vùng {idx} có tọa độ nằm ngoài ảnh gốc, bỏ qua")
                except (ValueError, IndexError) as e:
                    print(f"Lỗi khi xử lý file {basename}: {e}")
    
    return result_image

def create_output_dirs(image_path, output_dir=None):
    """
    Tạo các thư mục đầu ra.
    
    Args:
        image_path: Đường dẫn đến ảnh đầu vào
        output_dir: Thư mục đầu ra (nếu None, sẽ tạo thư mục mới trong cùng thư mục với ảnh)
        
    Returns:
        Dict chứa đường dẫn các thư mục
    """
    # Lấy tên file từ đường dẫn
    basename = os.path.basename(image_path)
    filename = os.path.splitext(basename)[0]
    
    # Xác định thư mục gốc
    if output_dir:
        root_dir = output_dir
    else:
        root_dir = os.path.join(os.path.dirname(image_path), f"{filename}_translated")
    
    # Tạo thư mục gốc
    os.makedirs(root_dir, exist_ok=True)
    
    # Tạo các thư mục con
    crops_dir = os.path.join(root_dir, "crops")
    preprocessed_dir = os.path.join(root_dir, "preprocessed")
    translated_dir = os.path.join(root_dir, "translated_crops")
    
    os.makedirs(crops_dir, exist_ok=True)
    os.makedirs(preprocessed_dir, exist_ok=True)
    os.makedirs(translated_dir, exist_ok=True)
    
    # Tạo đường dẫn cho các file
    paths = {
        "root_dir": root_dir,
        "crops_dir": crops_dir,
        "preprocessed_dir": preprocessed_dir,
        "translated_dir": translated_dir,
        "whitened_image": os.path.join(root_dir, "whitened_image.jpg"),
        "detected_boxes": os.path.join(root_dir, "detected_boxes.jpg"),
        "cropped_info": os.path.join(crops_dir, "cropped_info.txt"),
        "preprocessed_info": os.path.join(preprocessed_dir, "processed_info.txt"),
        "extracted_texts": os.path.join(root_dir, "extracted_texts.txt"),
        "translated_texts": os.path.join(root_dir, "translated_texts.txt"),
        "translated_info": os.path.join(translated_dir, "translated_info.txt"),
        "final_image": os.path.join(root_dir, "final_translated_image.jpg")
    }
    
    return paths

def parse_args():
    """Phân tích tham số dòng lệnh."""
    parser = argparse.ArgumentParser(description='Dịch văn bản từ ảnh')
    parser.add_argument('--image_path', required=True, help='Đường dẫn đến ảnh đầu vào')
    parser.add_argument('--output_dir', help='Thư mục lưu kết quả (mặc định: thư mục cùng với ảnh)')
    parser.add_argument('--source_lang', help='Ngôn ngữ nguồn (tự động phát hiện nếu không cung cấp)')
    parser.add_argument('--target_lang', default='Vietnamese', help='Ngôn ngữ đích (mặc định: tiếng Việt)')
    parser.add_argument('--use_fallback', action='store_true', help='Sử dụng phương pháp dự phòng nếu OCR không hoạt động')
    parser.add_argument('--skip_preprocess', action='store_true', help='Bỏ qua bước tiền xử lý ảnh')
    parser.add_argument('--merge_boxes', action='store_true', help='Hợp nhất các bounding box gần nhau')
    parser.add_argument('--horizontal_threshold', type=int, default=30, help='Ngưỡng khoảng cách ngang để hợp nhất (pixel)')
    parser.add_argument('--vertical_threshold', type=int, default=10, help='Ngưỡng khoảng cách dọc để hợp nhất (pixel)')
    parser.add_argument('--api_key', help='OpenAI API key (nếu không cung cấp, sẽ lấy từ biến môi trường)')
    parser.add_argument('--bg_color', default='white', help='Màu nền cho văn bản dịch')
    parser.add_argument('--text_color', default='black', help='Màu chữ cho văn bản dịch')
    parser.add_argument('--show', action='store_true', help='Hiển thị kết quả')
    return parser.parse_args()

def main():
    """Hàm chính của chương trình."""
    # Phân tích tham số
    args = parse_args()
    
    # Kiểm tra file ảnh đầu vào
    if not os.path.exists(args.image_path):
        print(f"Lỗi: Không tìm thấy ảnh {args.image_path}")
        return
    
    # Lấy API key
    api_key = args.api_key or os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("Lỗi: Không tìm thấy OpenAI API key. Cung cấp qua --api_key hoặc biến môi trường OPENAI_API_KEY")
        return
    
    # Tạo thư mục đầu ra
    paths = create_output_dirs(args.image_path, args.output_dir)
    
    print("\n===== BẮT ĐẦU QUY TRÌNH DỊCH VĂN BẢN TỪ ẢNH =====\n")
    start_time = time.time()
    
    # Bước 1: Đọc ảnh
    print("\n----- Bước 1: Đọc ảnh -----")
    image = read_image(args.image_path)
    if image is None:
        return
    
    # Sao chép ảnh gốc vào thư mục đầu ra
    original_copy = os.path.join(paths["root_dir"], os.path.basename(args.image_path))
    shutil.copy2(args.image_path, original_copy)
    print(f"Đã sao chép ảnh gốc vào: {original_copy}")
    
    # Nhận dạng ngôn ngữ nguồn nếu không được cung cấp
    if not args.source_lang:
        print("\n----- Phát hiện ngôn ngữ nguồn -----")
        source_lang_code, source_lang_name = detect_language_from_image(args.image_path, api_key)
    else:
        # Nếu đã cung cấp nguồn ngữ, chuyển đổi về dạng phù hợp
        if args.source_lang in LANGUAGE_TO_OCR_CODE:
            # Nếu là tên đầy đủ
            source_lang_name = args.source_lang
            source_lang_code = LANGUAGE_TO_OCR_CODE[args.source_lang]
        else:
            # Nếu là mã ngôn ngữ
            source_lang_code = args.source_lang
            # Tìm tên đầy đủ (nếu có)
            source_lang_name = next((name for name, code in LANGUAGE_TO_OCR_CODE.items() 
                                if code == source_lang_code), source_lang_code)
        
        print(f"Sử dụng ngôn ngữ nguồn: {source_lang_name} ({source_lang_code})")
    
    # Bước 2: Phát hiện vùng văn bản (sử dụng OCR)
    print("\n----- Bước 2: Phát hiện vùng văn bản -----")
    # Sử dụng mã ngôn ngữ cho OCR
    text_areas = detect_text_areas(args.image_path, source_lang_code, args.use_fallback)
    
    # Hợp nhất các bounding box gần nhau nếu được yêu cầu
    if args.merge_boxes:
        print("Hợp nhất các bounding box gần nhau...")
        text_areas = merge_nearby_boxes(text_areas, args.horizontal_threshold, args.vertical_threshold)
    
    if not text_areas:
        print("Không phát hiện được vùng văn bản nào")
        return
    
    # Lưu ảnh với các vùng đã phát hiện
    save_detected_boxes_image(args.image_path, text_areas, paths["detected_boxes"])
    
    # Bước 3: Cắt vùng bounding box và tô trắng
    print("\n----- Bước 3: Cắt vùng bounding box và tô trắng -----")
    # Cắt vùng văn bản
    cropped_images = crop_text_areas(image, text_areas, paths["crops_dir"])
    
    # Tô trắng vùng văn bản
    whitened_image = whiten_text_areas(image, text_areas)
    cv2.imwrite(paths["whitened_image"], whitened_image)
    
    # Bước 4: Tiền xử lý ảnh
    processed_images = cropped_images
    if not args.skip_preprocess:
        print("\n----- Bước 4: Tiền xử lý ảnh -----")
        processed_images = process_images(cropped_images, paths["preprocessed_dir"], args.show)
    else:
        print("\n----- Bỏ qua bước 4: Tiền xử lý ảnh -----")
    
    # Bước 5: Trích xuất văn bản từ các ảnh đã xử lý
    print("\n----- Bước 5: Trích xuất văn bản -----")
    processed_crops = process_crops(processed_images, api_key, paths["extracted_texts"])
    
    # Bước 6: Dịch văn bản đã trích xuất
    print("\n----- Bước 6: Dịch văn bản -----")
    # Sử dụng tên ngôn ngữ đầy đủ cho GPT
    translated_texts = process_translations(processed_crops, args.target_lang, api_key, paths["translated_texts"])
    
    # Bước 7: Vẽ văn bản đã dịch vào vùng crop
    print("\n----- Bước 7: Vẽ văn bản đã dịch -----")
    processed_images = process_translated_images(
        translated_texts,
        paths["crops_dir"],
        paths["translated_dir"],
        args.bg_color,
        args.text_color,
        0  # Kích thước font tự động
    )
    
    # Bước 8: Chèn vùng đã xử lý vào ảnh gốc
    print("\n----- Bước 8: Chèn vùng đã xử lý vào ảnh gốc -----")
    result_image = merge_translated_regions(whitened_image, processed_images, cropped_images)
    cv2.imwrite(paths["final_image"], result_image)
    
    end_time = time.time()
    print(f"\nTổng thời gian xử lý: {end_time - start_time:.2f} giây")
    print(f"Đã lưu ảnh kết quả cuối cùng vào: {paths['final_image']}")
    
    # Hiển thị kết quả nếu yêu cầu
    if args.show:
        cv2.imshow('Original Image', image)
        cv2.imshow('Translated Image', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    print("\n===== QUY TRÌNH HOÀN THÀNH =====\n")

if __name__ == "__main__":
    main() 