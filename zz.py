def preprocess_image(image_path, output_path=None):
    """
    Tiền xử lý hình ảnh để cải thiện kết quả OCR.
    
    Args:
        image_path: Đường dẫn đến ảnh gốc
        output_path: Đường dẫn lưu ảnh đã xử lý (nếu None, sẽ tạo tên file tạm)
        
    Returns:
        str: Đường dẫn đến ảnh đã xử lý
    """
    print("Đang tiền xử lý ảnh để tăng độ chính xác...")
    
    if output_path is None:
        output_path = f"temp_processed_{os.path.basename(image_path)}"
    
    try:
        # Đọc ảnh
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Không thể đọc ảnh từ {image_path}")
            return image_path
        
        # Chuyển sang ảnh xám
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Áp dụng bộ lọc Gaussian để giảm nhiễu
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Tăng cường độ tương phản bằng CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blur)
        
        # Áp dụng ngưỡng thích ứng
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Mở hình ảnh (erode sau đó dilate) để giảm nhiễu
        kernel = np.ones((1, 1), np.uint8)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Lưu ảnh đã xử lý
        cv2.imwrite(output_path, opening)
        print(f"Đã lưu ảnh đã tiền xử lý vào {output_path}")
        
        return output_path
    except Exception as e:
        logger.error(f"Lỗi khi tiền xử lý ảnh: {e}")
        return image_path#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Chương trình Xử lý Văn bản từ Ảnh với OpenAI GPT-4o-mini và PaddleOCR

Quy trình:
1. Xác định ngôn ngữ trong ảnh bằng GPT-4o-mini
2. Trích xuất văn bản và bounding box với PaddleOCR
3. Chỉnh sửa chính tả và định dạng văn bản với GPT-4o-mini
4. Ghép kết quả với bounding box
5. Xuất kết quả

Yêu cầu:
    - paddle==2.5.2
    - paddleocr>=2.6.0.1
    - opencv-python
    - numpy
    - requests

Cách sử dụng:
    python ocr_with_gpt.py --image_path anh.jpg --api_key YOUR_OPENAI_API_KEY --output_path result.jpg
"""

import os
import argparse
import cv2
import numpy as np
import logging
import time
import base64
import requests
import json

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def timer(name):
    """Decorator đo thời gian thực thi của hàm."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            logger.info(f"{name}: {end_time - start_time:.2f} giây")
            return result
        return wrapper
    return decorator

def parse_args():
    """Phân tích tham số dòng lệnh."""
    parser = argparse.ArgumentParser(description='Xử lý văn bản từ ảnh với GPT-4o-mini và PaddleOCR')
    
    parser.add_argument('--image_path', required=True, help='Đường dẫn đến ảnh đầu vào')
    parser.add_argument('--output_path', default='output_result.jpg', help='Đường dẫn để lưu ảnh kết quả')
    parser.add_argument('--bbox_file', default='bbox_coordinates.txt', help='Đường dẫn để lưu tọa độ bounding box')
    parser.add_argument('--api_key', help='OpenAI API key')
    parser.add_argument('--show', action='store_true', help='Hiển thị kết quả')
    
    return parser.parse_args()

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

@timer("Bước 1: Xác định ngôn ngữ")
def detect_language(image_path, api_key):
    """
    Xác định ngôn ngữ trong ảnh bằng GPT-4o-mini.
    
    Args:
        image_path: Đường dẫn đến ảnh
        api_key: OpenAI API key
        
    Returns:
        str: Mã ngôn ngữ (vi, en, fr...)
    """
    print("Bước 1: Đang xác định ngôn ngữ trong ảnh...")
    
    # Mã hóa ảnh thành Base64
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    
    # Chuẩn bị payload
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "What language is used in this image? Reply with only the language code (e.g., 'vi', 'en', 'fr')."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]}
        ],
        "max_tokens": 10
    }
    
    # Gọi API
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        response.raise_for_status()  # Kiểm tra lỗi HTTP
        result = response.json()
        
        # Lấy mã ngôn ngữ
        lang_code = result['choices'][0]['message']['content'].strip().lower()
        
        # Loại bỏ các ký tự không phải chữ cái
        lang_code = ''.join(c for c in lang_code if c.isalpha())
        
        print(f"Ngôn ngữ được xác định: {lang_code}")
        return lang_code
    except Exception as e:
        logger.error(f"Lỗi khi xác định ngôn ngữ: {e}")
        # Fallback về tiếng Anh
        print("Không thể xác định ngôn ngữ, sử dụng tiếng Anh mặc định")
        return 'en'

@timer("Bước 2: Trích xuất văn bản")
def extract_text_with_paddleocr(image_path, lang):
    """
    Trích xuất văn bản và bounding box bằng PaddleOCR.
    
    Args:
        image_path: Đường dẫn đến ảnh
        lang: Mã ngôn ngữ (vi, en, ch...)
        
    Returns:
        List[Dict]: Danh sách kết quả [{"bbox": box, "text": text, "confidence": confidence}, ...]
    """
    print("Bước 2: Đang trích xuất văn bản và bounding box...")
    
    try:
        from paddleocr import PaddleOCR
        
        # Cấu hình PaddleOCR tối ưu cho ngôn ngữ
        config = {
            'use_angle_cls': True,
            'lang': lang,
            'use_gpu': False,
            'show_log': False
        }
        
        # Tối ưu hóa cho các ngôn ngữ cụ thể
        if lang == 'vi':
            config['det_db_box_thresh'] = 0.6
            config['det_db_thresh'] = 0.3
        
        # Khởi tạo PaddleOCR
        ocr = PaddleOCR(**config)
        
        # Thực hiện OCR
        result = ocr.ocr(image_path, cls=True)
        
        # Kiểm tra kết quả
        if not result or len(result) == 0:
            print("Không phát hiện văn bản trong ảnh")
            return []
        
        # Xử lý kết quả
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], list):
                det_results = result[0]
            else:
                det_results = result
        
        # Chuyển đổi thành danh sách đối tượng
        ocr_results = []
        for box, (text, confidence) in det_results:
            ocr_results.append({
                "bbox": box,
                "text": text,
                "confidence": confidence
            })
        
        # In kết quả
        print(f"Đã phát hiện {len(ocr_results)} dòng văn bản:")
        for idx, item in enumerate(ocr_results):
            print(f"  {idx+1}: {item['text']}")
        
        return ocr_results
    
    except ImportError:
        logger.error("Thiếu thư viện PaddleOCR. Cài đặt: pip install paddlepaddle==2.5.2 paddleocr")
        return []
    except Exception as e:
        logger.error(f"Lỗi khi trích xuất văn bản: {e}")
        return []

@timer("Bước 3: Trích xuất văn bản với GPT-4o-mini")
def extract_text_with_gpt(image_path, lang, api_key):
    """
    Trích xuất văn bản từ ảnh bằng GPT-4o-mini.
    
    Args:
        image_path: Đường dẫn đến ảnh
        lang: Mã ngôn ngữ
        api_key: OpenAI API key
        
    Returns:
        str: Văn bản được trích xuất
    """
    print("Bước 3: Đang trích xuất văn bản từ ảnh với GPT-4o-mini...")
    
    # Mã hóa ảnh thành Base64
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    
    # Chuẩn bị prompt
    lang_hint = f" in {lang} language" if lang else ""
    prompt = f"Extract all text from this image{lang_hint}. Include all visible text in the order it appears."
    
    # Chuẩn bị payload
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]}
        ],
        "temperature": 0.2,
        "max_tokens": 1000
    }
    
    # Gọi API
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        response.raise_for_status()  # Kiểm tra lỗi HTTP
        result = response.json()
        
        # Lấy văn bản
        extracted_text = result['choices'][0]['message']['content'].strip()
        
        print("Văn bản được trích xuất bởi GPT-4o-mini:")
        print("-" * 50)
        print(extracted_text)
        print("-" * 50)
        
        return extracted_text
    
    except Exception as e:
        logger.error(f"Lỗi khi trích xuất văn bản với GPT-4o-mini: {e}")
        return ""

@timer("Bước 4: Chỉnh sửa văn bản")
def correct_texts(ocr_results, extracted_text, lang, api_key):
    """
    Chỉnh sửa văn bản sử dụng GPT-4o-mini dựa trên văn bản đã trích xuất.
    
    Args:
        ocr_results: Kết quả từ PaddleOCR
        extracted_text: Văn bản đã trích xuất bởi GPT-4o-mini
        lang: Mã ngôn ngữ
        api_key: OpenAI API key
        
    Returns:
        List[str]: Danh sách văn bản đã chỉnh sửa
    """
    print("Bước 4: Đang chỉnh sửa văn bản dựa trên kết quả trích xuất...")
    
    if not ocr_results:
        return []
    
    # Tạo danh sách văn bản
    texts = [item["text"] for item in ocr_results]
    num_items = len(texts)
    
    # Chuẩn bị prompt với ràng buộc rõ ràng về số lượng dòng
    lang_hint = f" in {lang} language" if lang else ""
    prompt = f"""I have an image with text{lang_hint} that was processed using OCR. The OCR system extracted exactly {num_items} text items with some errors:

Here are the {num_items} text items from OCR (with errors):
"""
    
    for i, text in enumerate(texts, 1):
        prompt += f"{i}. {text}\n"
    
    prompt += f"\nI also have a more accurate extraction of the complete text from the image by GPT-4o-mini:\n\n"
    prompt += extracted_text
    
    prompt += f"""

Please correct EACH of the {num_items} OCR text items based on the accurate extraction.

IMPORTANT REQUIREMENTS:
1. You MUST return EXACTLY {num_items} corrected text items
2. Follow the SAME ORDER as the original OCR items
3. Each corrected item should correspond to one of the original OCR items
4. Return ONLY the corrected text, ONE item per line, numbered from 1 to {num_items}
5. NO additional text, explanations or formatting
6. Preserve the basic structure of each original item while fixing errors

I need exactly {num_items} lines of output, not more, not less.
"""
    
    # Chuẩn bị payload
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,  # Giảm temperature để có kết quả nhất quán hơn
        "max_tokens": 1000
    }
    
    # Gọi API
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        response.raise_for_status()  # Kiểm tra lỗi HTTP
        result = response.json()
        
        # Xử lý kết quả
        corrected_text = result['choices'][0]['message']['content'].strip()
        
        # Xử lý các dòng trong kết quả
        corrected_lines = []
        for line in corrected_text.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Loại bỏ số thứ tự đầu dòng (nếu có)
            if line[0].isdigit() and '.' in line[:5]:
                line = line[line.find('.')+1:].strip()
            
            corrected_lines.append(line)
        
        # Kiểm tra số lượng dòng
        if len(corrected_lines) != len(texts):
            print(f"Cảnh báo: Số lượng dòng không khớp. Gốc: {len(texts)}, Sửa: {len(corrected_lines)}")
            print("Thử lại với yêu cầu nghiêm ngặt hơn...")
            
            # Thử lại với prompt nghiêm ngặt hơn
            strict_prompt = f"""CRITICAL TASK: I have exactly {num_items} text lines from OCR that need correction.

Original {num_items} OCR lines:
"""
            for i, text in enumerate(texts, 1):
                strict_prompt += f"{i}. {text}\n"
            
            strict_prompt += f"\nAccurate text extraction:\n{extracted_text}\n\n"
            strict_prompt += f"""
YOUR JOB: Correct each of the {num_items} OCR lines based on the accurate extraction.

STRICT REQUIREMENTS:
1. You MUST output EXACTLY {num_items} lines - no more, no less
2. Each line should be a corrected version of the corresponding OCR line
3. Format your response as a simple numbered list from 1 to {num_items}
4. No explanations or additional text

Example format:
1. [corrected text for line 1]
2. [corrected text for line 2]
...and so on until line {num_items}

This is critical: I need EXACTLY {num_items} numbered lines in your response.
"""
            
            payload["messages"][0]["content"] = strict_prompt
            payload["temperature"] = 0.0  # Giảm temperature xuống 0
            
            try:
                response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload
                )
                
                response.raise_for_status()
                result = response.json()
                corrected_text = result['choices'][0]['message']['content'].strip()
                
                # Xử lý các dòng trong kết quả mới
                corrected_lines = []
                for line in corrected_text.split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Loại bỏ số thứ tự đầu dòng (nếu có)
                    if line[0].isdigit() and '.' in line[:5]:
                        line = line[line.find('.')+1:].strip()
                    
                    corrected_lines.append(line)
                
                # Kiểm tra lại số lượng dòng
                if len(corrected_lines) != len(texts):
                    print(f"Vẫn không khớp số dòng sau khi thử lại: {len(corrected_lines)} vs {len(texts)}")
                    print("Sử dụng văn bản gốc")
                    return texts
            except Exception as e:
                logger.error(f"Lỗi khi thử lại: {e}")
                return texts
        
        # In kết quả chỉnh sửa
        print("Văn bản sau khi chỉnh sửa:")
        for idx, (original, corrected) in enumerate(zip(texts, corrected_lines)):
            print(f"  {idx+1}:")
            print(f"    Gốc: {original}")
            print(f"    Sửa: {corrected}")
        
        return corrected_lines
    
    except Exception as e:
        logger.error(f"Lỗi khi chỉnh sửa văn bản: {e}")
        # Fallback về văn bản gốc
        return texts

@timer("Bước 5: Ghép kết quả")
def combine_results(ocr_results, corrected_texts):
    """
    Ghép kết quả chỉnh sửa với bounding box.
    
    Args:
        ocr_results: Kết quả từ PaddleOCR
        corrected_texts: Văn bản đã chỉnh sửa
        
    Returns:
        List[Dict]: Kết quả cuối cùng
    """
    print("Bước 5: Đang ghép kết quả chỉnh sửa với bounding box...")
    
    # Kiểm tra số lượng phần tử
    if len(ocr_results) != len(corrected_texts):
        print("Số lượng dòng không khớp, sử dụng văn bản gốc")
        return ocr_results
    
    # Ghép kết quả
    combined_results = []
    for i, item in enumerate(ocr_results):
        combined_results.append({
            "bbox": item["bbox"],
            "text": corrected_texts[i],
            "confidence": item["confidence"]
        })
    
    return combined_results

@timer("Bước 6: Lưu kết quả")
def save_results(image_path, combined_results, original_results, extracted_text, output_path, bbox_file):
    """
    Lưu kết quả cuối cùng.
    
    Args:
        image_path: Đường dẫn đến ảnh gốc
        combined_results: Kết quả cuối cùng
        original_results: Kết quả gốc
        extracted_text: Văn bản đã trích xuất bởi GPT-4o-mini
        output_path: Đường dẫn lưu ảnh kết quả
        bbox_file: Đường dẫn lưu thông tin bounding box
    """
    print("Bước 6: Đang lưu kết quả...")
    
    if not combined_results:
        print("Không có kết quả để lưu")
        return
    
    try:
        from paddleocr import draw_ocr
        
        # Đọc ảnh gốc
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Không thể đọc ảnh từ {image_path}")
            return
        
        # Tìm font hiển thị
        font_path = find_font()
        
        # Chuẩn bị dữ liệu cho draw_ocr
        boxes = [item["bbox"] for item in combined_results]
        txts = [item["text"] for item in combined_results]
        scores = [item["confidence"] for item in combined_results]
        
        # Vẽ kết quả OCR lên ảnh
        im_show = draw_ocr(image, boxes, txts, scores, font_path=font_path)
        cv2.imwrite(output_path, im_show)
        print(f"Đã lưu ảnh kết quả vào {output_path}")
        
        # Lưu kết quả vào file văn bản
        with open(bbox_file, 'w', encoding='utf-8') as f:
            f.write("="*50 + "\n")
            f.write(" DANH SÁCH BOUNDING BOX VÀ VĂN BẢN ĐÃ CHỈNH SỬA \n")
            f.write("="*50 + "\n\n")
            
            # Lưu văn bản GPT trích xuất
            f.write("VĂN BẢN TRÍCH XUẤT BỞI GPT-4O-MINI:\n")
            f.write("-" * 50 + "\n")
            f.write(extracted_text + "\n\n")
            f.write("-" * 50 + "\n\n")
            
            # Lưu từng bounding box và văn bản
            for idx, (combined, original) in enumerate(zip(combined_results, original_results)):
                pts = np.array(combined["bbox"]).astype(np.int32).reshape(-1, 2)
                coord_str = " ".join([f"({p[0]},{p[1]})" for p in pts])
                f.write(f"Bounding box {idx+1}:\n")
                f.write(f"  Tọa độ: {coord_str}\n")
                f.write(f"  Văn bản gốc (PaddleOCR): {original['text']}\n")
                f.write(f"  Văn bản đã sửa: {combined['text']}\n\n")
            
            f.write("="*50 + "\n")
        
        print(f"Đã lưu tọa độ bounding box vào {bbox_file}")
        
        # Lưu kết quả dạng JSON
        json_output = {
            "language": original_results[0].get("lang", "unknown"),
            "gpt_extracted_text": extracted_text,
            "results": [
                {
                    "bbox": item["bbox"],
                    "text_original": orig["text"],
                    "text_corrected": item["text"],
                    "confidence": item["confidence"]
                }
                for item, orig in zip(combined_results, original_results)
            ]
        }
        
        json_file = f"{os.path.splitext(output_path)[0]}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_output, f, ensure_ascii=False, indent=2)
        
        print(f"Đã lưu kết quả dạng JSON vào {json_file}")
    
    except ImportError:
        logger.error("Thiếu thư viện PaddleOCR để vẽ kết quả")
    except Exception as e:
        logger.error(f"Lỗi khi lưu kết quả: {e}")

def display_results(output_path):
    """Hiển thị kết quả."""
    try:
        img = cv2.imread(output_path)
        cv2.imshow('OCR Result', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        logger.error(f"Không thể hiển thị ảnh: {e}")

@timer("Tổng thời gian xử lý")
def process_image(image_path, api_key, output_path, bbox_file, show=False):
    """
    Xử lý văn bản từ ảnh.
    
    Args:
        image_path: Đường dẫn đến ảnh
        api_key: OpenAI API key
        output_path: Đường dẫn lưu ảnh kết quả
        bbox_file: Đường dẫn lưu tọa độ bounding box
        show: Hiển thị kết quả
    """
    print("\n===== BẮT ĐẦU QUY TRÌNH XỬ LÝ VĂN BẢN TỪ ẢNH =====\n")
    
    # Kiểm tra file đầu vào
    if not os.path.exists(image_path):
        print(f"Lỗi: Không tìm thấy ảnh {image_path}")
        return
    
    # Bước 1: Xác định ngôn ngữ trong ảnh
    lang = detect_language(image_path, api_key)
    
    # Bước 2: Trích xuất văn bản và bounding box bằng PaddleOCR
    ocr_results = extract_text_with_paddleocr(image_path, lang)
    
    if not ocr_results:
        print("Không phát hiện văn bản trong ảnh")
        return
    
    # Lưu thông tin ngôn ngữ vào kết quả
    for item in ocr_results:
        item["lang"] = lang
    
    # Bước 3: Trích xuất văn bản từ ảnh bằng GPT-4o-mini
    extracted_text = extract_text_with_gpt(image_path, lang, api_key)
    
    # Bước 4: Chỉnh sửa văn bản từ PaddleOCR dựa trên văn bản từ GPT
    corrected_texts = correct_texts(ocr_results, extracted_text, lang, api_key)
    
    # Bước 5: Ghép kết quả
    combined_results = combine_results(ocr_results, corrected_texts)
    
    # Bước 6: Lưu kết quả
    save_results(image_path, combined_results, ocr_results, extracted_text, output_path, bbox_file)
    
    # Hiển thị kết quả nếu yêu cầu
    if show:
        display_results(output_path)
    
    print("\n===== QUY TRÌNH XỬ LÝ HOÀN TẤT =====\n")

def main():
    """Hàm chính của chương trình."""
    args = parse_args()
    
    # Lấy API key từ tham số hoặc biến môi trường
    api_key = args.api_key or os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("Lỗi: Thiếu OpenAI API key. Cung cấp qua --api_key hoặc biến môi trường OPENAI_API_KEY")
        return
    
    # Xử lý ảnh
    process_image(args.image_path, api_key, args.output_path, args.bbox_file, args.show)

if __name__ == "__main__":
    main()