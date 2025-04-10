#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bước 5: Trích xuất văn bản từ ảnh đã tiền xử lý sử dụng OpenAI GPT-4o-mini
"""

import os
import argparse
import base64
import requests
import json
import time
import cv2
from dotenv import load_dotenv

# Tải biến môi trường
load_dotenv()

def parse_args():
    """Phân tích tham số dòng lệnh."""
    parser = argparse.ArgumentParser(description='Trích xuất văn bản từ ảnh đã xử lý')
    parser.add_argument('--crop_dir', required=True, help='Thư mục chứa các ảnh đã cắt')
    parser.add_argument('--processed_dir', help='Thư mục chứa các ảnh đã tiền xử lý')
    parser.add_argument('--processed_info', help='Đường dẫn đến file chứa thông tin các ảnh đã xử lý')
    parser.add_argument('--cropped_info', help='Đường dẫn đến file chứa thông tin các ảnh đã cắt')
    parser.add_argument('--output_file', default='extracted_texts.txt', help='File lưu văn bản đã trích xuất')
    parser.add_argument('--api_key', help='OpenAI API key (nếu không cung cấp, sẽ lấy từ biến môi trường)')
    return parser.parse_args()

def load_processed_info(processed_info_path):
    """
    Đọc thông tin các ảnh đã xử lý từ file.
    
    Args:
        processed_info_path: Đường dẫn đến file thông tin
        
    Returns:
        List thông tin các ảnh đã xử lý
    """
    processed_images = []
    
    try:
        with open(processed_info_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) >= 7:
                    index = int(parts[0])
                    crop_path = parts[1]
                    processed_path = parts[2]
                    box_str = parts[3]
                    rect_str = parts[4]
                    text = parts[5]
                    confidence = float(parts[6])
                    
                    # Chuyển đổi chuỗi box thành danh sách tọa độ
                    coords = box_str.split(',')
                    box = []
                    for i in range(0, len(coords), 2):
                        box.append([float(coords[i]), float(coords[i+1])])
                    
                    # Chuyển đổi chuỗi rect thành danh sách
                    rect_parts = rect_str.split(',')
                    rect = [int(rect_parts[0]), int(rect_parts[1]), int(rect_parts[2]), int(rect_parts[3])]
                    
                    processed_images.append({
                        'index': index,
                        'crop_path': crop_path,
                        'processed_path': processed_path,
                        'box': box,
                        'rect': rect,
                        'text': text,
                        'confidence': confidence
                    })
        
        print(f"Đã đọc thông tin {len(processed_images)} ảnh đã xử lý từ {processed_info_path}")
    
    except Exception as e:
        print(f"Lỗi khi đọc file thông tin các ảnh đã xử lý: {e}")
    
    return processed_images

def load_cropped_info(cropped_info_path):
    """
    Đọc thông tin các ảnh đã cắt từ file.
    
    Args:
        cropped_info_path: Đường dẫn đến file thông tin
        
    Returns:
        List thông tin các ảnh đã cắt
    """
    cropped_images = []
    
    try:
        with open(cropped_info_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) >= 6:
                    index = int(parts[0])
                    path = parts[1]
                    box_str = parts[2]
                    rect_str = parts[3]
                    text = parts[4]
                    confidence = float(parts[5])
                    
                    # Chuyển đổi chuỗi box thành danh sách tọa độ
                    coords = box_str.split(',')
                    box = []
                    for i in range(0, len(coords), 2):
                        box.append([float(coords[i]), float(coords[i+1])])
                    
                    # Chuyển đổi chuỗi rect thành danh sách
                    rect_parts = rect_str.split(',')
                    rect = [int(rect_parts[0]), int(rect_parts[1]), int(rect_parts[2]), int(rect_parts[3])]
                    
                    cropped_images.append({
                        'path': path,
                        'index': index,
                        'box': box,
                        'rect': rect,
                        'text': text,
                        'confidence': confidence
                    })
        
        print(f"Đã đọc thông tin {len(cropped_images)} ảnh đã cắt từ {cropped_info_path}")
    
    except Exception as e:
        print(f"Lỗi khi đọc file thông tin các ảnh đã cắt: {e}")
    
    return cropped_images

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

def get_image_files(directory):
    """
    Lấy danh sách các file ảnh trong thư mục.
    
    Args:
        directory: Đường dẫn thư mục
        
    Returns:
        List đường dẫn các file ảnh
    """
    image_files = []
    
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_files.append(os.path.join(directory, filename))
    
    # Sắp xếp theo số thứ tự trong tên file
    def extract_number(filename):
        try:
            # Trích xuất số từ tên file (crop_X.jpg hoặc processed_X.jpg)
            parts = os.path.basename(filename).split('_')
            if len(parts) > 1:
                return int(parts[1].split('.')[0])
            return 0
        except (IndexError, ValueError):
            return 0
    
    return sorted(image_files, key=extract_number)

def main():
    # Phân tích tham số
    args = parse_args()
    
    # Lấy API key
    api_key = args.api_key or os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("Lỗi: Không tìm thấy OpenAI API key")
        return False
    
    # Kiểm tra thư mục ảnh đã cắt
    if not os.path.exists(args.crop_dir):
        print(f"Lỗi: Không tìm thấy thư mục {args.crop_dir}")
        return False
    
    # Ưu tiên sử dụng ảnh đã tiền xử lý
    images_to_process = []
    
    # Kiểm tra thông tin ảnh đã xử lý
    if args.processed_dir and os.path.exists(args.processed_dir):
        processed_info_path = args.processed_info or os.path.join(args.processed_dir, "processed_info.txt")
        if os.path.exists(processed_info_path):
            # Đọc từ file thông tin ảnh đã xử lý
            images_to_process = load_processed_info(processed_info_path)
            print(f"Sử dụng ảnh đã tiền xử lý từ {args.processed_dir}")
    
    # Nếu không có thông tin ảnh đã xử lý, sử dụng thông tin ảnh đã cắt
    if not images_to_process:
        print("Không tìm thấy thông tin ảnh đã tiền xử lý, sử dụng ảnh đã cắt...")
        
        cropped_info_path = args.cropped_info or os.path.join(args.crop_dir, "cropped_info.txt")
        if os.path.exists(cropped_info_path):
            # Đọc từ file thông tin ảnh đã cắt
            images_to_process = load_cropped_info(cropped_info_path)
        else:
            # Nếu không có file thông tin, tạo từ danh sách file ảnh
            print(f"Không tìm thấy file thông tin {cropped_info_path}, đang tạo thông tin từ các file ảnh...")
            
            # Ưu tiên sử dụng ảnh đã tiền xử lý nếu có
            source_dir = args.processed_dir if args.processed_dir and os.path.exists(args.processed_dir) else args.crop_dir
            image_files = get_image_files(source_dir)
            
            for idx, image_path in enumerate(image_files):
                # Đọc ảnh để lấy kích thước
                image = cv2.imread(image_path)
                if image is not None:
                    height, width = image.shape[:2]
                    
                    # Lấy tên file từ đường dẫn
                    filename = os.path.basename(image_path)
                    
                    # Tạo thông tin
                    is_processed = "processed_" in filename.lower()
                    image_info = {
                        'index': idx + 1,
                        'text': f"Unknown text in {filename}",
                        'confidence': 0.0
                    }
                    
                    if is_processed:
                        # Ảnh đã tiền xử lý
                        crop_filename = filename.replace("processed_", "crop_")
                        crop_path = os.path.join(args.crop_dir, crop_filename)
                        if os.path.exists(crop_path):
                            image_info['crop_path'] = crop_path
                        image_info['processed_path'] = image_path
                    else:
                        # Ảnh đã cắt
                        image_info['path'] = image_path
                    
                    # Tạo thông tin giả về bounding box (hình chữ nhật đầy đủ)
                    image_info['box'] = [[0, 0], [width, 0], [width, height], [0, height]]
                    image_info['rect'] = [0, 0, width, height]
                    
                    images_to_process.append(image_info)
    
    if not images_to_process:
        print("Không có ảnh nào để xử lý")
        return False
    
    # Xử lý các ảnh
    output_file = os.path.join(os.path.dirname(args.crop_dir), args.output_file)
    processed_crops = process_crops(images_to_process, api_key, output_file)
    
    print(f"\nĐã trích xuất văn bản từ {len(processed_crops)} ảnh")
    
    return True

if __name__ == "__main__":
    main() 